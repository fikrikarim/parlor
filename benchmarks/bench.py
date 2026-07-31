"""End-to-end latency benchmark — connects to the running server over WebSocket.

Start the server first:  uv run parlor
Then run:                uv run python benchmarks/bench.py --label baseline --out benchmarks/results/baseline.json

Uses real spoken audio (see fixtures.py). Measures client-perceived latency:
t0 is the moment the utterance is sent (≈ VAD speech-end in real usage; the
VAD redemption delay is additive on top and reported separately by the server
config). Compare two runs with:  uv run python benchmarks/compare.py A.json B.json

Works against both the legacy protocol (single "text" message) and the
streaming protocol ("text_delta" / "turn_incomplete" / "turn_final").
Correctness and quality live in the e2e test suite:  uv run pytest
"""

import argparse
import asyncio
import contextlib
import json
import subprocess
import time
import wave
from pathlib import Path

import websockets

import fixtures
from compare import median

SERVER_URL = "ws://localhost:8000/ws"

PERF_TESTS = [
    # (name, fixture audio | None, image mode (False/True/"prefetch"), text | None)
    # "prefetch" sends the frame ~1s before the audio, like the real client
    # does at VAD speech start.
    ("text_only", None, False, "Tell me a fun fact about the ocean."),
    ("audio_short", "capital_france", False, None),
    ("audio_long", "long_question", False, None),
    ("audio_short_image", "capital_france", True, None),
    ("audio_long_image", "long_question", True, None),
    ("audio_short_prefetch", "capital_france", "prefetch", None),
    ("audio_long_prefetch", "long_question", "prefetch", None),
    # "chunk_stream": speech streamed in ~3s chunks during the utterance
    # (llama.cpp overlap prefill); t0 is the final-tail send = speech end.
    ("audio_long_overlap", "long_question", "chunk_stream", None),
]

RUNS_PER_TEST = 2


def connect(url: str):
    """Shared connect params: a large max_size for base64 audio/image payloads."""
    return websockets.connect(url, max_size=64 * 1024 * 1024)


def fixture_duration_s(name: str) -> float:
    with wave.open(str(fixtures.FIXTURES_DIR / f"{name}.wav"), "rb") as w:
        return round(w.getnframes() / w.getframerate(), 2)


def build_payload(fixture: str | None, with_image: bool, text: str | None) -> dict:
    payload = {}
    if fixture:
        payload["audio"] = fixtures.load_wav_b64(fixture)
    if with_image:
        payload["image"] = fixtures.make_image_b64()
    if text:
        payload["text"] = text
    return payload


async def run_turn(ws, payload: dict, timeout: float = 45) -> dict:
    """Send one turn and collect messages until the turn ends."""
    t0 = time.time()
    await ws.send(json.dumps(payload))

    r = {
        "t_text": None,        # first visible text (text or text_delta)
        "t_first_audio": None, # first audio_chunk received
        "t_total": None,       # audio_end / turn_incomplete
        "text": "",
        "transcription": None,
        "marker": None,        # streaming protocol: complete / incomplete_short / incomplete_long
        "server": {},          # server-reported timings (llm_time, tts_time, prefill_s, ...)
        "audio_chunks": 0,
    }

    deadline = t0 + timeout
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.time()))
        except TimeoutError:
            r["timed_out"] = True  # keep partial metrics, don't kill the suite
            break
        msg = json.loads(raw)
        now = time.time() - t0
        mtype = msg.get("type")

        if mtype in ("text", "text_delta"):
            if r["t_text"] is None:
                r["t_text"] = now
            r["text"] += msg.get("text", "")
            for k in ("llm_time", "prefill_s", "decode_s"):
                if k in msg:
                    r["server"][k] = msg[k]
            if msg.get("transcription"):
                r["transcription"] = msg["transcription"]
        elif mtype == "audio_chunk":
            if r["t_first_audio"] is None:
                r["t_first_audio"] = now
            r["audio_chunks"] += 1
        elif mtype == "turn_incomplete":
            r["marker"] = f"incomplete_{msg.get('kind', 'short')}"
            r["t_total"] = now
            for k, v in msg.items():
                if isinstance(v, (int, float)) and (k.endswith("_s") or k.endswith("_time")):
                    r["server"][k] = v
            break
        elif mtype == "turn_final":
            if msg.get("transcription"):
                r["transcription"] = msg["transcription"]
            r["server"].update(msg.get("timings", {}))
            if not msg.get("spoke", True):
                # No audio_end will follow — this is the terminal frame.
                r["marker"] = r["marker"] or "complete"
                r["t_total"] = now
                break
        elif mtype == "audio_end":
            r["t_total"] = now
            if r["marker"] is None:
                r["marker"] = "complete"
            if "tts_time" in msg:
                r["server"]["tts_time"] = msg["tts_time"]
            break

    for k in ("t_text", "t_first_audio", "t_total"):
        if r[k] is not None:
            r[k] = round(r[k], 3)
    return r


async def run_perf_suite(url: str) -> dict:
    results = {}
    # Warmup: absorb first-turn GPU/pipeline compilation so runs are comparable.
    async with connect(url) as ws:
        await run_turn(ws, {"text": "Hi!"}, timeout=120)
    print("warmup done")

    for name, fixture, with_image, text in PERF_TESTS:
        runs = []
        for i in range(RUNS_PER_TEST):
            async with connect(url) as ws:
                if with_image == "prefetch":
                    await ws.send(json.dumps({"type": "frame", "image": fixtures.make_image_b64()}))
                    await asyncio.sleep(1.0)  # frame prefills while the "user speaks"
                if with_image == "chunk_stream":
                    chunks = fixtures.load_wav_chunks_b64(fixture)
                    for seq, c in enumerate(chunks[:-1]):
                        await ws.send(json.dumps({"type": "speech_chunk", "seq": seq, "audio": c}))
                        await asyncio.sleep(1.0)  # server primes while the "user speaks"
                    r = await run_turn(ws, {"audio": chunks[-1], "chunked": True})
                else:
                    r = await run_turn(ws, build_payload(fixture, with_image is True, text))
                runs.append(r)
            print(
                f"  {name} run {i + 1}: text={r['t_text']}s "
                f"first_audio={r['t_first_audio']}s total={r['t_total']}s"
            )
        entry = {"runs": runs}
        if fixture:
            entry["audio_duration_s"] = fixture_duration_s(fixture)
        results[name] = entry

    # Multi-turn on one connection (KV-cache growth across turns).
    multi = []
    async with connect(url) as ws:
        for i in range(3):
            r = await run_turn(ws, build_payload("capital_france", True, None))
            multi.append(r)
            print(f"  multi_turn {i + 1}: total={r['t_total']}s")
    results["multi_turn"] = {"runs": multi}
    return results


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip()
    except OSError:
        return "unknown"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=SERVER_URL)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    fixtures.generate_all()

    print(f"== perf suite ({args.label}) ==")
    perf = await run_perf_suite(args.url)

    out = {
        "label": args.label,
        "git_rev": git_rev(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "perf": perf,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")

    print("\n== summary ==")
    for name, entry in perf.items():
        med = median(r["t_total"] for r in entry["runs"])
        fmed = median(r["t_first_audio"] for r in entry["runs"])
        print(f"  {name:<20} first_audio={fmed}s total={med}s")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
