"""Turn-detection benchmark — which TURN_MODE judges "did the user finish?"
best, and how fast.

Scores the three modes in server.py against labelled human speech from
pipecat's smart-turn v3.2 test set (`endpoint_bool`: did the turn end?):

    smart      — the smart-turn-v3.2 ONNX classifier (acoustic, no LLM)
    marker     — Gemma starts its reply with FINISHED / WAIT
    two_phase  — a separate one-word Gemma request, then the response

Only real human recordings are used (`synthetic=false`). TTS-synthesized
clips read as finished no matter what the words are, which flatters the
LLM modes and unfairly punishes the acoustic one — the same effect that
makes tests/test_conversation.py::test_thinking_pause_is_held unrunnable.
Pass --synthetic to include them anyway; outside English they are often
all a language has (every Indonesian clip in the set is synthetic).

    # cache clips once, N per class per language (needs network)
    uv run python benchmarks/turnbench.py --fetch 100

    # the acoustic classifier — no LLM, so run it once for all models
    uv run python benchmarks/turnbench.py --modes smart \
        --out benchmarks/results/turn-smart.json

    # the LLM modes; each spawns its own llama-server on port 8099
    uv run python benchmarks/turnbench.py --model e4b --modes marker,two_phase \
        --out benchmarks/results/turn-e4b.json

Caveat: this is smart-turn's own test split, so it is in-domain for the
`smart` mode and its score here is optimistic. LiveKit's eot-bench
(Apache-2.0, 14 languages, real human-to-agent turns) is the independent
cross-check, and scores smart-turn v3.2 far more harshly.
"""

import argparse
import base64
import contextlib
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

import fixtures

DATASET = "pipecat-ai/smart-turn-data-v3.2-test"
FILTER_URL = "https://datasets-server.huggingface.co/filter"
CLIPS_DIR = Path(__file__).parent / "fixtures" / "turn"
MANIFEST = CLIPS_DIR / "manifest.json"

# GGUF repo, weights, mmproj — all three are Google's official QAT q4_0.
MODELS = {
    "e2b": ("google/gemma-4-E2B-it-qat-q4_0-gguf",
            "gemma-4-E2B_q4_0-it.gguf", "gemma-4-E2B-it-mmproj.gguf"),
    "e4b": ("google/gemma-4-E4B-it-qat-q4_0-gguf",
            "gemma-4-E4B_q4_0-it.gguf", "gemma-4-E4B-it-mmproj.gguf"),
    "12b": ("google/gemma-4-12B-it-qat-q4_0-gguf",
            "gemma-4-12b-it-qat-q4_0.gguf", "mmproj-gemma-4-12b-it-qat-q4_0.gguf"),
}

BENCH_PORT = "8099"  # not 8081, so a dev server can keep running alongside

# The two LLM-judged variants server.py used to ship as TURN_MODE=marker and
# TURN_MODE=two_phase. They live here now: they lost badly enough to be worth
# deleting from the server, but re-running them against the next Gemma is a
# one-line change, so the prompts stay reproducible rather than lost to git.
MARKER_SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone and may show you their camera. Your reply is spoken "
    "aloud, so write plain conversational text: 1-4 short sentences, no formatting.\n"
    "\n"
    "Your reply MUST start with exactly one of these words on its own line, "
    "judging the user's speech:\n"
    "- FINISHED — the user completed their thought. Continue with your spoken "
    "response on the next line.\n"
    "- WAIT — the user has not finished: they were cut off mid-sentence or are "
    "pausing to think. Say nothing else and let them continue.\n"
    "Also reply WAIT if the audio is just an echo of your own previous reply "
    "(the microphone picking up your voice) or is silence or noise.\n"
    "\n"
    "If the user sent audio, end your reply with a new line:\n"
    "###TRANSCRIPT: the exact words the user said\n"
    "\n"
    "Examples:\n"
    'User audio: "What\'s your favorite color?"\n'
    "You: FINISHED\n"
    "I really like deep blue. What about you?\n"
    "###TRANSCRIPT: What's your favorite color?\n"
    "\n"
    'User audio: "So the thing I wanted to say is"\n'
    "You: WAIT\n"
    "\n"
    'User audio: "Hmm, let me think about that for a second."\n'
    "You: WAIT\n"
)

MARKER_INSTRUCTION = (
    "The user just spoke to you. Start with FINISHED or WAIT, then respond to "
    "what they said. End with the ###TRANSCRIPT line."
)

DECISION_PROMPT = (
    "The user just spoke. Judge ONLY whether they finished their thought — "
    "do not answer them yet. Examples:\n"
    '"What is your favorite food?" -> FINISHED\n'
    '"I moved here last year and I want to know how I can make more friends." -> FINISHED\n'
    '"So the thing I wanted to ask is" -> WAIT\n'
    '"Hmm, let me think about that." -> WAIT\n'
    "Reply with exactly one word: FINISHED or WAIT."
)

# Close variants ("FINISH", trailing colon, markdown wrap) count too — they
# never open a real response in uppercase.
MARKER_RE = re.compile(r"[\s*_]*(FINISHED|FINISH|WAITING|WAIT)\b[:.]?[\s*_]*")


# ── clip fetching ─────────────────────────────────────────────────────────

def _query(lang: str, complete: bool, synthetic: bool, offset: int, length: int) -> dict:
    where = f"\"language\"='{lang}' AND \"endpoint_bool\"={'true' if complete else 'false'}"
    if not synthetic:
        where += ' AND "synthetic"=false'
    params = {
        "dataset": DATASET, "config": "default", "split": "train", "where": where,
        "orderby": '"id"',  # deterministic paging and a stable sample
        "offset": offset, "length": length,
    }
    url = FILTER_URL + "?" + urllib.parse.urlencode(params)
    problem = "no attempt made"
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read())
            if "error" not in data:
                return data
            problem = data["error"]  # a cold index builds server-side
        except urllib.error.HTTPError as e:
            problem = f"HTTP {e.code}"
        print(f"  datasets-server: {problem} (retry {attempt + 1})")
        time.sleep(15)
    raise RuntimeError(f"datasets-server never answered: {problem}")


def fetch_clips(langs: list[str], per_class: int, synthetic: bool) -> None:
    """Cache a balanced complete/incomplete sample as 16kHz mono WAVs."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lang in langs:
        for complete in (True, False):
            got = 0
            while got < per_class:
                page = _query(lang, complete, synthetic, got, min(100, per_class - got))
                rows = page["rows"]
                if not rows:
                    print(f"  {lang} complete={complete}: only {got} rows exist")
                    break
                print(f"  {lang} complete={complete}: {got + len(rows)}/{per_class} "
                      f"(of {page['num_rows_total']} available)")
                for entry in rows:
                    row = entry["row"]
                    path = CLIPS_DIR / f"{row['id']}.wav"
                    if not path.exists():
                        with urllib.request.urlopen(row["audio"][0]["src"], timeout=60) as r:
                            pcm, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
                        if pcm.ndim > 1:
                            pcm = pcm.mean(axis=1)
                        fixtures._write_wav(
                            path, fixtures._resample_linear(pcm, sr, fixtures.TARGET_SR),
                            fixtures.TARGET_SR)
                    manifest.append({"id": row["id"], "lang": lang,
                                     "complete": complete, "source": row["dataset"],
                                     "synthetic": row["synthetic"]})
                got += len(rows)
    # Merge, so real-speech and synthetic-language fetches can be combined.
    merged = {c["id"]: c for c in
              (json.loads(MANIFEST.read_text()) if MANIFEST.exists() else [])}
    merged.update({c["id"]: c for c in manifest})
    MANIFEST.write_text(json.dumps(list(merged.values()), indent=2))
    print(f"Cached {len(manifest)} clips ({len(merged)} total) in {CLIPS_DIR}")


def load_manifest(langs: list[str], limit: int | None) -> list[dict]:
    if not MANIFEST.exists():
        sys.exit("No clips cached — run with --fetch N first.")
    items = [c for c in json.loads(MANIFEST.read_text()) if c["lang"] in langs]
    if limit:  # keep the classes balanced when trimming
        keep, counts = [], {}
        for c in items:
            key = (c["lang"], c["complete"])
            if counts.get(key, 0) < limit // (2 * len(langs)):
                counts[key] = counts.get(key, 0) + 1
                keep.append(c)
        items = keep
    return items


# ── judges ────────────────────────────────────────────────────────────────

def judge_smart(detector, audio: np.ndarray, _b64: str) -> tuple[bool, float]:
    with contextlib.redirect_stdout(io.StringIO()):  # it logs every call
        complete, prob = detector.predict(audio)
    return complete, prob


def judge_marker(_detector, _audio, b64: str) -> tuple[bool, float]:
    """Stopped after the marker token: the reply text costs the same
    whichever way the marker went, so decision latency is what turn-taking
    actually feels."""
    messages = [
        {"role": "system", "content": MARKER_SYSTEM_PROMPT},
        {"role": "user", "content": [
            pipeline.audio_part(b64), pipeline.text_part(MARKER_INSTRUCTION)]},
    ]
    return _marker_verdict(llama.chat_blocking(messages, max_tokens=8))


def judge_two_phase(_detector, _audio, b64: str) -> tuple[bool, float]:
    messages = [
        {"role": "system", "content": server.SYSTEM_PROMPT},
        {"role": "user", "content": [
            pipeline.audio_part(b64), pipeline.text_part(DECISION_PROMPT)]},
    ]
    return _marker_verdict(llama.chat_blocking(messages, max_tokens=6))


def _marker_verdict(text: str) -> tuple[bool, float]:
    """A missing or garbled marker counts as 'complete' — the fallback the
    old StreamParser used when no marker arrived."""
    match = MARKER_RE.match(text.strip())
    complete = not match or match.group(1).startswith("FINISH")
    return complete, float(complete)


JUDGES = {"smart": judge_smart, "marker": judge_marker, "two_phase": judge_two_phase}


# ── scoring ───────────────────────────────────────────────────────────────

def percentile(values: list[float], pct: float) -> float:
    return round(float(np.percentile(values, pct)), 1) if values else 0.0


def score(results: list[dict]) -> dict:
    tp = sum(r["truth"] and r["pred"] for r in results)
    tn = sum(not r["truth"] and not r["pred"] for r in results)
    fp = sum(not r["truth"] and r["pred"] for r in results)
    fn = sum(r["truth"] and not r["pred"] for r in results)
    latencies = [r["ms"] for r in results]
    n_complete, n_incomplete = tp + fn, tn + fp
    return {
        "n": len(results),
        "accuracy": round((tp + tn) / len(results), 3),
        # Missing a finished turn = dead air; the user waits for a reply.
        "recall_complete": round(tp / n_complete, 3) if n_complete else None,
        # Cutting in on an unfinished turn — the interruption users hate.
        "recall_incomplete": round(tn / n_incomplete, 3) if n_incomplete else None,
        "interrupt_rate": round(fp / n_incomplete, 3) if n_incomplete else None,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "ms_p50": percentile(latencies, 50),
        "ms_p95": percentile(latencies, 95),
    }


def sweep_threshold(results: list[dict]) -> dict:
    """Accuracy vs the p(complete) cutoff in turn_detector.py. Raising it
    trades dead air (waiting on a finished turn) for interruptions, which
    are the more jarring failure — worth tuning against real speech."""
    if len({r["prob"] for r in results}) < 3:  # marker modes emit 0.0/1.0
        return {}
    sweep = {}
    for cut in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        scored = [{**r, "pred": r["prob"] > cut} for r in results]
        stats = score(scored)
        sweep[f"{cut:.1f}"] = {k: stats[k] for k in
                               ("accuracy", "recall_complete", "interrupt_rate")}
    return sweep


def score_by_lang(results: list[dict]) -> dict:
    langs = sorted({r["lang"] for r in results})
    if len(langs) < 2:
        return {}
    return {l: score([r for r in results if r["lang"] == l]) for l in langs}


def run_mode(mode: str, clips: list[dict], detector) -> list[dict]:
    judge = JUDGES[mode]
    results = []
    for i, clip in enumerate(clips, 1):
        raw = (CLIPS_DIR / f"{clip['id']}.wav").read_bytes()
        b64 = base64.b64encode(raw).decode()
        audio = pipeline.wav_to_float32(b64)
        t0 = time.time()
        pred, prob = judge(detector, audio, b64)
        results.append({"id": clip["id"], "lang": clip["lang"],
                        "truth": clip["complete"], "pred": bool(pred),
                        "prob": round(prob, 3), "ms": (time.time() - t0) * 1000})
        if i % 25 == 0 or i == len(clips):
            print(f"  {mode}: {i}/{len(clips)}")
    return results


def report(label: str, stats: dict) -> None:
    print(f"{label:<22} acc {stats['accuracy']:.3f}   "
          f"finished {stats['recall_complete']:.3f}   "
          f"unfinished {stats['recall_incomplete']:.3f}   "
          f"interrupts {stats['interrupt_rate']:.3f}   "
          f"{stats['ms_p50']:.0f}ms p50 / {stats['ms_p95']:.0f}ms p95")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, metavar="N",
                    help="cache N clips per class per language, then exit")
    ap.add_argument("--langs", default="eng", help="ISO 639-3 codes, comma separated")
    ap.add_argument("--synthetic", action="store_true",
                    help="also fetch TTS clips — the only kind that exists for most "
                         "non-English languages, but their prosody always reads as "
                         "finished, so acoustic scores from them are pessimistic")
    ap.add_argument("--modes", default="smart,marker,two_phase")
    ap.add_argument("--model", default="e2b", choices=list(MODELS))
    ap.add_argument("--limit", type=int, help="score only the first N clips")
    ap.add_argument("--out", help="write the full per-clip results here")
    args = ap.parse_args()

    langs = args.langs.split(",")
    if args.fetch:
        fetch_clips(langs, args.fetch, args.synthetic)
        return

    modes = args.modes.split(",")
    clips = load_manifest(langs, args.limit)
    print(f"{len(clips)} clips ({sum(c['complete'] for c in clips)} finished) "
          f"| model {args.model} | modes {', '.join(modes)}")

    if set(modes) - {"smart"}:
        from huggingface_hub import hf_hub_download
        repo, weights, mmproj = MODELS[args.model]
        os.environ["MODEL_PATH"] = hf_hub_download(repo, weights, local_files_only=True)
        os.environ["MMPROJ_PATH"] = hf_hub_download(repo, mmproj, local_files_only=True)
        os.environ["LLAMA_PORT"] = BENCH_PORT

    global llama, pipeline, server  # after the env above; llama reads LLAMA_PORT at import
    from parlor import llama, pipeline, server
    detector = None
    if "smart" in modes:
        from parlor.turn_detector import TurnDetector
        detector = TurnDetector()
    if set(modes) - {"smart"}:
        llama.start()

    try:
        out = {"model": args.model, "langs": langs, "modes": {}}
        for mode in modes:
            results = run_mode(mode, clips, detector)
            out["modes"][mode] = {"stats": score(results),
                                  "by_lang": score_by_lang(results),
                                  "threshold_sweep": sweep_threshold(results),
                                  "results": results}
    finally:
        llama.stop()

    print()
    for mode in modes:
        label = f"{mode} ({args.model})" if mode != "smart" else "smart (no LLM)"
        report(label, out["modes"][mode]["stats"])
        for lang, stats in out["modes"][mode]["by_lang"].items():
            report(f"  └ {lang}", stats)
        for cut, stats in out["modes"][mode]["threshold_sweep"].items():
            print(f"  └ p>{cut}                acc {stats['accuracy']:.3f}   "
                  f"finished {stats['recall_complete']:.3f}   "
                  f"interrupts {stats['interrupt_rate']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
