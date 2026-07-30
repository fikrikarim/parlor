"""Parlor — on-device, real-time multimodal AI (voice + vision).

LLM inference runs on llama.cpp (llama-server, spawned as a subprocess).
The server owns the conversation history and re-sends it every request;
llama-server's prefix cache makes that cheap, and it also enables two
speculative tricks: the camera frame and the user's speech (in ~3s chunks)
are pushed through cache-priming requests WHILE the user is still talking,
so the final request only pays for the tail of the utterance.
"""

import asyncio
import base64
import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import wave
from contextlib import asynccontextmanager
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # logs stream even when piped

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

try:  # raised by ws.send_text after the client goes away (uvicorn-specific)
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:  # pragma: no cover
    class ClientDisconnected(OSError):
        pass

DISCONNECT_ERRORS = (WebSocketDisconnect, ClientDisconnected)

import tts

from dotenv import load_dotenv
load_dotenv()

# Google's official QAT quant: q4_0 quality trained-in, faster than K-quants.
HF_GGUF_REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"
HF_GGUF_FILE = "gemma-4-E2B_q4_0-it.gguf"
HF_MMPROJ_FILE = "gemma-4-E2B-it-mmproj.gguf"

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8081"))
LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "")  # set to use an external server
LLAMA_CTX = int(os.environ.get("LLAMA_CTX", "16384"))

# Turn completeness is judged by the smart-turn audio classifier before the
# LLM is involved, so the prompt carries no FINISHED/WAIT machinery at all.
# Asking Gemma to judge it instead scores at chance on audio — see
# benchmarks/turnbench.py, which still reproduces those two variants.
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone and may show you their camera. Your replies are "
    "spoken aloud, so write plain conversational text without formatting."
)

# The reply streams straight into TTS, so the format is: response first
# (sentences are spoken as they decode), transcript last (it never delays
# first audio).
RESPOND_PROMPT = (
    "Respond to what the user said in their audio message: 1-4 short "
    "sentences, spoken aloud.{camera} Then end your reply with a new line: "
    "###TRANSCRIPT: followed by the exact words the user said."
)

NUDGE_PROMPT = (
    "(The user went quiet without finishing their thought. In one short, warm "
    "sentence, encourage them to continue. No transcript line.)"
)

TRANSCRIPT_TAG = "###TRANSCRIPT:"
SENTENCE_END_RE = re.compile(r"[.!?]+\s")
MAX_OUTPUT_TOKENS = 256

# Rotate history before the llama context fills. Rough token estimates are
# fine here — the guard just needs to fire before generation degrades.
CONTEXT_HEADROOM = 2000
AUDIO_TOKENS_PER_SEC = 32
IMAGE_TOKENS = 300

_DONE = object()

llama_proc = None
tts_backend = None
detector = None  # smart-turn end-of-turn classifier


def resolve_model_paths() -> tuple[str, str]:
    model = os.environ.get("MODEL_PATH", "")
    mmproj = os.environ.get("MMPROJ_PATH", "")
    if model and mmproj:
        return model, mmproj
    from huggingface_hub import hf_hub_download
    kw = {}
    try:
        model = model or hf_hub_download(HF_GGUF_REPO, HF_GGUF_FILE)
        mmproj = mmproj or hf_hub_download(HF_GGUF_REPO, HF_MMPROJ_FILE)
    except Exception:  # offline — use the local cache
        kw = {"local_files_only": True}
        model = model or hf_hub_download(HF_GGUF_REPO, HF_GGUF_FILE, **kw)
        mmproj = mmproj or hf_hub_download(HF_GGUF_REPO, HF_MMPROJ_FILE, **kw)
    return model, mmproj


def llama_host_port() -> tuple[str, int]:
    if LLAMA_URL:
        host = LLAMA_URL.split("//")[-1]
        h, _, p = host.partition(":")
        return h, int(p or 80)
    return "127.0.0.1", LLAMA_PORT


def start_llama_server():
    global llama_proc
    if LLAMA_URL:
        print(f"Using external llama-server at {LLAMA_URL}")
        return
    binary = shutil.which("llama-server")
    if not binary:
        raise RuntimeError("llama-server not found — install with: brew install llama.cpp")
    model, mmproj = resolve_model_paths()
    print(f"Starting llama-server with {Path(model).name} (ctx={LLAMA_CTX})...")
    llama_proc = subprocess.Popen(
        [binary, "-m", model, "--mmproj", mmproj, "-ngl", "99",
         "--port", str(LLAMA_PORT), "-c", str(LLAMA_CTX), "-np", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    host, port = llama_host_port()
    deadline = time.time() + 180
    while time.time() < deadline:
        if llama_proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {llama_proc.returncode}")
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            if ok:
                print("llama-server ready.")
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready in 180s")


def load_models():
    global tts_backend, detector
    start_llama_server()
    from turn_detector import TurnDetector
    detector = TurnDetector()
    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield
    if llama_proc:
        llama_proc.terminate()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


# ── llama-server chat API ─────────────────────────────────────────────────

def _chat_body(messages: list, max_tokens: int, stream: bool) -> dict:
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def chat_blocking(messages: list, max_tokens: int) -> str:
    """Non-streaming request; returns the message content ('' on discard)."""
    host, port = llama_host_port()
    conn = http.client.HTTPConnection(host, port, timeout=300)
    conn.request("POST", "/v1/chat/completions",
                 json.dumps(_chat_body(messages, max_tokens, stream=False)),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    if "error" in data:
        raise RuntimeError(f"llama-server: {data['error']}")
    return data["choices"][0]["message"].get("content") or ""


class ChatStream:
    """Streaming chat request, driven from an executor thread. cancel() is
    thread-safe and actually aborts generation server-side (the connection
    close is observed by llama-server)."""

    def __init__(self, messages: list, max_tokens: int):
        self.body = _chat_body(messages, max_tokens, stream=True)
        self.conn = None
        self.cancelled = False

    def run(self, on_delta):
        host, port = llama_host_port()
        self.conn = http.client.HTTPConnection(host, port, timeout=300)
        self.conn.request("POST", "/v1/chat/completions", json.dumps(self.body),
                          {"Content-Type": "application/json"})
        resp = self.conn.getresponse()
        if resp.status != 200:
            # Surface bad requests as errors: a silently-empty turn would get
            # stored in history and poison every subsequent request.
            body = resp.read()[:300]
            self.conn.close()
            raise RuntimeError(f"llama-server HTTP {resp.status}: {body!r}")
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                delta = json.loads(payload)["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    on_delta(text)
        except Exception as e:
            # Any failure here means the stream is dead — including
            # http.client's own cleanup racing a cancel() from another thread
            # (it can raise AttributeError from _close_conn). Truncation is
            # normal on abort; a genuinely dead llama-server surfaces on the
            # next request.
            if not self.cancelled:
                print(f"LLM stream ended early: {type(e).__name__}: {e}")
        finally:
            try:
                self.conn.close()
            except OSError:
                pass

    def cancel(self):
        self.cancelled = True
        try:
            if self.conn and self.conn.sock:
                self.conn.sock.shutdown(socket.SHUT_RDWR)
            if self.conn:
                self.conn.close()
        except OSError:
            pass


# ── content helpers ───────────────────────────────────────────────────────

def image_part(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}


def audio_part(b64: str) -> dict:
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


def valid_audio(b64: str | None) -> bool:
    """At least ~100ms of 16kHz s16 WAV — llama-server 400s on empty audio,
    and one bad message in history would poison every later request."""
    return bool(b64) and len(b64) * 3 // 4 > 44 + 3200


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4 + 8
            continue
        for p in content:
            if p["type"] == "text":
                total += len(p["text"]) // 4
            elif p["type"] == "input_audio":
                wav_bytes = len(p["input_audio"]["data"]) * 3 // 4
                total += (wav_bytes // 32000) * AUDIO_TOKENS_PER_SEC  # 16kHz s16
            else:
                total += IMAGE_TOKENS
        total += 8
    return total


# ── streaming turn parser (unchanged semantics from the litert version) ───

class StreamParser:
    """Incrementally parses '<response>\\n###TRANSCRIPT: <words>'.

    feed() returns complete response sentences as they become available;
    finalize() returns the trailing partial sentence and the transcript.
    """

    # Hold back enough of the tail to never TTS a partially-arrived tag.
    TAG_HOLDBACK = len(TRANSCRIPT_TAG) + 2

    def __init__(self):
        self.response = ""
        self.transcript = ""
        self._in_transcript = False
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        if self._in_transcript:
            self.transcript += delta
            return []

        self.response += delta
        tag_pos = self.response.find(TRANSCRIPT_TAG)
        if tag_pos != -1:
            self.transcript = self.response[tag_pos + len(TRANSCRIPT_TAG):]
            self.response = self.response[:tag_pos]
            self._in_transcript = True
        return self._complete_sentences()

    def _complete_sentences(self) -> list[str]:
        end = len(self.response)
        if not self._in_transcript:
            end = max(self._emitted, end - self.TAG_HOLDBACK)
        # A malformed transcript tag (e.g. missing colon) never matches
        # TRANSCRIPT_TAG — make sure we still never speak past a "###".
        hash_pos = self.response.find("###", self._emitted)
        if hash_pos != -1:
            end = min(end, hash_pos)
        sentences = []
        while True:
            m = SENTENCE_END_RE.search(self.response, self._emitted, end)
            if not m:
                break
            sentence = self.response[self._emitted:m.end()].strip()
            self._emitted = m.end()
            if sentence:
                sentences.append(sentence)
        return sentences

    def finalize(self) -> tuple[list[str], str | None]:
        sentences = self.feed("")
        # Cut any (possibly truncated) transcript tag — never speak it.
        tail = re.split(r"#{2,}", self.response[self._emitted:])[0].strip()
        transcript = self.transcript.strip() or None
        return sentences + ([tail] if tail else []), transcript


# ── turn execution ────────────────────────────────────────────────────────

async def run_turn(ws: WebSocket, messages: list, interrupted: asyncio.Event,
                   active: dict) -> str:
    """Stream one model turn: decode → sentences → TTS, all pipelined.
    Returns the raw generated text (stored verbatim in history so the next
    request gets a full prefix-cache hit)."""
    loop = asyncio.get_event_loop()
    t0 = time.time()
    timings: dict = {}

    chunk_q: asyncio.Queue = asyncio.Queue()
    stream = ChatStream(messages, MAX_OUTPUT_TOKENS)
    active["stream"] = stream
    raw = {"text": ""}

    def produce():
        try:
            def on_delta(text):
                raw["text"] += text
                loop.call_soon_threadsafe(chunk_q.put_nowait, text)
            stream.run(on_delta)
            loop.call_soon_threadsafe(chunk_q.put_nowait, _DONE)
        except Exception as e:  # surfaced to the consumer loop
            loop.call_soon_threadsafe(chunk_q.put_nowait, e)

    producer = loop.run_in_executor(None, produce)

    sentence_q: asyncio.Queue = asyncio.Queue()
    audio_state = {"started": False, "first_audio_at": None, "chunks": 0}

    async def tts_worker():
        while True:
            sentence = await sentence_q.get()
            if sentence is _DONE:
                return
            if interrupted.is_set():
                continue  # keep draining
            pcm = await loop.run_in_executor(None, lambda s=sentence: tts_backend.generate(s))
            if interrupted.is_set():
                continue
            if not audio_state["started"]:
                audio_state["started"] = True
                audio_state["first_audio_at"] = time.time()
                await ws.send_text(json.dumps({
                    "type": "audio_start",
                    "sample_rate": tts_backend.sample_rate,
                }))
            pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
            await ws.send_text(json.dumps({
                "type": "audio_chunk",
                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                "index": audio_state["chunks"],
            }))
            audio_state["chunks"] += 1

    tts_task = asyncio.create_task(tts_worker())
    parser = StreamParser()
    tts_started_at = None

    async def dispatch(sentences: list[str]):
        nonlocal tts_started_at
        for sentence in sentences:
            if tts_started_at is None:
                tts_started_at = time.time()
            await ws.send_text(json.dumps({"type": "text_delta", "text": sentence + " "}))
            sentence_q.put_nowait(sentence)

    try:
        while True:
            item = await chunk_q.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            if "prefill_s" not in timings:
                timings["prefill_s"] = round(time.time() - t0, 3)
            if not interrupted.is_set():
                await dispatch(parser.feed(item))

        tail, transcript = parser.finalize()
        timings["llm_time"] = round(time.time() - t0, 3)
        timings["decode_s"] = round(timings["llm_time"] - timings.get("prefill_s", 0), 3)

        if not interrupted.is_set():
            await dispatch(tail)
    finally:
        active["stream"] = None
        sentence_q.put_nowait(_DONE)
        try:
            await tts_task
        finally:
            # The producer thread must be done before anyone reuses the slot.
            await producer

    if audio_state["first_audio_at"]:
        timings["ttfa_s"] = round(audio_state["first_audio_at"] - t0, 3)
    if tts_started_at:
        timings["tts_time"] = round(time.time() - tts_started_at, 3)

    print(
        f"LLM ({timings['llm_time']:.2f}s, prefill {timings.get('prefill_s')}s) "
        f"heard: {transcript!r} → {parser.response.strip()!r}"
    )

    if interrupted.is_set():
        print("Interrupted mid-turn")
        return raw["text"]

    await ws.send_text(json.dumps({
        "type": "turn_final",
        "transcription": transcript,
        "timings": timings,
        "spoke": audio_state["started"],  # False → client must not wait for audio_end
    }))
    if audio_state["started"]:
        await ws.send_text(json.dumps({
            "type": "audio_end",
            "tts_time": timings.get("tts_time", 0),
        }))
    return raw["text"]


async def prime_cache(messages: list):
    """Fire-and-discard request that pushes a prompt prefix (camera frame,
    speech chunks) through llama-server's cache while the user is talking.
    Content must be media-only appends — a trailing text block would diverge
    the prefix and kill reuse."""
    t0 = time.time()
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: chat_blocking(messages, max_tokens=1))
        print(f"Primed cache ({time.time() - t0:.2f}s)")
        return True
    except Exception as e:
        print(f"Cache priming failed: {e}")
        return False


def user_content(image_b64: str | None, audio_b64s: list[str]) -> list:
    """Media parts of the current user turn, in canonical (cache-stable)
    order: image first, then audio segments oldest-to-newest."""
    parts = [image_part(image_b64)] if image_b64 else []
    parts += [audio_part(b) for b in audio_b64s]
    return parts


def wav_to_float32(b64: str) -> np.ndarray:
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def turn_instruction(msg: dict, has_image: bool, has_audio: bool) -> str:
    if msg.get("type") == "nudge":
        return NUDGE_PROMPT
    if has_audio:
        camera = " Mention what you see on their camera if relevant." if has_image else ""
        return RESPOND_PROMPT.format(camera=camera)
    if has_image:
        return "The user is showing you their camera. Describe what you see."
    return msg.get("text", "Hello!")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    history: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    interrupted = asyncio.Event()
    active = {"stream": None}
    msg_queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    stream = active.get("stream")
                    if stream:
                        stream.cancel()  # actually aborts generation
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except DISCONNECT_ERRORS:
            pass
        finally:
            # Always unblock the main loop, even on unexpected errors.
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    frame_image: str | None = None   # camera frame held for the current utterance
    speech_chunks: list[str] = []    # streamed-in speech, already cache-primed
    held_audio: list[str] = []       # incomplete-turn segments awaiting continuation

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            # Rotate history before the llama context fills: keep the system
            # prompt and the most recent exchanges.
            if estimate_tokens(history) > LLAMA_CTX - CONTEXT_HEADROOM:
                keep = 1 + max(2, (len(history) - 1) // 2)
                print(f"Context near limit — dropping {len(history) - keep} oldest messages")
                history = [history[0]] + history[-(keep - 1):]

            if msg.get("type") == "frame":
                if msg.get("image"):
                    frame_image = msg["image"]
                    speech_chunks = []
                    await prime_cache(history + [
                        {"role": "user", "content": user_content(frame_image, held_audio)}])
                continue

            if msg.get("type") == "speech_chunk":
                if msg.get("seq") == 0:
                    speech_chunks = []
                if valid_audio(msg.get("audio")):
                    speech_chunks.append(msg["audio"])
                    await prime_cache(history + [
                        {"role": "user",
                         "content": user_content(frame_image, held_audio + speech_chunks)}])
                continue

            interrupted.clear()
            chunks = speech_chunks if msg.get("chunked") else []
            speech_chunks = []
            audio_b64s = held_audio + chunks
            if valid_audio(msg.get("audio")):
                audio_b64s.append(msg["audio"])
            image = msg.get("image") or frame_image
            has_audio = bool(audio_b64s)

            if not audio_b64s and not image and not msg.get("text") and msg.get("type") != "nudge":
                # Mic glitch produced no usable media — release the client.
                await ws.send_text(json.dumps({
                    "type": "turn_final", "transcription": None,
                    "timings": {}, "spoke": False,
                }))
                continue

            # The audio classifier judges completeness before the LLM is
            # involved at all. Incomplete → hold the segments (they stay in
            # the next turn's content AND warm in the cache) and wait.
            if has_audio and msg.get("type") != "nudge":
                pcm = np.concatenate([wav_to_float32(b) for b in audio_b64s])
                t0d = time.time()
                complete, prob = await asyncio.get_event_loop().run_in_executor(
                    None, detector.predict, pcm)
                decision_s = round(time.time() - t0d, 3)
                if not complete and not interrupted.is_set():
                    held_audio = audio_b64s
                    await prime_cache(history + [
                        {"role": "user", "content": user_content(frame_image, held_audio)}])
                    await ws.send_text(json.dumps({
                        "type": "turn_incomplete", "kind": "short",
                        "decision_s": decision_s, "p_complete": round(prob, 2),
                    }))
                    continue

            content = user_content(image, audio_b64s)
            has_image = bool(image)
            frame_image = None
            held_audio = []

            try:
                instruction = turn_instruction(msg, has_image, has_audio)
                user_msg = {"role": "user", "content": content + [text_part(instruction)]}
                raw_text = await run_turn(ws, history + [user_msg], interrupted, active)
                # Store the turn verbatim (same bytes → full prefix-cache hit
                # on the next request). Never store a turn the model produced
                # nothing for — a degenerate message poisons all later requests.
                if raw_text.strip():
                    history.append(user_msg)
                    history.append({"role": "assistant", "content": raw_text})
            except DISCONNECT_ERRORS:
                raise
            except Exception:
                # Keep the session alive and release the client from
                # its 'processing' state.
                traceback.print_exc()
                await ws.send_text(json.dumps({
                    "type": "turn_final", "transcription": None,
                    "timings": {}, "spoke": False,
                }))
    except DISCONNECT_ERRORS:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
