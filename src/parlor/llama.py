"""llama.cpp backend: spawns/manages llama-server and speaks its
OpenAI-compatible chat API (blocking, streaming, and cache-priming)."""

import http.client
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from parlor import hf

load_dotenv()  # config below is read at import time — .env must apply first

# Google's official QAT quants: q4_0 quality trained-in, faster than
# K-quants. MODEL picks the size; MODEL_PATH/MMPROJ_PATH override entirely.
# Rough guide on an M3 Pro: e2b ≈ 0.6-1.0s to first audio, e4b ≈ 1.0-1.7s
# (noticeably better answers — the default), 12b needs ~8GB and is slower
# still.
MODELS = {
    "e2b": ("google/gemma-4-E2B-it-qat-q4_0-gguf",
            "gemma-4-E2B_q4_0-it.gguf", "gemma-4-E2B-it-mmproj.gguf"),
    "e4b": ("google/gemma-4-E4B-it-qat-q4_0-gguf",
            "gemma-4-E4B_q4_0-it.gguf", "gemma-4-E4B-it-mmproj.gguf"),
    "12b": ("google/gemma-4-12B-it-qat-q4_0-gguf",
            "gemma-4-12b-it-qat-q4_0.gguf", "mmproj-gemma-4-12b-it-qat-q4_0.gguf"),
}
MODEL = os.environ.get("MODEL", "e4b").lower()

PORT = int(os.environ.get("LLAMA_PORT", "8081"))
URL = os.environ.get("LLAMA_SERVER_URL", "")  # set to use an external server
CTX = int(os.environ.get("LLAMA_CTX", "16384"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

_proc: subprocess.Popen | None = None


def resolve_model_paths() -> tuple[str, str]:
    model = os.environ.get("MODEL_PATH", "")
    mmproj = os.environ.get("MMPROJ_PATH", "")
    if model and mmproj:
        return model, mmproj
    if MODEL not in MODELS:
        raise RuntimeError(f"MODEL={MODEL!r} — expected one of {', '.join(MODELS)}")
    repo, gguf, mmproj_file = MODELS[MODEL]
    return model or hf.download(repo, gguf), mmproj or hf.download(repo, mmproj_file)


def model_label() -> str:
    """Human-readable model name for the UI."""
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return Path(path).stem
    return f"Gemma 4 {MODEL.upper()}" if MODEL in MODELS else MODEL


def host_port() -> tuple[str, int]:
    if URL:
        # urlsplit needs the '//' to see an authority; without it a bare
        # 'myhost:8081' would parse as scheme 'myhost'. Handles a trailing
        # path too ('http://myhost:8081/v1' is how these URLs are usually
        # written).
        u = urlsplit(URL if "//" in URL else "//" + URL)
        if u.scheme == "https":
            # Everything here speaks plain HTTPConnection — silently
            # defaulting an https URL to port 80 would "work" wrongly.
            raise RuntimeError("LLAMA_SERVER_URL must be http:// (no TLS)")
        return u.hostname or "127.0.0.1", u.port or 80
    return "127.0.0.1", PORT


def _connect(timeout: float) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(*host_port(), timeout=timeout)


def start() -> None:
    global _proc
    if URL:
        print(f"Using external llama-server at {URL}")
        return
    binary = shutil.which("llama-server")
    if not binary:
        raise RuntimeError("llama-server not found — install with: brew install llama.cpp")
    model, mmproj = resolve_model_paths()
    print(f"Starting llama-server with {Path(model).name} (ctx={CTX})...")
    # Output goes to DEVNULL — un-silence here when debugging llama itself.
    _proc = subprocess.Popen(
        [binary, "-m", model, "--mmproj", mmproj, "-ngl", "99",
         "--port", str(PORT), "-c", str(CTX), "-np", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if _proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {_proc.returncode}")
        try:
            conn = _connect(timeout=2)
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


def stop() -> None:
    if _proc:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()  # a hung llama-server must not outlive us
            _proc.wait()


def _chat_body(messages: list, max_tokens: int, stream: bool) -> dict:
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": stream,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        # The final chunk then carries usage.prompt_tokens — the REAL
        # context size, which drives history rotation (estimates drift).
        body["stream_options"] = {"include_usage": True}
    return body


def chat_blocking(messages: list, max_tokens: int) -> str:
    """Non-streaming request; returns the message content ('' on discard)."""
    conn = _connect(timeout=300)
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
        self.prompt_tokens: int | None = None  # real count, from the usage chunk

    def run(self, on_delta):
        # self.conn is published before the request is sent, so a cancel()
        # landing mid-upload still tears the socket down.
        self.conn = _connect(timeout=300)
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
                chunk = json.loads(payload)
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens"):
                    self.prompt_tokens = usage["prompt_tokens"]
                choices = chunk.get("choices") or []
                text = choices[0].get("delta", {}).get("content") if choices else None
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
