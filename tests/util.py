"""Shared e2e helpers: a synchronous client for the parlor WebSocket
protocol, and a word-error-rate metric for transcript checks."""

import base64
import io
import json
import re
import time
import wave
from dataclasses import dataclass, field

import fixtures
import numpy as np
from websockets.sync.client import connect


def audio(name: str) -> dict:
    """Turn payload carrying one fixture utterance, as the client sends it."""
    return {"audio": fixtures.load_wav_b64(name)}


def norm_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate of hyp against ref (Levenshtein / ref length)."""
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return round(d[-1] / len(r), 3)


def silence_wav_b64(seconds: float = 0.05) -> str:
    """A too-short/no-speech WAV, like a mic glitch produces."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(seconds * 16000), dtype=np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class Turn:
    """Everything the server sent for one turn."""

    marker: str = "timeout"  # complete | incomplete | released | timeout
    text: str = ""
    transcription: str | None = None
    audio_chunks: int = 0
    p_complete: float | None = None
    timings: dict = field(default_factory=dict)


class Session:
    """One WebSocket connection = one server-side conversation."""

    def __init__(self, url: str):
        self.ws = connect(url, max_size=64 * 1024 * 1024)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.ws.close()

    def send(self, payload: dict) -> None:
        self.ws.send(json.dumps(payload))

    def recv(self, timeout: float) -> dict | None:
        try:
            return json.loads(self.ws.recv(timeout=timeout))
        except TimeoutError:
            return None

    def turn(self, payload: dict, timeout: float = 90) -> Turn:
        self.send(payload)
        return self.collect_turn(timeout)

    def collect_turn(self, timeout: float = 90) -> Turn:
        """Read messages until the current turn reaches a terminal state.
        Every terminal state also sends 'ready', mirroring the browser
        (which announces idle when playback ends) — the server holds
        delegation deliveries until it hears it."""
        t = Turn()
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv(timeout=max(0.1, deadline - time.time()))
            if msg is None:
                break
            kind = msg.get("type")
            if kind == "text_delta":
                t.text += msg.get("text", "")
            elif kind == "transcript":  # pushed early, before the response
                t.transcription = msg.get("transcription")
                t.p_complete = msg.get("p_complete")
            elif kind == "audio_chunk":
                t.audio_chunks += 1
            elif kind == "turn_incomplete":
                t.marker = "incomplete"
                t.p_complete = msg.get("p_complete")
                self.send({"type": "ready"})
                return t
            elif kind == "turn_final":
                t.transcription = msg.get("transcription")
                t.timings = msg.get("timings", {})
                if not msg.get("spoke", True):
                    t.marker = "released"
                    self.send({"type": "ready"})
                    return t
            elif kind == "audio_end":
                t.marker = "complete"
                self.send({"type": "ready"})
                return t
        return t

    def wait_for(self, kind: str, timeout: float = 30) -> dict | None:
        """Read messages until one of type `kind` arrives (others are
        discarded); None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv(timeout=max(0.1, deadline - time.time()))
            if msg and msg.get("type") == kind:
                return msg
        return None

    def send_speech_chunks(self, chunks: list[str], gap_s: float = 1.0) -> None:
        """Stream all but the last chunk like the client does during speech."""
        for seq, chunk in enumerate(chunks[:-1]):
            self.send({"type": "speech_chunk", "seq": seq, "audio": chunk})
            time.sleep(gap_s)

    def drain(self, quiet_s: float = 2.0) -> None:
        """Read and discard messages until the line goes quiet."""
        while self.recv(timeout=quiet_s) is not None:
            pass
