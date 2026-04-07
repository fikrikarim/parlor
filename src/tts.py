"""Platform-aware Kokoro TTS: mlx-audio on Apple Silicon, kokoro-onnx elsewhere.

Set MINIMAX_API_KEY to use MiniMax cloud TTS instead of local Kokoro models.
"""

import json
import os
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class MiniMaxTTSBackend(TTSBackend):
    """MiniMax cloud TTS backend (https://api.minimax.io/v1/t2a_v2).

    Activated when MINIMAX_API_KEY is set. Uses PCM audio format to avoid
    any extra decoding dependencies — the hex-encoded PCM bytes are converted
    directly to a float32 numpy array.

    Supported voices: English_Graceful_Lady, English_Insightful_Speaker,
    English_radiant_girl, English_Persuasive_Man, English_Lucky_Robot,
    English_expressive_narrator.

    Supported models: speech-2.8-hd (default), speech-2.8-turbo.
    """

    VOICES = [
        "English_Graceful_Lady",
        "English_Insightful_Speaker",
        "English_radiant_girl",
        "English_Persuasive_Man",
        "English_Lucky_Robot",
        "English_expressive_narrator",
    ]

    sample_rate: int = 32000

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not self._api_key:
            raise ValueError("MINIMAX_API_KEY is required for MiniMaxTTSBackend")
        self._base_url = (
            base_url or os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io")
        ).rstrip("/")

    def generate(
        self, text: str, voice: str = "English_Graceful_Lady", speed: float = 1.0
    ) -> np.ndarray:
        """Synthesize speech and return float32 PCM audio array."""
        url = f"{self._base_url}/v1/t2a_v2"
        model = os.environ.get("MINIMAX_TTS_MODEL", "speech-2.8-hd")
        payload = json.dumps(
            {
                "model": model,
                "text": text,
                "stream": False,
                "voice_setting": {
                    "voice_id": voice,
                    "speed": speed,
                    "vol": 1,
                    "pitch": 0,
                },
                "audio_setting": {
                    "sample_rate": self.sample_rate,
                    "format": "pcm",
                    "channel": 1,
                },
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())

        status = result.get("base_resp", {}).get("status_code", -1)
        if status != 0:
            msg = result.get("base_resp", {}).get("status_msg", "unknown error")
            raise RuntimeError(f"MiniMax TTS API error {status}: {msg}")

        audio_hex = result["data"]["audio"]
        audio_bytes = bytes.fromhex(audio_hex)
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0


class MLXBackend(TTSBackend):
    """mlx-audio backend (Apple Silicon GPU via MLX)."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(r.audio) for r in results])


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend (ONNX Runtime, CPU)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        pcm, _sr = self._model.create(text, voice=voice, speed=speed)
        return pcm


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform.

    Priority:
    1. MiniMax cloud TTS — if MINIMAX_API_KEY is set.
    2. mlx-audio (Apple Silicon GPU) — on macOS arm64 unless KOKORO_ONNX is set.
    3. kokoro-onnx (CPU) — fallback for all other platforms.
    """
    if os.environ.get("MINIMAX_API_KEY"):
        backend = MiniMaxTTSBackend()
        print(f"TTS: MiniMax cloud (sample_rate={backend.sample_rate})")
        return backend

    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = MLXBackend()
            print(f"TTS: mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to kokoro-onnx")

    backend = ONNXBackend()
    print(f"TTS: kokoro-onnx (CPU, sample_rate={backend.sample_rate})")
    return backend
