"""Unit tests for MiniMaxTTSBackend."""

import json
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path so we can import tts module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tts import MiniMaxTTSBackend


class TestMiniMaxTTSBackend:
    def test_init_with_explicit_key(self):
        backend = MiniMaxTTSBackend(api_key="test-key")
        assert backend._api_key == "test-key"
        assert backend._base_url == "https://api.minimax.io"
        assert backend.sample_rate == 32000

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        backend = MiniMaxTTSBackend()
        assert backend._api_key == "env-key"

    def test_init_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com")
        backend = MiniMaxTTSBackend(api_key="test-key")
        assert backend._base_url == "https://api.minimaxi.com"

    def test_init_strips_trailing_slash(self):
        backend = MiniMaxTTSBackend(api_key="test-key", base_url="https://api.minimax.io/")
        assert backend._base_url == "https://api.minimax.io"

    def test_init_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            MiniMaxTTSBackend()

    def test_generate_returns_float32_array(self):
        # Build a fake 100ms of silence at 32000 Hz (32000 * 0.1 = 3200 samples)
        pcm_int16 = np.zeros(3200, dtype=np.int16)
        audio_hex = pcm_int16.tobytes().hex()

        fake_response = json.dumps({
            "data": {"audio": audio_hex},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        backend = MiniMaxTTSBackend(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.generate("Hello world")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 3200
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_generate_normalizes_audio(self):
        # Max int16 value should become ~1.0 in float32
        pcm_int16 = np.array([32767, -32768, 0, 16383], dtype=np.int16)
        audio_hex = pcm_int16.tobytes().hex()

        fake_response = json.dumps({
            "data": {"audio": audio_hex},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        backend = MiniMaxTTSBackend(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.generate("Hello")

        assert result[0] == pytest.approx(32767 / 32767.0, abs=1e-5)
        assert result[1] == pytest.approx(-32768 / 32767.0, abs=1e-5)
        assert result[2] == pytest.approx(0.0, abs=1e-5)

    def test_generate_raises_on_api_error(self):
        fake_response = json.dumps({
            "data": {},
            "base_resp": {"status_code": 1001, "status_msg": "invalid api key"},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        backend = MiniMaxTTSBackend(api_key="bad-key")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="1001"):
                backend.generate("Hello")

    def test_generate_uses_correct_url(self):
        pcm_int16 = np.zeros(100, dtype=np.int16)
        fake_response = json.dumps({
            "data": {"audio": pcm_int16.tobytes().hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        backend = MiniMaxTTSBackend(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            backend.generate("Hello")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == "https://api.minimax.io/v1/t2a_v2"

    def test_generate_request_payload(self):
        pcm_int16 = np.zeros(100, dtype=np.int16)
        fake_response = json.dumps({
            "data": {"audio": pcm_int16.tobytes().hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        backend = MiniMaxTTSBackend(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            backend.generate("Hello world", voice="English_Persuasive_Man", speed=1.2)

        call_args = mock_urlopen.call_args[0][0]
        payload = json.loads(call_args.data)

        assert payload["text"] == "Hello world"
        assert payload["stream"] is False
        assert payload["voice_setting"]["voice_id"] == "English_Persuasive_Man"
        assert payload["voice_setting"]["speed"] == 1.2
        assert payload["audio_setting"]["format"] == "pcm"
        assert payload["audio_setting"]["sample_rate"] == 32000
        assert payload["audio_setting"]["channel"] == 1

    def test_generate_uses_bearer_auth(self):
        pcm_int16 = np.zeros(100, dtype=np.int16)
        fake_response = json.dumps({
            "data": {"audio": pcm_int16.tobytes().hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        backend = MiniMaxTTSBackend(api_key="my-secret-key")
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            backend.generate("Hello")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.get_header("Authorization") == "Bearer my-secret-key"

    def test_default_model_is_speech_28_hd(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_TTS_MODEL", raising=False)
        pcm_int16 = np.zeros(100, dtype=np.int16)
        fake_response = json.dumps({
            "data": {"audio": pcm_int16.tobytes().hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response

        backend = MiniMaxTTSBackend(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            backend.generate("Hello")

        call_args = mock_urlopen.call_args[0][0]
        payload = json.loads(call_args.data)
        assert payload["model"] == "speech-2.8-hd"


class TestLoadFunctionWithMiniMax:
    def test_load_uses_minimax_when_key_set(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        from tts import load
        backend = load()
        assert isinstance(backend, MiniMaxTTSBackend)

    def test_load_skips_minimax_when_no_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setenv("KOKORO_ONNX", "1")  # force ONNX path

        # Mock ONNXBackend init to avoid downloading models
        with patch("tts.ONNXBackend.__init__", return_value=None):
            with patch("tts.ONNXBackend.sample_rate", new=24000, create=True):
                from tts import load
                backend = load()
                assert not isinstance(backend, MiniMaxTTSBackend)
