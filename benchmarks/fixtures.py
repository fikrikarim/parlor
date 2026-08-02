"""Speech fixtures for the e2e benchmark.

Synthesizes real spoken audio with the local Kokoro TTS backend (so the
Gemma audio encoder gets actual speech, not sine waves), resamples to the
16 kHz the frontend sends, and caches WAVs in benchmarks/fixtures/.

Run directly to (re)generate:  uv run python benchmarks/fixtures.py
"""

import base64
import io
import sys
import wave
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TARGET_SR = 16000

# name -> (spoken text, keywords expected in the transcript). The e2e suite
# asserts on the spoken text via word error rate; the keywords remain for
# quick greps and ad-hoc scripts.
FIXTURES = {
    "capital_france": (
        "What is the capital of France?",
        ["capital", "france"],
    ),
    "long_question": (
        "I have been trying to learn English for a few months now, and I "
        "wonder if you could give me some advice on how to improve my "
        "pronunciation when I speak with other people.",
        ["english", "pronunciation"],
    ),
    # Truncated mid-utterance (see BASE_KWARGS) so the audio cuts off abruptly,
    # like a real interrupted speaker — TTS of a trailing-off sentence
    # otherwise sounds politely finished.
    "incomplete_cutoff": (
        "So the thing I wanted to ask you about is the weather in Paris "
        "for my trip next week.",
        [],
    ),
    "thinking_pause": (
        "That's a hard question. Hmm, let me think about it for a moment.",
        [],
    ),
    "describe_scene": (
        "Can you describe what you can see right now?",
        ["see"],
    ),
    # Continues incomplete_cutoff: held audio + this = the full question.
    "continuation": (
        "the weather in Paris for my trip next week.",
        ["paris"],
    ),
    # An uncommon name on purpose: asked cold, the model hallucinates "your
    # name is Alex" often enough to break history-isolation assertions.
    "name_intro": (
        "By the way, my name is Willow. It's nice to meet you!",
        ["willow"],
    ),
    "name_recall": (
        "Can you remind me what my name is?",
        ["name"],
    ),
    # Elapsed-time e2e (tests/test_elapsed.py): asked after a real pause.
    "how_long_quiet": (
        "Roughly how long was I quiet just now?",
        ["long"],
    ),
    # Transcript-stability fillers (tests/test_transcription_stability.py):
    # each carries content words unique across the whole fixture set, so a
    # transcript that borrows another turn's words is provably written from
    # history, not from the audio. Plain statements on purpose — no
    # search/current-info phrasing that would fire <delegate>.
    "filler_color": (
        "My favorite color is turquoise, like the sea near my childhood home.",
        ["turquoise"],
    ),
    "filler_pet": (
        "I adopted a golden retriever puppy last month and named him Biscuit.",
        ["biscuit"],
    ),
    "filler_baking": (
        "On weekends I like to bake sourdough bread with rosemary and olives.",
        ["sourdough"],
    ),
    "filler_novel": (
        "I have been reading a mystery novel set in Lisbon about a missing violin.",
        ["lisbon"],
    ),
    "filler_jazz": (
        "Lately I have been listening to a lot of jazz piano records in the evening.",
        ["jazz"],
    ),
    "filler_garden": (
        "This spring I want to plant tomatoes and basil on my balcony.",
        ["basil"],
    ),
    # Delegation e2e: explicit "search the web" phrasing so the model
    # reliably emits the <delegate> tag (see benchmarks/tagbench.py).
    "delegate_pizza": (
        "Can you search the web and find the best pizza place in Rome "
        "right now?",
        ["pizza", "rome"],
    ),
    # "stock" in the task makes the suite's mock reasoner fail the request
    # (tests/conftest.py) — the failure path must end in a spoken apology.
    "delegate_stock": (
        "Can you search the web for the current stock price of Apple?",
        ["stock", "apple"],
    ),
    # "naples" makes the mock stall 8s — long enough to prove the
    # conversation continues while a background task runs. Current-info
    # phrasing on purpose: for topics the model believes it knows (e.g.
    # "the history of pizza") it acks without the tag often enough to
    # flake, no matter how the request is worded.
    "delegate_naples": (
        "Can you search the web for today's weather in Naples, Italy?",
        ["weather", "naples"],
    ),
    # Alternate phrasings for the delegation tests' retry: re-sending the
    # SAME audio tends to reproduce the same tagless completion (cached
    # prefix, near-deterministic sampling), so a retry only decorrelates
    # with different words. Trigger keywords ('stock', 'naples') preserved.
    "delegate_pizza_alt": (
        "Please look up the best pizza places in Rome right now.",
        ["pizza", "rome"],
    ),
    "delegate_stock_alt": (
        "Please look up Apple's current stock price on the web for me.",
        ["stock", "apple"],
    ),
    "delegate_naples_alt": (
        "Please look up the current weather in Naples, Italy for me.",
        ["weather", "naples"],
    ),
    # Mode-switch voice commands (translation + listen e2e).
    "cmd_translate": (
        "From now on, please translate everything I say into English.",
        ["translate"],
    ),
    "cmd_translate_spanish": (
        "From now on, please translate everything I say into Spanish.",
        ["translate"],
    ),
    "cmd_translate_pair": (
        "Please translate between English and Spanish for us.",
        ["translate"],
    ),
    # Real Spanish speech (es→en direction of two-way interpreting),
    # synthesized on the Spanish voice's own pipeline.
    "es_train_station": (
        "¿Dónde está la estación de tren más cercana?",
        [],
    ),
    "cmd_stop_translate": (
        "Okay, stop translating now and go back to normal conversation.",
        ["stop"],
    ),
    "cmd_listen": (
        "From now on, please just listen quietly and don't respond. I "
        "want to think out loud for a bit.",
        ["listen"],
    ),
    "cmd_stop_listen": (
        "Okay, I'm done thinking out loud. What do you think about all "
        "of that?",
        ["done"],
    ),
    # Timer e2e: short durations so the ring lands inside test timeouts;
    # _alt phrasings decorrelate the retry like the delegation fixtures.
    "cmd_timer_short": (
        "Set a timer for ten seconds, please.",
        ["timer"],
    ),
    "cmd_timer_short_alt": (
        "Please give me a ten second timer.",
        ["timer"],
    ),
    "cmd_timer_minute": (
        "Please set a one minute timer for the tea.",
        ["timer"],
    ),
    "cmd_timer_minute_alt": (
        "Can you give me a one minute timer for the tea?",
        ["timer"],
    ),
}

# Synthesis kwargs (see _synthesize) for base fixtures that need them.
BASE_KWARGS = {"incomplete_cutoff": {"keep_frac": 0.55},
               "es_train_station": {"voice": "ef_dora"}}

# Degraded re-recordings of base fixtures, reproducing live-mic conditions
# clean synthesis can't: VAD-clipped word endings (the encoder hallucinates
# confident completions on abrupt cuts), background noise, other voices.
# name -> (base fixture, degradation kwargs for _synthesize)
VARIANTS = {
    "capital_france_clipped": ("capital_france", {"clip_end_s": 0.18}),
    "capital_france_noisy": ("capital_france", {"snr_db": 12}),
    # A mid-thought cutoff that does NOT address the assistant (unlike
    # incomplete_cutoff's "I wanted to ask you…") — listen mode must stay
    # silent through it (tests/test_listen.py).
    "filler_baking_cutoff": ("filler_baking", {"keep_frac": 0.55}),
    # am_adam holds a clear margin below smart-turn's 0.5 threshold after
    # the int16 WAV round-trip (0.25); am_michael sat at ~0.5 and flipped
    # with synthesis drift, breaking the held-turn tests.
    "long_question_male": ("long_question", {"voice": "am_adam"}),
}


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return pcm
    duration = len(pcm) / src_sr
    n_dst = int(duration * dst_sr)
    x_src = np.linspace(0.0, duration, num=len(pcm), endpoint=False)
    x_dst = np.linspace(0.0, duration, num=n_dst, endpoint=False)
    return np.interp(x_dst, x_src, pcm).astype(np.float32)


def _write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_int16.tobytes())


def _synthesize(backend, text: str, *, voice: str = "af_heart", keep_frac: float | None = None,
                clip_end_s: float | None = None, snr_db: float | None = None) -> np.ndarray:
    pcm = backend.generate(text, voice=voice, speed=1.0)
    pcm = _resample_linear(np.asarray(pcm, dtype=np.float32), backend.sample_rate, TARGET_SR)
    if keep_frac:  # chop mid-utterance, like an interrupted speaker
        pcm = pcm[: int(len(pcm) * keep_frac)]
        fade = min(len(pcm), int(0.01 * TARGET_SR))
        pcm[-fade:] *= np.linspace(1.0, 0.0, fade)
    if clip_end_s:  # trim trailing silence, then cut into the last word
        voiced = np.where(np.abs(pcm) > 0.01)[0]
        if len(voiced):
            pcm = pcm[: voiced[-1] + 1]
        pcm = pcm[: -int(clip_end_s * TARGET_SR)]
    if snr_db is not None:
        noise = np.random.default_rng(42).standard_normal(len(pcm)).astype(np.float32)
        noise *= np.sqrt(np.mean(pcm**2) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
        pcm = np.clip(pcm + noise, -1.0, 1.0)
    return pcm


def generate_all(force: bool = False) -> None:
    specs = {n: (FIXTURES[n][0], BASE_KWARGS.get(n, {})) for n in FIXTURES}
    specs |= {n: (FIXTURES[base][0], kwargs) for n, (base, kwargs) in VARIANTS.items()}
    missing = [n for n in specs if force or not (FIXTURES_DIR / f"{n}.wav").exists()]
    if not missing:
        return
    FIXTURES_DIR.mkdir(exist_ok=True)
    from parlor import tts

    backend = tts.load()
    for name in missing:
        text, kwargs = specs[name]
        pcm = _synthesize(backend, text, **kwargs)
        _write_wav(FIXTURES_DIR / f"{name}.wav", pcm, TARGET_SR)
        print(f"fixture {name}: {len(pcm) / TARGET_SR:.1f}s speech")


def breath_wav_b64(seconds: float = 1.2, level: float = 0.03, seed: int = 7) -> str:
    """Non-speech audio like a VAD false trigger captures: low-pass noise
    with a breath-like swell (level scales the pre-envelope RMS). Long
    enough to pass the server's valid_audio check, seeded so every run
    sends identical bytes."""
    n = int(seconds * TARGET_SR)
    rng = np.random.default_rng(seed)
    noise = np.convolve(rng.standard_normal(n).astype(np.float32),
                        np.ones(48, dtype=np.float32) / 48, mode="same")
    env = np.sin(np.linspace(0, np.pi, n)).astype(np.float32) ** 2
    pcm = noise / (np.sqrt(np.mean(noise**2)) + 1e-9) * level * env
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes((pcm * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def load_wav_b64(name: str) -> str:
    return base64.b64encode((FIXTURES_DIR / f"{name}.wav").read_bytes()).decode()


def load_wav_chunks_b64(name: str, chunk_s: float = 3.0) -> list[str]:
    """Fixture audio sliced into WAV-encoded chunks, as the streaming client
    sends them during speech (arbitrary sample boundaries)."""
    with wave.open(str(FIXTURES_DIR / f"{name}.wav"), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    step = int(chunk_s * sr)
    chunks = []
    for start in range(0, len(pcm), step):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sr)
            out.writeframes(pcm[start:start + step].tobytes())
        chunks.append(base64.b64encode(buf.getvalue()).decode())
    return chunks


def make_image_b64(width: int = 320, height: int = 240, scene: str = "day") -> str:
    """A scene the model can plausibly describe, drawn with PIL and
    JPEG-encoded like the frontend. "day": blue sky over a green field with a
    red sun-like circle. "night": dark sky with a white crescent moon —
    distinct enough to test frame freshness across turns."""
    from PIL import Image, ImageDraw

    if scene == "night":
        img = Image.new("RGB", (width, height), color=(12, 14, 40))
        draw = ImageDraw.Draw(img)
        draw.ellipse([width - 120, 25, width - 45, 100], fill=(240, 240, 220))
        draw.ellipse([width - 105, 20, width - 30, 95], fill=(12, 14, 40))  # crescent bite
    else:
        img = Image.new("RGB", (width, height), color=(90, 160, 220))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, height * 2 // 3, width, height], fill=(70, 150, 70))
        draw.ellipse([width - 110, 20, width - 40, 90], fill=(220, 60, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    generate_all(force="--force" in sys.argv)
    print("Fixtures ready:", ", ".join([*FIXTURES, *VARIANTS]))
