# Parlor

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine.

Parlor uses [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) for understanding speech and vision (switchable to E2B or 12B), and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech. You talk, show your camera, and it talks back, all locally.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

# Why?

I'm [self-hosting a totally free voice AI](https://www.fikrikarim.com/bule-ai-initial-release/) on my home server to help people learn speaking English. It has hundreds of monthly active users, and I've been thinking about how to keep it free while making it sustainable.

The obvious answer: run everything on-device, eliminating any server cost. Six months ago I needed an RTX 5090 to run just the voice models in real-time.

Google just released a super capable small model that I can run on my M3 Pro in real-time, with vision too! Sure you can't do agentic coding with this, but it is a game-changer for people learning a new language. Imagine a few years from now that people can run this locally on their phones. They can point their camera at objects and talk about them. And this model is multi-lingual, so people can always fallback to their native language if they want. This is essentially what OpenAI demoed a few years ago.

## AI disclosure

This software is developed with strong assistance from Claude and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you.

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
FastAPI server
    ├── smart-turn-v3 (~20ms)                   →  did you finish your thought?
    ├── Gemma 4 E4B via llama.cpp (QAT q4_0)    →  hears + sees, streams the reply
    │     └── action head (same model, JSON)    →  timers · modes · research
    ├── Kokoro TTS (MLX on Mac, ONNX on Linux)  →  speaks sentence-by-sentence
    └── background reasoner (optional)          →  frontier model for web research
    │
    │  WebSocket (transcript + streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Background research** (optional). "Find the best pizza in Rome right now" hands the task to a frontier model on any OpenAI-compatible endpoint (OpenRouter + web search by default) while the conversation continues; the answer is spoken when it arrives. Off unless `REASONER_API_KEY` is set — without it Parlor stays fully on-device.
- **Hands-free turn-taking.** Browser-side [Silero VAD](https://github.com/ricky0123/vad) with a ~200ms silence cutoff — no push-to-talk. Pipecat's [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) then judges whether you actually finished your thought: mid-thought pauses are held silently until you continue, or answered once you stay quiet.
- **Everything streams.** The reply opens with a transcript of what was heard (on screen immediately — committing to it first measurably improves accuracy), then is spoken sentence-by-sentence while still generating. Your camera frame and speech are pushed through llama.cpp's prompt cache while you're still talking, so even long questions start answering almost instantly.
- **Barge-in.** Speak over the AI to interrupt it; generation is aborted server-side. Echo is handled without the browser's echo canceller — or just wear headphones.
- **Actions never ride the speech.** Timers, mode switches, and research requests are decided by a separate grammar-forced JSON request over the same prompt cache, hidden under TTS playback. The spoken reply stays pure speech, so control markup can never leak into TTS and a promise made aloud is always kept (recall 1.0 vs 0.955 for in-band tags — `benchmarks/archbench.py`).
- **Timers.** "Set a timer for three minutes for the pasta" — the server owns the clock (a turn-based model can't ring into silence — `benchmarks/timerprobe.py`), the model announces the ring out loud in any mode, and a countdown chip with a cancel button tracks it.
- **Live translation mode.** "Translate everything I say into Spanish" turns Parlor into a consecutive interpreter: each utterance rendered into the named language (English by default) after a short silence, no conversational replies, in any language Gemma understands. "Translate between English and Spanish" makes it two-way: the model picks the direction per utterance from the language it hears, and the English voice reads both directions. Say "stop translating" — or hit the stop chip — to exit. Kokoro's non-English voices are noticeably weaker than English — the TTS backend (`src/parlor/tts.py`) is a small interface you can swap for a better multilingual TTS.
- **Just-listen mode.** "Just listen for a while, I want to think out loud" makes it a silent scribe: every utterance transcribed on screen, nothing spoken back, until you address it again.
- **A sense of time.** The model is told how much quiet preceded a turn, how long research took, and when the session started — so "how long was I gone?" gets a real answer.

## Requirements

- Python 3.12+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (`brew install llama.cpp` on macOS; other platforms: [install guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)). Needs build b9503 (June 2026) or newer, b9512 for `MODEL=12b` — older builds lack Gemma 4 audio or crash loading its mmproj
- macOS with Apple Silicon, or Linux with a supported GPU
- ~6 GB free RAM for the default E4B model (`MODEL=e2b` fits in ~4 GB)

## Quick start

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

# Install uv and llama.cpp if you don't have them
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install llama.cpp

uv sync
uv run parlor
```

Open [http://localhost:8000](http://localhost:8000), grant camera and microphone access, and start talking.

Models are downloaded automatically on first run (~5.7 GB for Gemma 4 E4B QAT + its multimodal projector, plus TTS models).

## Configuration

Set these in your shell or a `.env` at the repo root. The common ones:

| Variable           | Default                        | Description                                    |
| ------------------ | ------------------------------ | ---------------------------------------------- |
| `MODEL`            | `e4b`                          | Gemma 4 size: `e2b` (fastest), `e4b` (better answers, ~1.8x e2b latency), `12b` (needs ~8GB) |
| `PORT`             | `8000`                         | Server port                                    |
| `REASONER_API_KEY` | (unset — research off)         | API key enabling background research           |
| `REASONER_BASE_URL`| `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat endpoint            |
| `REASONER_MODEL`   | `openai/gpt-5.6-luna`          | Model the endpoint should run                  |

The full list — local model paths, llama.cpp tuning, TTS backend, test hooks — is in [docs/configuration.md](docs/configuration.md).

## Performance (Apple M3 Pro)

Measured with `MODEL=e2b` from end of utterance to first audio heard (add ~200ms of VAD silence detection on top; the default E4B is roughly 1.8x these numbers, with noticeably better answers). The camera frame and the speech itself are prefilled while you're still speaking, and the reply opens with the transcript line (its decode time is included below — the price of accurate transcripts):

| Turn                                  | First audio | Turn complete |
| ------------------------------------- | ----------- | ------------- |
| Short question (~2s speech)           | ~0.7s       | ~1.3s         |
| Short question + camera               | ~0.8s       | ~1.3s         |
| Long question (~9s speech), streamed  | ~1.3-1.4s   | ~2.1-2.3s     |
| Long question + camera                | ~1.5s       | ~2.3s         |

Reproduce with the end-to-end benchmark (real spoken audio, synthesized locally). Run it before and after a change to see the impact:

```bash
uv run parlor                    # terminal 1
uv run python benchmarks/bench.py --label before --out benchmarks/results/before.json   # terminal 2
# ...make changes, restart the server...
uv run python benchmarks/bench.py --label after --out benchmarks/results/after.json
uv run python benchmarks/compare.py benchmarks/results/before.json benchmarks/results/after.json
```

## Testing

An end-to-end suite spawns the real server (llama.cpp, TTS, turn detector)
and drives it over WebSocket with synthesized speech — including degraded
audio (clipped word endings, noise, other voices) that reproduces live-mic
failure modes:

```bash
uv run pytest            # ~1 minute + model load
```

Set `PARLOR_TEST_URL=ws://localhost:8000/ws` to run it against an
already-running server. Browser-only behavior (echo at speaker volume, VAD
feel, multilingual speech) still needs a live mic and ears.

## Project structure

```
src/parlor/
├── server.py              # FastAPI app + per-connection conversation loop
├── llama.py               # llama-server lifecycle + chat API client
├── pipeline.py            # Streaming turn pipeline (decode → sentences → TTS)
├── actions.py             # Action decider (grammar-forced JSON: timers, modes, research)
├── reasoner.py            # Background research delegation (OpenAI-compatible)
├── modes.py               # Session modes (conversation, translate, listen)
├── turn_detector.py       # smart-turn-v3 end-of-turn classifier
├── tts.py                 # Platform-aware TTS (MLX on Mac, ONNX on Linux)
└── web/                   # Frontend (markup, styles, app logic: VAD, camera, playback)
tests/                     # End-to-end test suite (uv run pytest)
benchmarks/
├── bench.py               # End-to-end latency benchmark
├── fixtures.py            # Spoken-audio fixtures (synthesized locally)
├── compare.py             # Diff two benchmark result files
├── turnbench.py           # Turn-detection accuracy benchmark
├── archbench.py           # In-band control tags vs decoupled action head
├── camerabench.py         # Attach-every-turn vs camera-as-tool-call
└── timerprobe.py          # Why the server owns the timer clock
```

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection
- [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) end-of-turn detection by Pipecat

## License

[Apache 2.0](LICENSE)
