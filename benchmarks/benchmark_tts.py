"""Benchmark: kokoro-onnx (CPU) vs mlx-audio (Apple GPU) for Kokoro TTS."""

import platform
import sys
import time
import statistics
import numpy as np

# Test sentences of varying length
SENTENCES = {
    "short": "Hello, how are you doing today?",
    "medium": "I can see you're sitting at your desk. It looks like you're working on something interesting. How can I help you today?",
    "long": (
        "That's a really great question! Let me think about this for a moment. "
        "The history of artificial intelligence goes back to the 1950s, when Alan Turing "
        "first proposed the idea of machines that could think. Since then, we've made "
        "incredible progress, from simple rule-based systems to today's large language models "
        "that can understand and generate human-like text."
    ),
}

VOICE = "af_heart"
SPEED = 1.1
WARMUP = 2
RUNS = 5


def benchmark_kokoro_onnx():
    """Benchmark kokoro-onnx (ONNX Runtime, CPU)."""
    import kokoro_onnx
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
    voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

    print("Loading kokoro-onnx...")
    t0 = time.time()
    tts = kokoro_onnx.Kokoro(model_path, voices_path)
    print(f"  Loaded in {time.time() - t0:.2f}s")

    results = {}
    for label, text in SENTENCES.items():
        # Warmup
        for _ in range(WARMUP):
            tts.create(text, voice=VOICE, speed=SPEED)

        # Timed runs
        times = []
        audio_duration = None
        for _ in range(RUNS):
            t0 = time.time()
            pcm, sr = tts.create(text, voice=VOICE, speed=SPEED)
            elapsed = time.time() - t0
            times.append(elapsed)
            audio_duration = len(pcm) / sr

        results[label] = {
            "times": times,
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "audio_sec": audio_duration,
            "rtf": statistics.mean(times) / audio_duration,
            "sample_rate": sr,
        }

    return results


def benchmark_mlx_audio():
    """Benchmark mlx-audio (MLX, Apple GPU)."""
    from mlx_audio.tts.generate import load_model

    print("Loading mlx-audio...")
    t0 = time.time()
    model = load_model("mlx-community/Kokoro-82M-bf16")
    sr = model.sample_rate
    # Warmup to trigger pipeline init
    list(model.generate(text="Hello", voice=VOICE, speed=SPEED))
    print(f"  Loaded in {time.time() - t0:.2f}s (sample_rate={sr})")

    results = {}
    for label, text in SENTENCES.items():
        # Warmup
        for _ in range(WARMUP):
            list(model.generate(text=text, voice=VOICE, speed=SPEED))

        # Timed runs
        times = []
        audio_duration = None
        for _ in range(RUNS):
            t0 = time.time()
            gen_results = list(model.generate(text=text, voice=VOICE, speed=SPEED))
            elapsed = time.time() - t0
            times.append(elapsed)
            pcm = np.concatenate([np.array(r.audio) for r in gen_results])
            audio_duration = len(pcm) / sr

        results[label] = {
            "times": times,
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "audio_sec": audio_duration,
            "rtf": statistics.mean(times) / audio_duration,
            "sample_rate": sr,
        }

    return results


def benchmark_mlx_audio_streaming():
    """Benchmark mlx-audio streaming: time-to-first-chunk on long text."""
    from mlx_audio.tts.generate import load_model

    print("\nBenchmarking mlx-audio streaming (time-to-first-chunk)...")
    model = load_model("mlx-community/Kokoro-82M-bf16")

    # Warmup
    list(model.generate(text="Hello", voice=VOICE, speed=SPEED))

    results = {}
    for label, text in SENTENCES.items():
        # Warmup streaming
        for _ in range(WARMUP):
            for r in model.generate(text=text, voice=VOICE, speed=SPEED, stream=True, streaming_interval=1.0):
                break

        ttfc_times = []
        total_times = []
        chunk_counts = []
        for _ in range(RUNS):
            t0 = time.time()
            first = True
            n_chunks = 0
            for r in model.generate(text=text, voice=VOICE, speed=SPEED, stream=True, streaming_interval=1.0):
                if first:
                    ttfc_times.append(time.time() - t0)
                    first = False
                n_chunks += 1
            total_times.append(time.time() - t0)
            chunk_counts.append(n_chunks)

        results[label] = {
            "ttfc_mean": statistics.mean(ttfc_times),
            "ttfc_min": min(ttfc_times),
            "total_mean": statistics.mean(total_times),
            "chunks": statistics.mean(chunk_counts),
        }

    return results


def print_results(name, results):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    for label, r in results.items():
        text = SENTENCES[label]
        print(f"\n  [{label}] ({len(text)} chars)")
        print(f"    Mean:   {r['mean']*1000:7.1f} ms  (+/-{r['stdev']*1000:.1f})")
        print(f"    Min:    {r['min']*1000:7.1f} ms")
        print(f"    Audio:  {r['audio_sec']:7.2f} s")
        print(f"    RTF:    {r['rtf']:7.3f}x  (< 1.0 = faster than real-time)")
        print(f"    SR:     {r['sample_rate']} Hz")


def print_streaming_results(results):
    print(f"\n{'=' * 60}")
    print(f"  mlx-audio: Streaming Mode")
    print(f"{'=' * 60}")
    for label, r in results.items():
        text = SENTENCES[label]
        print(f"\n  [{label}] ({len(text)} chars)")
        print(f"    TTFC Mean:  {r['ttfc_mean']*1000:7.1f} ms")
        print(f"    TTFC Min:   {r['ttfc_min']*1000:7.1f} ms")
        print(f"    Total Mean: {r['total_mean']*1000:7.1f} ms")
        print(f"    Chunks:     {r['chunks']:.1f}")


if __name__ == "__main__":
    is_apple = sys.platform == "darwin" and platform.machine() == "arm64"

    print("=" * 60)
    if is_apple:
        print("  TTS Benchmark: kokoro-onnx vs mlx-audio")
    else:
        print("  TTS Benchmark: kokoro-onnx")
    print(f"  Warmup: {WARMUP} runs, Measured: {RUNS} runs")
    print("=" * 60)

    onnx_results = benchmark_kokoro_onnx()
    print_results("kokoro-onnx (ONNX Runtime, CPU)", onnx_results)

    if is_apple:
        mlx_results = benchmark_mlx_audio()
        print_results("mlx-audio (MLX, Apple GPU)", mlx_results)

        streaming_results = benchmark_mlx_audio_streaming()
        print_streaming_results(streaming_results)

        # Comparison
        print(f"\n{'=' * 60}")
        print(f"  Comparison: speedup of mlx-audio over kokoro-onnx")
        print(f"{'=' * 60}")
        for label in SENTENCES:
            onnx_mean = onnx_results[label]["mean"]
            mlx_mean = mlx_results[label]["mean"]
            speedup = onnx_mean / mlx_mean
            print(f"  [{label}]  {onnx_mean*1000:.0f}ms -> {mlx_mean*1000:.0f}ms  ({speedup:.2f}x {'faster' if speedup > 1 else 'slower'})")
