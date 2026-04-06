# Parlor — SwiftUI + MLX Swift

High-performance on-device multimodal voice + vision AI, rebuilt as a native SwiftUI app with MLX inference on Apple Silicon.

## Requirements

- **Xcode 16+** with Swift 6
- **macOS 15 (Sequoia)** or **iOS 18+**
- **Apple Silicon** required (M1+ for Mac, A17+ for iPhone)

## Quick Start

```bash
cd swiftui
swift build
open Parlor.xcodeproj  # or create one (see below)
```

On first launch the app automatically downloads model weights from HuggingFace Hub (~3-5 GB) and caches them locally. Subsequent launches load from cache.

## Project Setup

### Option A: Xcode Project (Recommended)

1. Open Xcode → **File → New → Project → macOS App**
2. Set product name to `Parlor`, interface to **SwiftUI**, language to **Swift**
3. In Build Settings:
   - Set **Swift Language Version** to **Swift 6**
   - Set **Strict Concurrency Checking** to **Complete**
4. Delete the generated source files
5. Drag all files from `Parlor/` into your project
6. Add package dependencies (see below)
7. Set `Info.plist` path and `Parlor.entitlements` in target settings

### Option B: Swift Package (Already Configured)

The `Package.swift` is ready to go:

```bash
cd swiftui
swift build
swift run Parlor
```

> Note: Camera/microphone access requires running as a proper app bundle.

## Dependencies

The following Swift packages are used (declared in `Package.swift`):

| Package | Source | Purpose |
|---------|--------|---------|
| `MLXLLM` | [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) | LLM inference via MLX |
| `MLXLMCommon` | same | Shared LLM types and generation |
| `KokoroSwift` | [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios) | Kokoro-82M TTS via MLX |

To add manually in Xcode:
1. **File → Add Package Dependencies**
2. Add `https://github.com/ml-explore/mlx-swift-examples.git` (branch: `main`)
3. Add `https://github.com/mlalma/kokoro-ios.git` (branch: `main`)
4. Link `MLXLLM`, `MLXLMCommon`, and `KokoroSwift` to your target

## Models

Models are downloaded automatically from HuggingFace Hub on first launch:

### LLM — Gemma 3 4B (default)

- **Model ID:** `mlx-community/gemma-3-4b-it-4bit`
- **Size:** ~2.5 GB (4-bit quantized)
- **Supports:** Text + vision (image understanding)

Change the model in `AppState.swift`:
```swift
static let llmModelID = "mlx-community/gemma-3-4b-it-4bit"
```

Other compatible models from [mlx-community](https://huggingface.co/mlx-community):
- `mlx-community/Llama-3.2-3B-Instruct-4bit` — Llama 3.2 3B
- `mlx-community/Qwen2.5-7B-Instruct-4bit` — Qwen 2.5 7B
- `mlx-community/Phi-4-mini-instruct-4bit` — Phi-4 Mini
- Any MLX-format model on HuggingFace

### TTS — Kokoro 82M

- **Model ID:** `mlx-community/Kokoro-82M-bf16`
- **Size:** ~164 MB (bf16)
- **Output:** 24kHz PCM audio
- **Voice:** `af_heart` (configurable)
- **Speed:** ~3x real-time on Apple Silicon

## Architecture

```
ParlorApp
 └── ContentView
      ├── Header (model name + status + download progress)
      ├── CameraPreview + GlowView + WaveformView
      ├── TranscriptView (message bubbles)
      └── ControlBar (camera toggle, phase indicator)

ConversationViewModel (@MainActor, @Observable)
 ├── AudioCapture → VoiceActivityDetector → speech events
 ├── CameraManager → frame capture on demand
 ├── LLMEngine (actor) → MLX LLM inference via MLXLLM
 ├── TTSEngine (actor) → MLX TTS via KokoroSwift
 └── AudioPlayer → streaming playback with level metering
```

### Swift 6 Concurrency Design

| Component | Isolation | Rationale |
|-----------|-----------|-----------|
| `ConversationViewModel` | `@MainActor` | Drives all UI state |
| `LLMEngine` | `actor` | Thread-safe model container access |
| `TTSEngine` | `actor` | Thread-safe TTS generation |
| `AudioCapture` | `@unchecked Sendable` | AVAudioEngine requires specific threading |
| `AudioPlayer` | `@unchecked Sendable` | AVAudioEngine + playback scheduling |
| `CameraManager` | `@unchecked Sendable` | AVCaptureSession delegate callbacks |
| `VoiceActivityDetector` | `Sendable` struct | Pure value-type processing |
| All data types | `Sendable` | Safe to pass across isolation boundaries |

### Performance

- **MLX**: GPU-accelerated inference on Apple Silicon via Metal
- **Unified Memory**: Models stay in shared CPU/GPU memory — zero-copy
- **4-bit Quantization**: LLM runs with minimal memory footprint
- **Accelerate/vDSP**: FFT, RMS energy, and audio resampling
- **Canvas**: Metal-backed waveform visualization
- **AsyncStream**: Zero-copy audio pipeline bridging callbacks to async
- **LazyVStack**: Efficient transcript rendering for long conversations
