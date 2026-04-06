# Parlor — SwiftUI + CoreML

High-performance on-device multimodal voice + vision AI, rebuilt as a native SwiftUI app with CoreML inference.

## Requirements

- **Xcode 16+** with Swift 6
- **macOS 15 (Sequoia)** or **iOS 18+**
- Apple Silicon recommended (Neural Engine + GPU acceleration)

## Project Setup

### Option A: Xcode Project (Recommended)

1. Open Xcode → **File → New → Project → macOS/iOS App**
2. Set product name to `Parlor`, interface to **SwiftUI**, language to **Swift**
3. In Build Settings:
   - Set **Swift Language Version** to **Swift 6**
   - Set **Strict Concurrency Checking** to **Complete**
4. Delete the generated `ContentView.swift` and `ParlorApp.swift`
5. Drag all files from `Parlor/` into your Xcode project
6. Add your CoreML models to the project (see below)
7. Set `Info.plist` path and `Parlor.entitlements` in target settings

### Option B: Swift Package

```bash
cd swiftui
swift build
```

> Note: Camera/microphone access requires running as a proper app bundle, not a CLI.

## Adding CoreML Models

The app expects two CoreML model bundles in the app's resources:

### 1. LLM — Gemma 4 E2B (`GemmaE2B.mlmodelc`)

Convert using [coremltools](https://github.com/apple/coremltools):

```python
import coremltools as ct
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it")
mlmodel = ct.convert(
    model,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
)
mlmodel.save("GemmaE2B.mlpackage")
```

Then compile: **Xcode → Add Files → select .mlpackage → Xcode compiles to .mlmodelc**

**Expected input features:**
- `audio_input`: MLMultiArray `[1, N]` Float32 — PCM audio at 16kHz
- `image_input`: CVPixelBuffer BGRA — Camera frame (320px width)
- `text_input`: MLMultiArray `[1, N]` Float32 — UTF-8 encoded prompt

**Expected output features:**
- `output_tokens`: MLMultiArray `[1, N]` Int32 — Generated token IDs

> Adjust `LLMEngine.swift` `buildInput()`/`parseOutput()` to match your model's actual schema.

### 2. TTS — Kokoro 82M (`Kokoro82M.mlmodelc`)

```python
import coremltools as ct
import torch
from kokoro import KokoroModel

model = KokoroModel.from_pretrained("hexgrad/Kokoro-82M")
traced = torch.jit.trace(model, example_inputs)
mlmodel = ct.convert(
    traced,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
)
mlmodel.save("Kokoro82M.mlpackage")
```

**Expected input features:**
- `input_ids`: MLMultiArray `[1, N]` Int32 — Phoneme/token IDs
- `speed`: MLMultiArray `[1]` Float32 — Speed factor

**Expected output features:**
- `audio_output`: MLMultiArray `[1, N]` Float32 — PCM samples at 24kHz

> The app runs without models in "demo mode" — audio capture and camera work, but no inference.

## Architecture

```
ParlorApp
 └── ContentView
      ├── Header (model name + status)
      ├── CameraPreview + GlowView + WaveformView
      ├── TranscriptView (message bubbles)
      └── ControlBar (camera toggle, phase indicator)

ConversationViewModel (@MainActor, @Observable)
 ├── AudioCapture → VoiceActivityDetector → speech events
 ├── CameraManager → frame capture on demand
 ├── LLMEngine (actor) → CoreML multimodal inference
 ├── TTSEngine (actor) → CoreML speech synthesis
 └── AudioPlayer → streaming playback with level metering
```

### Swift 6 Concurrency Design

| Component | Isolation | Rationale |
|-----------|-----------|-----------|
| `ConversationViewModel` | `@MainActor` | Drives all UI state |
| `LLMEngine` | `actor` | Thread-safe model inference |
| `TTSEngine` | `actor` | Thread-safe TTS synthesis |
| `AudioCapture` | `@unchecked Sendable` | AVAudioEngine requires specific threading |
| `AudioPlayer` | `@unchecked Sendable` | AVAudioEngine + playback scheduling |
| `CameraManager` | `@unchecked Sendable` | AVCaptureSession delegate callbacks |
| `VoiceActivityDetector` | `Sendable` struct | Pure value-type processing |
| All data types | `Sendable` | Safe to pass across isolation boundaries |

### Performance

- **CoreML**: Runs on Neural Engine + GPU via `computeUnits: .all`
- **Accelerate**: vDSP for FFT, RMS energy, and audio resampling
- **Canvas rendering**: Metal-backed waveform visualization
- **AsyncStream**: Zero-copy audio pipeline bridging callbacks to structured concurrency
- **LazyVStack**: Efficient transcript rendering for long conversations
