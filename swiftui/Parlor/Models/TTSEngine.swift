import Foundation
import KokoroSwift
import MLX

/// MLX-powered Text-to-Speech engine using Kokoro-82M via kokoro-ios.
///
/// Runs entirely on Apple Silicon GPU via Metal. Achieves ~3x real-time
/// performance on modern devices.
///
/// Model weights are downloaded from HuggingFace on first use and cached.
actor TTSEngine {

    // MARK: - Properties

    private var kokoro: KokoroTTS?
    private var isLoaded = false

    nonisolated let sampleRate = ModelConfig.ttsSampleRate

    /// Voice preset identifier (e.g. "af_heart")
    var voice: String = "af_heart"

    /// Speech speed multiplier (1.0 = normal, 1.1 = slightly faster)
    var speed: Float = 1.1

    // MARK: - Model Loading

    /// Load Kokoro model. Downloads weights on first use.
    ///
    /// - Parameter modelPath: Optional local path to model weights.
    ///   If nil, downloads from HuggingFace Hub.
    func loadModel(modelPath: String? = nil) async throws {
        guard !isLoaded else { return }

        let tts: KokoroTTS
        if let modelPath {
            let url = URL(fileURLWithPath: modelPath)
            tts = try KokoroTTS(modelPath: url)
        } else {
            // Download from HuggingFace Hub
            tts = try await KokoroTTS.downloadAndLoad(
                from: ModelConfig.ttsModelID
            )
        }

        self.kokoro = tts
        self.isLoaded = true
    }

    nonisolated func isModelAvailable() -> Bool {
        // MLX models download on demand
        true
    }

    // MARK: - Synthesis

    /// Synthesize speech from text, returning PCM float32 samples at 24kHz.
    func synthesize(text: String) async throws -> [Float] {
        guard let kokoro else {
            throw TTSError.modelNotLoaded
        }

        let sentences = splitIntoSentences(text)
        var allSamples: [Float] = []
        allSamples.reserveCapacity(Int(ModelConfig.ttsSampleRate) * 10)

        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let audio = try kokoro.generate(
                text: trimmed,
                voice: voice,
                speed: speed
            )

            // Convert MLXArray to [Float]
            let samples = audio.asArray(Float.self)
            allSamples.append(contentsOf: samples)
        }

        return allSamples
    }

    /// Synthesize sentence by sentence, yielding audio chunks progressively.
    func synthesizeStreaming(text: String) -> AsyncStream<[Float]> {
        let sentences = splitIntoSentences(text)
        let capturedVoice = voice
        let capturedSpeed = speed

        return AsyncStream { [weak self] continuation in
            let task = Task {
                guard let self else {
                    continuation.finish()
                    return
                }

                for sentence in sentences {
                    guard !Task.isCancelled else { break }
                    let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { continue }

                    do {
                        guard let kokoro = await self.kokoro else { break }
                        let audio = try kokoro.generate(
                            text: trimmed,
                            voice: capturedVoice,
                            speed: capturedSpeed
                        )
                        let samples = audio.asArray(Float.self)
                        continuation.yield(samples)
                    } catch {
                        break
                    }
                }
                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Text Processing

    private func splitIntoSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""

        for char in text {
            current.append(char)
            if ".!?".contains(char) {
                sentences.append(current)
                current = ""
            }
        }

        if !current.trimmingCharacters(in: .whitespaces).isEmpty {
            sentences.append(current)
        }

        return sentences
    }
}

// MARK: - KokoroTTS convenience extension

extension KokoroTTS {
    /// Download model from HuggingFace and initialize.
    static func downloadAndLoad(from modelID: String) async throws -> KokoroTTS {
        // kokoro-ios handles download and caching internally
        // when initialized with a HuggingFace model identifier
        return try KokoroTTS(modelID: modelID)
    }
}

// MARK: - Errors

enum TTSError: LocalizedError, Sendable {
    case modelNotFound(String)
    case modelNotLoaded
    case synthesizeFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            "TTS model '\(name)' not found"
        case .modelNotLoaded:
            "TTS model not loaded. Call loadModel() first."
        case .synthesizeFailed(let reason):
            "Speech synthesis failed: \(reason)"
        }
    }
}
