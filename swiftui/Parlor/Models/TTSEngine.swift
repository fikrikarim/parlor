@preconcurrency import CoreML
import Accelerate
import Foundation

/// CoreML-based Text-to-Speech engine wrapping Kokoro-82M.
///
/// To use, convert the Kokoro model to CoreML format and place the
/// `.mlmodelc` bundle in the app's resources.
///
/// The model should accept:
///   - Phoneme/token IDs as MLMultiArray (Int32)
///   - Voice embedding as MLMultiArray (Float32) [optional]
///   - Speed factor as Double [optional]
/// And output:
///   - PCM audio samples as MLMultiArray (Float32) at 24kHz
///
/// Conversion example with coremltools:
/// ```python
/// import coremltools as ct
/// # See README.md for conversion instructions
/// ```
actor TTSEngine {

    // MARK: - Properties

    private var model: MLModel?
    private var isLoaded = false

    nonisolated let sampleRate = ModelConfig.ttsSampleRate
    nonisolated let modelName = ModelConfig.ttsModelName

    // MARK: - Configuration

    /// Voice identifier for the TTS model
    var voice: String = "af_heart"

    /// Speech speed multiplier (1.0 = normal, 1.1 = slightly faster)
    var speed: Float = 1.1

    // MARK: - Model Loading

    func loadModel() async throws {
        guard !isLoaded else { return }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        guard let modelURL = Bundle.main.url(
            forResource: ModelConfig.ttsModelName,
            withExtension: "mlmodelc"
        ) else {
            throw TTSError.modelNotFound(ModelConfig.ttsModelName)
        }

        let compiledModel = try await MLModel.load(
            contentsOf: modelURL,
            configuration: config
        )
        self.model = compiledModel
        self.isLoaded = true
    }

    nonisolated func isModelAvailable() -> Bool {
        Bundle.main.url(
            forResource: ModelConfig.ttsModelName,
            withExtension: "mlmodelc"
        ) != nil
    }

    // MARK: - Synthesis

    /// Synthesize speech from text, returning PCM float32 samples at 24kHz.
    func synthesize(text: String) async throws -> [Float] {
        guard let model else {
            throw TTSError.modelNotLoaded
        }

        // Split into sentences for streaming-style generation
        let sentences = splitIntoSentences(text)
        var allSamples: [Float] = []
        allSamples.reserveCapacity(Int(ModelConfig.ttsSampleRate) * 10) // ~10 seconds

        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let input = try buildInput(text: trimmed)
            // Run prediction off-actor for performance
            let capturedModel = model
            let output = try await Task.detached(priority: .userInitiated) {
                try capturedModel.prediction(from: input)
            }.value

            let samples = try extractSamples(from: output)
            allSamples.append(contentsOf: samples)
        }

        return allSamples
    }

    /// Synthesize speech sentence by sentence, yielding audio chunks progressively.
    func synthesizeStreaming(text: String) -> AsyncStream<[Float]> {
        let sentences = splitIntoSentences(text)

        return AsyncStream { continuation in
            let task = Task { [weak self] in
                for sentence in sentences {
                    guard !Task.isCancelled else { break }
                    let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { continue }

                    do {
                        guard let self else { break }
                        let input = try await self.buildInput(text: trimmed)
                        guard let model = await self.model else { break }
                        let output = try await Task.detached(priority: .userInitiated) {
                            try model.prediction(from: input)
                        }.value
                        let samples = try await self.extractSamples(from: output)
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

    // MARK: - Input Construction

    private func buildInput(text: String) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [:]

        // Convert text to phoneme/token IDs
        // This is a simplified tokenizer. Replace with the actual Kokoro tokenizer.
        let tokenIDs = tokenize(text)
        let tokenArray = try MLMultiArray(
            shape: [1, NSNumber(value: tokenIDs.count)],
            dataType: .int32
        )
        for (i, token) in tokenIDs.enumerated() {
            tokenArray[i] = NSNumber(value: token)
        }
        features["input_ids"] = MLFeatureValue(multiArray: tokenArray)

        // Speed factor
        let speedArray = try MLMultiArray(shape: [1], dataType: .float32)
        speedArray[0] = NSNumber(value: speed)
        features["speed"] = MLFeatureValue(multiArray: speedArray)

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    // MARK: - Output Parsing

    private func extractSamples(from output: MLFeatureProvider) throws -> [Float] {
        // Try standard output names
        for name in ["audio_output", "waveform", "output", "audio"] {
            if let multiArray = output.featureValue(for: name)?.multiArrayValue {
                return multiArrayToFloats(multiArray)
            }
        }
        throw TTSError.invalidOutput
    }

    private func multiArrayToFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        result.withUnsafeMutableBufferPointer { dest in
            dest.baseAddress!.update(from: ptr, count: count)
        }
        return result
    }

    // MARK: - Text Processing

    private func splitIntoSentences(_ text: String) -> [String] {
        // Split on sentence-ending punctuation followed by whitespace
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

    /// Simplified tokenizer. Replace with actual Kokoro tokenizer (IPA/phoneme based).
    private func tokenize(_ text: String) -> [Int32] {
        // Placeholder: convert UTF-8 bytes to token IDs
        // In production, use the Kokoro phonemizer + tokenizer
        Array(text.utf8).map { Int32($0) }
    }
}

// MARK: - Errors

enum TTSError: LocalizedError, Sendable {
    case modelNotFound(String)
    case modelNotLoaded
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            "CoreML model '\(name).mlmodelc' not found in app bundle"
        case .modelNotLoaded:
            "TTS model not loaded. Call loadModel() first."
        case .invalidOutput:
            "Could not extract audio samples from model output"
        }
    }
}
