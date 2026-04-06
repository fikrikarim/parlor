@preconcurrency import CoreML
import Foundation

/// CoreML-based LLM engine for multimodal inference.
///
/// To use, convert your model to CoreML format and place the `.mlmodelc` bundle
/// in the app's resources. The model should accept:
///   - Audio input (16kHz PCM float32)
///   - Image input (pixel buffer or multi-array)
///   - Text tokens (multi-array of Int32)
/// And output generated token IDs.
///
/// For Gemma 4 E2B, convert using `coremltools`:
/// ```python
/// import coremltools as ct
/// # See README.md for full conversion instructions
/// ```
actor LLMEngine {

    // MARK: - Properties

    private var model: MLModel?
    private var isLoaded = false

    nonisolated let modelName = ModelConfig.llmModelName

    // MARK: - Model Loading

    /// Load the CoreML model from the app bundle.
    func loadModel() async throws {
        guard !isLoaded else { return }

        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine + GPU + CPU

        guard let modelURL = Bundle.main.url(
            forResource: ModelConfig.llmModelName,
            withExtension: "mlmodelc"
        ) else {
            throw LLMError.modelNotFound(ModelConfig.llmModelName)
        }

        let compiledModel = try await MLModel.load(
            contentsOf: modelURL,
            configuration: config
        )
        self.model = compiledModel
        self.isLoaded = true
    }

    /// Check if a model bundle exists in the app resources.
    nonisolated func isModelAvailable() -> Bool {
        Bundle.main.url(
            forResource: ModelConfig.llmModelName,
            withExtension: "mlmodelc"
        ) != nil
    }

    // MARK: - Inference

    /// Run multimodal inference with audio, optional image, and conversation history.
    ///
    /// - Parameters:
    ///   - audio: WAV audio data at 16kHz mono
    ///   - image: Optional camera frame
    ///   - history: Conversation history for context
    /// - Returns: LLM response with transcription and generated text
    func generate(
        audio: Data,
        image: CGImage?,
        history: [Message]
    ) async throws -> LLMResponse {
        guard let model else {
            throw LLMError.modelNotLoaded
        }

        let startTime = ContinuousClock.now

        // Build model input from multimodal data
        let input = try buildInput(audio: audio, image: image, history: history)

        // Run prediction off-actor for performance
        let capturedModel = model
        let output = try await Task.detached(priority: .userInitiated) {
            try capturedModel.prediction(from: input)
        }.value

        let inferenceTime = (ContinuousClock.now - startTime).seconds

        // Parse output
        let result = try parseOutput(output)

        return LLMResponse(
            transcription: result.transcription,
            response: result.response,
            inferenceTime: inferenceTime
        )
    }

    // MARK: - Input Construction

    /// Build MLFeatureProvider from multimodal inputs.
    ///
    /// Override this method to match your specific CoreML model's input schema.
    /// The default implementation creates a dictionary-based feature provider
    /// with standard input names.
    private func buildInput(
        audio: Data,
        image: CGImage?,
        history: [Message]
    ) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [:]

        // Audio input: convert WAV data to MLMultiArray
        let audioSamples = try extractPCMSamples(from: audio)
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: audioSamples.count)], dataType: .float32)
        for (i, sample) in audioSamples.enumerated() {
            audioArray[i] = NSNumber(value: sample)
        }
        features["audio_input"] = MLFeatureValue(multiArray: audioArray)

        // Image input: convert CGImage to pixel buffer
        if let image {
            let pixelBuffer = try createPixelBuffer(from: image)
            features["image_input"] = MLFeatureValue(pixelBuffer: pixelBuffer)
        }

        // Context: build prompt from history
        let prompt = buildPrompt(from: history)
        let promptData = Array(prompt.utf8).map { Float($0) }
        let promptArray = try MLMultiArray(
            shape: [1, NSNumber(value: promptData.count)],
            dataType: .float32
        )
        for (i, val) in promptData.enumerated() {
            promptArray[i] = NSNumber(value: val)
        }
        features["text_input"] = MLFeatureValue(multiArray: promptArray)

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    // MARK: - Output Parsing

    private struct ParsedOutput {
        let transcription: String
        let response: String
    }

    /// Parse model output into transcription and response text.
    ///
    /// Override to match your CoreML model's output schema.
    private func parseOutput(_ output: MLFeatureProvider) throws -> ParsedOutput {
        // Attempt to read from standard output feature names
        if let outputArray = output.featureValue(for: "output_tokens")?.multiArrayValue {
            let tokens = (0..<outputArray.count).map { outputArray[$0].int32Value }
            let text = decodeTokens(tokens)

            // Try to split into transcription and response
            // Format: "Transcription: <text>\nResponse: <text>"
            if let sepRange = text.range(of: "\nResponse: ") {
                let transcription = String(text[text.startIndex..<sepRange.lowerBound])
                    .replacingOccurrences(of: "Transcription: ", with: "")
                let response = String(text[sepRange.upperBound...])
                return ParsedOutput(transcription: transcription, response: response)
            }

            return ParsedOutput(transcription: "", response: text)
        }

        // Fallback: try text output
        if let textValue = output.featureValue(for: "output_text")?.stringValue {
            return ParsedOutput(transcription: "", response: textValue)
        }

        throw LLMError.invalidOutput
    }

    // MARK: - Helpers

    private func extractPCMSamples(from wavData: Data) throws -> [Float] {
        // Skip WAV header (44 bytes) and read PCM float32 samples
        guard wavData.count > 44 else {
            throw LLMError.invalidAudioData
        }
        let pcmData = wavData.dropFirst(44)
        let sampleCount = pcmData.count / MemoryLayout<Float>.size
        return pcmData.withUnsafeBytes { buffer in
            guard let baseAddress = buffer.baseAddress else { return [] }
            let floatPtr = baseAddress.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: floatPtr, count: sampleCount))
        }
    }

    private func createPixelBuffer(from image: CGImage) throws -> CVPixelBuffer {
        let width = image.width
        let height = image.height
        var pixelBuffer: CVPixelBuffer?

        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw LLMError.pixelBufferCreationFailed
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw LLMError.pixelBufferCreationFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private func buildPrompt(from history: [Message]) -> String {
        var prompt = ModelConfig.systemPrompt + "\n\n"
        for msg in history.suffix(10) {
            let role = msg.role == .user ? "User" : "Assistant"
            prompt += "\(role): \(msg.text)\n"
        }
        return prompt
    }

    /// Placeholder token decoder. Replace with your model's actual tokenizer.
    private func decodeTokens(_ tokens: [Int32]) -> String {
        // This is a placeholder. In production, use the model's actual tokenizer
        // (e.g., SentencePiece, BPE) to decode token IDs back to text.
        String(tokens.compactMap { UnicodeScalar(Int($0)).map(Character.init) })
    }
}

// MARK: - Errors

enum LLMError: LocalizedError, Sendable {
    case modelNotFound(String)
    case modelNotLoaded
    case invalidAudioData
    case invalidOutput
    case pixelBufferCreationFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            "CoreML model '\(name).mlmodelc' not found in app bundle"
        case .modelNotLoaded:
            "LLM model not loaded. Call loadModel() first."
        case .invalidAudioData:
            "Audio data is too short or corrupted"
        case .invalidOutput:
            "Could not parse model output"
        case .pixelBufferCreationFailed:
            "Failed to create pixel buffer from image"
        }
    }
}

// MARK: - Duration Helper

extension Duration {
    var seconds: TimeInterval {
        let (secs, attos) = components
        return Double(secs) + Double(attos) * 1e-18
    }
}
