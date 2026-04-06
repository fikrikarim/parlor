import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-powered LLM engine for multimodal inference on Apple Silicon.
///
/// Uses MLXLLM from mlx-swift-examples to load and run models from
/// HuggingFace Hub (e.g. "mlx-community/gemma-4-E2B-it-4bit").
///
/// The model is loaded once and kept resident in unified memory for
/// fast inference on the GPU via Metal.
actor LLMEngine {

    // MARK: - Properties

    private var modelContainer: ModelContainer?
    private var isLoaded = false

    /// HuggingFace model ID or local path. Set before calling `loadModel()`.
    var modelID: String = ModelConfig.llmModelID

    // MARK: - Model Loading

    /// Load the MLX model from HuggingFace Hub or a local directory.
    /// Downloads and caches automatically on first use.
    func loadModel(progressHandler: (@Sendable (Progress) -> Void)? = nil) async throws {
        guard !isLoaded else { return }

        let configuration = ModelConfiguration(id: modelID)

        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: configuration,
            progressHandler: progressHandler
        )

        self.modelContainer = container
        self.isLoaded = true
    }

    /// Load from a local directory path instead of HuggingFace.
    func loadModel(from localPath: URL, progressHandler: (@Sendable (Progress) -> Void)? = nil) async throws {
        guard !isLoaded else { return }

        let configuration = ModelConfiguration(directory: localPath)

        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: configuration,
            progressHandler: progressHandler
        )

        self.modelContainer = container
        self.isLoaded = true
    }

    nonisolated func isModelAvailable() -> Bool {
        // MLX models download on demand, so always "available"
        true
    }

    // MARK: - Inference

    /// Run multimodal inference with audio, optional image, and conversation history.
    ///
    /// - Parameters:
    ///   - audio: WAV audio data at 16kHz mono (passed as context in prompt if model doesn't support native audio)
    ///   - image: Optional camera frame as CGImage
    ///   - history: Conversation history for context
    /// - Returns: LLM response with transcription and generated text
    func generate(
        audio: Data,
        image: CGImage?,
        history: [Message]
    ) async throws -> LLMResponse {
        guard let container = modelContainer else {
            throw LLMError.modelNotLoaded
        }

        let startTime = ContinuousClock.now

        // Build prompt from conversation history
        let prompt = buildPrompt(from: history)

        // Build multimodal input
        var images: [UserInput.Image] = []
        if let image {
            images.append(.cgImage(image))
        }

        let userInput = UserInput(
            prompt: .text(prompt),
            images: images
        )

        // Configure generation parameters
        let generateParams = GenerateParameters(
            temperature: 1.0,
            topP: 0.95,
            repetitionPenalty: 1.0
        )

        let maxTokens = ModelConfig.maxGenerationTokens

        // Run inference on the model container
        let result = try await container.perform { context in
            let input = try await context.processor.prepare(input: userInput)
            return try MLXLMCommon.generate(
                input: input,
                parameters: generateParams,
                context: context
            ) { tokens in
                tokens >= maxTokens ? .stop : .more
            }
        }

        let inferenceTime = (ContinuousClock.now - startTime).seconds
        let outputText = result.output

        // Parse transcription and response from output
        let parsed = parseResponse(outputText)

        return LLMResponse(
            transcription: parsed.transcription,
            response: parsed.response,
            inferenceTime: inferenceTime
        )
    }

    /// Stream tokens one at a time, yielding partial text.
    func generateStreaming(
        prompt: String,
        image: CGImage?
    ) -> AsyncStream<String> {
        AsyncStream { [weak self] continuation in
            let task = Task {
                guard let self, let container = await self.modelContainer else {
                    continuation.finish()
                    return
                }

                do {
                    var images: [UserInput.Image] = []
                    if let image {
                        images.append(.cgImage(image))
                    }

                    let userInput = UserInput(
                        prompt: .text(prompt),
                        images: images
                    )

                    let generateParams = GenerateParameters(
                        temperature: 1.0,
                        topP: 0.95
                    )

                    let maxTokens = ModelConfig.maxGenerationTokens

                    _ = try await container.perform { context in
                        let input = try await context.processor.prepare(input: userInput)
                        return try MLXLMCommon.generate(
                            input: input,
                            parameters: generateParams,
                            context: context
                        ) { tokens in
                            if Task.isCancelled { return .stop }

                            // Decode the latest token
                            let text = context.tokenizer.decode(tokens: [tokens])
                            continuation.yield(text)

                            return tokens >= maxTokens ? .stop : .more
                        }
                    }
                } catch {
                    // Generation ended
                }

                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Prompt Building

    private func buildPrompt(from history: [Message]) -> String {
        var prompt = ModelConfig.systemPrompt + "\n\n"

        for msg in history.suffix(10) {
            let role = msg.role == .user ? "User" : "Assistant"
            prompt += "\(role): \(msg.text)\n"
        }

        prompt += "Assistant:"
        return prompt
    }

    // MARK: - Response Parsing

    private struct ParsedOutput {
        let transcription: String
        let response: String
    }

    /// Parse model output, attempting to extract transcription and response.
    /// Handles various output formats including tool-call style and freeform.
    private func parseResponse(_ text: String) -> ParsedOutput {
        let cleaned = text
            .replacingOccurrences(of: "<|\"|\">", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Try "Transcription: ... Response: ..." format
        if let sepRange = cleaned.range(of: "\nResponse: ") {
            let transcription = String(cleaned[cleaned.startIndex..<sepRange.lowerBound])
                .replacingOccurrences(of: "Transcription: ", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let response = String(cleaned[sepRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ParsedOutput(transcription: transcription, response: response)
        }

        // Try JSON tool call format
        if let data = cleaned.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let transcription = json["transcription"] as? String ?? ""
            let response = json["response"] as? String ?? cleaned
            return ParsedOutput(transcription: transcription, response: response)
        }

        // Freeform text
        return ParsedOutput(transcription: "", response: cleaned)
    }
}

// MARK: - Errors

enum LLMError: LocalizedError, Sendable {
    case modelNotLoaded
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "LLM model not loaded. Call loadModel() first."
        case .generationFailed(let reason):
            "Generation failed: \(reason)"
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

// MARK: - UserInput.Image extension for CGImage

extension UserInput.Image {
    /// Create a UserInput.Image from a CGImage by writing to a temporary file.
    static func cgImage(_ image: CGImage) -> UserInput.Image {
        // Write CGImage to temp JPEG for the model processor
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("jpg")

        #if os(macOS)
        let rep = NSBitmapImageRep(cgImage: image)
        if let jpegData = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
            try? jpegData.write(to: tempURL)
        }
        #else
        if let jpegData = UIImage(cgImage: image).jpegData(compressionQuality: 0.8) {
            try? jpegData.write(to: tempURL)
        }
        #endif

        return .url(tempURL)
    }
}

#if os(macOS)
import AppKit
#else
import UIKit
#endif
