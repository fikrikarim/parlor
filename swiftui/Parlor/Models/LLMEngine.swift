import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXLMHuggingFace
import MLXLMTokenizers
import MLXVLM

#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// MLX-powered LLM engine for multimodal inference on Apple Silicon.
///
/// Uses mlx-swift-lm v3 with `ChatSession` for multi-turn conversation
/// management including KV cache persistence across turns.
///
/// For vision models (image understanding), uses `MLXVLM`. For text-only
/// models, uses `MLXLLM`. The factory is selected automatically based
/// on the model configuration.
actor LLMEngine {

    // MARK: - Properties

    private var modelContainer: ModelContainer?
    private var chatSession: ChatSession?
    private var isLoaded = false

    /// HuggingFace model ID. Set before calling `loadModel()`.
    var modelID: String = ModelConfig.llmModelID

    // MARK: - Model Loading

    /// Load the MLX model from HuggingFace Hub.
    /// Downloads and caches weights on first use (~2-5 GB depending on model).
    func loadModel(progressHandler: (@Sendable (Progress) -> Void)? = nil) async throws {
        guard !isLoaded else { return }

        let hub = HubClient.default

        // Try VLM first (for vision+language), fall back to LLM
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                from: hub,
                using: TokenizersLoader(),
                id: modelID
            )
        } catch {
            // Not a VLM — try as text-only LLM
            container = try await LLMModelFactory.shared.loadContainer(
                from: hub,
                using: TokenizersLoader(),
                id: modelID
            )
        }

        self.modelContainer = container

        // Create ChatSession for multi-turn conversation with KV cache
        let session = ChatSession(container)
        session.instructions = ModelConfig.systemPrompt
        self.chatSession = session

        self.isLoaded = true
    }

    /// Load from a local directory path.
    func loadModel(from localPath: URL) async throws {
        guard !isLoaded else { return }

        let container = try await loadModelContainer(
            from: localPath,
            using: TokenizersLoader()
        )

        self.modelContainer = container

        let session = ChatSession(container)
        session.instructions = ModelConfig.systemPrompt
        self.chatSession = session

        self.isLoaded = true
    }

    nonisolated func isModelAvailable() -> Bool {
        // MLX models download on demand from HuggingFace
        true
    }

    // MARK: - Inference

    /// Run inference with optional image and conversation context.
    ///
    /// Uses `ChatSession` which automatically manages multi-turn KV cache,
    /// so conversation history does not need to be re-processed each turn.
    func generate(
        audio: Data,
        image: CGImage?,
        history: [Message]
    ) async throws -> LLMResponse {
        guard let session = chatSession else {
            throw LLMError.modelNotLoaded
        }

        let startTime = ContinuousClock.now

        // Build the user prompt
        // Since MLX Swift doesn't have native audio input for most models,
        // we pass audio context as a note. For models with native audio,
        // extend UserInput accordingly.
        let prompt = buildTurnPrompt(from: history)

        // Generate response — streaming to collect full text
        var responseText = ""

        if let image {
            // VLM path: include image
            let imageURL = writeImageToTemp(image)
            let messages: [Chat.Message] = [
                .user(prompt, images: [.url(imageURL)])
            ]

            for try await chunk in try await session.streamDetails(
                to: messages,
                parameters: GenerateParameters(
                    maxTokens: ModelConfig.maxGenerationTokens,
                    temperature: 1.0,
                    topP: 0.95
                )
            ) {
                if let text = chunk.chunk {
                    responseText += text
                }
            }

            // Clean up temp image
            try? FileManager.default.removeItem(at: imageURL)
        } else {
            // Text-only path
            for try await chunk in try await session.streamResponse(
                to: prompt,
                parameters: GenerateParameters(
                    maxTokens: ModelConfig.maxGenerationTokens,
                    temperature: 1.0,
                    topP: 0.95
                )
            ) {
                responseText += chunk
            }
        }

        let inferenceTime = (ContinuousClock.now - startTime).seconds

        // Parse transcription and response
        let parsed = parseResponse(responseText)

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
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { [weak self] continuation in
            let task = Task {
                guard let self, let session = await self.chatSession else {
                    continuation.finish()
                    return
                }

                do {
                    if let image {
                        let imageURL = await self.writeImageToTemp(image)
                        let messages: [Chat.Message] = [
                            .user(prompt, images: [.url(imageURL)])
                        ]
                        for try await chunk in try await session.streamDetails(to: messages) {
                            if let text = chunk.chunk {
                                continuation.yield(text)
                            }
                        }
                        try? FileManager.default.removeItem(at: imageURL)
                    } else {
                        for try await chunk in try await session.streamResponse(to: prompt) {
                            continuation.yield(chunk)
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                    return
                }

                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// Reset conversation state and KV cache.
    func resetConversation() {
        chatSession?.clear()
    }

    // MARK: - Prompt Building

    /// Build the current turn's prompt from the latest user message context.
    private func buildTurnPrompt(from history: [Message]) -> String {
        // ChatSession manages history internally via KV cache,
        // so we only need the current turn's content.
        guard let lastUser = history.last(where: { $0.role == .user }) else {
            return "Please respond."
        }

        var prompt = ""
        if lastUser.hasImage {
            prompt += "The user is showing you something through their camera. "
        }
        prompt += "The user said: \(lastUser.text)"
        prompt += "\n\nRespond naturally and concisely."
        return prompt
    }

    // MARK: - Response Parsing

    private struct ParsedOutput {
        let transcription: String
        let response: String
    }

    private func parseResponse(_ text: String) -> ParsedOutput {
        let cleaned = text
            .replacingOccurrences(of: "<|\"|\">", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Try JSON tool call format
        if let data = cleaned.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let transcription = json["transcription"] as? String ?? ""
            let response = json["response"] as? String ?? cleaned
            return ParsedOutput(transcription: transcription, response: response)
        }

        return ParsedOutput(transcription: "", response: cleaned)
    }

    // MARK: - Image Helpers

    private func writeImageToTemp(_ image: CGImage) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("jpg")

        #if os(macOS)
        let rep = NSBitmapImageRep(cgImage: image)
        if let data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
            try? data.write(to: url)
        }
        #else
        if let data = UIImage(cgImage: image).jpegData(compressionQuality: 0.8) {
            try? data.write(to: url)
        }
        #endif

        return url
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
