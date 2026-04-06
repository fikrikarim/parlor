@preconcurrency import CoreGraphics
import Foundation

// MARK: - App Phase State Machine

enum AppPhase: String, Sendable, CaseIterable {
    case loading
    case listening
    case processing
    case speaking
}

// MARK: - Conversation Message

struct Message: Identifiable, Sendable {
    let id: UUID
    var role: Role
    var text: String
    var transcription: String?
    var timestamp: Date
    var llmTime: TimeInterval?
    var ttsTime: TimeInterval?
    var hasImage: Bool

    enum Role: String, Sendable {
        case user
        case assistant
    }

    init(
        id: UUID = UUID(),
        role: Role,
        text: String,
        transcription: String? = nil,
        timestamp: Date = Date(),
        llmTime: TimeInterval? = nil,
        ttsTime: TimeInterval? = nil,
        hasImage: Bool = false
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.transcription = transcription
        self.timestamp = timestamp
        self.llmTime = llmTime
        self.ttsTime = ttsTime
        self.hasImage = hasImage
    }
}

// MARK: - LLM Response

struct LLMResponse: Sendable {
    let transcription: String
    let response: String
    let inferenceTime: TimeInterval
}

// MARK: - Audio Events

enum AudioEvent: Sendable {
    case speechStart
    case speechEnd(audio: Data)
    case level(Float)
    case frequencyData([Float])
}

// MARK: - VAD Events

enum VADEvent: Sendable {
    case speechStarted
    case speechContinuing
    case speechEnded
}

// MARK: - Model Configuration

enum ModelConfig {
    /// HuggingFace model ID for the LLM. Change to your preferred MLX model.
    static let llmModelID = "mlx-community/gemma-3-4b-it-4bit"

    /// HuggingFace model ID for Kokoro TTS.
    static let ttsModelID = "mlx-community/Kokoro-82M-bf16"

    static let ttsSampleRate: Double = 24_000
    static let captureSampleRate: Double = 16_000
    static let fftBinCount = 40
    static let maxAudioDurationSeconds: Double = 30.0
    static let maxGenerationTokens = 512

    static let systemPrompt = """
        You are a helpful AI assistant in a real-time voice and vision conversation. \
        The user is speaking to you through their microphone and may be showing you \
        things through their camera. Listen carefully to what they say, observe what \
        they show you, and respond naturally and concisely. Keep responses brief \
        (1-3 sentences) unless asked for detail.
        """
}
