@preconcurrency import AVFoundation
import Foundation
import SwiftUI

/// Central coordinator for the Parlor conversation loop.
///
/// Orchestrates: Audio capture → VAD → LLM inference → TTS → Playback.
/// All UI-bound state is `@MainActor`-isolated via `@Observable`.
@MainActor
@Observable
final class ConversationViewModel {

    // MARK: - Observable State

    var phase: AppPhase = .loading
    var messages: [Message] = []
    var isCameraEnabled = true
    var audioLevel: Float = 0
    var frequencyBins: [Float] = Array(repeating: 0, count: ModelConfig.fftBinCount)
    var isModelLoaded = false
    var errorMessage: String?
    var statusText: String = "Initializing..."

    // MARK: - Private

    private let llm = LLMEngine()
    private let tts = TTSEngine()
    private let audioCapture = AudioCapture()
    private let audioPlayer = AudioPlayer()
    private let camera = CameraManager()

    private var listeningTask: Task<Void, Never>?
    private var outputLevelTask: Task<Void, Never>?
    private var isInterrupted = false
    private var bargeInGraceDeadline: ContinuousClock.Instant?

    // MARK: - Lifecycle

    /// Initialize all subsystems and start the conversation loop.
    func start() async {
        phase = .loading
        statusText = "Loading models..."

        // Load MLX models (downloads from HuggingFace on first use)
        do {
            statusText = "Loading LLM (\(ModelConfig.llmModelID))..."
            try await llm.loadModel()
            statusText = "LLM loaded. Loading TTS..."

            try await tts.loadModel()
            statusText = "Models loaded"
        } catch {
            errorMessage = error.localizedDescription
            statusText = "Model loading failed"
            return
        }

        // Start audio capture
        statusText = "Starting audio..."
        do {
            try audioCapture.start()
        } catch {
            errorMessage = "Microphone: \(error.localizedDescription)"
            statusText = "Microphone access failed"
            return
        }

        // Start camera
        statusText = "Starting camera..."
        do {
            try await camera.start()
        } catch {
            // Camera failure is non-fatal
            isCameraEnabled = false
        }

        isModelLoaded = true
        phase = .listening
        statusText = "Ready"

        // Start listening for audio events
        listeningTask = Task { [weak self] in
            guard let self else { return }
            for await event in self.audioCapture.events {
                guard !Task.isCancelled else { break }
                await self.handleAudioEvent(event)
            }
        }

        // Monitor output audio levels during playback
        outputLevelTask = Task { [weak self] in
            guard let self else { return }
            for await level in self.audioPlayer.outputLevels {
                guard !Task.isCancelled else { break }
                if self.phase == .speaking {
                    self.audioLevel = level
                }
            }
        }
    }

    /// Shut down all subsystems.
    func shutdown() {
        listeningTask?.cancel()
        outputLevelTask?.cancel()
        audioCapture.stop()
        audioPlayer.stop()
        camera.stop()
    }

    // MARK: - Audio Event Handling

    private func handleAudioEvent(_ event: AudioEvent) async {
        switch event {
        case .speechStart:
            if phase == .speaking {
                // Barge-in: user is interrupting the AI
                if let deadline = bargeInGraceDeadline,
                   ContinuousClock.now < deadline {
                    // Within grace period — ignore to prevent echo false trigger
                    return
                }
                await interrupt()
            }

        case .speechEnd(let audioData):
            guard phase == .listening else { return }
            await processUserInput(audio: audioData)

        case .level(let level):
            if phase != .speaking {
                audioLevel = level
            }

        case .frequencyData(let bins):
            if phase != .speaking {
                frequencyBins = bins
            }
        }
    }

    // MARK: - Conversation Processing

    private func processUserInput(audio: Data) async {
        phase = .processing
        isInterrupted = false
        audioCapture.echoSuppression = false
        statusText = "Thinking..."

        // Capture camera frame
        let frame: CGImage? = isCameraEnabled ? camera.captureFrame() : nil

        // Add placeholder user message
        let userMsgId = UUID()
        messages.append(Message(
            id: userMsgId,
            role: .user,
            text: "...",
            hasImage: frame != nil
        ))

        do {
            // Run LLM inference
            let response = try await llm.generate(
                audio: audio,
                image: frame,
                history: messages
            )

            // Update user message with transcription
            if let idx = messages.firstIndex(where: { $0.id == userMsgId }) {
                messages[idx].text = response.transcription.isEmpty
                    ? "(audio)"
                    : response.transcription
                messages[idx].llmTime = response.inferenceTime
            }

            guard !isInterrupted else {
                phase = .listening
                statusText = "Ready"
                return
            }

            // Add assistant message
            let assistantMsgId = UUID()
            messages.append(Message(
                id: assistantMsgId,
                role: .assistant,
                text: response.response,
                llmTime: response.inferenceTime
            ))

            // Synthesize and play speech
            phase = .speaking
            statusText = "Speaking"
            audioCapture.echoSuppression = true
            bargeInGraceDeadline = ContinuousClock.now + .milliseconds(800)

            let ttsStart = ContinuousClock.now
            let audioSamples = try await tts.synthesize(text: response.response)
            let ttsTime = (ContinuousClock.now - ttsStart).seconds

            guard !isInterrupted else {
                phase = .listening
                statusText = "Ready"
                audioCapture.echoSuppression = false
                return
            }

            // Update message with TTS timing
            if let idx = messages.firstIndex(where: { $0.id == assistantMsgId }) {
                messages[idx].ttsTime = ttsTime
            }

            // Play audio
            await audioPlayer.play(audioSamples)

            // Return to listening
            phase = .listening
            statusText = "Ready"
            audioCapture.echoSuppression = false

        } catch {
            errorMessage = error.localizedDescription
            phase = .listening
            statusText = "Ready"
            audioCapture.echoSuppression = false
        }
    }

    // MARK: - Interruption

    func interrupt() async {
        isInterrupted = true
        audioPlayer.stop()
        audioCapture.echoSuppression = false
        phase = .listening
        statusText = "Ready"
    }

    // MARK: - Controls

    func toggleCamera() {
        isCameraEnabled.toggle()
        if isCameraEnabled {
            Task {
                try? await camera.start()
            }
        } else {
            camera.stop()
        }
    }

    // MARK: - Camera Session Access

    var captureSession: AVCaptureSession {
        camera.session
    }
}
