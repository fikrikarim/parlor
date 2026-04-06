import Foundation

/// Energy-based Voice Activity Detector with hysteresis.
/// Processes RMS energy values and emits speech start/end events.
/// Designed to run synchronously in an audio processing loop.
struct VoiceActivityDetector: Sendable {

    // MARK: - Configuration

    /// RMS threshold to trigger speech detection
    var speechThreshold: Float = 0.015

    /// RMS threshold below which silence is detected
    var silenceThreshold: Float = 0.008

    /// Number of consecutive frames above threshold to confirm speech
    var speechConfirmFrames: Int = 3

    /// Number of consecutive frames below threshold to confirm silence
    var silenceConfirmFrames: Int = 20

    /// Raised threshold during speaking state (echo prevention)
    var echoSuppressedThreshold: Float = 0.06

    /// Whether to use raised threshold (set during AI speaking)
    var echoSuppression: Bool = false

    // MARK: - State

    private enum State: Sendable {
        case idle
        case pendingSpeech(count: Int)
        case inSpeech
        case pendingSilence(count: Int)
    }

    private var state: State = .idle

    // MARK: - Processing

    /// Process a single RMS energy value. Returns a VAD event if a state transition occurred.
    mutating func process(energy: Float) -> VADEvent? {
        let threshold = echoSuppression ? echoSuppressedThreshold : speechThreshold

        switch state {
        case .idle:
            if energy > threshold {
                state = .pendingSpeech(count: 1)
            }
            return nil

        case .pendingSpeech(let count):
            if energy > threshold {
                let next = count + 1
                if next >= speechConfirmFrames {
                    state = .inSpeech
                    return .speechStarted
                }
                state = .pendingSpeech(count: next)
            } else {
                state = .idle
            }
            return nil

        case .inSpeech:
            if energy < silenceThreshold {
                state = .pendingSilence(count: 1)
            }
            return .speechContinuing

        case .pendingSilence(let count):
            if energy >= silenceThreshold {
                state = .inSpeech
                return .speechContinuing
            }
            let next = count + 1
            if next >= silenceConfirmFrames {
                state = .idle
                return .speechEnded
            }
            state = .pendingSilence(count: next)
            return .speechContinuing
        }
    }

    /// Reset the detector to idle state.
    mutating func reset() {
        state = .idle
    }
}
