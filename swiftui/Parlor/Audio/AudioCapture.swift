@preconcurrency import AVFoundation
import Accelerate
import Foundation

/// Manages microphone input capture with integrated VAD and frequency analysis.
///
/// Bridges AVAudioEngine callbacks to structured concurrency via AsyncStream.
/// All AVFoundation interaction happens on the audio engine's internal threads;
/// results are forwarded as Sendable `AudioEvent` values.
final class AudioCapture: @unchecked Sendable {

    // MARK: - Public Interface

    /// Stream of high-level audio events (speech start/end, levels, FFT data).
    let events: AsyncStream<AudioEvent>

    // MARK: - Private State

    private let engine = AVAudioEngine()
    private let eventContinuation: AsyncStream<AudioEvent>.Continuation
    private var processingTask: Task<Void, Never>?

    // Raw audio stream bridging the tap callback to async processing
    private let rawStream: AsyncStream<RawChunk>
    private let rawContinuation: AsyncStream<RawChunk>.Continuation

    private struct RawChunk: Sendable {
        let samples: [Float]
        let rms: Float
        let sampleRate: Double
    }

    // MARK: - Configuration

    /// Set to true during AI speaking to raise VAD threshold (echo suppression).
    /// Thread-safe via atomic-like access pattern (written from main, read from processing task).
    private let _echoSuppression = ManagedAtomic<Bool>(false)

    var echoSuppression: Bool {
        get { _echoSuppression.value }
        set { _echoSuppression.value = newValue }
    }

    // MARK: - Init

    init() {
        let (events, eventCont) = AsyncStream.makeStream(of: AudioEvent.self, bufferingPolicy: .bufferingNewest(64))
        self.events = events
        self.eventContinuation = eventCont

        let (raw, rawCont) = AsyncStream.makeStream(of: RawChunk.self, bufferingPolicy: .bufferingNewest(128))
        self.rawStream = raw
        self.rawContinuation = rawCont
    }

    deinit {
        stop()
    }

    // MARK: - Lifecycle

    func start() throws {
        #if os(macOS)
        // Request microphone permission on macOS
        if #available(macOS 14.0, *) {
            // Permission handled by entitlements
        }
        #endif

        let inputNode = engine.inputNode
        let hardwareFormat = inputNode.outputFormat(forBus: 0)
        let rawCont = self.rawContinuation

        // Install tap to capture audio buffers
        inputNode.installTap(
            onBus: 0,
            bufferSize: 4096,
            format: hardwareFormat
        ) { buffer, _ in
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0 else { return }

            // Compute RMS energy using Accelerate
            var rms: Float = 0
            vDSP_rmsqv(channelData, 1, &rms, vDSP_Length(frameCount))

            // Copy samples
            let samples = Array(UnsafeBufferPointer(start: channelData, count: frameCount))

            rawCont.yield(RawChunk(
                samples: samples,
                rms: rms,
                sampleRate: hardwareFormat.sampleRate
            ))
        }

        try engine.start()

        // Start async processing of raw audio
        processingTask = Task.detached(priority: .userInitiated) { [weak self] in
            await self?.processRawAudio()
        }
    }

    func stop() {
        processingTask?.cancel()
        processingTask = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        rawContinuation.finish()
        eventContinuation.finish()
    }

    // MARK: - Audio Processing Loop

    private func processRawAudio() async {
        var vad = VoiceActivityDetector()
        var recordingBuffer: [Float] = []
        recordingBuffer.reserveCapacity(Int(ModelConfig.captureSampleRate) * 30)
        var currentSampleRate: Double = 16_000

        for await chunk in rawStream {
            guard !Task.isCancelled else { break }

            currentSampleRate = chunk.sampleRate

            // Update VAD echo suppression state
            vad.echoSuppression = _echoSuppression.value

            // Emit audio level
            eventContinuation.yield(.level(chunk.rms))

            // Compute and emit frequency data periodically
            if chunk.samples.count >= 256 {
                let bins = computeFFTBins(chunk.samples, binCount: ModelConfig.fftBinCount)
                eventContinuation.yield(.frequencyData(bins))
            }

            // Process through VAD
            let event = vad.process(energy: chunk.rms)

            switch event {
            case .speechStarted:
                recordingBuffer.removeAll(keepingCapacity: true)
                recordingBuffer.append(contentsOf: chunk.samples)
                eventContinuation.yield(.speechStart)

            case .speechContinuing:
                recordingBuffer.append(contentsOf: chunk.samples)
                // Enforce max duration
                let maxSamples = Int(currentSampleRate * ModelConfig.maxAudioDurationSeconds)
                if recordingBuffer.count > maxSamples {
                    let wavData = createWAV(
                        from: recordingBuffer,
                        sampleRate: currentSampleRate
                    )
                    eventContinuation.yield(.speechEnd(audio: wavData))
                    recordingBuffer.removeAll(keepingCapacity: true)
                    vad.reset()
                }

            case .speechEnded:
                recordingBuffer.append(contentsOf: chunk.samples)
                let wavData = createWAV(
                    from: recordingBuffer,
                    sampleRate: currentSampleRate
                )
                eventContinuation.yield(.speechEnd(audio: wavData))
                recordingBuffer.removeAll(keepingCapacity: true)

            case nil:
                break
            }
        }
    }

    // MARK: - FFT

    private func computeFFTBins(_ samples: [Float], binCount: Int) -> [Float] {
        let fftSize = 256
        guard samples.count >= fftSize else {
            return [Float](repeating: 0, count: binCount)
        }

        let log2n = vDSP_Length(log2(Float(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return [Float](repeating: 0, count: binCount)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Take last fftSize samples and apply Hann window
        var windowed = [Float](repeating: 0, count: fftSize)
        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

        let offset = samples.count - fftSize
        samples.withUnsafeBufferPointer { buf in
            vDSP_vmul(
                buf.baseAddress! + offset, 1,
                window, 1,
                &windowed, 1,
                vDSP_Length(fftSize)
            )
        }

        // Split into real/imaginary for FFT
        let halfSize = fftSize / 2
        var realPart = [Float](repeating: 0, count: halfSize)
        var imagPart = [Float](repeating: 0, count: halfSize)

        // Pack interleaved data into split complex
        windowed.withUnsafeBufferPointer { src in
            realPart.withUnsafeMutableBufferPointer { real in
                imagPart.withUnsafeMutableBufferPointer { imag in
                    var split = DSPSplitComplex(realp: real.baseAddress!, imagp: imag.baseAddress!)
                    src.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(halfSize))
                    }
                }
            }
        }

        // Forward FFT
        realPart.withUnsafeMutableBufferPointer { real in
            imagPart.withUnsafeMutableBufferPointer { imag in
                var split = DSPSplitComplex(realp: real.baseAddress!, imagp: imag.baseAddress!)
                vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
            }
        }

        // Compute magnitudes
        var magnitudes = [Float](repeating: 0, count: halfSize)
        realPart.withUnsafeMutableBufferPointer { real in
            imagPart.withUnsafeMutableBufferPointer { imag in
                var split = DSPSplitComplex(realp: real.baseAddress!, imagp: imag.baseAddress!)
                vDSP_zvabs(&split, 1, &magnitudes, 1, vDSP_Length(halfSize))
            }
        }

        // Bin magnitudes into output bins
        let binsPerOutput = max(1, halfSize / binCount)
        var result = [Float](repeating: 0, count: binCount)
        for i in 0..<binCount {
            let start = i * binsPerOutput
            let end = min(start + binsPerOutput, halfSize)
            guard end > start else { continue }
            var sum: Float = 0
            vDSP_sve(Array(magnitudes[start..<end]), 1, &sum, vDSP_Length(end - start))
            result[i] = sum / Float(end - start)
        }

        // Normalize to 0...1
        var maxVal: Float = 0
        vDSP_maxv(result, 1, &maxVal, vDSP_Length(binCount))
        if maxVal > 0.001 {
            var scale = 1.0 / maxVal
            vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(binCount))
        }

        return result
    }

    // MARK: - WAV Creation

    /// Create a WAV file from float32 PCM samples, resampling to 16kHz if needed.
    private func createWAV(from samples: [Float], sampleRate: Double) -> Data {
        let targetRate = ModelConfig.captureSampleRate
        let finalSamples: [Float]

        if abs(sampleRate - targetRate) > 1.0 {
            // Resample using vDSP
            finalSamples = resample(samples, from: sampleRate, to: targetRate)
        } else {
            finalSamples = samples
        }

        // Build WAV header + PCM data
        let dataSize = finalSamples.count * MemoryLayout<Float>.size
        let fileSize = 44 + dataSize
        var data = Data(capacity: fileSize)

        // RIFF header
        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize - 8).littleEndian) { Array($0) })
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"

        // fmt chunk
        data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // "fmt "
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(3).littleEndian) { Array($0) })  // IEEE float
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // mono
        data.append(contentsOf: withUnsafeBytes(of: UInt32(targetRate).littleEndian) { Array($0) })
        let byteRate = UInt32(targetRate) * UInt32(MemoryLayout<Float>.size)
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(MemoryLayout<Float>.size).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(32).littleEndian) { Array($0) }) // 32-bit

        // data chunk
        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // "data"
        data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })
        finalSamples.withUnsafeBytes { data.append(contentsOf: $0) }

        return data
    }

    /// Simple linear interpolation resampler using vDSP.
    private func resample(_ samples: [Float], from srcRate: Double, to dstRate: Double) -> [Float] {
        let ratio = dstRate / srcRate
        let outputCount = Int(Double(samples.count) * ratio)
        guard outputCount > 0 else { return [] }

        var result = [Float](repeating: 0, count: outputCount)
        var control = (0..<outputCount).map { Float(Double($0) / ratio) }
        vDSP_vlint(samples, &control, 1, &result, 1, vDSP_Length(outputCount), vDSP_Length(samples.count))
        return result
    }
}

// MARK: - Atomic Bool (lock-free)

/// Minimal atomic boolean for cross-thread flag communication.
private final class ManagedAtomic<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Value

    init(_ initialValue: Value) {
        _value = initialValue
    }

    var value: Value {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _value
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _value = newValue
        }
    }
}
