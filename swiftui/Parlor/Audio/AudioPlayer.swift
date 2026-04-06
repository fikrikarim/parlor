@preconcurrency import AVFoundation
import Accelerate
import Foundation

/// Manages streaming audio playback with level metering for visualization.
///
/// Audio chunks are scheduled for gapless playback on an AVAudioPlayerNode.
/// Provides real-time output levels and frequency data via AsyncStream.
final class AudioPlayer: @unchecked Sendable {

    // MARK: - Public Streams

    /// Output audio levels for waveform visualization during playback.
    let outputLevels: AsyncStream<Float>

    /// Output frequency bins for spectrum visualization during playback.
    let outputFrequencyData: AsyncStream<[Float]>

    // MARK: - Private State

    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let levelContinuation: AsyncStream<Float>.Continuation
    private let freqContinuation: AsyncStream<[Float]>.Continuation

    private var completionContinuation: CheckedContinuation<Void, Never>?
    private var pendingBuffers = 0
    private let lock = NSLock()

    // MARK: - Init

    init(sampleRate: Double = ModelConfig.ttsSampleRate) {
        let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
        self.format = fmt

        let (levels, levelCont) = AsyncStream.makeStream(of: Float.self, bufferingPolicy: .bufferingNewest(32))
        self.outputLevels = levels
        self.levelContinuation = levelCont

        let (freq, freqCont) = AsyncStream.makeStream(of: [Float].self, bufferingPolicy: .bufferingNewest(16))
        self.outputFrequencyData = freq
        self.freqContinuation = freqCont

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: fmt)

        // Install tap on main mixer for output level metering
        let binCount = ModelConfig.fftBinCount
        let fCont = freqCont
        let lCont = levelCont
        engine.mainMixerNode.installTap(
            onBus: 0,
            bufferSize: 1024,
            format: fmt
        ) { buffer, _ in
            guard let data = buffer.floatChannelData?[0] else { return }
            let count = Int(buffer.frameLength)
            guard count > 0 else { return }

            var rms: Float = 0
            vDSP_rmsqv(data, 1, &rms, vDSP_Length(count))
            lCont.yield(rms)

            // Simple frequency estimation from buffer
            if count >= 256 {
                let bins = AudioPlayer.quickFFTBins(data, frameCount: count, binCount: binCount)
                fCont.yield(bins)
            }
        }
    }

    deinit {
        engine.mainMixerNode.removeTap(onBus: 0)
        engine.stop()
        levelContinuation.finish()
        freqContinuation.finish()
    }

    // MARK: - Playback

    /// Play a complete audio buffer and wait for completion.
    func play(_ samples: [Float]) async {
        guard !samples.isEmpty else { return }

        ensureEngineRunning()

        let buffer = makePCMBuffer(from: samples)
        guard let buffer else { return }

        lock.lock()
        pendingBuffers = 1
        lock.unlock()

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.lock()
            completionContinuation = continuation
            lock.unlock()

            playerNode.scheduleBuffer(buffer) { [weak self] in
                self?.bufferCompleted()
            }
            playerNode.play()
        }
    }

    /// Schedule multiple audio chunks for gapless streaming playback.
    /// Returns when all chunks have finished playing.
    func playStreaming(_ chunks: AsyncStream<[Float]>) async {
        ensureEngineRunning()

        lock.lock()
        pendingBuffers = 0
        lock.unlock()

        var didScheduleAny = false

        for await chunk in chunks {
            guard !chunk.isEmpty else { continue }
            guard let buffer = makePCMBuffer(from: chunk) else { continue }

            lock.lock()
            pendingBuffers += 1
            lock.unlock()

            playerNode.scheduleBuffer(buffer) { [weak self] in
                self?.bufferCompleted()
            }

            if !didScheduleAny {
                playerNode.play()
                didScheduleAny = true
            }
        }

        // Wait for all scheduled buffers to finish
        if didScheduleAny {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                lock.lock()
                if pendingBuffers <= 0 {
                    lock.unlock()
                    continuation.resume()
                } else {
                    completionContinuation = continuation
                    lock.unlock()
                }
            }
        }
    }

    /// Stop playback immediately.
    func stop() {
        playerNode.stop()

        lock.lock()
        pendingBuffers = 0
        let cont = completionContinuation
        completionContinuation = nil
        lock.unlock()

        cont?.resume()
    }

    // MARK: - Private

    private func ensureEngineRunning() {
        if !engine.isRunning {
            try? engine.start()
        }
    }

    private func makePCMBuffer(from samples: [Float]) -> AVAudioPCMBuffer? {
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        )
        guard let buffer else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        return buffer
    }

    private func bufferCompleted() {
        lock.lock()
        pendingBuffers -= 1
        let shouldResume = pendingBuffers <= 0
        let cont = shouldResume ? completionContinuation : nil
        if shouldResume { completionContinuation = nil }
        lock.unlock()

        cont?.resume()
    }

    // MARK: - Static FFT Helper

    fileprivate static func quickFFTBins(
        _ data: UnsafePointer<Float>,
        frameCount: Int,
        binCount: Int
    ) -> [Float] {
        let fftSize = 256
        guard frameCount >= fftSize else {
            return [Float](repeating: 0, count: binCount)
        }

        let log2n = vDSP_Length(log2(Float(fftSize)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return [Float](repeating: 0, count: binCount)
        }
        defer { vDSP_destroy_fftsetup(setup) }

        let halfSize = fftSize / 2
        var real = [Float](repeating: 0, count: halfSize)
        var imag = [Float](repeating: 0, count: halfSize)

        // Use last fftSize samples
        let offset = frameCount - fftSize

        real.withUnsafeMutableBufferPointer { realBuf in
            imag.withUnsafeMutableBufferPointer { imagBuf in
                var split = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                (data + offset).withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { ptr in
                    vDSP_ctoz(ptr, 2, &split, 1, vDSP_Length(halfSize))
                }
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                vDSP_zvabs(&split, 1, &real, 1, vDSP_Length(halfSize))
            }
        }

        // Bin into output
        let binsPerOutput = max(1, halfSize / binCount)
        var result = [Float](repeating: 0, count: binCount)
        for i in 0..<binCount {
            let start = i * binsPerOutput
            let end = min(start + binsPerOutput, halfSize)
            guard end > start else { continue }
            var sum: Float = 0
            for j in start..<end { sum += real[j] }
            result[i] = sum / Float(end - start)
        }

        var maxVal: Float = 0
        vDSP_maxv(result, 1, &maxVal, vDSP_Length(binCount))
        if maxVal > 0.001 {
            var scale = 1.0 / maxVal
            vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(binCount))
        }

        return result
    }
}
