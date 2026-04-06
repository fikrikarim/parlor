@preconcurrency import AVFoundation
import CoreImage
import Foundation

#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// Manages camera capture session and provides frame snapshots.
///
/// Uses AVCaptureSession to maintain a live camera feed and captures
/// individual frames on demand for LLM input.
final class CameraManager: @unchecked Sendable {

    // MARK: - Properties

    let session = AVCaptureSession()

    /// The latest captured video frame, scaled to 320px width.
    private var latestFrame: CGImage?
    private let frameLock = NSLock()

    private let captureQueue = DispatchQueue(label: "camera.capture", qos: .userInteractive)
    private let outputDelegate = VideoOutputDelegate()
    private var videoOutput: AVCaptureVideoDataOutput?
    private var isConfigured = false

    // MARK: - Init

    init() {
        outputDelegate.onFrame = { [weak self] image in
            self?.frameLock.lock()
            self?.latestFrame = image
            self?.frameLock.unlock()
        }
    }

    // MARK: - Lifecycle

    func start() async throws {
        guard !isConfigured else {
            if !session.isRunning {
                session.startRunning()
            }
            return
        }

        // Request camera permission
        let authorized = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                continuation.resume(returning: true)
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    continuation.resume(returning: granted)
                }
            default:
                continuation.resume(returning: false)
            }
        }

        guard authorized else {
            throw CameraError.notAuthorized
        }

        try configureSession()
        isConfigured = true
        session.startRunning()
    }

    func stop() {
        session.stopRunning()
    }

    /// Capture the latest camera frame, scaled to a maximum width of 320px.
    func captureFrame() -> CGImage? {
        frameLock.lock()
        defer { frameLock.unlock() }
        return latestFrame
    }

    // MARK: - Session Configuration

    private func configureSession() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .medium

        // Find camera device
        #if os(macOS)
        guard let device = AVCaptureDevice.default(for: .video) else {
            throw CameraError.noCamera
        }
        #else
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            throw CameraError.noCamera
        }
        #endif

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraError.configurationFailed
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(outputDelegate, queue: captureQueue)

        guard session.canAddOutput(output) else {
            throw CameraError.configurationFailed
        }
        session.addOutput(output)
        self.videoOutput = output
    }
}

// MARK: - Video Output Delegate

private final class VideoOutputDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {

    var onFrame: (@Sendable (CGImage) -> Void)?

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Scale to 320px width
        let originalWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let scale = 320.0 / originalWidth
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        let rect = scaled.extent
        guard let cgImage = ciContext.createCGImage(scaled, from: rect) else { return }

        onFrame?(cgImage)
    }
}

// MARK: - Errors

enum CameraError: LocalizedError, Sendable {
    case notAuthorized
    case noCamera
    case configurationFailed

    var errorDescription: String? {
        switch self {
        case .notAuthorized: "Camera access not authorized"
        case .noCamera: "No camera device found"
        case .configurationFailed: "Failed to configure camera session"
        }
    }
}
