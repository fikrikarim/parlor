@preconcurrency import AVFoundation
import SwiftUI

/// Live camera preview using AVCaptureVideoPreviewLayer.
/// Automatically mirrors the feed horizontally (selfie-style).

#if os(macOS)

struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession

    func makeNSView(context: Context) -> CameraPreviewNSView {
        let view = CameraPreviewNSView()
        view.session = session
        return view
    }

    func updateNSView(_ nsView: CameraPreviewNSView, context: Context) {
        nsView.session = session
    }
}

final class CameraPreviewNSView: NSView {
    var session: AVCaptureSession? {
        didSet {
            guard let session, session != oldValue else { return }
            setupPreviewLayer(session: session)
        }
    }

    private var previewLayer: AVCaptureVideoPreviewLayer?

    override func layout() {
        super.layout()
        previewLayer?.frame = bounds
    }

    private func setupPreviewLayer(session: AVCaptureSession) {
        previewLayer?.removeFromSuperlayer()

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = bounds

        // Mirror horizontally for selfie view
        layer.setAffineTransform(CGAffineTransform(scaleX: -1, y: 1))

        self.wantsLayer = true
        self.layer?.addSublayer(layer)
        self.previewLayer = layer
    }
}

#else

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.session = session
        return view
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.session = session
    }
}

final class CameraPreviewUIView: UIView {
    var session: AVCaptureSession? {
        didSet {
            guard let session, session != oldValue else { return }
            setupPreviewLayer(session: session)
        }
    }

    private var previewLayer: AVCaptureVideoPreviewLayer?

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }

    private func setupPreviewLayer(session: AVCaptureSession) {
        previewLayer?.removeFromSuperlayer()

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = bounds

        // Front camera is already mirrored on iOS
        self.layer.addSublayer(layer)
        self.previewLayer = layer
    }
}

#endif
