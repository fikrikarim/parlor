import SwiftUI

struct ContentView: View {
    @Environment(ConversationViewModel.self) private var viewModel

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            VStack(spacing: 0) {
                headerBar
                Divider().background(Theme.border)

                GeometryReader { geo in
                    let isCompact = geo.size.height < 500

                    VStack(spacing: 0) {
                        // Camera viewport with glow
                        ZStack {
                            CameraPreview(session: viewModel.captureSession)
                                .opacity(viewModel.isCameraEnabled ? 1 : 0)
                                .clipShape(RoundedRectangle(cornerRadius: 12))

                            if !viewModel.isCameraEnabled {
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Theme.surface)
                                    .overlay {
                                        Image(systemName: "video.slash")
                                            .font(.system(size: 32))
                                            .foregroundStyle(Theme.textMuted)
                                    }
                            }

                            GlowView(phase: viewModel.phase, audioLevel: viewModel.audioLevel)

                            // Waveform overlay at bottom of viewport
                            VStack {
                                Spacer()
                                WaveformView(
                                    bins: viewModel.frequencyBins,
                                    phase: viewModel.phase
                                )
                                .frame(height: 48)
                                .padding(.horizontal, 8)
                                .padding(.bottom, 8)
                            }
                        }
                        .aspectRatio(isCompact ? 16/9 : 4/3, contentMode: .fit)
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, 16)
                        .padding(.top, 12)

                        // Transcript
                        TranscriptView(messages: viewModel.messages)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .padding(.horizontal, 16)
                            .padding(.top, 8)

                        Divider().background(Theme.border)

                        // Controls
                        ControlBar(
                            phase: viewModel.phase,
                            isCameraEnabled: viewModel.isCameraEnabled,
                            onToggleCamera: { viewModel.toggleCamera() }
                        )
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                    }
                }
            }
        }
        .preferredColorScheme(.dark)
        .task {
            await viewModel.start()
        }
        .overlay {
            if let error = viewModel.errorMessage {
                errorBanner(error)
            }
        }
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 12) {
            Text("Parlor")
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(Theme.textPrimary)

            Text("Gemma 4 E2B")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.textSecondary)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Theme.surface, in: Capsule())

            Spacer()

            statusPill
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var statusPill: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 6, height: 6)
                .shadow(color: statusColor.opacity(0.6), radius: 3)

            Text(viewModel.statusText)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Theme.surface, in: Capsule())
    }

    private var statusColor: Color {
        if viewModel.errorMessage != nil { return .red }
        if viewModel.isModelLoaded { return Theme.listening }
        return Theme.loadingColor
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        VStack {
            Spacer()
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.yellow)
                Text(message)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(2)
                Spacer()
                Button {
                    viewModel.errorMessage = nil
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            .padding(12)
            .background(Theme.surface.opacity(0.95), in: RoundedRectangle(cornerRadius: 10))
            .padding(16)
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .animation(.easeOut(duration: 0.3), value: viewModel.errorMessage)
    }
}
