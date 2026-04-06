import SwiftUI

/// Bottom control bar with camera toggle, state indicator, and on-device badge.
struct ControlBar: View {
    let phase: AppPhase
    let isCameraEnabled: Bool
    let onToggleCamera: () -> Void

    var body: some View {
        HStack(spacing: 16) {
            // Camera toggle
            Button(action: onToggleCamera) {
                HStack(spacing: 6) {
                    Image(systemName: isCameraEnabled ? "video.fill" : "video.slash.fill")
                        .font(.system(size: 13))
                    Text(isCameraEnabled ? "Camera On" : "Camera Off")
                        .font(.system(size: 12, weight: .medium))
                }
                .foregroundStyle(isCameraEnabled ? Theme.textPrimary : Theme.textMuted)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Theme.surface, in: Capsule())
            }
            .buttonStyle(.plain)

            Spacer()

            // Phase indicator
            HStack(spacing: 6) {
                Circle()
                    .fill(Theme.phaseColor(for: phase))
                    .frame(width: 8, height: 8)
                    .shadow(color: Theme.phaseColor(for: phase).opacity(0.5), radius: 4)
                    .overlay {
                        if phase == .processing {
                            Circle()
                                .stroke(Theme.phaseColor(for: phase).opacity(0.4), lineWidth: 2)
                                .scaleEffect(1.8)
                                .opacity(0.5)
                                .animation(
                                    .easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                                    value: phase
                                )
                        }
                    }

                Text(Theme.phaseLabel(for: phase))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.textSecondary)
            }

            Spacer()

            // On-device badge
            HStack(spacing: 4) {
                Image(systemName: "lock.shield.fill")
                    .font(.system(size: 10))
                Text("On-Device")
                    .font(.system(size: 11, weight: .medium))
            }
            .foregroundStyle(Theme.textMuted)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Theme.surface, in: Capsule())
        }
    }
}
