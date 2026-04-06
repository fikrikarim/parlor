import SwiftUI

/// Three-layer animated glow effect overlay for the camera viewport.
/// Color and animation intensity respond to the current app phase and audio level.
struct GlowView: View {
    let phase: AppPhase
    let audioLevel: Float

    @State private var breathe = false

    var body: some View {
        let color = Theme.phaseColor(for: phase)
        let intensity = glowIntensity

        ZStack {
            // Layer 1: Outer soft glow
            RoundedRectangle(cornerRadius: 16)
                .stroke(color.opacity(0.15 * intensity), lineWidth: 40)
                .blur(radius: 30)
                .scaleEffect(breathe ? 1.02 : 0.98)

            // Layer 2: Mid glow
            RoundedRectangle(cornerRadius: 14)
                .stroke(color.opacity(0.25 * intensity), lineWidth: 20)
                .blur(radius: 15)
                .scaleEffect(breathe ? 1.01 : 0.99)

            // Layer 3: Inner sharp glow
            RoundedRectangle(cornerRadius: 12)
                .stroke(color.opacity(0.4 * intensity), lineWidth: 3)
                .blur(radius: 4)
        }
        .allowsHitTesting(false)
        .animation(phaseAnimation, value: breathe)
        .animation(Theme.phaseTransition, value: phase)
        .onAppear { breathe = true }
        .onChange(of: phase) { _, _ in
            // Reset breathing cycle on phase change
            breathe = false
            withAnimation(phaseAnimation) {
                breathe = true
            }
        }
    }

    private var glowIntensity: Double {
        switch phase {
        case .loading: 0.3
        case .listening: 0.6
        case .processing: 0.8
        case .speaking: Double(0.5 + min(1.0, audioLevel * 8.0))
        }
    }

    private var phaseAnimation: Animation {
        switch phase {
        case .loading: .easeInOut(duration: 3.0).repeatForever(autoreverses: true)
        case .listening: .easeInOut(duration: 2.0).repeatForever(autoreverses: true)
        case .processing: .easeInOut(duration: 0.8).repeatForever(autoreverses: true)
        case .speaking: .easeInOut(duration: 1.2).repeatForever(autoreverses: true)
        }
    }
}
