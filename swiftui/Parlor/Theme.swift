import SwiftUI

// MARK: - Color Theme

enum Theme {
    // Background
    static let background = Color(red: 0.02, green: 0.02, blue: 0.035)
    static let surface = Color(red: 0.08, green: 0.08, blue: 0.12)
    static let surfaceLight = Color(red: 0.12, green: 0.12, blue: 0.17)

    // Text
    static let textPrimary = Color(red: 0.93, green: 0.93, blue: 0.95)
    static let textSecondary = Color(red: 0.55, green: 0.56, blue: 0.62)
    static let textMuted = Color(red: 0.35, green: 0.36, blue: 0.42)

    // Phase colors
    static let listening = Color(red: 0.29, green: 0.87, blue: 0.50)    // #4ade80
    static let processing = Color(red: 0.96, green: 0.62, blue: 0.04)   // #f59e0b
    static let speaking = Color(red: 0.51, green: 0.55, blue: 0.97)     // #818cf8
    static let loadingColor = Color(red: 0.23, green: 0.24, blue: 0.27) // #3a3d46

    // Borders
    static let border = Color.white.opacity(0.06)
    static let borderLight = Color.white.opacity(0.1)

    static func phaseColor(for phase: AppPhase) -> Color {
        switch phase {
        case .loading: loadingColor
        case .listening: listening
        case .processing: processing
        case .speaking: speaking
        }
    }

    static func phaseLabel(for phase: AppPhase) -> String {
        switch phase {
        case .loading: "Loading..."
        case .listening: "Listening"
        case .processing: "Thinking..."
        case .speaking: "Speaking"
        }
    }

    // Animation
    static let glowAnimation = Animation.easeInOut(duration: 2.0).repeatForever(autoreverses: true)
    static let fastGlowAnimation = Animation.easeInOut(duration: 0.8).repeatForever(autoreverses: true)
    static let phaseTransition = Animation.easeInOut(duration: 0.3)
}
