import SwiftUI

/// Real-time frequency spectrum visualizer rendered with Canvas.
/// Displays animated bars driven by FFT frequency data.
struct WaveformView: View {
    let bins: [Float]
    let phase: AppPhase

    // Smoothed bins for animation
    @State private var smoothedBins: [Float] = []
    @State private var ambientOffset: Double = 0

    private let barSpacing: CGFloat = 2
    private let cornerRadius: CGFloat = 1.5
    private let smoothingFactor: Float = 0.3

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
            Canvas { context, size in
                let binCount = max(bins.count, 1)
                let totalSpacing = barSpacing * CGFloat(binCount - 1)
                let barWidth = max(1, (size.width - totalSpacing) / CGFloat(binCount))
                let maxHeight = size.height

                let color = Theme.phaseColor(for: phase)

                for i in 0..<binCount {
                    let value = i < currentBins.count ? CGFloat(currentBins[i]) : 0

                    // Minimum bar height for ambient animation
                    let ambient = ambientHeight(index: i, total: binCount, time: timeline.date.timeIntervalSinceReferenceDate)
                    let height = max(ambient, value * maxHeight)
                    let clampedHeight = min(height, maxHeight)

                    let x = CGFloat(i) * (barWidth + barSpacing)
                    let y = (maxHeight - clampedHeight) / 2

                    let rect = CGRect(x: x, y: y, width: barWidth, height: max(2, clampedHeight))
                    let path = RoundedRectangle(cornerRadius: cornerRadius)
                        .path(in: rect)

                    let opacity = 0.4 + 0.6 * Double(value)
                    context.fill(path, with: .color(color.opacity(opacity)))
                }
            }
        }
        .onChange(of: bins) { _, newBins in
            updateSmoothedBins(newBins)
        }
        .onAppear {
            smoothedBins = Array(repeating: 0, count: bins.count)
        }
    }

    private var currentBins: [Float] {
        smoothedBins.isEmpty ? bins : smoothedBins
    }

    private func updateSmoothedBins(_ newBins: [Float]) {
        if smoothedBins.count != newBins.count {
            smoothedBins = newBins
            return
        }
        for i in 0..<newBins.count {
            smoothedBins[i] = smoothedBins[i] * (1 - smoothingFactor) + newBins[i] * smoothingFactor
        }
    }

    /// Generates subtle ambient bar movement when there's no active audio.
    private func ambientHeight(index: Int, total: Int, time: TimeInterval) -> CGFloat {
        let phase1 = sin(time * 1.5 + Double(index) * 0.3) * 0.5 + 0.5
        let phase2 = sin(time * 0.8 + Double(index) * 0.7) * 0.5 + 0.5
        let combined = (phase1 + phase2) / 2.0
        return CGFloat(combined) * 4.0 + 2.0 // 2-6px ambient
    }
}
