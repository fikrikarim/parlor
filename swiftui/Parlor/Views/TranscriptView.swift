import SwiftUI

/// Scrollable conversation transcript with user and assistant message bubbles.
struct TranscriptView: View {
    let messages: [Message]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                }
                .padding(.vertical, 8)
            }
            .scrollIndicators(.hidden)
            .onChange(of: messages.count) { _, _ in
                if let last = messages.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }
}

// MARK: - Message Bubble

private struct MessageBubble: View {
    let message: Message

    private var isUser: Bool { message.role == .user }

    var body: some View {
        HStack {
            if isUser { Spacer(minLength: 60) }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .font(.system(size: 14))
                    .foregroundStyle(Theme.textPrimary)
                    .textSelection(.enabled)

                metadataLine
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(bubbleBackground, in: bubbleShape)

            if !isUser { Spacer(minLength: 60) }
        }
    }

    private var bubbleBackground: Color {
        isUser
            ? Theme.listening.opacity(0.1)
            : Theme.surface
    }

    private var bubbleShape: some InsettableShape {
        RoundedRectangle(cornerRadius: 12, style: .continuous)
    }

    @ViewBuilder
    private var metadataLine: some View {
        HStack(spacing: 6) {
            if message.hasImage {
                Image(systemName: "camera.fill")
                    .font(.system(size: 9))
                    .foregroundStyle(Theme.textMuted)
            }

            if let llmTime = message.llmTime {
                Text(String(format: "LLM %.1fs", llmTime))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Theme.textMuted)
            }

            if let ttsTime = message.ttsTime {
                Text(String(format: "TTS %.1fs", ttsTime))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Theme.textMuted)
            }

            Text(message.timestamp, style: .time)
                .font(.system(size: 10))
                .foregroundStyle(Theme.textMuted)
        }
    }
}
