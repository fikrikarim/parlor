import SwiftUI

@main
struct ParlorApp: App {
    @State private var viewModel = ConversationViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(viewModel)
        }
        #if os(macOS)
        .defaultSize(width: 900, height: 700)
        #endif
    }
}
