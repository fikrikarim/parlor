// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Parlor",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    dependencies: [
        // MLX Swift core + LLM inference
        .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main"),
        // Kokoro TTS via MLX Swift
        .package(url: "https://github.com/mlalma/kokoro-ios.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "Parlor",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "KokoroSwift", package: "kokoro-ios"),
            ],
            path: "Parlor",
            resources: [
                .process("Info.plist"),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
    ]
)
