// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Parlor",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    dependencies: [
        // MLX Swift LLM/VLM inference (v3 — decoupled tokenizer/downloader)
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.0.0"),
        // Kokoro-82M TTS via MLX Swift
        .package(url: "https://github.com/mlalma/kokoro-ios.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "Parlor",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLMHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMTokenizers", package: "mlx-swift-lm"),
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
