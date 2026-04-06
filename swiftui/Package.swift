// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Parlor",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    targets: [
        .executableTarget(
            name: "Parlor",
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
