// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "KokoroConsumerFixture",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .executable(name: "kokoro-consumer-fixture", targets: ["KokoroConsumerFixture"]),
    ],
    dependencies: [
        .package(path: "../../swift-tts"),
    ],
    targets: [
        .executableTarget(
            name: "KokoroConsumerFixture",
            dependencies: [
                .product(name: "KokoroTTS", package: "swift-tts"),
            ]
        ),
    ]
)
