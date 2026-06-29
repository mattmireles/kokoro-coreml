// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "kokoro-coreml",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0"),
    ],
    products: [
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
        .executable(name: "kokoro-sdk-smoke", targets: ["KokoroSDKSmoke"]),
        .executable(name: "kokoro-misaki-probe", targets: ["KokoroMisakiProbe"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/mattmireles/MisakiSwift",
            revision: "3a27756a780fc138e328a96e533fb440a3419d5b"
        ),
    ],
    targets: [
        .target(
            name: "KokoroPipeline",
            path: "swift/Sources/KokoroPipeline"
        ),
        .target(
            name: "KokoroTTS",
            dependencies: [
                "KokoroPipeline",
                .product(name: "MisakiSwift", package: "MisakiSwift"),
            ],
            path: "swift-tts/Sources/KokoroTTS",
            resources: [
                .process("Resources"),
            ]
        ),
        .executableTarget(
            name: "KokoroSDKSmoke",
            dependencies: ["KokoroTTS"],
            path: "swift-tts/Sources/KokoroSDKSmoke"
        ),
        .executableTarget(
            name: "KokoroMisakiProbe",
            dependencies: ["KokoroTTS"],
            path: "swift-tts/Sources/KokoroMisakiProbe"
        ),
    ]
)
