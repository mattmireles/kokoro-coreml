// swift-tools-version: 5.9
// Root manifest so apps can depend on this repo straight from GitHub:
//   .package(url: "https://github.com/mattmireles/kokoro-coreml.git", revision: "<commit>")
// then `.product(name: "KokoroTTS", package: "kokoro-coreml")` for raw text ->
// speech, or KokoroPipeline / KokoroG2P for the low-level runtime.
// `swift run -c release kokoro-say "Hello"` from the repo root speaks to
// ./kokoro.wav. Benchmarks, other CLIs and tests live in swift/Package.swift and
// swift-tts/Package.swift. Keep the target paths and the MisakiSwift pin below
// in sync with those two manifests.
//
// macOS 15 / iOS 18: KokoroTTS's G2P dependency (MisakiSwift) requires them.

import PackageDescription

let package = Package(
    name: "KokoroCoreML",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0"),
    ],
    products: [
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
        .library(name: "KokoroPipeline", targets: ["KokoroPipeline"]),
        .library(name: "KokoroG2P", targets: ["KokoroG2P"]),
        .executable(name: "kokoro-say", targets: ["KokoroSay"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/mattmireles/MisakiSwift",
            revision: "3a27756a780fc138e328a96e533fb440a3419d5b"
        ),
    ],
    targets: [
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
            name: "KokoroSay",
            dependencies: ["KokoroTTS"],
            path: "swift-tts/Sources/KokoroSay"
        ),
        .target(
            name: "KokoroPipeline",
            path: "swift/Sources/KokoroPipeline"
        ),
        // English text -> Kokoro IPA (ported from FluidAudio v0.17.4, Apache-2.0).
        .target(
            name: "KokoroG2P",
            dependencies: ["NemoTextProcessing"],
            path: "swift/Sources/KokoroG2P"
        ),
        // Byte-exact NeMo text normalization (prebuilt FST engine, Apache-2.0):
        // FluidInference/text-processing-rs v0.3.1, macOS + iOS slices.
        .binaryTarget(
            name: "NemoTextProcessing",
            url: "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.1/NemoTextProcessing.xcframework.zip",
            checksum: "5fa8c10d4ec26c1bb2413125f351a7222a4c68a23b74476680fbada7e26fc6aa"
        ),
    ]
)
