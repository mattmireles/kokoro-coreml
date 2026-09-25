// swift-tools-version: 5.9
// Root manifest so apps can depend on this repo straight from GitHub:
//   .package(url: "https://github.com/mattmireles/kokoro-coreml.git", revision: "<commit>")
// It exposes the two runtime libraries only. Benchmarks, CLIs and tests live
// in swift/Package.swift. Keep the target paths below in sync with it.

import PackageDescription

let package = Package(
    name: "KokoroCoreML",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "KokoroPipeline", targets: ["KokoroPipeline"]),
        .library(name: "KokoroG2P", targets: ["KokoroG2P"]),
    ],
    targets: [
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
