// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift Package Manager required to build this package.

import PackageDescription

let package = Package(
    name: "KokoroPipeline",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "KokoroPipeline", targets: ["KokoroPipeline"]),
        .library(name: "KokoroG2P", targets: ["KokoroG2P"]),
        .executable(name: "kokoro-g2p", targets: ["KokoroG2PCLI"]),
        .executable(name: "kokoro-bench", targets: ["KokoroBenchmark"]),
        .executable(name: "kokoro-hnsf-bench", targets: ["KokoroHnsfBenchmark"]),
    ],
    targets: [
        .target(
            name: "KokoroPipeline",
            path: "Sources/KokoroPipeline"
        ),
        // English text -> Kokoro IPA (ported from FluidAudio v0.17.4, Apache-2.0).
        .target(
            name: "KokoroG2P",
            dependencies: ["NemoTextProcessing"],
            path: "Sources/KokoroG2P"
        ),
        // Byte-exact NeMo text normalization (prebuilt FST engine, Apache-2.0):
        // FluidInference/text-processing-rs v0.3.1, macOS + iOS slices.
        .binaryTarget(
            name: "NemoTextProcessing",
            url: "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.1/NemoTextProcessing.xcframework.zip",
            checksum: "5fa8c10d4ec26c1bb2413125f351a7222a4c68a23b74476680fbada7e26fc6aa"
        ),
        .executableTarget(
            name: "KokoroG2PCLI",
            dependencies: ["KokoroG2P"],
            path: "Sources/KokoroG2PCLI"
        ),
        .executableTarget(
            name: "KokoroBenchmark",
            dependencies: ["KokoroPipeline"],
            path: "Sources/KokoroBenchmark"
        ),
        .executableTarget(
            name: "KokoroHnsfBenchmark",
            dependencies: ["KokoroPipeline"],
            path: "Sources/KokoroHnsfBenchmark"
        ),
        .testTarget(
            name: "KokoroPipelineTests",
            dependencies: ["KokoroPipeline"],
            path: "Tests/KokoroPipelineTests"
        ),
    ]
)
