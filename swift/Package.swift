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
        .executable(name: "kokoro-bench", targets: ["KokoroBenchmark"]),
        .executable(name: "kokoro-hnsf-bench", targets: ["KokoroHnsfBenchmark"]),
    ],
    targets: [
        .target(
            name: "KokoroPipeline",
            path: "Sources/KokoroPipeline"
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
