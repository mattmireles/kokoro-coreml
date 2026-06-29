// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "KokoroTTS",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0"),
    ],
    products: [
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
        .executable(name: "kokoro-misaki-probe", targets: ["KokoroMisakiProbe"]),
        .executable(name: "kokoro-sdk-smoke", targets: ["KokoroSDKSmoke"]),
    ],
    dependencies: [
        .package(name: "KokoroPipeline", path: "../swift"),
        .package(
            url: "https://github.com/mattmireles/MisakiSwift",
            revision: "3a27756a780fc138e328a96e533fb440a3419d5b"
        ),
    ],
    targets: [
        .target(
            name: "KokoroTTS",
            dependencies: [
                .product(name: "KokoroPipeline", package: "KokoroPipeline"),
                .product(name: "MisakiSwift", package: "MisakiSwift"),
            ],
            path: "Sources/KokoroTTS",
            resources: [
                .process("Resources"),
            ]
        ),
        .executableTarget(
            name: "KokoroMisakiProbe",
            dependencies: ["KokoroTTS"],
            path: "Sources/KokoroMisakiProbe"
        ),
        .executableTarget(
            name: "KokoroSDKSmoke",
            dependencies: ["KokoroTTS"],
            path: "Sources/KokoroSDKSmoke"
        ),
        .testTarget(
            name: "KokoroTTSTests",
            dependencies: ["KokoroTTS"],
            path: "Tests/KokoroTTSTests"
        ),
    ]
)
