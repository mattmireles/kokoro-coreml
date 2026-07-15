import XCTest
@testable import KokoroTTS

final class KokoroMisakiPhonemizerTests: XCTestCase {
    /// Environment variable required before tests invoke Misaki/MLX at runtime.
    private static let runtimeTestsFlag = "KOKORO_RUN_MISAKI_RUNTIME_TESTS"

    /// Skips Misaki runtime tests unless the caller provided an app-style MLX environment.
    ///
    /// SwiftPM shell tests compile MisakiSwift but do not build or expose the
    /// `mlx-swift_Cmlx.bundle` shader resources required by MLX. Phase 1 uses
    /// the xcodebuild-built probe for runtime proof and keeps these tests as an
    /// explicit opt-in guard for developers running inside that environment.
    private func skipUnlessRuntimeProbeEnabled() throws {
        guard ProcessInfo.processInfo.environment[Self.runtimeTestsFlag] == "1" else {
            throw XCTSkip("Set \(Self.runtimeTestsFlag)=1 only when MLX shader resources are available.")
        }
    }

    /// Verifies that the default U.S. English Misaki path returns usable phonemes.
    func testUSPhonemizerReturnsNonEmptyPhonemes() throws {
        try skipUnlessRuntimeProbeEnabled()
        let phonemizer = KokoroMisakiPhonemizer()

        let result = try phonemizer.phonemize("Hello world.")

        XCTAssertFalse(result.phonemes.isEmpty)
        XCTAssertEqual(result.utf16Count, result.phonemes.utf16.count)
    }

    /// Verifies that the British English path is wired and produces phonemes.
    func testBritishPhonemizerReturnsNonEmptyPhonemes() throws {
        try skipUnlessRuntimeProbeEnabled()
        let phonemizer = KokoroMisakiPhonemizer(british: true)

        let result = try phonemizer.phonemize("I live in Reading.")

        XCTAssertFalse(result.phonemes.isEmpty)
        XCTAssertEqual(result.utf16Count, result.phonemes.utf16.count)
    }

    /// Verifies the raw-text SDK package can see the low-level pipeline package.
    func testFacadeExposesPipelineTokenBoundary() {
        XCTAssertEqual(KokoroTTS.maxCallerChunkTokens, 450)
    }

    /// Verifies the SDK default diagnostics do not expose raw caller payloads.
    func testDiagnosticsDefaultIsPrivacySafe() {
        let policy = KokoroDiagnosticsPolicy.privacySafeDefault

        XCTAssertFalse(policy.includesRawText)
        XCTAssertFalse(policy.includesPhonemes)
        XCTAssertFalse(policy.persistsRawPayloads)
    }

    /// Verifies explicit debug diagnostics may expose text without persistence.
    func testDiagnosticsDebugPayloadsStillDisablePersistence() {
        let policy = KokoroDiagnosticsPolicy.interactiveDebugPayloads

        XCTAssertTrue(policy.includesRawText)
        XCTAssertTrue(policy.includesPhonemes)
        XCTAssertFalse(policy.persistsRawPayloads)
    }

    /// Verifies SwiftPM bundles every checked runtime asset with the SDK target.
    func testRuntimeAssetsAreBundled() throws {
        for asset in KokoroRuntimeAsset.allCases {
            let url = try KokoroRuntimeAssets.url(for: asset)

            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "\(asset.fileName) is missing")
        }
    }

    /// Verifies the checked hn-NSF resource does not carry the old unverified hash marker.
    func testRuntimeHnsfWeightsCarryVerifiedHash() throws {
        let url = try KokoroRuntimeAssets.url(for: .hnsfWeights)
        let payload = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]

        XCTAssertEqual(
            payload?["weights_sha256"] as? String,
            "25a471a6fc81fc9c5ff7c46e4be9d9ec3710dbbfea6e121a99fac75e4a97ad99"
        )
    }
}
