import Foundation
import XCTest
@testable import KokoroTTS

/// Guards the one-verification-per-launch contract for an unchanged bundle.
///
/// `KokoroTTS.load` builds a fresh `KokoroSDKModelProvider` on every call, and
/// apps that drop the facade between playbacks call it repeatedly. Each provider
/// used to re-hash every model package from scratch — about 172 MB of SHA-256
/// per listen on top of the hydration sweep that had just verified the same
/// bytes. These tests pin both the reuse and its invalidation.
///
/// The digest counter is process-wide, so these assertions assume XCTest's
/// default serial execution within the test bundle.
final class SingleValidationPassTests: XCTestCase {
    /// Verifies a second provider re-validates an unchanged package without hashing it.
    func testSecondPrepareSkipsPackageRehash() throws {
        let root = try KokoroBundleFixture.makeBundleRoot()
        let packageURL = root.appendingPathComponent(
            "coreml/kokoro_duration_t128.mlpackage",
            isDirectory: true
        )

        let baseline = KokoroFileDigest.computedDigestCount
        let first = try KokoroSDKModelProvider(resources: .directory(root))
        _ = try first.validateModelPackageIfNeeded(packageURL)
        let afterFirstPass = KokoroFileDigest.computedDigestCount
        XCTAssertGreaterThan(
            afterFirstPass,
            baseline,
            "the first preparation must actually verify the bundle"
        )

        let second = try KokoroSDKModelProvider(resources: .directory(root))
        _ = try second.validateModelPackageIfNeeded(packageURL)

        XCTAssertEqual(
            KokoroFileDigest.computedDigestCount,
            afterFirstPass,
            "a second preparation re-hashed an unchanged bundle"
        )
    }

    /// Verifies a rewritten package file still fails validation on the next pass.
    func testChangedPackageFileIsRehashedAndRejected() throws {
        let root = try KokoroBundleFixture.makeBundleRoot()
        let packageURL = root.appendingPathComponent(
            "coreml/kokoro_duration_t128.mlpackage",
            isDirectory: true
        )
        let payloadURL = packageURL.appendingPathComponent(
            KokoroBundleFixture.packagePayloadRelativePath
        )
        let first = try KokoroSDKModelProvider(resources: .directory(root))
        _ = try first.validateModelPackageIfNeeded(packageURL)

        // Same byte count, different bytes: only the modification date can tell
        // the memo that this file is no longer the one it verified.
        try Data("tampered-128".utf8).write(to: payloadURL)
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(1)],
            ofItemAtPath: payloadURL.path
        )

        let second = try KokoroSDKModelProvider(resources: .directory(root))

        XCTAssertThrowsError(try second.validateModelPackageIfNeeded(packageURL)) { error in
            XCTAssertEqual(
                error as? KokoroError,
                .badHash(path: "coreml/kokoro_duration_t128.mlpackage")
            )
        }
    }
}
