import CryptoKit
import Foundation
import XCTest
@testable import KokoroTTS

/// Guards the promise that a warm cache never reports a download.
///
/// Gist's iPhone app captioned every preparation "Downloading the voice — one
/// time, about 165 MB" because hydration emitted the same bare `Double` whether
/// it was re-verifying cached bytes or pulling them over the network. These
/// tests pin the phase contract that makes an honest caption possible.
///
/// The fixtures hydrate from a `file://` hosted manifest: the store treats local
/// sources as transfers just like HTTP sources, so cold and warm behaviour can be
/// exercised with no network and no stub server.
final class HydrationProgressPhaseTests: XCTestCase {
    /// Verifies a fully cached bundle is never captioned as a download.
    func testCacheHitHydrationNeverReportsDownloadingPhase() async throws {
        let source = try makeHostedSource()
        let cacheDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let store = KokoroDownloadedModelStore(
            manifestURL: source.manifestURL,
            expectedManifestSHA256: source.manifestSHA256,
            cacheDirectory: cacheDirectory
        )
        _ = try await store.hydrate()

        let recorder = PhaseRecorder()
        _ = try await store.hydrate(phaseProgress: { recorder.append($0) })

        let updates = recorder.updates
        XCTAssertFalse(updates.isEmpty)
        XCTAssertFalse(
            updates.contains { $0.phase == .downloading },
            "a fully cached hydration reported a download phase"
        )
        XCTAssertTrue(updates.contains { $0.phase == .verifying })
        XCTAssertEqual(try XCTUnwrap(updates.last).fraction, 1, accuracy: 0.0001)
    }

    /// Verifies the first hydration of an empty cache does report a download.
    func testColdHydrationReportsDownloadingPhase() async throws {
        let source = try makeHostedSource()
        let cacheDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let store = KokoroDownloadedModelStore(
            manifestURL: source.manifestURL,
            expectedManifestSHA256: source.manifestSHA256,
            cacheDirectory: cacheDirectory
        )

        let recorder = PhaseRecorder()
        _ = try await store.hydrate(phaseProgress: { recorder.append($0) })

        let updates = recorder.updates
        XCTAssertTrue(
            updates.contains { $0.phase == .downloading },
            "a cold hydration never reported a download phase"
        )
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: cacheDirectory.appendingPathComponent("voices/af_heart.bin").path
            )
        )
    }

    /// Verifies the back-compatible `Double` overload still receives fractions.
    func testFractionOverloadStillReportsCompletion() async throws {
        let source = try makeHostedSource()
        let cacheDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let store = KokoroDownloadedModelStore(
            manifestURL: source.manifestURL,
            expectedManifestSHA256: source.manifestSHA256,
            cacheDirectory: cacheDirectory
        )

        let recorder = FractionRecorder()
        _ = try await store.hydrate(progress: { recorder.append($0) })

        XCTAssertEqual(recorder.values.last ?? 0, 1, accuracy: 0.0001)
    }

    /// Writes a hosted-manifest source tree that hydration can pull from locally.
    ///
    /// - Returns: Manifest URL and the digest a store must be pinned to.
    private func makeHostedSource() throws -> (manifestURL: URL, manifestSHA256: String) {
        let source = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let voices = source.appendingPathComponent("voices", isDirectory: true)
        try FileManager.default.createDirectory(at: voices, withIntermediateDirectories: true)

        let payload = Data((0..<2_048).map { UInt8($0 % 251) })
        try payload.write(to: voices.appendingPathComponent("af_heart.bin"))

        let manifest: [String: Any] = [
            "version": "test-1",
            "files": [[
                "path": "voices/af_heart.bin",
                "bytes": payload.count,
                "sha256": SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined(),
            ]],
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        let manifestURL = source.appendingPathComponent("HostedManifest.json")
        try data.write(to: manifestURL)
        return (
            manifestURL,
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        )
    }
}

/// Collects phase-tagged hydration updates from URLSession delegate queues.
private final class PhaseRecorder: @unchecked Sendable {
    /// Guards ``storage``.
    private let lock = NSLock()

    /// Updates in arrival order.
    private var storage: [KokoroHydrationProgress] = []

    /// Records one update.
    ///
    /// - Parameter update: Hydration progress update.
    func append(_ update: KokoroHydrationProgress) {
        lock.lock()
        storage.append(update)
        lock.unlock()
    }

    /// Updates recorded so far.
    var updates: [KokoroHydrationProgress] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

/// Collects bare fractions from the back-compatible hydration overload.
private final class FractionRecorder: @unchecked Sendable {
    /// Guards ``storage``.
    private let lock = NSLock()

    /// Fractions in arrival order.
    private var storage: [Double] = []

    /// Records one fraction.
    ///
    /// - Parameter value: Hydration completion in `0...1`.
    func append(_ value: Double) {
        lock.lock()
        storage.append(value)
        lock.unlock()
    }

    /// Fractions recorded so far.
    var values: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
