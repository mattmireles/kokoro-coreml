import CoreML
import XCTest
@testable import KokoroPipeline

/// Regression tests for lazy model loading and the compiled-model cache.
///
/// The botnet iPad worker (`botnet/apps/ios-worker`, `KokoroNativeRuntime.swift`)
/// depends on both: init must not compile the model set, and compiled bundles
/// must survive a relaunch. A 2026-09-24 sync from a main without them broke
/// the worker's build.
final class LazyLoadingTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokoro-lazy-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    /// Init discovers packages without opening them: these fake packages would
    /// throw on compile, so an eager init would fail.
    func testInitDiscoversPackagesWithoutOpeningThem() throws {
        for name in [
            "kokoro_duration_t128", "kokoro_f0ntrain_t600",
            "kokoro_decoder_pre_15s", "kokoro_decoder_har_post_15s",
            "kokoro_f0ntrain_t120", "kokoro_decoder_pre_3s",  // 3 s has no generator
        ] {
            try FileManager.default.createDirectory(
                at: scratch.appendingPathComponent("\(name).mlpackage"), withIntermediateDirectories: true)
        }

        let pipeline = try KokoroPipeline(
            modelsDirectory: scratch, linearWeights: Array(repeating: 0, count: 9), linearBias: 0)

        XCTAssertEqual(pipeline.availableBucketSeconds(), [15])
        XCTAssertEqual(pipeline.durationModelChoices().map(\.cacheKey), ["padded_t128"])
        XCTAssertThrowsError(try pipeline.prepareForBucket(bucketSec: 15, tFrames: 600))
    }

    /// A bucket missing one of its stages is not advertised.
    func testBucketMissingAStageIsNotAvailable() throws {
        for name in ["kokoro_duration_t128", "kokoro_decoder_pre_15s", "kokoro_decoder_har_post_15s"] {
            try FileManager.default.createDirectory(
                at: scratch.appendingPathComponent("\(name).mlpackage"), withIntermediateDirectories: true)
        }
        let pipeline = try KokoroPipeline(
            modelsDirectory: scratch, linearWeights: Array(repeating: 0, count: 9), linearBias: 0)
        XCTAssertEqual(pipeline.availableBucketSeconds(), [])
    }

    /// Runs where the repo's gitignored coreml/ holds a real package. Covers a
    /// cache miss landing in the cache, a corrupt cached bundle being
    /// recompiled rather than thrown, and an unwritable cache still loading.
    func testCompiledModelCacheKeepsRepairsAndToleratesAnUnwritableCache() throws {
        let package = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("coreml/kokoro_f0ntrain_t120.mlpackage")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: package.path), "coreml/ packages not present")
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let cache = scratch.appendingPathComponent("compiled", isDirectory: true)
        let cached = cache.appendingPathComponent("kokoro_f0ntrain_t120.mlmodelc")

        _ = try CompiledModelCache.load(package: package, configuration: config, cache: cache)
        XCTAssertTrue(FileManager.default.fileExists(atPath: cached.path), "first load keeps the compile")

        // Corrupt the cached bundle: the next load must recompile, not throw.
        for item in try FileManager.default.contentsOfDirectory(at: cached, includingPropertiesForKeys: nil) {
            try FileManager.default.removeItem(at: item)
        }
        _ = try CompiledModelCache.load(package: package, configuration: config, cache: cache)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: cached.path).isEmpty, "corrupt bundle is replaced")

        // A cache path that is a file cannot be written: slow, not failed.
        let blocked = scratch.appendingPathComponent("blocked")
        try Data().write(to: blocked)
        _ = try CompiledModelCache.load(package: package, configuration: config, cache: blocked)
    }
}
