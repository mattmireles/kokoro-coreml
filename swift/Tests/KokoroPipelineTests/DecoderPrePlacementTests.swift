import CoreML
import XCTest
@testable import KokoroPipeline

final class DecoderPrePlacementTests: XCTestCase {
    private func tempCache() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("placement-\(UUID().uuidString)").appendingPathComponent("decoder-pre-placement.json")
    }

    func testOverrideWinsAndSkipsTheMeasurement() {
        var measured = false
        let p = DecoderPrePlacement.resolve(override: .init(decoderPreSeconds: 30), environment: [:], cacheURL: tempCache(), deviceName: "Test GPU", measure: { measured = true; return .init(decoderPreSeconds: 10) })
        XCTAssertEqual(p.neuralEngineMaxBucketSeconds, 30)
        XCTAssertEqual(p.source, "override")
        XCTAssertFalse(measured)
    }

    func testEnvironmentVariableOverrides() {
        let p = DecoderPrePlacement.resolve(environment: [DecoderPrePlacement.overrideEnvironmentVariable: "10"], cacheURL: tempCache(), deviceName: "Test GPU")
        XCTAssertEqual(p.neuralEngineMaxBucketSeconds, 10)
        XCTAssertTrue(p.source.hasPrefix("override"), p.source)  // one variable set: the other thresholds come from the default
    }

    func testMeasurementIsCachedPerDeviceOSAndPackage() {
        let cache = tempCache()
        var calls = 0
        let m3 = DecoderPrePlacement.cacheKey(deviceName: "Apple M3 Max", osVersion: "26.5.0", packageFingerprint: "100b@1")
        let first = DecoderPrePlacement.resolve(environment: [:], cacheURL: cache, cacheKey: m3, measure: { calls += 1; return .init(decoderPreSeconds: 10, durationTokens: 256, f0ntrainFrames: 0) })
        XCTAssertEqual(first.source, "measured"); XCTAssertEqual(first.neuralEngineMaxBucketSeconds, 10); XCTAssertEqual(first.durationNeuralEngineMaxTokens, 256)
        let second = DecoderPrePlacement.resolve(environment: [:], cacheURL: cache, cacheKey: m3, measure: { calls += 1; return .init(decoderPreSeconds: 30) })
        XCTAssertEqual(second.source, "cache"); XCTAssertEqual(second.neuralEngineMaxBucketSeconds, 10); XCTAssertEqual(second.durationNeuralEngineMaxTokens, 256)
        // Another device, a newer OS or a re-exported package each measure again.
        for key in [DecoderPrePlacement.cacheKey(deviceName: "Apple M1", osVersion: "26.5.0", packageFingerprint: "100b@1"),
                    DecoderPrePlacement.cacheKey(deviceName: "Apple M3 Max", osVersion: "26.6.0", packageFingerprint: "100b@1"),
                    DecoderPrePlacement.cacheKey(deviceName: "Apple M3 Max", osVersion: "26.5.0", packageFingerprint: "101b@2")] {
            let p = DecoderPrePlacement.resolve(environment: [:], cacheURL: cache, cacheKey: key, measure: { calls += 1; return .init(decoderPreSeconds: 30) })
            XCTAssertEqual(p.source, "measured"); XCTAssertEqual(p.neuralEngineMaxBucketSeconds, 30)
        }
        XCTAssertEqual(calls, 4)
    }

    func testPackageFingerprintChangesWithContent() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("pkg-\(UUID().uuidString).mlpackage")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: dir.appendingPathComponent("a.bin"))
        let before = DecoderPrePlacement.packageFingerprint(of: dir)
        try Data([1, 2, 3, 4]).write(to: dir.appendingPathComponent("a.bin"))
        XCTAssertNotEqual(before, DecoderPrePlacement.packageFingerprint(of: dir))
        XCTAssertEqual(DecoderPrePlacement.packageFingerprint(of: dir.appendingPathComponent("missing")), "missing")
    }

    func testDefaultIsTheNeuralEngineThroughout() {
        let p = DecoderPrePlacement.resolve(environment: [:], cacheURL: tempCache(), deviceName: "Unknown")
        XCTAssertEqual(p.source, "default")
        XCTAssertEqual(p.neuralEngineMaxBucketSeconds, 30)
        for sec in [3, 7, 10, 15, 30] { XCTAssertEqual(p.computeUnits(bucketSec: sec), .cpuAndNeuralEngine) }
        XCTAssertEqual(p.computeUnits(durationTokens: 64), .cpuAndGPU)
        XCTAssertEqual(p.computeUnits(f0ntrainFrames: 120), .cpuAndGPU)
    }

    func testLargestNeuralEngineWinSearchesFromTheTopAndStopsAtTheFirstWin() throws {
        var asked: [Int] = []
        // M3 Max shape: t512 loses, t256 wins -> 256 after two comparisons.
        XCTAssertEqual(try DecoderPrePlacement.largestNeuralEngineWin(sizes: [512, 256]) { size in asked.append(size); return size <= 256 }, 256)
        XCTAssertEqual(asked, [512, 256])
        // M1 shape: t512 wins -> 512 after one.
        asked = []
        XCTAssertEqual(try DecoderPrePlacement.largestNeuralEngineWin(sizes: [512, 256]) { size in asked.append(size); return true }, 512)
        XCTAssertEqual(asked, [512])
        // Nothing wins, or nothing can be compared -> 0.
        XCTAssertEqual(try DecoderPrePlacement.largestNeuralEngineWin(sizes: [1200, 600, 280]) { _ in false }, 0)
        XCTAssertEqual(try DecoderPrePlacement.largestNeuralEngineWin(sizes: [1200, 600, 280]) { _ in nil }, 0)
    }

    func testPartialEnvironmentOverrideKeepsTheOtherThresholdsFromTheCache() {
        let cache = tempCache()
        _ = DecoderPrePlacement.resolve(environment: [:], cacheURL: cache, cacheKey: "k", measure: { .init(decoderPreSeconds: 10, durationTokens: 256, f0ntrainFrames: 600) })
        let p = DecoderPrePlacement.resolve(environment: [DecoderPrePlacement.durationOverrideEnvironmentVariable: "0"], cacheURL: cache, cacheKey: "k")
        XCTAssertEqual(p.durationNeuralEngineMaxTokens, 0)
        XCTAssertEqual(p.f0ntrainNeuralEngineMaxFrames, 600)
        XCTAssertEqual(p.neuralEngineMaxBucketSeconds, 10)
        XCTAssertEqual(p.source, "override+cache")
    }

    func testGPUWinsThresholdPlacesTheLongBucketsOnTheGPU() {
        let p = DecoderPrePlacement(neuralEngineMaxBucketSeconds: DecoderPrePlacement.gpuWinsThresholdSeconds, source: "measured", deviceName: "Apple M3 Max")
        for sec in [3, 7, 10] { XCTAssertEqual(p.computeUnits(bucketSec: sec), .cpuAndNeuralEngine, "bucket \(sec)s") }
        for sec in [15, 30] { XCTAssertEqual(p.computeUnits(bucketSec: sec), .cpuAndGPU, "bucket \(sec)s") }
        XCTAssertTrue(usesFlexibleDecoderPre(bucketSec: 15, flexibleLoaded: true, neuralEngineMaxBucketSeconds: 10))
        XCTAssertFalse(usesFlexibleDecoderPre(bucketSec: 15, flexibleLoaded: true, neuralEngineMaxBucketSeconds: 30))
        // A bucket with no fixed model falls back to the flexible program whatever the placement.
        XCTAssertTrue(usesFlexibleDecoderPre(bucketSec: 15, flexibleLoaded: true, neuralEngineMaxBucketSeconds: 30, fixedAvailable: false))
        XCTAssertFalse(usesFlexibleDecoderPre(bucketSec: 15, flexibleLoaded: false, neuralEngineMaxBucketSeconds: 30, fixedAvailable: false))
    }

    func testMissingEngineYieldsNoMeasurementAndNothingCached() throws {
        XCTAssertNil(try DecoderPrePlacement.measuredThreshold(neuralEngine: nil, gpu: nil))
        let cache = tempCache()
        let p = DecoderPrePlacement.resolve(environment: [:], cacheURL: cache, cacheKey: "k", measure: { nil })
        XCTAssertEqual(p.source, "default")
        XCTAssertFalse(FileManager.default.fileExists(atPath: cache.path))
    }

    func testSetWithoutTheLargestDecoderPreKeepsTheNeuralEngineAndStillMeasuresTheOtherStages() throws {
        var asked: [String] = []
        let t = try DecoderPrePlacement.measuredThresholds(largestBucketSeconds: 30) { stage, size, _ in asked.append("\(stage)\(size)"); return nil }
        XCTAssertEqual(t?.decoderPreSeconds, 30)
        XCTAssertEqual(t?.durationTokens, 0); XCTAssertEqual(t?.f0ntrainFrames, 0)
        XCTAssertTrue(asked.contains("duration256") && asked.contains("f0ntrain280"))
        XCTAssertFalse(asked.contains("duration512"), "t512 stays on the GPU: its Neural Engine rounding differs from PyTorch by one frame")
    }

    func testNeuralEngineMustWinByTheMargin() {
        XCTAssertTrue(DecoderPrePlacement.neuralEngineWins(neuralEngineTime: 9.0, gpuTime: 10.0))
        XCTAssertFalse(DecoderPrePlacement.neuralEngineWins(neuralEngineTime: 9.9, gpuTime: 10.0))
        XCTAssertFalse(DecoderPrePlacement.neuralEngineWins(neuralEngineTime: 10.0, gpuTime: 10.0))
    }

    func testPackageFingerprintIgnoresTheManifestCoremltoolsRewrites() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("pkg-\(UUID().uuidString).mlpackage")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: dir.appendingPathComponent("a.bin"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("Manifest.json"))
        let before = DecoderPrePlacement.packageFingerprint(of: dir)
        sleep(1)
        try Data("{ }".utf8).write(to: dir.appendingPathComponent("Manifest.json"))
        XCTAssertEqual(before, DecoderPrePlacement.packageFingerprint(of: dir))
    }

    func testNeuralEngineLoadFailureRetriesOnTheGPU() throws {
        let ne = MLModelConfiguration(); ne.computeUnits = .cpuAndNeuralEngine; ne.allowLowPrecisionAccumulationOnGPU = true
        var attempts: [MLComputeUnits] = []
        let loaded: String = try MLModel.withGPUFallback(ne) { config in
            attempts.append(config.computeUnits)
            if config.computeUnits == .cpuAndNeuralEngine { throw PipelineError.modelNotLoaded("ANECCompile() FAILED") }
            XCTAssertTrue(config.allowLowPrecisionAccumulationOnGPU)  // the retry keeps everything but the units
            return "gpu"
        }
        XCTAssertEqual(loaded, "gpu"); XCTAssertEqual(attempts, [.cpuAndNeuralEngine, .cpuAndGPU])
        XCTAssertEqual(ne.computeUnits, .cpuAndNeuralEngine)  // the caller's configuration is untouched
        // A GPU failure is not retried.
        let gpu = MLModelConfiguration(); gpu.computeUnits = .cpuAndGPU
        XCTAssertThrowsError(try MLModel.withGPUFallback(gpu) { _ -> String in throw PipelineError.modelNotLoaded("gpu") })
        XCTAssertNil(MLModel.fallbackComputeUnits(after: .all))
    }
}
