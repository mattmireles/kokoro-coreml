import CoreML
import XCTest
@testable import KokoroPipeline

/// Regression tests for running a schema-2 SDK bundle's `coreml/` directory
/// directly: split duration/F0Ntrain pairs and no whole models, a folded
/// generator, and the hybrid generator with its fallback.
@available(macOS 15.0, iOS 18.0, *)
final class SchemaTwoBundleTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokoro-schema2-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        HybridGenerator.resetFailureForTesting()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
        HybridGenerator.resetFailureForTesting()
    }

    private func makePackages(_ names: [String]) throws {
        for name in names {
            try FileManager.default.createDirectory(
                at: scratch.appendingPathComponent("\(name).mlpackage"), withIntermediateDirectories: true)
        }
    }

    /// Regression: a split pair alone used to yield no duration choice, so a
    /// bundle without kokoro_duration_t128 failed with "duration" missing.
    func testSplitDurationPairAloneYieldsT128Choice() throws {
        try makePackages(["kokoro_duration_ane_front_t128", "kokoro_duration_back_t128"])

        let choices = KokoroPipeline.discoverDurationChoices(modelsDirectory: scratch)

        XCTAssertEqual(choices.map(\.cacheKey), ["padded_t128"])
        XCTAssertTrue(choices[0].allowsPadding)
        XCTAssertEqual(choices[0].packageURL.lastPathComponent, "kokoro_duration_t128.mlpackage")
    }

    /// Regression: a bucket whose F0Ntrain ships only as a split pair was not
    /// advertised by KokoroPipeline.
    func testSplitF0NPairMakesBucketAvailable() throws {
        try makePackages([
            "kokoro_duration_ane_front_t128", "kokoro_duration_back_t128",
            "kokoro_f0ntrain_front_t600", "kokoro_f0ntrain_ane_back_t600",
            "kokoro_decoder_pre_15s", "kokoro_generator_fold_poly_15s",
        ])

        let pipeline = try KokoroPipeline(
            modelsDirectory: scratch, linearWeights: Array(repeating: 0, count: 9), linearBias: 0)

        XCTAssertEqual(pipeline.availableBucketSeconds(), [15])
        XCTAssertEqual(pipeline.durationModelChoices().map(\.tokenLength), [128])
    }

    /// Regression: only a Metal failure used to fall back; any other hybrid
    /// error (plan, compile, ANE load) failed the synthesis. Now any error
    /// getting the generator is remembered and the hybrid stays off.
    func testAnyHybridErrorTurnsHybridOffForTheProcess() throws {
        struct Broken: Error {}
        final class Provider: KokoroModelProvider {
            var loads = 0
            func durationModelChoices() -> [DurationModelChoice] { [] }
            func availableBucketSeconds() -> [Int] { [15] }
            func durationModel(choice: DurationModelChoice) throws -> MLModel { throw Broken() }
            func f0ntrainModel(tFrames: Int) throws -> MLModel { throw Broken() }
            func decoderPreModel(bucketSec: Int) throws -> MLModel { throw Broken() }
            func generatorModel(bucketSec: Int) throws -> MLModel { throw Broken() }
            // Like the real providers: gate on `allowed`, then try to load.
            func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? {
                guard HybridGenerator.allowed else { return nil }
                loads += 1
                throw Broken()
            }
        }
        let provider = Provider()
        XCTAssertTrue(HybridGenerator.allowed)

        XCTAssertNil(try resolveHybridGenerator(provider, bucketSec: 15, xPreFrames: 800, requested: false))
        XCTAssertNil(try resolveHybridGenerator(provider, bucketSec: 15, xPreFrames: 800, requested: false))

        XCTAssertEqual(provider.loads, 1, "the failure is remembered, not retried")
        XCTAssertTrue(HybridGenerator.failedThisProcess)
        XCTAssertFalse(HybridGenerator.allowed)
    }

    /// Regression: the resolver skipped the provider once a failure was
    /// recorded, so the provider never got the call in which it releases the
    /// failed hybrid; after an iOS-background Metal failure the hybrid stayed
    /// resident next to the folded generator.
    func testFailedHybridIsReleasedOnNextResolve() throws {
        struct Broken: Error {}
        final class Provider: KokoroModelProvider {
            var held: AnyObject? = NSObject()
            func durationModelChoices() -> [DurationModelChoice] { [] }
            func availableBucketSeconds() -> [Int] { [15] }
            func durationModel(choice: DurationModelChoice) throws -> MLModel { throw Broken() }
            func f0ntrainModel(tFrames: Int) throws -> MLModel { throw Broken() }
            func decoderPreModel(bucketSec: Int) throws -> MLModel { throw Broken() }
            func generatorModel(bucketSec: Int) throws -> MLModel { throw Broken() }
            func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? {
                guard HybridGenerator.allowed else { held = nil; return nil }
                return held
            }
        }
        let provider = Provider()
        weak var generator = try resolveHybridGenerator(provider, bucketSec: 15, xPreFrames: 800, requested: false)
        XCTAssertNotNil(generator)
        HybridGenerator.recordFailure(Broken())   // e.g. metalFailed during predict
        XCTAssertNil(try resolveHybridGenerator(provider, bucketSec: 15, xPreFrames: 800, requested: false))
        XCTAssertNil(generator, "the provider must drop the failed hybrid")
    }

    /// Regression (benchmark mislabel): with the hybrid requested by a
    /// benchmark (KOKORO_HYBRID_GENERATOR=1, ios-bench --hybrid-generator) a
    /// hybrid error fell back to folded and was timed under the hybrid label.
    /// Now it is rethrown and not remembered.
    func testRequestedHybridErrorIsRethrownNotFallenBack() throws {
        struct Broken: Error {}
        let provider = ThrowingGeneratorProvider()

        XCTAssertThrowsError(try resolveHybridGenerator(provider, bucketSec: 15, xPreFrames: 800, requested: true)) {
            XCTAssertTrue($0 is ThrowingGeneratorProvider.HybridBroken)
        }
        XCTAssertThrowsError(try hybridFailed(Broken(), requested: true)) { XCTAssertTrue($0 is Broken) }
        XCTAssertFalse(HybridGenerator.failedThisProcess, "a requested hybrid never silently turns itself off")

        XCTAssertNil(try resolveHybridGenerator(provider, bucketSec: 15, xPreFrames: 800, requested: false))
        XCTAssertTrue(HybridGenerator.failedThisProcess)
    }

    /// Regression: prepareForBucket opened the whole kokoro_f0ntrain package,
    /// which a schema-2 directory does not ship, before every bucket.
    func testPrepareForBucketDoesNotDemandWholeF0N() throws {
        try makePackages([
            "kokoro_duration_ane_front_t128", "kokoro_duration_back_t128",
            "kokoro_f0ntrain_front_t600", "kokoro_f0ntrain_ane_back_t600",
            "kokoro_decoder_pre_15s", "kokoro_generator_fold_poly_15s",
        ])
        let pipeline = try KokoroPipeline(
            modelsDirectory: scratch, linearWeights: Array(repeating: 0, count: 9), linearBias: 0)

        // The empty split packages fail to compile; the whole package must not be asked for.
        XCTAssertThrowsError(try pipeline.prepareForBucket(bucketSec: 15, tFrames: 600)) { error in
            XCTAssertNotEqual("\(error)", "\(PipelineError.modelNotLoaded("kokoro_f0ntrain_t600"))")
        }
    }

    /// Regression: adain.bin offsets were trusted, so a plan.json paired with a
    /// shorter table read past the end of it during synthesis.
    func testAdaINOffsetsOutsideTheTableAreRejected() throws {
        let entry = HybridPlan.AdaIN(name: "rb0_n1_0", channels: 2, eps: 1e-5, weight: 0, bias: 512, alpha: 516)
        let needed = 518   // weight 2*2*128, bias 2*2, alpha 2

        XCTAssertNoThrow(try HybridGenerator.checkAdaINOffsets([entry], tableCount: needed))
        XCTAssertThrowsError(try HybridGenerator.checkAdaINOffsets([entry], tableCount: needed - 1))
        let negative = HybridPlan.AdaIN(name: "x", channels: 2, eps: 1e-5, weight: -1, bias: 512, alpha: 516)
        XCTAssertThrowsError(try HybridGenerator.checkAdaINOffsets([negative], tableCount: 10_000))
    }
}

/// A provider whose generators all throw, counting which ones were asked for.
private final class ThrowingGeneratorProvider: KokoroModelProvider {
    struct Broken: Error {}
    struct HybridBroken: Error {}
    var foldedCalls = 0
    var generatorCalls = 0
    func durationModelChoices() -> [DurationModelChoice] { [] }
    func availableBucketSeconds() -> [Int] { [15] }
    func durationModel(choice: DurationModelChoice) throws -> MLModel { throw Broken() }
    func f0ntrainModel(tFrames: Int) throws -> MLModel { throw Broken() }
    func decoderPreModel(bucketSec: Int) throws -> MLModel { throw Broken() }
    func generatorModel(bucketSec: Int) throws -> MLModel { generatorCalls += 1; throw Broken() }
    func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? { throw HybridBroken() }
    func foldedGeneratorModel(bucketSec: Int) throws -> MLModel? { foldedCalls += 1; return nil }
}
