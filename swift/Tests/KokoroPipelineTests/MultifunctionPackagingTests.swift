import CoreML
import XCTest
@testable import KokoroPipeline

final class MultifunctionPackagingTests: XCTestCase {
    func testFunctionNamesFollowTheBuildScriptConvention() {
        XCTAssertEqual(PipelineConstants.durationFunctionName(tokenLength: 64), "t64")
        XCTAssertEqual(PipelineConstants.f0ntrainFunctionName(tFrames: 1200), "t1200")
        XCTAssertEqual(PipelineConstants.decoderPreFunctionName(bucketSec: 7), "bucket_7s")
    }

    func testOpeningAMissingPackageYieldsNil() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("missing-\(UUID().uuidString).mlpackage")
        XCTAssertNil(try MultifunctionPackage.open(at: url))
    }

    func testDurationChoicesWithoutAMultifunctionPackageHaveNoFunctionName() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("choices-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("kokoro_duration_t64.mlpackage"), withIntermediateDirectories: true)
        let choices = KokoroPipeline.discoverDurationChoices(modelsDirectory: dir, useExactDurationModels: false)
        XCTAssertEqual(choices.map { $0.cacheKey }, ["padded_t64"])
        XCTAssertNil(choices[0].functionName)
    }

    /// Runs only where the built package exists (outputs/ is not tracked).
    func testDurationChoicesPreferTheMultifunctionPackage() throws {
        let built = URL(fileURLWithPath: "/Users/willem/Documents/Repositories/kokoro-coreml-masked/outputs/dynamic-length/coreml/multifunction/kokoro_duration_multifunction.mlpackage")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: built.path), "multifunction duration package not built")
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("choices-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent(PipelineConstants.durationMultifunctionPackage), withDestinationURL: built)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("kokoro_duration_t64.mlpackage"), withIntermediateDirectories: true)
        let choices = KokoroPipeline.discoverDurationChoices(modelsDirectory: dir, useExactDurationModels: false)
        XCTAssertEqual(choices.map { $0.cacheKey }, ["padded_t64", "padded_t128", "padded_t256", "padded_t512"])
        XCTAssertEqual(choices.map { $0.functionName }, ["t64", "t128", "t256", "t512"])
        XCTAssertTrue(choices.allSatisfy { $0.packageURL.lastPathComponent == PipelineConstants.durationMultifunctionPackage })
    }
}
