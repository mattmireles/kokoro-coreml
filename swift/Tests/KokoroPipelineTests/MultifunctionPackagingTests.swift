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

    /// Runs only where the repo's gitignored coreml/ holds the built package
    /// (`scripts/build_multifunction_packages.py --models-dir coreml`).
    func testDurationChoicesPreferTheMultifunctionPackage() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // KokoroPipelineTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // swift
            .deletingLastPathComponent()  // repo root
        let built = repoRoot.appendingPathComponent("coreml").appendingPathComponent(PipelineConstants.durationMultifunctionPackage)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: built.path), "multifunction duration package not built")
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("choices-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent(PipelineConstants.durationMultifunctionPackage), withDestinationURL: built)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("kokoro_duration_t64.mlpackage"), withIntermediateDirectories: true)
        let choices = KokoroPipeline.discoverDurationChoices(modelsDirectory: dir, useExactDurationModels: false)
        // The shipped package packs every size of the Swift ladder (t32-t512).
        let sizes = PipelineConstants.durationTokenSizes
        XCTAssertEqual(choices.map { $0.cacheKey }, sizes.map { "padded_t\($0)" })
        XCTAssertEqual(choices.map { $0.functionName }, sizes.map { "t\($0)" })
        XCTAssertTrue(choices.allSatisfy { $0.packageURL.lastPathComponent == PipelineConstants.durationMultifunctionPackage })
    }
}
