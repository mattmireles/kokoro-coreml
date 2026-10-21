import CoreML
import XCTest
@testable import KokoroPipeline

final class FlexibleShapeTests: XCTestCase {
    /// Regression: MPSGraph on macOS 27.0.1 aborts the process on the flexible
    /// generator's first predict, so OS 27 must fall back to bucket generators.
    func testFlexibleGeneratorGatedOffOnOS27() {
        let os26 = OperatingSystemVersion(majorVersion: 26, minorVersion: 7, patchVersion: 1)
        let os27 = OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 1)
        let os28 = OperatingSystemVersion(majorVersion: 28, minorVersion: 0, patchVersion: 0)
        XCTAssertTrue(PipelineConstants.flexibleGeneratorAllowed(osVersion: os26, environment: [:]))
        XCTAssertFalse(PipelineConstants.flexibleGeneratorAllowed(osVersion: os27, environment: [:]))
        XCTAssertFalse(PipelineConstants.flexibleGeneratorAllowed(osVersion: os28, environment: [:]))
    }

    func testFlexibleGeneratorEnvOverrideForcesEitherWay() {
        let key = PipelineConstants.flexibleGeneratorEnvKey
        let os26 = OperatingSystemVersion(majorVersion: 26, minorVersion: 7, patchVersion: 1)
        let os27 = OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 1)
        XCTAssertTrue(PipelineConstants.flexibleGeneratorAllowed(osVersion: os27, environment: [key: "1"]))
        XCTAssertFalse(PipelineConstants.flexibleGeneratorAllowed(osVersion: os26, environment: [key: "0"]))
        XCTAssertFalse(PipelineConstants.flexibleGeneratorAllowed(osVersion: os27, environment: [key: "yes"]))
    }

    func testFlexibleTimeAxisRoundsUpToTheGranule() throws {
        XCTAssertEqual(try flexibleTimeAxis(realFrames: 1095, accepted: 40...2400, granule: 40, stage: "generator"), 1120)
        XCTAssertEqual(try flexibleTimeAxis(realFrames: 1120, accepted: 40...2400, granule: 40, stage: "generator"), 1120)
        XCTAssertEqual(try flexibleTimeAxis(realFrames: 2400, accepted: 40...2400, granule: 40, stage: "generator"), 2400)
        XCTAssertEqual(try flexibleTimeAxis(realFrames: 556, accepted: 20...1200, granule: 20, stage: "decoder-pre"), 560)
    }

    func testFlexibleTimeAxisRaisesVeryShortInputsToTheLowerBound() throws {
        XCTAssertEqual(try flexibleTimeAxis(realFrames: 12, accepted: 40...2400, granule: 40, stage: "generator"), 40)
    }

    func testFlexibleTimeAxisCapsRoundingAtTheUpperBoundAndRejectsLongerInputs() {
        XCTAssertEqual(try? flexibleTimeAxis(realFrames: 2390, accepted: 40...2400, granule: 40, stage: "generator"), 2400)
        XCTAssertThrowsError(try flexibleTimeAxis(realFrames: 2401, accepted: 40...2400, granule: 40, stage: "generator"))
    }

    func testFlexibleMaskInputsFollowTheAlignmentRule() throws {
        // A mask of 10 frames at factor 10 with one extra frame: valid 4 of 10 -> 40 of 101, the extra frame invalid.
        let total = 10 * 10 + 1
        let m = try makeBucketMask(validFrames: 10 * 4, totalFrames: total)
        XCTAssertEqual(floatValues(from: m).reduce(0, +), 40)
        // Full fill carries the extra frame.
        let full = try makeBucketMask(validFrames: 10 * 10 + 1, totalFrames: total)
        XCTAssertEqual(floatValues(from: full).reduce(0, +), 101)
    }

    func testGranuleIsHalfASecondOfXPreFrames() {
        XCTAssertEqual(PipelineConstants.flexibleXPreGranuleFrames, 40)
    }

    func testHarFramesPerXPreFrameMatchesTheGeneratorContract() {
        // 2,400 x_pre frames -> 144,001 har frames (hop 5 over 300 samples per frame).
        XCTAssertEqual(HarmonicConstants.stftFramesPerXPreFrame * 2400 + 1, 144_001)
    }

    func testZeroPad3DCutsALongerSourceToTheTargetLength() throws {
        let source = try makeZeroArray3D(channels: 2, time: 6)
        let ptr = source.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<12 { ptr[i] = Float(i) }
        let cut = try zeroPad3D(source: source, channels: 2, targetTime: 4)
        XCTAssertEqual(cut.shape.map { $0.intValue }, [1, 2, 4])
        XCTAssertEqual(floatValues(from: cut), [0, 1, 2, 3, 6, 7, 8, 9])
    }

    func testZeroPad1DCutsALongerSourceToTheTargetLength() {
        XCTAssertEqual(zeroPad1D(source: [1, 2, 3, 4, 5], targetLength: 3), [1, 2, 3])
        XCTAssertEqual(zeroPad1D(source: [1, 2], targetLength: 4), [1, 2, 0, 0])
    }
}
