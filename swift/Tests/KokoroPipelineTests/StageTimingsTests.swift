import XCTest
@testable import KokoroPipeline

final class StageTimingsTests: XCTestCase {
    func testTotalSubtractsDecoderPreHnsfOverlap() {
        var timings = StageTimings()
        timings.durationCoreML = 0.010
        timings.alignment = 0.001
        timings.matrixOps = 0.002
        timings.f0ntrainCoreML = 0.020
        timings.padding = 0.003
        timings.decoderPre = 0.006
        timings.hnsfSwift = 0.015
        timings.decoderPreHnsfOverlap = 0.005
        timings.generatorCoreML = 0.050
        timings.trim = 0.004

        XCTAssertEqual(timings.total, 0.106, accuracy: 1e-12)
    }
}
