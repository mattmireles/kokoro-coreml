import CoreML
import XCTest
@testable import KokoroPipeline

final class DecoderPrePlacementTests: XCTestCase {
    func testDecoderPreRunsOnTheNeuralEngineUpToTenSecondsAndOnTheGPUAbove() {
        XCTAssertEqual(PipelineConstants.decoderPreNeuralEngineMaxBucketSeconds, 10)
        for sec in [3, 7, 10] {
            XCTAssertEqual(PipelineConstants.decoderPreComputeUnits(bucketSec: sec), .cpuAndNeuralEngine, "bucket \(sec)s")
        }
        for sec in [15, 30] {
            XCTAssertEqual(PipelineConstants.decoderPreComputeUnits(bucketSec: sec), .cpuAndGPU, "bucket \(sec)s")
        }
    }
}
