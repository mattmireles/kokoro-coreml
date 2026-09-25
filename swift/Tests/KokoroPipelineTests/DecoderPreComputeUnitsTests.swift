import CoreML
import XCTest
@testable import KokoroPipeline

final class DecoderPreComputeUnitsTests: XCTestCase {
    /// Decoder-pre stays bucketed on the Neural Engine at every size (owner
    /// decision 2026-09-24); no bucket may route it to the GPU.
    func testDecoderPreRunsOnTheNeuralEngineAtEveryBucket() {
        XCTAssertEqual(PipelineConstants.decoderPreComputeUnits, .cpuAndNeuralEngine)
    }
}
