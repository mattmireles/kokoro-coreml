import CoreML
import XCTest
@testable import KokoroPipeline

final class DurationInputPaddingTests: XCTestCase {
    /// Regression: MLMultiArray does not zero its buffer, so a short input padded
    /// into a longer duration choice used to carry stale heap in the tail.
    func testPaddedTailOfIdsAndMaskIsZero() throws {
        let tokenLength = 128
        // Dirty the heap with same-size nonzero arrays so a reused buffer shows.
        for _ in 0..<8 {
            let dirty = try MLMultiArray(shape: [1, NSNumber(value: tokenLength)], dataType: .int32)
            dirty.dataPointer.assumingMemoryBound(to: Int32.self)
                .initialize(repeating: 7, count: tokenLength)
        }
        let choice = DurationModelChoice(
            cacheKey: "padded_t128",
            tokenLength: tokenLength,
            packageURL: URL(fileURLWithPath: "/dev/null"),
            requiresAttentionMask: true,
            allowsPadding: true
        )
        let realCount = 44
        let features = try makeDurationInputFeatures(
            inputIds: Array(repeating: 5, count: realCount),
            attentionMask: Array(repeating: 1, count: realCount),
            refS: Array(repeating: 0.1, count: PipelineConstants.voiceEmbeddingDim),
            speed: 1,
            choice: choice
        )
        for name in ["input_ids", "attention_mask"] {
            let array = try XCTUnwrap(features.featureValue(for: name)?.multiArrayValue)
            let ptr = array.dataPointer.assumingMemoryBound(to: Int32.self)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: ptr, count: realCount)),
                           Array(repeating: name == "input_ids" ? 5 : 1, count: realCount), name)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: ptr + realCount, count: tokenLength - realCount)),
                           Array(repeating: 0, count: tokenLength - realCount), name)
        }
    }
}
