import XCTest
@testable import KokoroPipeline

final class KokoroPreparedInputTests: XCTestCase {
    /// Verifies the prepared-input value bridges back to the executor request.
    func testPreparedInputCreatesSynthesisRequest() {
        let inputIds: [Int32] = [0, 50, 4, 0]
        let attentionMask: [Int32] = [1, 1, 1, 1]
        let refS = Array(repeating: Float(0.5), count: PipelineConstants.voiceEmbeddingDim)
        let prepared = KokoroPreparedInput(
            key: "hello",
            text: "Hello.",
            voice: "af_heart",
            inputIds: inputIds,
            attentionMask: attentionMask,
            refS: refS,
            speed: 1.2,
            canonicalDurationSeconds: nil,
            numTokens: inputIds.count,
            hnsfWeightsSHA256: "hash"
        )

        let request = prepared.synthesisRequest()

        XCTAssertEqual(request.inputIds, inputIds)
        XCTAssertEqual(request.attentionMask, attentionMask)
        XCTAssertEqual(request.refS, refS)
        XCTAssertEqual(request.speed, 1.2)
    }
}
