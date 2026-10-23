/// PcmStreamJoiner: streamed chunks must concatenate to exactly what
/// PcmJoiner.join produces for the same segments (the SDK's streaming
/// synthesis plays them as they arrive).
import XCTest
@testable import KokoroPipeline

final class PcmStreamJoinerTests: XCTestCase {
    func testStreamConcatenationEqualsJoin() {
        let sr = 1000   // 5 ms fade = 5 samples
        let cases: [[[Float]]] = [
            [(0..<40).map { Float($0) }],
            [(0..<40).map { Float($0) }, (0..<30).map { Float(-$0) }],
            [(0..<12).map { Float($0) }, [], (0..<3).map { Float(100 + $0) }, (0..<20).map { Float(-$0) }],
            [(0..<2).map { Float($0) }, (0..<2).map { Float(5 + $0) }, (0..<9).map { Float(9 - $0) }],
        ]
        for segments in cases {
            var joiner = PcmStreamJoiner(sampleRate: sr)
            var streamed: [Float] = []
            for s in segments { streamed += joiner.push(s) }
            streamed += joiner.finish()
            let joined = PcmJoiner.join(segments: segments, sampleRate: sr)
            XCTAssertEqual(streamed.count, joined.count)
            for (a, b) in zip(streamed, joined) { XCTAssertEqual(a, b, accuracy: 1e-6) }
        }
    }
}
