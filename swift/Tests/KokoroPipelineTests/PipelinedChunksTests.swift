import Foundation
import XCTest
@testable import KokoroPipeline

/// `runPipelinedChunks`: order, overlap and failure handling of the two-chunk
/// pipeline used by executeKokoroSynthesisPipelined.
final class PipelinedChunksTests: XCTestCase {
    /// Regression: semaphores created with value 1 and left at 0 trapped in
    /// `_dispatch_semaphore_dispose` (iPhone 12 Pro, 2026-10-22). Many runs
    /// create and free them.
    func testPipelinedChunksReturnInOrderAndFreeCleanly() throws {
        for _ in 0..<50 {
            var got: [Int] = []
            try runPipelinedChunks(count: 5, work: { i, gate in
                usleep(UInt32(1000 * (5 - i)))   // later items finish their first stage sooner
                gate()
                usleep(500)
                return i * 10
            }, onResult: { i, v in
                XCTAssertEqual(v, i * 10)
                got.append(i)
            })
            XCTAssertEqual(got, [0, 1, 2, 3, 4])
        }
    }

    /// Second stages never overlap each other; a first stage overlaps the
    /// previous item's second stage.
    func testSecondStagesAreSerialAndOverlapNextFirstStage() throws {
        let lock = NSLock()
        var inSecond = 0, maxInSecond = 0, overlapped = false
        try runPipelinedChunks(count: 4, work: { i, gate in
            lock.lock(); if inSecond > 0 { overlapped = true }; lock.unlock()   // first stage starts
            usleep(2000)
            gate()
            lock.lock(); inSecond += 1; maxInSecond = max(maxInSecond, inSecond); lock.unlock()
            usleep(20000)
            lock.lock(); inSecond -= 1; lock.unlock()
            return i
        }, onResult: { _, _ in })
        XCTAssertEqual(maxInSecond, 1)
        XCTAssertTrue(overlapped)
    }

    /// A failure throws, starts no later item, and does not hang.
    func testFailureStopsLaterChunksAndThrows() {
        struct Boom: Error {}
        let lock = NSLock()
        var started: [Int] = []
        var delivered: [Int] = []
        XCTAssertThrowsError(try runPipelinedChunks(count: 6, work: { i, gate in
            lock.lock(); started.append(i); lock.unlock()
            if i == 2 { throw Boom() }
            gate()
            return i
        }, onResult: { i, _ in delivered.append(i) })) { XCTAssertTrue($0 is Boom) }
        XCTAssertEqual(delivered, [0, 1])
        XCTAssertFalse(started.contains(5))
    }

    /// An item that never calls the gate still hands over in order.
    func testItemWithoutGateKeepsOrder() throws {
        var got: [Int] = []
        try runPipelinedChunks(count: 3, work: { i, gate in
            if i != 1 { gate() }
            return i
        }, onResult: { i, _ in got.append(i) })
        XCTAssertEqual(got, [0, 1, 2])
    }
}
