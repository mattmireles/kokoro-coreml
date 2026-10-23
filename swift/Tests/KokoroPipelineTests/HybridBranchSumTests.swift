/// HybridGenerator.sumBranches: the three resblock branches of a stage share a
/// few chains of scratch tensors. When the first finished branch's output holds
/// the running sum, its chain must never be handed to a later branch, or that
/// branch overwrites the sum. This shipped once: stage 2 got garbage input
/// (-4.6 dB) whenever the branches ran one after another.
import XCTest
@testable import KokoroPipeline

final class HybridBranchSumTests: XCTestCase {
    private final class Box {
        var value: Int
        init(_ v: Int) { value = v }
    }

    private func serial(_ work: [() throws -> Void]) throws { for w in work { try w() } }

    @available(macOS 15.0, iOS 18.0, *)
    private func run(chains: [Box], accumulator: Box?) throws -> (Box, [Int]) {
        var used: [Int] = []
        let sum = try HybridGenerator.sumBranches(chains: chains, accumulator: accumulator, parallel: serial,
                                                  body: { j, ch in
            used.append(ObjectIdentifier(ch).hashValue)
            ch.value = j + 1              // the branch writes its output into its chain
            return ch
        }, add: { y, acc, mode in
            acc.value = mode == 0 ? y.value : acc.value + y.value
        })
        return (sum, used)
    }

    func testSumHolderChainIsNotReusedWhenBranchesRunSerially() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("needs macOS 15") }
        let (sum, used) = try run(chains: [Box(0), Box(0), Box(0)], accumulator: nil)
        XCTAssertFalse(used.dropFirst().contains(used[0]), "the chain holding the sum was handed out again")
        XCTAssertEqual(sum.value, 1 + 2 + 3)
    }

    func testOwnAccumulatorLetsChainsBeReused() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("needs macOS 15") }
        let (sum, used) = try run(chains: [Box(0)], accumulator: Box(0))
        XCTAssertEqual(Set(used).count, 1)
        XCTAssertEqual(sum.value, 1 + 2 + 3)
    }

    /// `claim` runs exactly once, for the branch whose output becomes the
    /// accumulator (it still owes its deferred residual), never with an own accumulator.
    func testClaimRunsOnceForTheHolder() throws {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("needs macOS 15") }
        for (chains, own) in [([Box(0), Box(0), Box(0)], nil as Box?), ([Box(0)], Box(0))] {
            var claimed: [Int] = []
            let sum = try HybridGenerator.sumBranches(chains: chains, accumulator: own, parallel: serial,
                                                      body: { j, ch in ch.value = j + 1; return ch },
                                                      claim: { y in claimed.append(y.value) },
                                                      add: { y, acc, mode in acc.value = mode == 0 ? y.value : acc.value + y.value })
            XCTAssertEqual(claimed.count, own == nil ? 1 : 0)
            XCTAssertEqual(sum.value, 1 + 2 + 3)
        }
    }
}
