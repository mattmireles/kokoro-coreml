// Runs only with -DKOKORO_E5, like HybridArenaLayout itself.
#if KOKORO_E5
import XCTest
@testable import KokoroPipeline

/// HybridArenaLayout: tensors that are live at the same time never share
/// bytes, and no tensor crosses an arena slot (one IOSurface per slot).
final class HybridArenaLayoutTests: XCTestCase {
    private func layout(rows: Int) -> HybridArenaLayout {
        HybridArenaLayout(rows: rows, w0: 240, w1: 2400, w2s: 14400, q6: 2496)
    }

    private func range(_ l: HybridArenaLayout, _ key: String) -> Range<Int> {
        let o = l.offsets[key]!
        return o..<(o + l.lengths[key]!)
    }

    private func assertDisjoint(_ l: HybridArenaLayout, _ keys: [String], _ phase: String) {
        for (i, a) in keys.enumerated() {
            for b in keys[(i + 1)...] {
                XCTAssertFalse(range(l, a).overlaps(range(l, b)), "\(phase): \(a) overlaps \(b)")
            }
        }
    }

    private let chain = { (name: String) in ["t1", "p0", "p1"].map { "\(name).\($0)" } }
    private let small = ["x_pre", "har", "har6", "sm1", "zm1", "sm2", "zm2", "logits"]

    func testTensorsLiveTogetherDoNotOverlap() {
        for rows in [2, 3, 4, 6, 11] {
            let l = layout(rows: rows)
            // Stage 1: the stage-2 noise chain (n1) runs alongside every stage-1 tensor.
            let stage1 = chain("n1") + chain("s1b0") + chain("s1b1") + chain("s1b2") + ["x1", "poly0"] + small
            assertDisjoint(l, stage1, "stage 1 r\(rows)")
            // ups1 reads the stage-1 sum (a stage-1 chain buffer) and writes poly1;
            // unshuffle reads poly1 and the noise branch (n1) and writes x2.
            let handoff = chain("s1b0") + chain("s1b1") + chain("s1b2") + chain("n1") + ["poly1", "x2"] + small
            for key in handoff where key != "poly1" {
                XCTAssertFalse(range(l, "poly1").overlaps(range(l, key)), "r\(rows): poly1 overlaps \(key)")
            }
            assertDisjoint(l, chain("n1") + ["x2", "poly1"], "hand-off r\(rows)")
            // Stage 2: three chains, their shared input x2, and the output.
            let stage2 = chain("n1") + chain("s2b1") + chain("s2b2") + ["x2"] + small
            assertDisjoint(l, stage2, "stage 2 r\(rows)")
        }
    }

    func testNoTensorCrossesASlot() {
        let l = layout(rows: 11)
        let ends = l.slotStarts.dropFirst() + [l.size]
        for (key, r) in l.offsets.keys.map({ ($0, range(l, $0)) }) {
            let slot = zip(l.slotStarts, ends).first { $0.0 <= r.lowerBound && r.lowerBound < $0.1 }
            XCTAssertNotNil(slot, key)
            XCTAssertLessThanOrEqual(r.upperBound, slot!.1, "\(key) crosses a slot")
        }
    }

    /// The arena is the point: it must be smaller than one surface per tensor.
    func testArenaIsSmallerThanSeparateTensors() {
        let l = layout(rows: 11)
        let separate = l.lengths.values.reduce(0) { $0 + HybridArenaLayout.pad($1) }
        XCTAssertLessThan(Double(l.size), 0.8 * Double(separate))
    }

    /// Separate surfaces on A17-class iPhones, shared arena elsewhere; the
    /// environment override wins.
    func testSharingChoicePerDevice() {
        XCTAssertTrue(HybridArenaLayout.sharesMemory(machine: "iPhone13,3", override: nil))    // A14
        XCTAssertFalse(HybridArenaLayout.sharesMemory(machine: "iPhone16,2", override: nil))   // A17 Pro
        XCTAssertTrue(HybridArenaLayout.sharesMemory(machine: "arm64", override: nil))         // Mac
        XCTAssertTrue(HybridArenaLayout.sharesMemory(machine: "iPhone16,2", override: "1"))
        XCTAssertFalse(HybridArenaLayout.sharesMemory(machine: "iPhone13,3", override: "0"))
    }

    /// Separate layout: nothing overlaps at all.
    func testSeparateLayoutHasNoOverlap() {
        let l = HybridArenaLayout(rows: 4, w0: 240, w1: 2400, w2s: 14400, q6: 2496, shared: false)
        assertDisjoint(l, Array(l.offsets.keys), "separate")
    }
}
#endif
