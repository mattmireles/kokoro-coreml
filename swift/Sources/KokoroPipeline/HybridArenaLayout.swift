// Compiled only with -DKOKORO_E5, off by default: the private e5rt path names
// private frameworks (Espresso), which App Store builds must not contain.
// Benchmarks: swift build -c release --package-path swift -Xswiftc -DKOKORO_E5,
// then run kokoro-bench with KOKORO_HYBRID_GENERATOR=1 KOKORO_HYBRID_E5=1.
#if KOKORO_E5
import Foundation

/// Byte offsets of the hybrid generator's pooled tensors in one arena, laid
/// out by lifetime (used with the private e5rt path, E5Runtime.swift).
///
/// With Core ML every tensor is its own IOSurface, so the pool is the sum of
/// all tensors. Here stage-1 tensors and `poly1` live inside the memory of
/// the two stage-2 branch chains, which are idle until stage 2 starts. S2 is
/// one stage-2 tensor (rows x 128 x w2s fp16):
///
///   A [0, 3 S2)    n1 chain: the stage-2 noise branch, runs during stage 1,
///                  then serves as stage-2 chain 0
///   B [3 S2, 9 S2) s2b1, s2b2 chains (stage 2). During stage 1: x1, poly0
///                  and the three stage-1 chains (11 stage-1 tensors from
///                  the start of B), and poly1 at B + 4 S2, clear of them:
///                  ups1 reads a stage-1 chain buffer (the branch sum) and
///                  writes poly1, which is dead before the stage-2 chains start
///   C [9 S2, 10 S2) x2: input of all three stage-2 branches
///   tail            small tensors (inputs, masks, logits, accumulators)
///
/// Keys and shapes match `HybridGenerator.pooled(_:_:_:)` call sites.
struct HybridArenaLayout {
    let offsets: [String: Int]
    let lengths: [String: Int]
    let size: Int
    /// Arena split points: one IOSurface per stage-2 slot (no tensor straddles
    /// one), then one per tail tensor. Small surfaces keep the ANE's per-call
    /// cost after idle like Core ML's; one ~445 MB surface ran ~50% slower cold.
    let slotStarts: [Int]

    static let page = 16384
    static func pad(_ n: Int) -> Int { (n + page - 1) / page * page }

    /// - Parameter chains: concurrent branch chains; below 3 the generator
    ///   pools a separate accumulator per stage (`acc.256`, `acc.128`).
    ///   - shared: lay tensors out by lifetime (true) or give each its own slot.
    init(rows: Int, w0: Int, w1: Int, w2s: Int, q6: Int, chains: Int = 3, shared: Bool = true) {
        func bytes(_ c: Int, _ w: Int) -> Int { rows * c * w * 2 }
        let s2 = Self.pad(bytes(128, w2s)), s1 = Self.pad(bytes(256, w1))
        var offsets: [String: Int] = [:], lengths: [String: Int] = [:]
        func put(_ key: String, _ offset: Int, _ c: Int, _ w: Int) { offsets[key] = offset; lengths[key] = bytes(c, w) }
        // A: n1 chain.
        for (i, part) in ["t1", "p0", "p1"].enumerated() { put("n1.\(part)", i * s2, 128, w2s) }
        // B: stage-2 chains 1 and 2.
        let b = 3 * s2
        for (j, chain) in ["s2b1", "s2b2"].enumerated() {
            for (i, part) in ["t1", "p0", "p1"].enumerated() { put("\(chain).\(part)", b + (3 * j + i) * s2, 128, w2s) }
        }
        // Stage-1 tensors inside B (x1, poly0, three chains), then poly1.
        precondition(bytes(2560, w0) <= s1, "poly0 must fit a stage-1 slot")
        precondition(s2 % s1 == 0, "stage-1 tensors must tile stage-2 slots")
        put("x1", b, 256, w1)
        put("poly0", b + s1, 2560, w0)
        var slot = 2
        for chain in ["s1b0", "s1b1", "s1b2"] {
            for part in ["t1", "p0", "p1"] { put("\(chain).\(part)", b + slot * s1, 256, w1); slot += 1 }
        }
        precondition(b + slot * s1 <= b + 4 * s2, "stage-1 tensors must end before poly1")
        precondition(bytes(768, w1) <= s2, "poly1 must fit one stage-2 slot")
        put("poly1", b + 4 * s2, 768, w1)
        // C: x2.
        put("x2", 9 * s2, 128, w2s)
        // Tail.
        var end = 10 * s2
        var starts = (0..<10).map { $0 * s2 }
        var tail = [("x_pre", 512, w0), ("har", 22, w2s), ("har6", 22 * 6, q6), ("sm1", 1, w1), ("zm1", 1, w1),
                    ("sm2", 1, w2s), ("zm2", 1, w2s), ("logits", 22, w2s)]
        if chains < 3 { tail += [("acc.256", 256, w1), ("acc.128", 128, w2s)] }
        for (key, c, w) in tail {
            put(key, end, c, w)
            starts.append(end)
            end += Self.pad(bytes(c, w))
        }
        if !shared {   // every tensor in its own slot
            var at = 0
            starts = []
            for key in offsets.keys.sorted() {
                offsets[key] = at
                starts.append(at)
                at += Self.pad(lengths[key]!)
            }
            end = at
        }
        self.offsets = offsets
        self.lengths = lengths
        self.size = end
        self.slotStarts = starts
    }

    /// Whether to share memory between tensors on this device. Measured
    /// (README/Notes/ane-folded-generator.md, "Private e5rt arena", 30 s):
    /// A14 shared 401-408 ms / 534 MB vs Core ML 417-432 ms / 745 MB; A17
    /// shared costs 10-15 ms, separate surfaces match Core ML (257-269 ms) at
    /// 840 MB vs 1,765 MB; M1 shared 568 MB vs separate 748 / Core ML 779 MB at
    /// equal speed within noise. So: separate on iPhone16,x and newer, shared
    /// elsewhere. KOKORO_E5_SHARE=0/1 overrides.
    static func sharesMemory(machine: String, override: String?) -> Bool {
        if let override { return override == "1" }
        guard machine.hasPrefix("iPhone"), let major = Int(machine.dropFirst(6).prefix { $0 != "," }) else { return true }
        return major < 16
    }

    static var sharesMemoryOnThisDevice: Bool {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return sharesMemory(machine: String(cString: machine), override: ProcessInfo.processInfo.environment["KOKORO_E5_SHARE"])
    }
}
#endif
