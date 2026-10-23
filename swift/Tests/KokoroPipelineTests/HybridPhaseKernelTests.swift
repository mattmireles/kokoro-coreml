/// Phase-major rows in the hybrid generator's Metal kernels (HybridGenerator.swift).
/// A dilated conv on the ANE reads and writes each row phase-major: natural sample
/// w = q*d + p sits at p*(W/d) + q. `stats` must weight each sample by the mask at
/// its natural position, and `apply_phase` must convert between layouts. A wrong
/// index here still runs and sounds like noise, so pin it against a CPU reference.
import Metal
import XCTest
@testable import KokoroPipeline

final class HybridPhaseKernelTests: XCTestCase {
    private let rows = 2, channels = 3, width = 60  // W/d % 4 == 0 for d = 3 and 5

    func testApplyPhaseConvertsLayoutsAndAppliesSnake() throws {
        let k = try kernels()
        let x = (0..<(rows * channels * width)).map { Float(($0 * 37) % 101) / 50 - 1 }
        let zm = (0..<(rows * width)).map { Float(($0 * 7) % 5 == 0 ? 0 : 1) }
        let ss: [SIMD2<Float>] = [SIMD2(0.5, 0.1), SIMD2(-1.2, 0.3), SIMD2(2, -0.4)]
        for (dIn, dOut) in [(1, 3), (3, 1), (1, 5), (5, 1)] {
            let got = try k.applyPhase(x: x, zm: zm, ss: ss, dIn: dIn, dOut: dOut, rows: rows, channels: channels, width: width)
            for n in 0..<rows { for c in 0..<channels { for w in 0..<width {
                let base = (n * channels + c) * width
                let z = (x[base + position(w, dIn)] * ss[c].x + ss[c].y) * zm[n * width + w]
                let want = z + sin(z) * sin(z)
                XCTAssertEqual(got[base + position(w, dOut)], want, accuracy: 0.01, "dIn \(dIn) dOut \(dOut) n \(n) c \(c) w \(w)")
            }}}
        }
    }

    /// Wide rows (W/3 = 520, W/5 = 312 phase columns): many threads per row and
    /// the layout conversion at every block boundary.
    func testApplyPhaseAcrossTiles() throws {
        let k = try kernels()
        let rows = 1, channels = 2, width = 1560   // W/3 = 520, W/5 = 312 phase columns
        let x = (0..<(rows * channels * width)).map { Float(($0 * 31) % 97) / 48 - 1 }
        let zm = (0..<(rows * width)).map { Float($0 % 11 == 0 ? 0 : 1) }
        let ss: [SIMD2<Float>] = [SIMD2(0.7, -0.2), SIMD2(-0.9, 0.4)]
        func pos(_ w: Int, _ d: Int) -> Int { d == 1 ? w : (w % d) * (width / d) + w / d }
        for (dIn, dOut) in [(1, 3), (3, 1), (1, 5), (5, 1)] {
            let got = try k.applyPhase(x: x, zm: zm, ss: ss, dIn: dIn, dOut: dOut, rows: rows, channels: channels, width: width)
            for c in 0..<channels { for w in 0..<width {
                let base = c * width
                let z = (x[base + pos(w, dIn)] * ss[c].x + ss[c].y) * zm[w]
                XCTAssertEqual(got[base + pos(w, dOut)], z + sin(z) * sin(z), accuracy: 0.01, "dIn \(dIn) dOut \(dOut) c \(c) w \(w)")
            }}
        }
    }

    /// stats_add writes x = y + r and returns the statistics of that sum.
    /// The shift-read race it once had (thread 0 overwrote x[0] before other
    /// simdgroups read it: 19.4 dB audio in the full generator vs 46.0 fixed)
    /// does not reproduce in this harness. Not testable here: it depends on GPU
    /// scheduling; guarded by the end-to-end hybrid-check quality gate instead.
    func testStatsAddMatchesStatsOfTheSum() throws {
        let k = try kernels()
        let rows = 4, channels = 128, width = 14400   // stage-2 rows, as in production
        let y = (0..<(rows * channels * width)).map { Float(($0 * 17) % 53) / 20 + 3 }   // |mean| >> std
        let r = (0..<(rows * channels * width)).map { Float(($0 * 5) % 23) / 40 - 0.3 }
        let sm = (0..<(rows * width)).map { Float($0 % 9 == 0 ? 0 : 1) }
        let (x, got) = try k.statsAdd(y: y, r: r, sm: sm, rows: rows, channels: channels, width: width, threads: 256)
        let sum = zip(y, r).map { Float(Float16(Float(Float16($0)) + Float(Float16($1)))) }
        for i in stride(from: 0, to: x.count, by: 97) { XCTAssertEqual(x[i], sum[i], accuracy: 0.01, "x \(i)") }
        let want = try k.stats(x: sum, sm: sm, d: 1, rows: rows, channels: channels, width: width)
        for i in 0..<want.count {
            XCTAssertEqual(got[i].x, want[i].x, accuracy: abs(want[i].x) * 1e-3 + 0.5, "sum \(i)")
            XCTAssertEqual(got[i].y, want[i].y, accuracy: abs(want[i].y) * 1e-3 + 0.5, "sumsq \(i)")
        }
    }

    func testStatsWeightPhaseMajorRowsByNaturalMask() throws {
        let k = try kernels()
        let natural = (0..<(rows * channels * width)).map { Float(($0 * 13) % 29) / 10 }
        let sm = (0..<(rows * width)).map { Float($0 % 6 < 4 ? 1 : 0) }
        let d = 3
        var phased = natural
        for r in 0..<(rows * channels) { for w in 0..<width {
            phased[r * width + position(w, d)] = natural[r * width + w]
        }}
        let want = try k.stats(x: natural, sm: sm, d: 1, rows: rows, channels: channels, width: width)
        let got = try k.stats(x: phased, sm: sm, d: d, rows: rows, channels: channels, width: width)
        for i in 0..<want.count {
            XCTAssertEqual(got[i].x, want[i].x, accuracy: 0.01)
            XCTAssertEqual(got[i].y, want[i].y, accuracy: 0.01)
        }
    }

    /// join mode 1 (out += a): the residual add after each resblock's second conv.
    func testJoinAddsResidualInPlace() throws {
        let k = try kernels()
        let a = (0..<(rows * channels * width)).map { Float($0 % 13) / 4 }
        let o = (0..<(rows * channels * width)).map { Float($0 % 7) / 2 - 1 }
        let got = try k.join(a: a, out: o, zm: [Float](repeating: 0, count: rows * width), mode: 1,
                             rows: rows, channels: channels, width: width)
        for i in 0..<a.count { XCTAssertEqual(got[i], a[i] + o[i], accuracy: 0.01, "i \(i)") }
    }

    /// Join modes 4-6 add a branch's deferred last residual b in the branch sum:
    /// 4 out = a + b, 5 out += a + b, 6 out = (out + a + b) * zm * scale.
    func testJoinAddsDeferredResidual() throws {
        let k = try kernels()
        let a = (0..<(rows * channels * width)).map { Float($0 % 13) / 4 }
        let b = (0..<(rows * channels * width)).map { Float($0 % 5) / 3 - 0.5 }
        let o = (0..<(rows * channels * width)).map { Float($0 % 7) / 2 - 1 }
        let zm = (0..<(rows * width)).map { Float($0 % 4 == 0 ? 0 : 1) }
        let scale: Float = 1.0 / 3
        for mode in 4...6 {
            let got = try k.join(a: a, b: b, out: o, zm: zm, mode: mode, scale: scale, rows: rows, channels: channels, width: width)
            for i in 0..<a.count {
                let m = zm[(i / (channels * width)) * width + i % width]
                let want = mode == 4 ? a[i] + b[i] : mode == 5 ? o[i] + a[i] + b[i] : (o[i] + a[i] + b[i]) * m * scale
                XCTAssertEqual(got[i], want, accuracy: 0.02, "mode \(mode) i \(i)")
            }
        }
    }

    /// unshuffle with a column offset: stage 2 runs on a crop of the polyphase
    /// row (plan `crop2`), so output column w must read polyphase sample w + offset
    /// (shifted by the reflect pad), not w.
    func testUnshuffleReadsPolyphaseAtOffset() throws {
        let k = try kernels()
        let s = 3, q = 8, c = 2, n = 1, offset = 4, outW = 12
        let poly = (0..<(n * c * s * q)).map { Float($0) }
        let xs = [Float](repeating: 0.5, count: n * c * outW)
        let zm = [Float](repeating: 1, count: n * outW)
        for reflect in [false, true] {
            let got = try k.unshuffle(poly: poly, xs: xs, zm: zm, s: s, q: q, channels: c, rows: n, width: outW,
                                      reflect: reflect, offset: offset)
            for ch in 0..<c { for w in 0..<outW {
                let g = w + offset
                let j = reflect ? (g == 0 ? 1 : g - 1) : g
                let y: Float = j < q * s ? poly[(ch * s + j % s) * q + j / s] : 0
                XCTAssertEqual(got[ch * outW + w], y + 0.5, accuracy: 0.05, "reflect \(reflect) ch \(ch) w \(w)")
            }}
        }
    }

    /// Where natural sample w sits in a row stored phase-major with dilation d.
    private func position(_ w: Int, _ d: Int) -> Int { d == 1 ? w : (w % d) * (width / d) + w / d }

    private func kernels() throws -> Kernels {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("needs Metal") }
        return try Kernels(device: device)
    }
}

/// Runs single kernels from `hybridKernelSource` on small fp16 buffers.
private struct Kernels {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let lib: MTLLibrary

    init(device: MTLDevice) throws {
        self.device = device
        queue = device.makeCommandQueue()!
        lib = try device.makeLibrary(source: hybridKernelSource, options: nil)
    }

    func half(_ v: [Float]) -> MTLBuffer {
        let h = v.map { Float16($0) }
        return device.makeBuffer(bytes: h, length: h.count * 2, options: .storageModeShared)!
    }

    func run(_ name: String, _ body: (MTLComputeCommandEncoder) -> Void) throws {
        let pipe = try device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipe)
        body(e)
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    func applyPhase(x: [Float], zm: [Float], ss: [SIMD2<Float>], dIn: Int, dOut: Int,
                    rows: Int, channels: Int, width: Int) throws -> [Float] {
        let out = device.makeBuffer(length: x.count * 2, options: .storageModeShared)!
        var dims = SIMD4<UInt32>(UInt32(rows), UInt32(channels), UInt32(width), 1)
        var pd = SIMD2<UInt32>(UInt32(dIn), UInt32(dOut))
        try run("apply_phase") { e in
            e.setBuffer(half(x), offset: 0, index: 0)
            e.setBuffer(half(zm), offset: 0, index: 1)
            e.setBytes(ss, length: ss.count * 8, index: 2)
            e.setBuffer(out, offset: 0, index: 3)
            e.setBytes(&dims, length: 16, index: 4)
            e.setBytes(&pd, length: 8, index: 5)
            let q4 = width / max(dIn, dOut) / 4   // as HybridGenerator.adaSnakeBody dispatches it
            e.dispatchThreadgroups(MTLSize(width: (q4 + 63) / 64, height: channels, depth: rows),
                                   threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        let p = out.contents().assumingMemoryBound(to: Float16.self)
        return (0..<x.count).map { Float(p[$0]) }
    }

    func join(a: [Float], b: [Float]? = nil, out: [Float], zm: [Float], mode: Int, scale: Float = 1,
              rows: Int, channels: Int, width: Int) throws -> [Float] {
        let ob = half(out)
        var dims = SIMD4<UInt32>(UInt32(rows), UInt32(channels), UInt32(width), 0)
        var m = SIMD2<Float>(Float(mode), scale)
        try run("join") { e in
            e.setBuffer(half(a), offset: 0, index: 0)
            e.setBuffer(half(b ?? a), offset: 0, index: 1)
            e.setBuffer(half(zm), offset: 0, index: 2)
            e.setBuffer(ob, offset: 0, index: 3)
            e.setBytes(&dims, length: 16, index: 4)
            e.setBytes(&m, length: 8, index: 5)
            e.dispatchThreads(MTLSize(width: width / 4, height: channels, depth: rows),
                              threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
        }
        let p = ob.contents().assumingMemoryBound(to: Float16.self)
        return (0..<out.count).map { Float(p[$0]) }
    }

    func unshuffle(poly: [Float], xs: [Float], zm: [Float], s: Int, q: Int, channels: Int, rows: Int, width: Int,
                   reflect: Bool, offset: Int) throws -> [Float] {
        let out = device.makeBuffer(length: rows * channels * width * 2, options: .storageModeShared)!
        var dims = SIMD4<UInt32>(UInt32(rows), UInt32(channels), UInt32(width), 0)
        var p = SIMD4<UInt32>(UInt32(s), reflect ? 1 : 0, UInt32(q), UInt32(offset))
        try run("unshuffle") { e in
            e.setBuffer(half(poly), offset: 0, index: 0)
            e.setBuffer(half(xs), offset: 0, index: 1)
            e.setBuffer(half(zm), offset: 0, index: 2)
            e.setBuffer(out, offset: 0, index: 3)
            e.setBytes(&dims, length: 16, index: 4)
            e.setBytes(&p, length: 16, index: 5)
            e.dispatchThreads(MTLSize(width: width, height: channels, depth: rows),
                              threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
        }
        let ptr = out.contents().assumingMemoryBound(to: Float16.self)
        return (0..<(rows * channels * width)).map { Float(ptr[$0]) }
    }

    /// Per (channel, row): (shifted masked sum, sum of squares), shift = first stored value.
    /// stats_add: returns (x after the in-place add, centred stats like `stats`).
    func statsAdd(y: [Float], r: [Float], sm: [Float], rows: Int, channels: Int, width: Int, threads: Int) throws -> ([Float], [SIMD2<Float>]) {
        let partial = device.makeBuffer(length: channels * rows * 16, options: .storageModeShared)!
        let xb = half(y)
        var dims = SIMD4<UInt32>(UInt32(rows), UInt32(channels), UInt32(width), 1)
        try run("stats_add") { e in
            e.setBuffer(xb, offset: 0, index: 0)
            e.setBuffer(half(sm), offset: 0, index: 1)
            e.setBuffer(partial, offset: 0, index: 2)
            e.setBytes(&dims, length: 16, index: 3)
            e.setBuffer(half(r), offset: 0, index: 4)
            e.dispatchThreadgroups(MTLSize(width: channels, height: rows, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
        let xp = xb.contents().assumingMemoryBound(to: Float16.self)
        let p = partial.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let st: [SIMD2<Float>] = (0..<(channels * rows)).map { i in
            let (sq, k) = (p[i * 2], p[i * 2 + 1].x)
            let n = i % rows
            let cnt = sm[(n * width)..<((n + 1) * width)].reduce(0, +)
            return SIMD2(sq.x + k * cnt, sq.y + 2 * k * sq.x + k * k * cnt)
        }
        return ((0..<y.count).map { Float(xp[$0]) }, st)
    }

    func stats(x: [Float], sm: [Float], d: Int, rows: Int, channels: Int, width: Int) throws -> [SIMD2<Float>] {
        let partial = device.makeBuffer(length: channels * rows * 16, options: .storageModeShared)!
        var dims = SIMD4<UInt32>(UInt32(rows), UInt32(channels), UInt32(width), UInt32(d))
        try run("stats") { e in
            e.setBuffer(half(x), offset: 0, index: 0)
            e.setBuffer(half(sm), offset: 0, index: 1)
            e.setBuffer(partial, offset: 0, index: 2)
            e.setBytes(&dims, length: 16, index: 3)
            e.dispatchThreadgroups(MTLSize(width: channels, height: rows, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        let p = partial.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        // Rows start with different stored values, so compare sums around a common zero:
        // s0 = s + k*cnt, q0 = q + 2k*s + k^2*cnt.
        return (0..<(channels * rows)).map { i in
            let (sq, k) = (p[i * 2], p[i * 2 + 1].x)
            let n = i % rows
            let cnt = sm[(n * width)..<((n + 1) * width)].reduce(0, +)
            return SIMD2(sq.x + k * cnt, sq.y + 2 * k * sq.x + k * k * cnt)
        }
    }
}
