/// Hybrid generator: Neural Engine convolutions + Metal AdaIN/Snake.
///
/// On the ANE an elementwise pass costs about as much as a whole k=11 conv,
/// and the generator does ~10 of them per conv. Measured on an M1 per stage-2
/// unit: ANE conv 11 ms + Metal AdaIN+Snake 5 ms, against ~45 ms with
/// everything on the ANE (README/Notes/ane-folded-generator.md).
///
/// So the generator is cut at every AdaIN:
/// - the ANE runs only convolutions: one multifunction package, one function
///   per conv layer, exported by scripts/export_hybrid_generator.py;
/// - Metal runs the exact masked AdaIN statistics (fp32), the AdaIN affine,
///   Snake (native sin) and the branch joins.
/// Tensors stay row-folded, (rows, C, 1, W) fp16, in shared MTLBuffers that
/// both engines read and write without copies.
///
/// The step order mirrors `run_hybrid` in scripts/export_hybrid_generator.py;
/// keep the two in step.
import Accelerate
import CoreML
import CoreVideo
import Foundation
import Metal
import QuartzCore

public enum HybridGeneratorError: Error {
    case noMetal
    case badPlan(String)
    /// A Metal command buffer failed (e.g. iOS refuses GPU work in the
    /// background). Callers fall back to the folded ANE generator.
    case metalFailed(String)
}

/// plan.json written by scripts/export_hybrid_generator.py.
struct HybridPlan: Decodable {
    struct AdaIN: Decodable {
        let name: String
        let channels: Int
        let eps: Float
        let weight: Int
        let bias: Int
        let alpha: Int
    }
    let core: Int
    let halo: Int
    let w0: Int
    let w1: Int
    let w2: Int
    let extra2: Int
    let rows: [Int]
    let functions: [String]
    let adain: [AdaIN]
    /// Space-to-depth columns of the nc0 input (har6) and its left zero pad.
    let q6: Int
    let harPad: Int
    /// Conv functions whose input/output rows are phase-major (dilation d).
    let phase: [String: Int]?
    /// Stage 2 runs on columns [crop2, w2 - crop2) of the host's w2-wide rows
    /// (the outer halo is only needed by stage 1); nil in older plans.
    let crop2: Int?
    /// Second convs exported without the residual add; the host adds it on the
    /// GPU (export_hybrid_generator.py --resid-gpu).
    let residGPU: Bool?
    /// Width of the stage-2 tensors.
    var w2s: Int { w2 - 2 * (crop2 ?? 0) }

    private enum CodingKeys: String, CodingKey {
        case core, halo, w0, w1, w2, extra2, rows, functions, adain, q6, phase, crop2, harPad = "har_pad", residGPU = "resid_gpu"
    }
}

/// One fp16 tensor (rows, C, 1, W) in an IOSurface that Core ML and Metal share.
///
/// IOSurface-backed arrays are what make the hybrid fast: on an M1 a stage-2
/// ANE conv took 10.2 ms with pointer-backed MLMultiArrays (Core ML repacks
/// them on every call) and 5.7 ms with IOSurface-backed ones, with no extra
/// cost when Metal work runs between calls (README/Notes/ane-folded-generator.md).
/// W * 2 bytes must be a multiple of 64 so the surface has no row padding;
/// other widths fall back to pointer-backed memory.
final class SharedTensor {
    let buffer: MTLBuffer
    /// Byte offset of this tensor in `buffer` (non-zero only in an e5rt arena).
    let offset: Int
    /// e5rt buffer object (alias of the arena range) when running through E5Runtime.
    let e5: UnsafeMutableRawPointer?
    let array: MLMultiArray
    /// Nil for widths an IOSurface would pad (then `array` is pointer-backed).
    let pixelBuffer: CVPixelBuffer?
    let channels: Int
    let width: Int
    let rows: Int
    /// A tensor at `offset` in an e5rt arena (no Core ML array of its own).
    init(arena: E5Arena, offset: Int, rows: Int, channels: Int, width: Int) throws {
        self.rows = rows
        self.channels = channels
        self.width = width
        let length = rows * channels * width * 2
        (buffer, self.offset) = try arena.metal(offset: offset, length: length)
        e5 = try arena.buffer(offset: offset, length: length)
        pixelBuffer = nil
        array = try MLMultiArray(dataPointer: buffer.contents() + self.offset, shape: [rows, channels, 1, width].map { NSNumber(value: $0) },
                                 dataType: .float16, strides: [channels * width, width, width, 1].map { NSNumber(value: $0) })
    }

    init(device: MTLDevice, rows: Int, channels: Int, width: Int) throws {
        offset = 0
        e5 = nil
        self.rows = rows
        self.channels = channels
        self.width = width
        let shape = [rows, channels, 1, width].map { NSNumber(value: $0) }
        guard (width * 2) % 64 == 0 else {
            // Rows an IOSurface would pad (x_pre, the ups0 output): plain shared
            // memory. Core ML copies these on each call; both are small and used once.
            guard let buffer = device.makeBuffer(length: rows * channels * width * 2, options: .storageModeShared) else {
                throw HybridGeneratorError.noMetal
            }
            self.buffer = buffer
            pixelBuffer = nil
            array = try MLMultiArray(dataPointer: buffer.contents(), shape: shape, dataType: .float16,
                                     strides: [channels * width, width, width, 1].map { NSNumber(value: $0) })
            return
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                                    kCVPixelBufferMetalCompatibilityKey as String: true]
        guard CVPixelBufferCreate(nil, width, rows * channels, kCVPixelFormatType_OneComponent16Half,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess, let pb else {
            throw HybridGeneratorError.noMetal
        }
        CVPixelBufferLockBaseAddress(pb, [])
        let base = CVPixelBufferGetBaseAddress(pb)!
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        CVPixelBufferUnlockBaseAddress(pb, [])
        guard rowBytes == width * 2 else {
            throw HybridGeneratorError.badPlan("width \(width) pads IOSurface rows to \(rowBytes) bytes")
        }
        let length = (rowBytes * rows * channels + 16383) / 16384 * 16384
        guard let buffer = device.makeBuffer(bytesNoCopy: base, length: length, options: .storageModeShared, deallocator: nil) else {
            throw HybridGeneratorError.noMetal
        }
        self.buffer = buffer
        pixelBuffer = pb
        array = MLMultiArray(pixelBuffer: pb, shape: shape)
    }

    /// A pointer-backed view of the same memory for host readers. Reading
    /// `array.dataPointer` locks the pixel buffer, and Core ML then rejects it
    /// as an output backing on the next call ("pixel buffer ... has been locked").
    var hostView: MLMultiArray {
        // The memory outlives the view: the tensor is pooled by the generator.
        try! MLMultiArray(dataPointer: buffer.contents() + offset, shape: [rows, channels, 1, width].map { NSNumber(value: $0) },
                          dataType: .float16, strides: [channels * width, width, width, 1].map { NSNumber(value: $0) })
    }

    /// First byte of this tensor.
    var base: UnsafeMutableRawPointer { buffer.contents() + offset }

    /// Writes Float values (same layout) as fp16.
    func write(_ values: UnsafePointer<Float>, count: Int) {
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: values), height: 1,
                                width: vImagePixelCount(count), rowBytes: count * 4)
        var dst = vImage_Buffer(data: buffer.contents() + offset, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
    }
}


/// The hybrid generator for one bucket's row count. Needs multifunction
/// packages, so macOS 15 / iOS 18 (the minimum the SDK already ships for).
@available(macOS 15.0, iOS 18.0, *)
public final class HybridGenerator {
    let plan: HybridPlan
    let rows: Int
    let device: MTLDevice
    let queue: MTLCommandQueue
    let statsPipe: MTLComputePipelineState
    let statsAddPipe: MTLComputePipelineState
    let finalizePipe: MTLComputePipelineState
    let applyPipe: MTLComputePipelineState
    let joinPipe: MTLComputePipelineState
    let unshufflePipe: MTLComputePipelineState
    let applyPhasePipe: MTLComputePipelineState
    let models: [String: MLModel]
    let table: [Float]
    private var e5Library: E5Library?
    private var e5Arena: E5Arena?
    private var e5Layout: HybridArenaLayout?
    /// Accumulated seconds per step kind when KOKORO_HYBRID_PROFILE=1.
    public private(set) var profile: [String: Double] = [:]
    private let profiling = ProcessInfo.processInfo.environment["KOKORO_HYBRID_PROFILE"] == "1"
    private let profileLock = NSLock()
    private func timed<T>(_ key: String, _ f: () throws -> T) rethrows -> T {
        guard profiling else { return try f() }
        let t = CFAbsoluteTimeGetCurrent()
        let r = try f()
        profileLock.lock()
        profile[key, default: 0] += CFAbsoluteTimeGetCurrent() - t
        profileLock.unlock()
        return r
    }
    /// KOKORO_HYBRID_TIMELINE=1: record when each ANE call (host time) and GPU
    /// command buffer (gpuStartTime/gpuEndTime, same clock) ran, and summarize
    /// busy/idle per engine into `profile` (tl.* keys) after each run.
    private let timeline = ProcessInfo.processInfo.environment["KOKORO_HYBRID_TIMELINE"] == "1"
    private var aneSpans: [(Double, Double)] = []
    private var gpuSpans: [(Double, Double)] = []
    /// KOKORO_HYBRID_CALLS=1: print every ANE call (name, start, duration in ms
    /// from the run start) after each run, to see whether a slow run is slow
    /// from the first call or ramps (README/Notes, "Cold generator after a break").
    private let logCalls = ProcessInfo.processInfo.environment["KOKORO_HYBRID_CALLS"] == "1"
    private var calls: [(String, Double, Double)] = []
    private var runStartTime = 0.0
    private func noteCall(_ name: String, _ a: Double, _ b: Double) {
        guard logCalls else { return }
        profileLock.lock(); calls.append((name, a, b)); profileLock.unlock()
    }
    private func printCalls() {
        guard logCalls else { return }
        let line = calls.sorted { $0.1 < $1.1 }
            .map { String(format: "%@:%.1f:%.2f", $0.0, ($0.1 - runStartTime) * 1000, ($0.2 - $0.1) * 1000) }
            .joined(separator: " ")
        print("HYBRIDCALLS \(line)")
        calls = []
    }
    private func noteSpan(ane: Bool, _ a: Double, _ b: Double) {
        guard timeline, b > a else { return }
        profileLock.lock()
        if ane { aneSpans.append((a, b)) } else { gpuSpans.append((a, b)) }
        profileLock.unlock()
    }
    /// Merges spans and returns (union length, merged spans).
    static func union(_ spans: [(Double, Double)]) -> (Double, [(Double, Double)]) {
        var merged: [(Double, Double)] = []
        for s in spans.sorted(by: { $0.0 < $1.0 }) {
            if let last = merged.last, s.0 <= last.1 { merged[merged.count - 1].1 = max(last.1, s.1) } else { merged.append(s) }
        }
        return (merged.reduce(0) { $0 + $1.1 - $1.0 }, merged)
    }
    private func summarizeTimeline(start: Double, end: Double) {
        guard timeline else { return }
        let (ane, am) = Self.union(aneSpans), (gpu, gm) = Self.union(gpuSpans)
        var both = 0.0
        for a in am { for g in gm { both += max(0, min(a.1, g.1) - max(a.0, g.0)) } }
        let busy = ane + gpu - both
        profile["tl.span"] = end - start
        profile["tl.ane_busy"] = ane
        profile["tl.gpu_busy"] = gpu
        profile["tl.both_busy"] = both
        profile["tl.neither"] = (end - start) - busy
        profile["tl.ane_calls_sum"] = aneSpans.reduce(0) { $0 + $1.1 - $1.0 }
        aneSpans = []
        gpuSpans = []
    }

    /// One-line timeline summary of the last run (ms).
    public func timelineLine() -> String {
        func ms(_ k: String) -> String { String(format: "%.0f", (profile[k] ?? 0) * 1000) }
        return "hybrid timeline: span \(ms("tl.span")) ane \(ms("tl.ane_busy")) gpu \(ms("tl.gpu_busy")) both \(ms("tl.both_busy")) neither \(ms("tl.neither")) (ane call sum \(ms("tl.ane_calls_sum"))) ms\n"
    }

    private func gpuTimed(_ key: String, _ cb: MTLCommandBuffer) {
        guard profiling else { return }
        profileLock.lock()
        profile["gpu." + key, default: 0] += cb.gpuEndTime - cb.gpuStartTime
        profileLock.unlock()
    }
    /// Reused across calls: allocating ~1 GB of buffers per call cost more than the math.
    private var pool: [String: SharedTensor] = [:]
    /// Per-AdaIN (g, b, row counts) tables and the space-to-depth scratch, reused across runs.
    private var gbBuffers: [MTLBuffer] = []
    private var har6Scratch: [Float] = []
    private func pooled(_ key: String, _ c: Int, _ w: Int) throws -> SharedTensor {
        if let t = pool[key] { return t }
        let t: SharedTensor
        if let arena = e5Arena, let layout = e5Layout {
            guard let offset = layout.offsets[key], layout.lengths[key] == rows * c * w * 2 else {
                throw HybridGeneratorError.badPlan("arena layout has no slot for \(key) (\(c) x \(w))")
            }
            t = try SharedTensor(arena: arena, offset: offset, rows: rows, channels: c, width: w)
        } else {
            t = try SharedTensor(device: device, rows: rows, channels: c, width: w)
        }
        pool[key] = t
        return t
    }

    public var geometry: FoldGeometry { FoldGeometry(rows: rows, core: plan.core, halo: plan.halo) }
    /// Concurrent branch chains per stage (KOKORO_HYBRID_CHAINS, default 3).
    /// Each stage-2 chain holds three ~40 MB tensors at 30 s.
    let maxChains = max(1, min(3, Int(ProcessInfo.processInfo.environment["KOKORO_HYBRID_CHAINS"] ?? "") ?? 3))

    /// Compiled .mlmodelc per package, once per process (compiling takes ~7 s).
    private static var compiledURLs: [URL: URL] = [:]
    private static let compileLock = NSLock()
    private static func compiled(_ package: URL) throws -> URL {
        compileLock.lock()
        defer { compileLock.unlock() }
        if let url = compiledURLs[package] { return url }
        let url = try MLModel.compileModel(at: package)
        compiledURLs[package] = url
        return url
    }

    /// Environment switch: `1` lets providers use the hybrid generator.
    public static let envKey = "KOKORO_HYBRID_GENERATOR"
    public static var enabled: Bool { ProcessInfo.processInfo.environment[envKey] == "1" }

    /// Shortest bucket the hybrid serves. Below it the folded generator wins:
    /// the hybrid's ~100 Core ML/Metal round trips are a fixed cost (3 s,
    /// wall: M1 211 vs 135 ms folded, M2 Air 200 vs 103 ms), while from 7 s up
    /// the hybrid is 1.5-2x faster than folded.
    public static let minBucketSec = 7

    /// Row count for `xPreFrames` (80 Hz) in a bucket: the smallest packaged row
    /// count that holds them. Ship few packages that match the chunker (e.g. r4
    /// for 10 s phone chunks, r6 for 15 s, r11 for 30 s): each row-count switch
    /// reloads 53 function models (~0.35-0.5 s on an M1).
    public static func rows(bucketSec: Int, xPreFrames: Int, directory: URL) -> Int? {
        guard bucketSec >= minBucketSec else { return nil }
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("plan.json")),
              let plan = try? JSONDecoder().decode(HybridPlan.self, from: data) else { return nil }
        let needed = (xPreFrames + plan.core - 1) / plan.core
        let fm = FileManager.default
        func hasPackage(_ r: Int) -> Bool {
            fm.fileExists(atPath: directory.appendingPathComponent("hybrid_generator_r\(r).mlpackage").path)
                || fm.fileExists(atPath: directory.appendingPathComponent("hybrid_generator_r\(r).mlmodelc").path)
        }
        // KOKORO_HYBRID_ROWS=4,11: only these packaged row counts (benchmarks that
        // mimic an app shipping fewer packages).
        let only = ProcessInfo.processInfo.environment["KOKORO_HYBRID_ROWS"].map { Set($0.split(separator: ",").compactMap { Int($0) }) }
        return plan.rows.sorted().first { $0 >= needed && hasPackage($0) && (only?.contains($0) ?? true) }
    }

    /// - Parameters:
    ///   - directory: folder with hybrid_generator_r{rows}.mlpackage (or .mlmodelc), plan.json, adain.bin.
    ///   - rows: one of plan.rows (the bucket's row count).
    public init(directory: URL, rows: Int) throws {
        plan = try JSONDecoder().decode(HybridPlan.self, from: Data(contentsOf: directory.appendingPathComponent("plan.json")))
        guard plan.rows.contains(rows) else { throw HybridGeneratorError.badPlan("rows \(rows) not in \(plan.rows)") }
        // The Metal kernels read phase-major rows four samples at a time.
        for (name, d) in plan.phase ?? [:] where plan.w1 % (4 * d) != 0 || plan.w2s % (4 * d) != 0 {
            throw HybridGeneratorError.badPlan("\(name): widths \(plan.w1), \(plan.w2) not multiples of 4 * \(d)")
        }
        self.rows = rows
        let tableData = try Data(contentsOf: directory.appendingPathComponent("adain.bin"))
        table = tableData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw HybridGeneratorError.noMetal
        }
        self.device = device
        self.queue = queue
        let lib = try device.makeLibrary(source: hybridKernelSource, options: nil)
        statsPipe = try device.makeComputePipelineState(function: lib.makeFunction(name: "stats")!)
        statsAddPipe = try device.makeComputePipelineState(function: lib.makeFunction(name: "stats_add")!)
        finalizePipe = try device.makeComputePipelineState(function: lib.makeFunction(name: "finalize")!)
        joinPipe = try device.makeComputePipelineState(function: lib.makeFunction(name: "join")!)
        unshufflePipe = try device.makeComputePipelineState(function: lib.makeFunction(name: "unshuffle")!)
        applyPipe = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply")!)
        applyPhasePipe = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply_phase")!)

        // One fixed-shape package per row count: the ANE ran enumerated shapes ~5x slower.
        var modelURL = directory.appendingPathComponent("hybrid_generator_r\(rows).mlmodelc")
        if !FileManager.default.fileExists(atPath: modelURL.path) {
            modelURL = try Self.compiled(directory.appendingPathComponent("hybrid_generator_r\(rows).mlpackage"))
        }
        // KOKORO_HYBRID_E5=1: private e5rt dispatch with all tensors in one
        // lifetime-planned arena (E5Runtime.swift, HybridArenaLayout.swift).
        if ProcessInfo.processInfo.environment["KOKORO_HYBRID_E5"] == "1" {
            let layout = HybridArenaLayout(rows: rows, w0: plan.w0, w1: plan.w1, w2s: plan.w2s, q6: plan.q6, chains: maxChains,
                                           shared: HybridArenaLayout.sharesMemoryOnThisDevice)
            e5Library = try E5Library(mil: modelURL.appendingPathComponent("model.mil"), functions: plan.functions)
            e5Arena = try E5Arena(device: device, slotStarts: layout.slotStarts, size: layout.size)
            e5Layout = layout
            models = [:]
            logFootprint("hybrid.loaded e5 r\(rows)")
            return
        }
        // Load the functions concurrently: a serial loop took ~1.4 s on an M1,
        // paid again on every bucket switch (one generator stays resident).
        let names = plan.functions
        var slots = [Result<MLModel, Error>?](repeating: nil, count: names.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: names.count) { i in
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            cfg.functionName = names[i]
            let r = Result { try MLModel(contentsOf: modelURL, configuration: cfg) }
            lock.lock(); slots[i] = r; lock.unlock()
        }
        var loaded: [String: MLModel] = [:]
        for (i, name) in names.enumerated() { loaded[name] = try slots[i]!.get() }
        // A package exported for another plan (e.g. before the stage-2 crop) fails
        // later with a shape error deep in a run; check the output width up front.
        let postWidth = loaded["post"]?.modelDescription.outputDescriptionsByName["y"]?.multiArrayConstraint?.shape.last?.intValue
        if let postWidth, postWidth != plan.w2s {
            throw HybridGeneratorError.badPlan("hybrid_generator_r\(rows) has width \(postWidth), plan.json says \(plan.w2s): re-export or copy matching packages")
        }
        models = loaded
        logFootprint("hybrid.loaded r\(rows)")
    }

    // MARK: - Metal steps

    private func dims(_ t: SharedTensor) -> SIMD4<UInt32> {
        SIMD4(UInt32(t.rows), UInt32(t.channels), UInt32(t.width), 0)
    }

    /// out = z + sin(z)^2 with z the AdaIN of x (masked statistics), scaled by alpha.
    private func adaSnake(_ x: SharedTensor, out: SharedTensor, sm: SharedTensor, zm: SharedTensor,
                          gb: MTLBuffer, count: Float, eps: Float, partial: MTLBuffer, ss: MTLBuffer,
                          dIn: Int = 1, dOut: Int = 1, residual: SharedTensor? = nil) {
        timed("metal.adasnake.\(x.channels == 256 ? "s1" : "s2")") {
            adaSnakeBody(x, out: out, sm: sm, zm: zm, gb: gb, count: count, eps: eps, partial: partial, ss: ss,
                         dIn: dIn, dOut: dOut, residual: residual)
        }
    }

    private func adaSnakeBody(_ x: SharedTensor, out: SharedTensor, sm: SharedTensor, zm: SharedTensor,
                              gb: MTLBuffer, count: Float, eps: Float, partial: MTLBuffer, ss: MTLBuffer,
                              dIn: Int, dOut: Int, residual: SharedTensor?) {
        var d = dims(x)
        d.w = UInt32(dIn)
        var params = SIMD4<Float>(count, eps, 0, 0)
        var cb = queue.makeCommandBuffer()!
        var e = cb.makeComputeCommandEncoder()!
        // With `residual`, x += residual first (fused into the stats pass; --resid-gpu).
        precondition(residual == nil || dIn == 1, "stats_add reads natural rows")
        e.setComputePipelineState(residual == nil ? statsPipe : statsAddPipe)
        if let residual { e.setBuffer(residual.buffer, offset: residual.offset, index: 4) }
        e.setBuffer(x.buffer, offset: x.offset, index: 0)
        e.setBuffer(sm.buffer, offset: sm.offset, index: 1)
        e.setBuffer(partial, offset: 0, index: 2)
        e.setBytes(&d, length: 16, index: 3)
        e.dispatchThreadgroups(MTLSize(width: x.channels, height: x.rows, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if residual != nil { e.memoryBarrier(scope: .buffers) }   // apply reads the x stats_add wrote (arena buffers are untracked)
        e.setComputePipelineState(finalizePipe)
        e.setBuffer(partial, offset: 0, index: 0)
        e.setBuffer(gb, offset: 0, index: 1)
        e.setBuffer(ss, offset: 0, index: 2)
        e.setBytes(&d, length: 16, index: 3)
        e.setBytes(&params, length: 16, index: 4)
        e.dispatchThreads(MTLSize(width: x.channels, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: min(x.channels, 256), height: 1, depth: 1))
        if profiling {  // separate command buffers so the profile can split stats from apply
            e.endEncoding(); cb.commit(); wait(cb)
            gpuTimed(x.channels == 256 ? "stats.s1" : "stats.s2", cb)
            cb = queue.makeCommandBuffer()!; e = cb.makeComputeCommandEncoder()!
        }
        if dIn == 1 && dOut == 1 {
            e.setComputePipelineState(applyPipe)
        } else {
            e.setComputePipelineState(applyPhasePipe)
            var pd = SIMD2<UInt32>(UInt32(dIn), UInt32(dOut))
            e.setBytes(&pd, length: 8, index: 5)
        }
        e.setBuffer(x.buffer, offset: x.offset, index: 0)
        e.setBuffer(zm.buffer, offset: zm.offset, index: 1)
        e.setBuffer(ss, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: out.offset, index: 3)
        e.setBytes(&d, length: 16, index: 4)
        if dIn == 1 && dOut == 1 {
            e.dispatchThreads(MTLSize(width: x.width / 4, height: x.channels, depth: x.rows),
                              threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        } else {
            let q4 = x.width / max(dIn, dOut) / 4   // apply_phase: tiles of 64 phase half4 columns
            e.dispatchThreadgroups(MTLSize(width: (q4 + 63) / 64, height: x.channels, depth: x.rows),
                                   threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        e.endEncoding()
        cb.commit()
        wait(cb)
        gpuTimed((x.channels == 256 ? "adasnake.s1" : "adasnake.s2") + ".d\(dIn)\(dOut)", cb)
        noteFailure(cb)
    }

    /// Waits for a Metal command buffer. iOS spins instead of sleeping in
    /// `waitUntilCompleted`: on an iPhone 12 Pro the Core ML call after a
    /// sleeping wait cost ~10 ms more than an immediate repeat, and spinning cut
    /// the 30 s generator from 686-713 to 553-588 ms (end to end 903-945 →
    /// 804-834 ms). Thread QoS user-interactive alone gave ~4%. On an M1 it made
    /// no clear difference, so macOS keeps sleeping (shared machines).
    /// KOKORO_HYBRID_SPIN=0/1 overrides.
    private let spinWait: Bool = {
        if let v = ProcessInfo.processInfo.environment["KOKORO_HYBRID_SPIN"] { return v == "1" }
        #if os(iOS)
        return true
        #else
        return false
        #endif
    }()
    private func wait(_ cb: MTLCommandBuffer) {
        if spinWait {
            while cb.status != .completed && cb.status != .error {}
        } else {
            cb.waitUntilCompleted()
        }
        noteSpan(ane: false, cb.gpuStartTime, cb.gpuEndTime)
    }

    private let failureLock = NSLock()
    private var metalFailure: String?
    private func noteFailure(_ cb: MTLCommandBuffer) {
        guard cb.status == .error else { return }
        failureLock.lock()
        if metalFailure == nil { metalFailure = cb.error.map { "\($0)" } ?? "command buffer error" }
        failureLock.unlock()
    }

    /// mode 0: out = a; 1: out += a; 2: out = (out + a) * zm * scale; 3: out = (a + b) * zm.
    private func join(_ a: SharedTensor, _ b: SharedTensor?, zm: SharedTensor, out: SharedTensor, mode: Int, scale: Float = 1) {
        timed("metal.join") { joinBody(a, b, zm: zm, out: out, mode: mode, scale: scale) }
    }

    private func joinBody(_ a: SharedTensor, _ b: SharedTensor?, zm: SharedTensor, out: SharedTensor, mode: Int, scale: Float) {
        var d = dims(out)
        var m = SIMD2<Float>(Float(mode), scale)
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(joinPipe)
        e.setBuffer(a.buffer, offset: a.offset, index: 0)
        e.setBuffer((b ?? a).buffer, offset: (b ?? a).offset, index: 1)
        e.setBuffer(zm.buffer, offset: zm.offset, index: 2)
        e.setBuffer(out.buffer, offset: out.offset, index: 3)
        e.setBytes(&d, length: 16, index: 4)
        e.setBytes(&m, length: 8, index: 5)
        e.dispatchThreads(MTLSize(width: out.width / 4, height: out.channels, depth: out.rows),
                          threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        e.endEncoding()
        cb.commit()
        wait(cb)
        noteFailure(cb)
    }

    /// out = (interleave(poly) + xs) * zm; see the `unshuffle` kernel.
    private func unshuffle(_ poly: SharedTensor, stride sTaps: Int, reflect: Bool, xs: SharedTensor, zm: SharedTensor, out: SharedTensor,
                           offset: Int = 0) {
        timed("metal.unshuffle") {
            var d = dims(out)
            var p = SIMD4<UInt32>(UInt32(sTaps), reflect ? 1 : 0, UInt32(poly.width), UInt32(offset))
            let cb = queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(unshufflePipe)
            e.setBuffer(poly.buffer, offset: poly.offset, index: 0)
            e.setBuffer(xs.buffer, offset: xs.offset, index: 1)
            e.setBuffer(zm.buffer, offset: zm.offset, index: 2)
            e.setBuffer(out.buffer, offset: out.offset, index: 3)
            e.setBytes(&d, length: 16, index: 4)
            e.setBytes(&p, length: 16, index: 5)
            e.dispatchThreads(MTLSize(width: out.width, height: out.channels, depth: out.rows),
                              threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
            e.endEncoding()
            cb.commit()
            wait(cb)
            noteFailure(cb)
        }
    }

    // MARK: - ANE steps

    private func ane(_ name: String, _ inputs: [String: SharedTensor], out: SharedTensor) throws {
        let perFunction = ProcessInfo.processInfo.environment["KOKORO_HYBRID_PROFILE_FN"] == "1"
        let key = perFunction || !(name.hasPrefix("rb") || name.hasPrefix("nr")) ? "ane.\(name)" : "ane.\(out.channels == 256 ? "s1" : "s2")"
        let a = CACurrentMediaTime()
        if let op = e5Library?.operation(name) {   // private e5rt path (E5Runtime.swift)
            try timed(key) { try op.run(inputs: inputs.mapValues { $0.e5! }, output: ("y", out.e5!)) }
            let b = CACurrentMediaTime()
            noteSpan(ane: true, a, b)
            noteCall(name, a, b)
            return
        }
        let opts = MLPredictionOptions()
        opts.outputBackings = ["y": out.array]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0.array) })
        // autoreleasepool: Core ML's per-call objects otherwise pile up until the
        // caller's pool drains; a CLI loop reached a 4.3 GB peak footprint.
        try autoreleasepool { _ = try timed(key) { try models[name]!.prediction(from: provider, options: opts) } }
        let b = CACurrentMediaTime()
        noteSpan(ane: true, a, b)
        noteCall(name, a, b)
        // Diagnostic: KOKORO_HYBRID_REPEAT=1 times a second, back-to-back call of
        // the same function, to separate ANE wake-up cost from the call itself.
        if profiling, ProcessInfo.processInfo.environment["KOKORO_HYBRID_REPEAT"] == "1" {
            _ = try timed("warm.\(key)") { try models[name]!.prediction(from: provider, options: opts) }
        }
    }

    // MARK: - Synthesis

    /// Scratch buffers of one independent chain (a resblock branch or a noise
    /// branch): three tensors. Pair k writes conv1 into ping[k % 2], AdaIN+Snake
    /// reads it back into t1, and conv2 (+ residual) writes into ping[k % 2]
    /// again, which is free by then; the residual it reads is the other ping
    /// (or the chain's start), so nothing aliases.
    private struct Chain {
        let t1: SharedTensor, ping: [SharedTensor]
        let partial: MTLBuffer, ss: MTLBuffer
    }

    private func chain(_ name: String, _ c: Int, _ w: Int) throws -> Chain {
        Chain(t1: try pooled("\(name).t1", c, w),
              ping: [try pooled("\(name).p0", c, w), try pooled("\(name).p1", c, w)],
              partial: device.makeBuffer(length: 256 * rows * 16, options: .storageModeShared)!,
              ss: device.makeBuffer(length: 256 * 8, options: .storageModeShared)!)
    }

    /// Runs the generator on folded inputs and returns conv_post logits (rows, 22, 1, w2) fp16.
    ///
    /// Independent chains run on their own threads so one chain's conv keeps the
    /// ANE busy while another chain's AdaIN runs on the GPU: the stage-2 noise
    /// branch runs during all of stage 1, and each stage's three resblock
    /// branches run together. Strict alternation left each engine idle, and
    /// waking it cost ~10 ms per ANE call on an M1.
    ///
    /// - Parameters:
    ///   - xPre: (rows, 512, 1, w0) folded decoder output, Float.
    ///   - har: (rows, 22, 1, w2) folded harmonic features, Float.
    ///   - style: the 128-dim AdaIN style vector (ref_s[0..<128]).
    ///   - sm1, zm1, sm2, zm2: Float stats/zero masks at stage 1 (rows x w1) and stage 2 (rows x w2).
    public func run(xPre: [Float], har: [Float], style: [Float],
                    sm1: [Float], zm1: [Float], sm2: [Float], zm2: [Float]) throws -> MLMultiArray {
        profile = [:]
        metalFailure = nil
        let runStart = CACurrentMediaTime()
        runStartTime = runStart
        defer { summarizeTimeline(start: runStart, end: CACurrentMediaTime()); printCalls() }

        let setupStart = CFAbsoluteTimeGetCurrent()
        let r = rows
        let w0 = plan.w0, w1 = plan.w1, w2 = plan.w2, w2s = plan.w2s, crop = plan.crop2 ?? 0
        let xIn = try pooled("x_pre", 512, w0), harIn = try pooled("har", 22, w2s)
        let har6In = try pooled("har6", 22 * 6, plan.q6)
        let m1s = try pooled("sm1", 1, w1), m1z = try pooled("zm1", 1, w1)
        let m2s = try pooled("sm2", 1, w2s), m2z = try pooled("zm2", 1, w2s)
        xPre.withUnsafeBufferPointer { xIn.write($0.baseAddress!, count: $0.count) }
        har.withUnsafeBufferPointer {
            HybridInputs.writeCropped($0.baseAddress!, rows: r * 22, srcWidth: w2, crop: crop, width: w2s, into: harIn.base)
        }
        if har6Scratch.count != r * 22 * 6 * plan.q6 { har6Scratch = [Float](repeating: 0, count: r * 22 * 6 * plan.q6) }
        har.withUnsafeBufferPointer { h in
            har6Scratch.withUnsafeMutableBufferPointer { sc in
                HybridInputs.writeSpaceToDepth6(h.baseAddress!, rows: r, channels: 22, w2: w2, q6: plan.q6, pad: plan.harPad,
                                                scratch: sc.baseAddress!, into: har6In.base)
            }
        }
        sm1.withUnsafeBufferPointer { m1s.write($0.baseAddress!, count: $0.count) }
        zm1.withUnsafeBufferPointer { m1z.write($0.baseAddress!, count: $0.count) }
        sm2.withUnsafeBufferPointer { HybridInputs.writeCropped($0.baseAddress!, rows: r, srcWidth: w2, crop: crop, width: w2s, into: m2s.base) }
        zm2.withUnsafeBufferPointer { HybridInputs.writeCropped($0.baseAddress!, rows: r, srcWidth: w2, crop: crop, width: w2s, into: m2z.base) }

        // Per-AdaIN g = (1 + gamma) * alpha, b = beta * alpha (style projection on the host),
        // followed by the per-row stats-mask counts the finalize kernel needs.
        let rowCounts1 = (0..<r).map { n in sm1[(n * w1)..<((n + 1) * w1)].reduce(0, +) }
        let rowCounts2 = (0..<r).map { n in sm2[(n * w2)..<((n + 1) * w2)].reduce(0, +) }
        if gbBuffers.isEmpty {   // allocated once; contents rewritten per run
            gbBuffers = try plan.adain.map { entry in
                guard let b = device.makeBuffer(length: (2 * entry.channels + r) * 4, options: .storageModeShared) else {
                    throw HybridGeneratorError.noMetal
                }
                return b
            }
        }
        for (i, entry) in plan.adain.enumerated() {
            let c = entry.channels
            var h = [Float](repeating: 0, count: 2 * c)
            table.withUnsafeBufferPointer { t in
                style.withUnsafeBufferPointer { s in
                    vDSP_mmul(t.baseAddress! + entry.weight, 1, s.baseAddress!, 1, &h, 1, vDSP_Length(2 * c), 1, vDSP_Length(style.count))
                }
            }
            let counts = c == 256 ? rowCounts1 : rowCounts2
            var gb = [Float](repeating: 0, count: 2 * c + counts.count)
            for k in 0..<c {
                let alpha = table[entry.alpha + k]
                gb[2 * k] = (1 + h[k] + table[entry.bias + k]) * alpha
                gb[2 * k + 1] = (h[c + k] + table[entry.bias + c + k]) * alpha
            }
            for n in 0..<counts.count { gb[2 * c + n] = counts[n] }
            gbBuffers[i].contents().copyMemory(from: gb, byteCount: gb.count * 4)
        }
        let adaBase = Dictionary(uniqueKeysWithValues: plan.adain.enumerated().map { ($0.element.name, $0.offset) })
        let count1 = sm1.reduce(0, +), count2 = sm2.reduce(0, +)
        if profiling { profile["setup"] = CFAbsoluteTimeGetCurrent() - setupStart }

        /// One resblock (3 AdaIN-Snake-conv pairs) from `start`; returns the buffer holding the result.
        // Branch outputs whose last residual (--resid-gpu) is added by the branch sum.
        let deferredLock = NSLock()
        var deferred: [ObjectIdentifier: SharedTensor] = [:]
        func takeDeferred(_ y: SharedTensor) -> SharedTensor? {
            deferredLock.lock(); defer { deferredLock.unlock() }
            return deferred.removeValue(forKey: ObjectIdentifier(y))
        }
        func resblock(_ prefix: String, start: SharedTensor, ch: Chain, sm: SharedTensor, zm: SharedTensor, count: Float,
                      deferResidual: Bool = false) throws -> SharedTensor {
            var x = start
            // --resid-gpu: x holds conv2's output and `pending` the residual still
            // to add; the next pair's stats pass adds it (stats_add), the last
            // pair's is added by a join.
            var pending: SharedTensor?
            for k in 0..<3 {
                let i1 = adaBase["\(prefix)_n1_\(k)"]!, i2 = adaBase["\(prefix)_n2_\(k)"]!
                let next = ch.ping[k % 2]
                // Dilated convs read and write phase-major rows (plan.phase); see PhaseConv in the exporter.
                let d = plan.phase?["\(prefix)_c1_\(k)"] ?? 1
                adaSnake(x, out: ch.t1, sm: sm, zm: zm, gb: gbBuffers[i1], count: count, eps: plan.adain[i1].eps,
                         partial: ch.partial, ss: ch.ss, dIn: 1, dOut: d, residual: pending)
                pending = nil
                try ane("\(prefix)_c1_\(k)", ["t": ch.t1], out: next)
                adaSnake(next, out: ch.t1, sm: sm, zm: zm, gb: gbBuffers[i2], count: count, eps: plan.adain[i2].eps,
                         partial: ch.partial, ss: ch.ss, dIn: d, dOut: 1)
                if plan.residGPU == true {
                    try ane("\(prefix)_c2_\(k)", ["t": ch.t1], out: next)
                    pending = x
                } else {
                    try ane("\(prefix)_c2_\(k)", ["t": ch.t1, "r": x], out: next)   // conv2 + residual
                }
                x = next
            }
            if let pending {
                if deferResidual {
                    deferredLock.lock(); deferred[ObjectIdentifier(x)] = pending; deferredLock.unlock()
                } else {
                    join(pending, nil, zm: zm, out: x, mode: 1)
                }
            }
            return x
        }

        /// Runs `work` items concurrently; rethrows the first error.
        func parallel(_ work: [() throws -> Void]) throws {
            // KOKORO_HYBRID_SERIAL=1 runs chains one after another (profiling only).
            if ProcessInfo.processInfo.environment["KOKORO_HYBRID_SERIAL"] == "1" {
                for w in work { try w() }
                return
            }
            var firstError: Error?
            let lock = NSLock()
            DispatchQueue.concurrentPerform(iterations: work.count) { i in
                do { try work[i]() } catch { lock.lock(); if firstError == nil { firstError = error }; lock.unlock() }
            }
            if let firstError { throw firstError }
        }

        /// The three resblock branches of a stage, averaged (masked); see `sumBranches`.
        func branches(chains: [Chain], _ body: @escaping (Int, Chain) throws -> SharedTensor) throws -> SharedTensor {
            let first = chains[0].ping[0]
            let own = chains.count >= 3 ? nil : try pooled("acc.\(first.channels)", first.channels, first.width)
            let zm = first.channels == 256 ? m1z : m2z
            return try Self.sumBranches(chains: chains, accumulator: own, parallel: parallel, body: body, claim: { y in
                if let r = takeDeferred(y) { self.join(r, nil, zm: zm, out: y, mode: 1) }
            }) { y, acc, mode in
                // A deferred last residual is added in the same pass (join modes 4-6).
                let r = takeDeferred(y)
                self.join(y, r, zm: zm, out: acc, mode: r == nil ? mode : mode + 4, scale: 1.0 / 3.0)
            }
        }

        // Stage-2 noise branch: depends only on har, so it runs during all of stage 1.
        let n1Chain = try chain("n1", 128, w2s)
        let n1In = n1Chain.ping[1]   // pair 0 writes ping[0]; ping[1] is free until pair 1
        var xs2: SharedTensor?
        var n1Error: Error?
        let n1Group = DispatchGroup()
        DispatchQueue.global(qos: .userInitiated).async(group: n1Group) {
            do {
                try self.ane("nc1", ["har": harIn], out: n1In)
                xs2 = try resblock("nr1", start: n1In, ch: n1Chain, sm: m2s, zm: m2z, count: count2)
            } catch { n1Error = error }
        }

        // Stage 1.
        let x1 = try pooled("x1", 256, w1)
        let poly0 = try pooled("poly0", 256 * 10, w0)
        let n0Chain = try chain("s1b0", 256, w1)
        let n0In = n0Chain.ping[1]
        var xs1: SharedTensor?
        try parallel([
            { try self.ane("ups0", ["x": xIn], out: poly0) },
            {
                try self.ane("nc0", ["har6": har6In], out: n0In)
                xs1 = try resblock("nr0", start: n0In, ch: n0Chain, sm: m1s, zm: m1z, count: count1)
            },
        ])
        unshuffle(poly0, stride: 10, reflect: false, xs: xs1!, zm: m1z, out: x1)   // x = (ups + noise) * zm
        let s1Chains = try (0..<maxChains).map { try chain("s1b\($0)", 256, w1) }
        let s1Out = try branches(chains: s1Chains) { j, ch in
            try resblock("rb\(j)", start: x1, ch: ch, sm: m1s, zm: m1z, count: count1, deferResidual: true)
        }
        let acc1 = s1Out
        logFootprint("hybrid.stage1")

        // Stage 2.
        let x2 = try pooled("x2", 128, w2s)
        let poly1 = try pooled("poly1", 128 * 6, w1)
        try ane("ups1", ["x": acc1], out: poly1)
        n1Group.wait()
        if let n1Error { throw n1Error }
        unshuffle(poly1, stride: 6, reflect: true, xs: xs2!, zm: m2z, out: x2, offset: crop)
        let s2Chains = [n1Chain] + (try (1..<maxChains).map { try chain("s2b\($0)", 128, w2s) })
        let s2Out = try branches(chains: s2Chains) { j, ch in
            try resblock("rb\(j + 3)", start: x2, ch: ch, sm: m2s, zm: m2z, count: count2, deferResidual: true)
        }
        let acc2 = s2Out
        logFootprint("hybrid.stage2")
        let logits = try pooled("logits", 22, w2s)
        try ane("post", ["x": acc2], out: logits)

        if let metalFailure { throw HybridGeneratorError.metalFailed(metalFailure) }
        guard crop > 0 else { return logits.hostView }
        // Back to the host's w2-wide layout (the crop margins stay zero; the unfold reads cores only).
        let wide = try MLMultiArray(shape: [r, 22, 1, w2].map { NSNumber(value: $0) }, dataType: .float16)
        let src = (logits.buffer.contents() + logits.offset).assumingMemoryBound(to: UInt16.self)
        let dst = wide.dataPointer.assumingMemoryBound(to: UInt16.self)
        memset(dst, 0, r * 22 * w2 * 2)
        for i in 0..<(r * 22) { memcpy(dst + i * w2 + crop, src + i * w2s, w2s * 2) }
        return wide
    }

    /// Folds the executor's tensors, runs the generator, and returns audio.
    ///
    /// - Parameters:
    ///   - xPre: decoder-pre output (1, 512, T), at least `validXPreFrames` frames.
    ///   - har: harmonic features, channel-major (22, harFrames).
    ///   - refS: 256-dim voice embedding (style = refS[0..<128]).
    /// - Returns: `validXPreFrames * 300` samples at 24 kHz.
    public func synthesize(xPre: MLMultiArray, validXPreFrames: Int, har: [Float], harFrames: Int, refS: [Float]) throws -> [Float] {
        let g = geometry
        let prepStart = CFAbsoluteTimeGetCurrent()
        guard validXPreFrames <= g.capacity else {
            throw PipelineError.modelContractMismatch("hybrid generator holds \(g.capacity) x_pre frames, got \(validXPreFrames)")
        }
        let channels = xPre.shape[1].intValue
        let channelStride = xPre.strides[1].intValue
        var xValid = [Float](repeating: 0, count: channels * validXPreFrames)
        xValid.withUnsafeMutableBufferPointer { dst in
            for c in 0..<channels {
                readRun(xPre, offset: c * channelStride, count: validXPreFrames, into: dst.baseAddress! + c * validXPreFrames)
            }
        }
        let harValid = validXPreFrames * FoldGeometry.stage2Rate + 1
        var harTrim = [Float](repeating: 0, count: HarmonicConstants.harChannels * harValid)
        for c in 0..<HarmonicConstants.harChannels {
            let n = min(harValid, harFrames)
            for t in 0..<n { harTrim[c * harValid + t] = har[c * harFrames + t] }
        }
        func dense(_ a: MLMultiArray) -> [Float] {
            Array(UnsafeBufferPointer(start: a.dataPointer.assumingMemoryBound(to: Float.self), count: a.count))
        }
        let extra = plan.extra2
        let xFold = try foldRows(xValid, channels: channels, time: validXPreFrames, rate: 1, extra: 0, geometry: g)
        let harFold = try foldRows(harTrim, channels: HarmonicConstants.harChannels, time: harValid,
                                   rate: FoldGeometry.stage2Rate, extra: extra, geometry: g)
        let m1 = try foldMasks(rate: FoldGeometry.stage1Rate, extra: 0, valid: validXPreFrames * FoldGeometry.stage1Rate, geometry: g)
        let m2 = try foldMasks(rate: FoldGeometry.stage2Rate, extra: extra, valid: harValid, geometry: g)
        let inputs = (dense(xFold), dense(harFold), dense(m1.stats), dense(m1.zero), dense(m2.stats), dense(m2.zero))
        let prep = CFAbsoluteTimeGetCurrent() - prepStart
        let runStart = CFAbsoluteTimeGetCurrent()
        let logits = try run(xPre: inputs.0, har: inputs.1, style: Array(refS[0..<128]),
                             sm1: inputs.2, zm1: inputs.3, sm2: inputs.4, zm2: inputs.5)
        let body = CFAbsoluteTimeGetCurrent() - runStart
        let tailStart = CFAbsoluteTimeGetCurrent()
        let audio = foldedGeneratorAudio(logits: logits, validXPreFrames: validXPreFrames, geometry: g)
        if timeline { FileHandle.standardError.write(Data(timelineLine().utf8)); print(timelineLine(), terminator: "") }
        if profiling {
            let ms = { (t: Double) in String(format: "%.1f", t * 1000) }
            let setup = profile["setup"].map(ms) ?? "?"
            var lines = "hybrid synth: prep \(ms(prep)) run \(ms(body)) (setup \(setup)) tail \(ms(CFAbsoluteTimeGetCurrent() - tailStart)) ms\n"
            for (k, v) in profile.sorted(by: { $0.value > $1.value }) where k != "setup" { lines += "  hybrid \(k) \(ms(v)) ms\n" }
            FileHandle.standardError.write(Data(lines.utf8))
        }
        return audio
    }
}
