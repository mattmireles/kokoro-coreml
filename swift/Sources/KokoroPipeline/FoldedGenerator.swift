/// Host side of the row-folded generator (the Neural Engine generator).
///
/// The Neural Engine rejects any tensor axis over 16,384, and the generator's
/// time axis reaches 144,001 at 30 s. The folded package therefore takes the
/// x_pre time axis cut into `rows` rows of `core` frames, each widened by
/// `halo` frames of real neighbour data on both sides, as (rows, C, 1, W)
/// tensors (rows in the batch axis, so the package can reshape W without a
/// transpose). This file builds those inputs, the masks that keep AdaIN
/// statistics global and exact, and turns the package's conv_post logits back
/// into audio (exp/sin + iSTFT, which the package leaves out).
///
/// Mirrors ``fold``/``masks``/``unfold``/``tail`` in
/// scripts/probe_generator_ane_fold.py; keep the two in step.
import Accelerate
import CoreML
import Foundation

/// Geometry of a folded generator package, read from its input shapes.
public struct FoldGeometry: Equatable {
    /// x_pre samples per frame at each generator axis.
    public static let stage1Rate = 10
    public static let stage2Rate = 60
    /// Stage-2 width is 60 * w0 + 5: one reflect-pad sample plus four zeros so
    /// the width is a multiple of 5 (the dilation-5 convs run space-to-batch).
    public static let stage2Extra = 5

    public let rows: Int
    public let core: Int
    public let halo: Int

    public var w0: Int { core + 2 * halo }
    public var w1: Int { w0 * Self.stage1Rate }
    public var w2: Int { w0 * Self.stage2Rate + Self.stage2Extra }
    /// x_pre frames the package can hold.
    public var capacity: Int { rows * core }

    public init(rows: Int, core: Int, halo: Int) {
        self.rows = rows
        self.core = core
        self.halo = halo
    }

    /// Reads rows and width from the package's `x_pre` input (rows, 512, 1, w0)
    /// and the halo from its `kokoro.fold.halo` metadata. Nil for any other package.
    public init?(model: MLModel) {
        guard let shape = inputShapes(from: model)["x_pre"], shape.count == 4,
              let haloText = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
              let halo = haloText["kokoro.fold.halo"].flatMap(Int.init) else { return nil }
        self.init(rows: shape[0], core: shape[3] - 2 * halo, halo: halo)
    }

    /// First global sample (at `rate` samples per frame) of row `r`'s window.
    func rowStart(_ r: Int, rate: Int) -> Int { (r * core - halo) * rate }
}

/// Copies `count` contiguous elements of `array` starting at element `offset`
/// into `dst` as Float. fp16 runs convert with vImage. The innermost axis must
/// have stride 1 (true for Core ML outputs; padding lives in the outer strides).
///
/// Do not use ``floatValues(from:)`` here: on a strided fp16 Neural Engine
/// output it walks element by element and took ~1 s per 30 s clip on an M1.
func readRun(_ array: MLMultiArray, offset: Int, count: Int, into dst: UnsafeMutablePointer<Float>) {
    switch array.dataType {
    case .float32:
        dst.update(from: array.dataPointer.assumingMemoryBound(to: Float.self) + offset, count: count)
    case .float16:
        var src = vImage_Buffer(data: array.dataPointer.assumingMemoryBound(to: UInt16.self) + offset,
                                height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var out = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&src, &out, vImage_Flags(kvImageNoFlags))
    default:
        for i in 0..<count { dst[i] = array[offset + i].floatValue }
    }
}

/// Copies a channel-major (C, T) signal at `rate` samples per x_pre frame into
/// (rows, C, 1, W). Window positions outside [0, T) stay zero.
func foldRows(_ values: [Float], channels: Int, time: Int, rate: Int, extra: Int, geometry g: FoldGeometry) throws -> MLMultiArray {
    let w = g.w0 * rate + extra
    let out = try MLMultiArray(shape: [NSNumber(value: g.rows), NSNumber(value: channels), 1, NSNumber(value: w)], dataType: .float32)
    let dst = out.dataPointer.assumingMemoryBound(to: Float.self)
    dst.update(repeating: 0, count: out.count)
    values.withUnsafeBufferPointer { src in
        for c in 0..<channels {
            for r in 0..<g.rows {
                let g0 = g.rowStart(r, rate: rate)
                let lo = max(g0, 0), hi = min(g0 + w, time)
                guard hi > lo else { continue }
                (dst + (r * channels + c) * w + (lo - g0)).update(from: src.baseAddress! + c * time + lo, count: hi - lo)
            }
        }
    }
    return out
}

/// Zero mask (global index in [0, valid)) and stats mask (zero mask and owned
/// by this row, so every index is counted once), each (rows, 1, 1, W).
func foldMasks(rate: Int, extra: Int, valid: Int, geometry g: FoldGeometry) throws -> (stats: MLMultiArray, zero: MLMultiArray) {
    let w = g.w0 * rate + extra
    let shape: [NSNumber] = [NSNumber(value: g.rows), 1, 1, NSNumber(value: w)]
    let sm = try MLMultiArray(shape: shape, dataType: .float32)
    let zm = try MLMultiArray(shape: shape, dataType: .float32)
    let sp = sm.dataPointer.assumingMemoryBound(to: Float.self)
    let zp = zm.dataPointer.assumingMemoryBound(to: Float.self)
    let span = g.core * rate
    for r in 0..<g.rows {
        let g0 = g.rowStart(r, rate: rate)
        for j in 0..<w {
            let gi = g0 + j
            let inside = gi >= 0 && gi < valid
            let owner = min(max(gi, 0) / span, g.rows - 1)
            zp[r * w + j] = inside ? 1 : 0
            sp[r * w + j] = inside && owner == r ? 1 : 0
        }
    }
    return (sm, zm)
}

/// Builds the folded package's inputs.
///
/// - Parameters:
///   - xPre: decoder-pre output, (1, 512, T) with at least `validXPreFrames` frames.
///   - har: harmonic source features, channel-major (22, harFrames).
///   - refS: 256-dim voice embedding.
public func foldedGeneratorInputs(
    xPre: MLMultiArray, validXPreFrames: Int, har: [Float], harFrames: Int, refS: [Float], geometry g: FoldGeometry
) throws -> MLFeatureProvider {
    guard validXPreFrames <= g.capacity else {
        throw PipelineError.modelContractMismatch("folded generator holds \(g.capacity) x_pre frames, got \(validXPreFrames)")
    }
    let channels = xPre.shape[1].intValue
    let channelStride = xPre.strides[1].intValue
    // Only the valid frames: the rows' zero padding stands in for bucket padding.
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
    let ref = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    copyInto(array: ref, from: refS)
    let m1 = try foldMasks(rate: FoldGeometry.stage1Rate, extra: 0, valid: validXPreFrames * FoldGeometry.stage1Rate, geometry: g)
    let m2 = try foldMasks(rate: FoldGeometry.stage2Rate, extra: FoldGeometry.stage2Extra, valid: harValid, geometry: g)
    return try MLDictionaryFeatureProvider(dictionary: [
        "x_pre": MLFeatureValue(multiArray: try foldRows(xValid, channels: channels, time: validXPreFrames, rate: 1, extra: 0, geometry: g)),
        "har": MLFeatureValue(multiArray: try foldRows(harTrim, channels: HarmonicConstants.harChannels, time: harValid, rate: FoldGeometry.stage2Rate, extra: FoldGeometry.stage2Extra, geometry: g)),
        "ref_s": MLFeatureValue(multiArray: ref),
        "sm1": MLFeatureValue(multiArray: m1.stats), "zm1": MLFeatureValue(multiArray: m1.zero),
        "sm2": MLFeatureValue(multiArray: m2.stats), "zm2": MLFeatureValue(multiArray: m2.zero),
    ])
}

/// Precomputed inverse-STFT basis of CustomSTFT(n_fft 20, hop 5, Hann) with
/// the one-sided scale and overlap-add normalisation folded in.
enum InverseSTFTBasis {
    static let nfft = HarmonicConstants.stftNfft
    static let hop = HarmonicConstants.stftHop
    static let bins = HarmonicConstants.stftFreqBins

    /// (nfft, 2 * bins) row-major: columns [cos-basis | -sin-basis], so a frame
    /// is basis x [real; imag].
    static let matrix: [Float] = {
        var window = [Double](repeating: 0, count: nfft)
        for n in 0..<nfft { window[n] = 0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(nfft)) }
        var powerSum = [Double](repeating: 0, count: hop)
        for offset in stride(from: 0, to: nfft, by: hop) {
            for i in 0..<hop { powerSum[i] += window[offset + i] * window[offset + i] }
        }
        let ola = powerSum.reduce(0, +) / Double(hop)
        var m = [Float](repeating: 0, count: nfft * 2 * bins)
        for n in 0..<nfft {
            for k in 0..<bins {
                let scale = (k == 0 || k == bins - 1) ? 1.0 / Double(nfft) : 2.0 / Double(nfft)
                let w = window[n] * scale / ola
                let angle = 2 * Double.pi * Double(k * n) / Double(nfft)
                m[n * 2 * bins + k] = Float(cos(angle) * w)
                m[n * 2 * bins + bins + k] = Float(-sin(angle) * w)
            }
        }
        return m
    }()
}

/// Unfolds conv_post logits (rows, 22, 1, w2) and runs the generator tail:
/// magnitude = exp(logits[0..<11]), phase = sin(logits[11..<22]), iSTFT.
///
/// Runs in `chunks` frame ranges on parallel threads (each does its own
/// overlap-add; the 15-sample seams are summed afterwards): single-threaded it
/// took 16-19 ms of a 30 s clip on an A14, on the critical path.
///
/// - Returns: `validXPreFrames * 300` audio samples at 24 kHz.
public func foldedGeneratorAudio(logits: MLMultiArray, validXPreFrames: Int, geometry g: FoldGeometry,
                                 chunks: Int = 4) -> [Float] {
    let bins = InverseSTFTBasis.bins
    let frames = validXPreFrames * FoldGeometry.stage2Rate + 1
    let rowStride = logits.strides[0].intValue
    let channelStride = logits.strides[1].intValue
    let span = g.core * FoldGeometry.stage2Rate
    let nfft = InverseSTFTBasis.nfft
    let hop = InverseSTFTBasis.hop
    let full = (frames - 1) * hop + nfft
    let parts = max(1, min(chunks, frames / 64))
    let per = (frames + parts - 1) / parts
    var locals = [[Float]](repeating: [], count: parts)
    let lock = NSLock()

    DispatchQueue.concurrentPerform(iterations: parts) { part in
        let f0 = part * per, f1 = min(frames, f0 + per), nf = f1 - f0
        guard nf > 0 else { return }
        // Unfold frames [f0, f1) into (22, nf), each global frame from the row whose core owns it.
        var spec = [Float](repeating: 0, count: 2 * bins * nf)
        spec.withUnsafeMutableBufferPointer { dst in
            for c in 0..<(2 * bins) {
                for r in 0..<g.rows {
                    let lo = max(f0, r * span)
                    let hi = min(f1, r == g.rows - 1 ? frames : min((r + 1) * span, frames))
                    guard hi > lo else { continue }
                    let j0 = lo - g.rowStart(r, rate: FoldGeometry.stage2Rate)
                    readRun(logits, offset: r * rowStride + c * channelStride + j0, count: hi - lo,
                            into: dst.baseAddress! + c * nf + (lo - f0))
                }
            }
        }
        // [real; imag] = [mag * cos(phase); mag * sin(phase)].
        var n = Int32(bins * nf)
        var mag = [Float](repeating: 0, count: bins * nf)
        var phase = [Float](repeating: 0, count: bins * nf)
        spec.withUnsafeBufferPointer { s in
            vvexpf(&mag, s.baseAddress!, &n)
            vvsinf(&phase, s.baseAddress! + bins * nf, &n)
        }
        var cosP = [Float](repeating: 0, count: bins * nf)
        var sinP = [Float](repeating: 0, count: bins * nf)
        vvsincosf(&sinP, &cosP, phase, &n)
        var coef = [Float](repeating: 0, count: 2 * bins * nf)
        vDSP_vmul(mag, 1, cosP, 1, &coef, 1, vDSP_Length(bins * nf))
        coef.withUnsafeMutableBufferPointer { c in
            vDSP_vmul(mag, 1, sinP, 1, c.baseAddress! + bins * nf, 1, vDSP_Length(bins * nf))
        }
        // Frame samples (nfft, nf) = basis (nfft, 22) x coef (22, nf), then overlap-add at hop 5.
        var frameSamples = [Float](repeating: 0, count: nfft * nf)
        vDSP_mmul(InverseSTFTBasis.matrix, 1, coef, 1, &frameSamples, 1,
                  vDSP_Length(nfft), vDSP_Length(nf), vDSP_Length(2 * bins))
        var ola = [Float](repeating: 0, count: (nf - 1) * hop + nfft)
        for i in 0..<nfft {
            frameSamples.withUnsafeBufferPointer { f in
                ola.withUnsafeMutableBufferPointer { o in
                    vDSP_vadd(o.baseAddress! + i, hop, f.baseAddress! + i * nf, 1, o.baseAddress! + i, hop, vDSP_Length(nf))
                }
            }
        }
        lock.lock(); locals[part] = ola; lock.unlock()
    }

    var out = [Float](repeating: 0, count: full)
    out.withUnsafeMutableBufferPointer { o in
        for part in 0..<parts where !locals[part].isEmpty {
            let start = part * per * hop
            locals[part].withUnsafeBufferPointer { l in
                vDSP_vadd(o.baseAddress! + start, 1, l.baseAddress!, 1, o.baseAddress! + start, 1, vDSP_Length(l.count))
            }
        }
    }
    // Drop the n_fft/2 centre padding at each end.
    let pad = nfft / 2
    return Array(out[pad..<(full - pad)])
}
