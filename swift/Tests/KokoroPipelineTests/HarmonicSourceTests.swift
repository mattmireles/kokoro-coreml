import Accelerate
import XCTest
@testable import KokoroPipeline

final class HarmonicSourceTests: XCTestCase {

    // MARK: - F0 Upsample

    func testF0UpsampleLength() {
        let f0: [Float] = [100.0, 200.0, 300.0]
        let result = f0Upsample(f0)
        XCTAssertEqual(result.count, 3 * HarmonicConstants.upsampleScale)
    }

    func testF0UpsampleNearestNeighbor() {
        let f0: [Float] = [100.0, 200.0]
        let result = f0Upsample(f0)
        // First 300 samples should all be 100.0
        for i in 0..<HarmonicConstants.upsampleScale {
            XCTAssertEqual(result[i], 100.0, "Sample \(i) should be 100.0")
        }
        // Next 300 should be 200.0
        for i in HarmonicConstants.upsampleScale..<(2 * HarmonicConstants.upsampleScale) {
            XCTAssertEqual(result[i], 200.0, "Sample \(i) should be 200.0")
        }
    }

    // MARK: - STFT

    func testSTFTOutputShape() {
        // 600 samples -> with center padding (20/2=10 each side) -> 620
        // nFrames = (620 - 20) / 5 + 1 = 121
        let signal = [Float](repeating: 0.5, count: 600)
        let (mag, phase) = stftTransform(signal)
        let expectedFrames = 121
        XCTAssertEqual(mag.count, HarmonicConstants.stftFreqBins * expectedFrames)
        XCTAssertEqual(phase.count, HarmonicConstants.stftFreqBins * expectedFrames)
    }

    func testSTFTDCBin() {
        // A constant signal should have energy only in the DC bin (k=0)
        let signal = [Float](repeating: 1.0, count: 100)
        let (mag, _) = stftTransform(signal)
        let nFrames = mag.count / HarmonicConstants.stftFreqBins

        // DC bin (k=0) should have nonzero magnitude
        let dcMag = mag[0]  // First frame, DC bin
        XCTAssertGreaterThan(dcMag, 0.1, "DC bin should have energy for constant signal")

        // Higher frequency bins should be near zero for a constant signal
        for k in 2..<HarmonicConstants.stftFreqBins {
            let binMag = mag[k * nFrames]  // First frame, bin k
            XCTAssertLessThan(binMag, 0.01, "Bin \(k) should be near zero for constant signal")
        }
    }

    // MARK: - buildHar

    func testSTFTEdgeBinsReportPositivePiForNegativeRealParts() {
        // A negative constant has a negative DC real part; a negative Nyquist
        // alternation (-cos(pi n)) has a negative Nyquist real part. Both have a
        // mathematically zero imaginary part, and torch.stft reports +pi there.
        let n = 400
        let negativeDC = [Float](repeating: -0.5, count: n)
        let negativeNyquist = (0..<n).map { -0.5 * Float(($0 % 2 == 0) ? 1 : -1) }
        let (_, dcPhase) = stftTransform(negativeDC)
        let (_, nyPhase) = stftTransform(negativeNyquist)
        let frames = dcPhase.count / HarmonicConstants.stftFreqBins
        let nyquist = HarmonicConstants.stftFreqBins - 1
        for t in 2..<(frames - 2) {
            XCTAssertEqual(dcPhase[t], Float.pi, accuracy: 1e-5, "DC phase at frame \(t)")
            // Hop 5 is odd, so the alternation flips sign every frame: the
            // Nyquist real part is negative on even frames (+pi) and positive
            // on odd frames (0). Never -pi.
            let expected: Float = (t % 2 == 0) ? Float.pi : 0
            XCTAssertEqual(nyPhase[nyquist * frames + t], expected, accuracy: 1e-5, "Nyquist phase at frame \(t)")
        }
    }

    func testBuildHarOutputShape() {
        // 80 F0 frames -> upsample 300x -> 24000 samples
        // STFT: (24000 + 20 - 20) / 5 + 1 = 4801 frames (with center padding)
        let f0 = [Float](repeating: 200.0, count: 80)
        let weights: [Float] = [-0.08, -0.19, -0.18, -0.18, -0.10, 0.08, 0.09, -0.39, -0.55]
        let bias: Float = -0.03

        let (har, nFrames) = buildHar(
            f0Padded: f0,
            linearWeights: weights,
            linearBias: bias,
            seed: 42
        )

        XCTAssertEqual(nFrames, 4801, "80 F0 frames -> 24000 samples -> 4801 STFT frames")
        XCTAssertEqual(har.count, HarmonicConstants.harChannels * nFrames)
    }

    func testBuildHarComponentsMatchBuildHar() {
        let f0 = [Float](repeating: 200.0, count: 4)
        let weights: [Float] = [-0.08, -0.19, -0.18, -0.18, -0.10, 0.08, 0.09, -0.39, -0.55]
        let bias: Float = -0.03

        let direct = buildHar(
            f0Padded: f0,
            linearWeights: weights,
            linearBias: bias,
            seed: 42
        )
        let components = buildHarComponents(
            f0Padded: f0,
            linearWeights: weights,
            linearBias: bias,
            seed: 42
        )

        XCTAssertEqual(components.harSource.count, f0.count * HarmonicConstants.upsampleScale)
        XCTAssertEqual(components.nFrames, direct.nFrames)
        XCTAssertEqual(components.magnitude.count, HarmonicConstants.stftFreqBins * components.nFrames)
        XCTAssertEqual(components.phase.count, HarmonicConstants.stftFreqBins * components.nFrames)
        XCTAssertEqual(components.har, direct.har)
    }

    func testGaussianNoiseMatchesScalarBoxMullerReference() {
        let count = 1025
        let seed: UInt64 = 42
        var candidate = [Float](repeating: 0, count: count)
        generateGaussianNoise(into: &candidate, count: count, seed: seed)

        let reference = scalarGaussianNoiseReference(count: count, seed: seed)
        XCTAssertEqual(candidate.count, reference.count)
        for index in 0..<count {
            XCTAssertEqual(candidate[index], reference[index], accuracy: 2e-6, "noise sample \(index)")
        }
    }

    /// The noise generator streams 32,768 Box-Muller pairs per chunk; a count
    /// above one chunk must still match the scalar draw order.
    func testGaussianNoiseMatchesScalarReferenceAcrossChunkBoundary() {
        let count = 2 * 32_768 + 3
        let seed: UInt64 = 7
        var candidate = [Float](repeating: 0, count: count)
        generateGaussianNoise(into: &candidate, count: count, seed: seed)
        let reference = scalarGaussianNoiseReference(count: count, seed: seed)
        for index in [0, 65_535, 65_536, 65_537, count - 1] {
            XCTAssertEqual(candidate[index], reference[index], accuracy: 2e-6, "noise sample \(index)")
        }
        let maxError = zip(candidate, reference).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxError, 2e-6)
    }

    /// `sineGenFromF0Frames` runs its harmonics concurrently through 65,536-sample
    /// chunks and must stay bit-identical to the sequential full-length form
    /// (PR #11) when the source crosses a chunk boundary.
    func testSineGenMatchesSequentialReferenceAcrossChunkBoundary() {
        let frameCount = 223  // 66,900 samples: one full chunk plus a partial one
        let f0 = (0..<frameCount).map { t -> Float in t % 37 < 5 ? 0 : 90 + Float(t % 50) * 4 }
        let weights: [Float] = (0..<HarmonicConstants.harmonicDim).map { 0.3 - Float($0) * 0.05 }
        let bias: Float = 0.01
        let seed: UInt64 = 11

        let candidate = sineGenFromF0Frames(f0Frames: f0, linearWeights: weights, linearBias: bias, seed: seed)
        let reference = sequentialSineGenReference(f0Frames: f0, linearWeights: weights, linearBias: bias, seed: seed)
        XCTAssertEqual(candidate.count, reference.count)
        XCTAssertTrue(candidate == reference, "concurrent sine generation diverged from the sequential form")
    }

    /// Regression test for the SplitMix64 seed scrambler in `SeededRNG.init`
    /// (HarmonicSource.swift, fixed 2026-07-14).
    ///
    /// Before the fix the xorshift64 state was seeded directly, and 0 is
    /// xorshift64's absorbing state: every draw returned 0 forever. Through
    /// Box-Muller that collapsed the "Gaussian" noise into a deterministic
    /// two-value impulse train (~5.65, 0, 5.65, 0, ...) — pure DC + Nyquist,
    /// no broadband noise — which is what the CS1 evaluation clips (rendered
    /// with --seed 0) suffered from; see
    /// README/Notes/cs1-audio-quality-evaluation-2026-07-14.md.
    ///
    /// `testGaussianNoiseMatchesScalarBoxMullerReference` cannot catch a
    /// removed scrambler because both sides construct the same `SeededRNG`,
    /// so a degenerate stream matches its own degenerate reference. This
    /// test pins the OUTPUT distribution for the exact absorbing seed
    /// instead: if the SplitMix64 finalizer is deleted, the pre-fix output
    /// has mean ~2.82, variance ~7.97, and exactly 2 distinct values, so the
    /// mean, variance, and distinct-count assertions below all fail (the
    /// max-run check does not — alternating values have run length 1 — it
    /// guards the separate constant-output degeneracy).
    func testSeedZeroProducesBroadbandGaussianNoise() {
        // Large enough that the statistical bands below are many sigma wide,
        // small enough to stay instant. For a standard Gaussian at n = 4096
        // the sample mean has sigma ~ 1/sqrt(n) = 0.016 and the sample
        // variance has sigma ~ sqrt(2/n) = 0.022, so 0.15 / [0.7, 1.3] are
        // ~10-sigma bands: they never flake, and fail hard on degeneracy.
        let count = 4096
        let absorbingSeed: UInt64 = 0  // pre-fix xorshift64 absorbing state
        var noise = [Float](repeating: 0, count: count)
        generateGaussianNoise(into: &noise, count: count, seed: absorbingSeed)

        var sum = 0.0
        for x in noise { sum += Double(x) }
        let mean = sum / Double(count)
        var sumSq = 0.0
        for x in noise {
            let d = Double(x) - mean
            sumSq += d * d
        }
        let variance = sumSq / Double(count)

        XCTAssertEqual(mean, 0.0, accuracy: 0.15,
                       "seed-0 noise mean should be ~0 (pre-fix degenerate stream had mean ~2.82)")
        XCTAssertGreaterThan(variance, 0.7,
                             "seed-0 noise variance collapsed — RNG produced a near-constant stream")
        XCTAssertLessThan(variance, 1.3,
                          "seed-0 noise variance should be ~1.0 (pre-fix impulse train had variance ~7.97)")

        // Broadband Gaussian floats are essentially all distinct; the
        // pre-fix stream took exactly two values (the DC + Nyquist train).
        let distinctValues = Set(noise).count
        XCTAssertGreaterThan(distinctValues, count / 2,
                             "seed-0 noise repeats values — degenerate periodic pattern, not broadband noise")

        // No long run of identical samples (a constant/DC segment).
        var maxRun = 1
        var run = 1
        for i in 1..<count {
            run = noise[i] == noise[i - 1] ? run + 1 : 1
            maxRun = max(maxRun, run)
        }
        XCTAssertLessThanOrEqual(maxRun, 3,
                                 "seed-0 noise has a run of \(maxRun) identical samples — DC segment, not noise")
    }

    // MARK: - Linear interpolation

    func testLinearInterpolateIdentity() {
        let input: [Double] = [1.0, 2.0, 3.0, 4.0]
        let result = linearInterpolateDown(input, targetLen: 4)
        for i in 0..<4 {
            XCTAssertEqual(result[i], input[i], accuracy: 1e-10)
        }
    }

    func testLinearInterpolateHalf() {
        let input: [Double] = [0.0, 1.0, 2.0, 3.0]
        let result = linearInterpolateDown(input, targetLen: 2)
        XCTAssertEqual(result.count, 2)
        // With align_corners=False: src_idx = (dst_idx + 0.5) * 4/2 - 0.5
        // dst=0: src_idx = 0.5 * 2 - 0.5 = 0.5 -> lerp(0, 1, 0.5) = 0.5
        // dst=1: src_idx = 1.5 * 2 - 0.5 = 2.5 -> lerp(2, 3, 0.5) = 2.5
        XCTAssertEqual(result[0], 0.5, accuracy: 1e-10)
        XCTAssertEqual(result[1], 2.5, accuracy: 1e-10)
    }

    private func scalarGaussianNoiseReference(count: Int, seed: UInt64) -> [Float] {
        var rng = SeededRNG(seed: seed)
        var output = [Float](repeating: 0, count: count)
        var i = 0
        while i < count - 1 {
            let u1 = max(Float.ulpOfOne, Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF))
            let u2 = Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF)
            let r = sqrt(-2.0 * log(u1))
            let theta = 2.0 * Float.pi * u2
            output[i] = r * cos(theta)
            output[i + 1] = r * sin(theta)
            i += 2
        }
        if i < count {
            let u1 = max(Float.ulpOfOne, Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF))
            let u2 = Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF)
            output[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
        }
        return output
    }
}

/// Sequential, full-length form of `sineGenFromF0Frames` (pre-PR #11): same
/// per-value arithmetic, no chunking and no concurrency.
private func sequentialSineGenReference(f0Frames: [Float], linearWeights: [Float], linearBias: Float, seed: UInt64) -> [Float] {
    let frameCount = f0Frames.count
    let scale = HarmonicConstants.upsampleScale
    let L = frameCount * scale
    let dim = HarmonicConstants.harmonicDim
    let sr = HarmonicConstants.sampleRate
    var noise = [Float](repeating: 0, count: dim * L)
    generateGaussianNoise(into: &noise, count: dim * L, seed: seed)
    var sines = [Float](repeating: 0, count: dim * L)
    for h in 0..<dim {
        let invSr = Double(h + 1) / sr
        var cum = [Double](repeating: 0, count: frameCount)
        var acc = 0.0
        for t in 0..<frameCount {
            let r = (Double(f0Frames[t]) * invSr).truncatingRemainder(dividingBy: 1.0)
            acc = t == 0 ? (r < 0 ? r + 1.0 : r) : acc + (r < 0 ? r + 1.0 : r)
            cum[t] = acc
        }
        var scaled = [Double](repeating: 0, count: frameCount)
        vDSP_vsmulD(cum, 1, [2.0 * Double.pi * Double(scale)], &scaled, 1, vDSP_Length(frameCount))
        var up = [Double](repeating: 0, count: L)
        linearInterpolateInto(from: scaled, count: frameCount, into: &up, targetLen: L)
        var sinD = [Double](repeating: 0, count: L)
        var n32 = Int32(L)
        vvsin(&sinD, up, &n32)
        var sinF = [Float](repeating: 0, count: L)
        vDSP_vdpsp(sinD, 1, &sinF, 1, vDSP_Length(L))
        for i in 0..<L {
            let f0 = f0Frames[i / scale]
            let uv: Float = f0 > HarmonicConstants.voicedThreshold ? 1 : 0
            let amp = uv * HarmonicConstants.noiseStd + (1 - uv) * (HarmonicConstants.sineAmp / 3)
            sines[h * L + i] = (sinF[i] * HarmonicConstants.sineAmp) * uv + noise[h * L + i] * amp
        }
    }
    var merged = [Float](repeating: 0, count: L)
    vDSP_mmul(linearWeights, 1, sines, 1, &merged, 1, 1, vDSP_Length(L), vDSP_Length(dim))
    var bias = linearBias
    vDSP_vsadd(merged, 1, &bias, &merged, 1, vDSP_Length(L))
    var tanhCount = Int32(L)
    vvtanhf(&merged, merged, &tanhCount)
    return merged
}
