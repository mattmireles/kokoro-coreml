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
