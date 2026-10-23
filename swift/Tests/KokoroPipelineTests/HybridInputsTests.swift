import Accelerate
import XCTest
@testable import KokoroPipeline

/// HybridInputs must reproduce the original scalar input preparation exactly
/// (the vectorized version replaced bounds-checked loops on the critical path).
final class HybridInputsTests: XCTestCase {
    private func halfs(_ p: UnsafeMutableRawPointer, _ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var s = vImage_Buffer(data: p, height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
        out.withUnsafeMutableBytes { d in
            var dd = vImage_Buffer(data: d.baseAddress!, height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
            vImageConvert_Planar16FtoPlanarF(&s, &dd, vImage_Flags(kvImageNoFlags))
        }
        return out
    }

    func testWriteCroppedMatchesScalarCrop() {
        let rows = 5, w2 = 40, crop = 4, width = 32
        let src = (0..<(rows * w2)).map { Float($0 % 97) * 0.25 - 3 }
        let dst = UnsafeMutableRawPointer.allocate(byteCount: rows * width * 2, alignment: 16); defer { dst.deallocate() }
        src.withUnsafeBufferPointer { HybridInputs.writeCropped($0.baseAddress!, rows: rows, srcWidth: w2, crop: crop, width: width, into: dst) }
        var want: [Float] = []
        for i in 0..<rows { for t in 0..<width { want.append(src[i * w2 + crop + t]) } }
        XCTAssertEqual(halfs(dst, rows * width), want)
    }

    func testSpaceToDepthMatchesScalarLoop() {
        let rows = 2, channels = 3, w2 = 50, q6 = 10, pad = 3
        let har = (0..<(rows * channels * w2)).map { Float($0 % 31) * 0.5 - 7 }
        let n = rows * channels * 6 * q6
        let dst = UnsafeMutableRawPointer.allocate(byteCount: n * 2, alignment: 16); defer { dst.deallocate() }
        var scratch = [Float](repeating: 99, count: n)   // stale values must be overwritten
        har.withUnsafeBufferPointer { h in
            scratch.withUnsafeMutableBufferPointer { sc in
                HybridInputs.writeSpaceToDepth6(h.baseAddress!, rows: rows, channels: channels, w2: w2, q6: q6, pad: pad,
                                                scratch: sc.baseAddress!, into: dst)
            }
        }
        var want = [Float](repeating: 0, count: n)   // the original loop from HybridGenerator.run
        for r in 0..<rows { for c in 0..<channels {
            let src = (r * channels + c) * w2
            for e in 0..<6 {
                let d = ((r * channels + c) * 6 + e) * q6
                var q = 0, t = e - pad
                while q < q6 { if t >= 0 && t < w2 { want[d + q] = har[src + t] }; q += 1; t += 6 }
            }
        } }
        XCTAssertEqual(halfs(dst, n), want)
    }
}
