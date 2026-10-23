import Accelerate
import Foundation

/// Host-side input preparation for the hybrid generator, on its critical path:
/// nothing runs on the ANE until these are written. The first version used
/// bounds-checked Swift array loops over ~7 M elements: on an A14 the first ANE
/// call started 16-19 ms into a warm run and 48-52 ms into one after 300 ms
/// idle (README/Notes/ane-folded-generator.md, "Cold generator after a break").
enum HybridInputs {
    /// Writes columns [crop, crop + width) of each `srcWidth`-wide Float row as
    /// fp16 rows of `width` values, rows in parallel.
    static func writeCropped(_ src: UnsafePointer<Float>, rows: Int, srcWidth: Int, crop: Int, width: Int,
                             into dst: UnsafeMutableRawPointer) {
        let chunk = max(1, rows / 8)
        DispatchQueue.concurrentPerform(iterations: (rows + chunk - 1) / chunk) { part in
            for i in (part * chunk)..<min(rows, (part + 1) * chunk) {
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src + i * srcWidth + crop), height: 1,
                                      width: vImagePixelCount(width), rowBytes: width * 4)
                var d = vImage_Buffer(data: dst + i * width * 2, height: 1, width: vImagePixelCount(width), rowBytes: width * 2)
                vImageConvert_PlanarFtoPlanar16F(&s, &d, vImage_Flags(kvImageNoFlags))
            }
        }
    }

    /// Space-to-depth for the stride-6 noise conv, as fp16:
    /// out[n, c*6+e, q] = har[n, c, q*6 + e - pad] (0 outside [0, w2)).
    /// `scratch` holds `rows * channels * 6 * q6` Floats and is overwritten.
    static func writeSpaceToDepth6(_ har: UnsafePointer<Float>, rows: Int, channels: Int, w2: Int, q6: Int, pad: Int,
                                   scratch: UnsafeMutablePointer<Float>, into dst: UnsafeMutableRawPointer) {
        let planes = rows * channels
        DispatchQueue.concurrentPerform(iterations: planes) { plane in
            let src = har + plane * w2
            for e in 0..<6 {
                let out = scratch + (plane * 6 + e) * q6
                var t = e - pad
                for q in 0..<q6 {
                    out[q] = t >= 0 && t < w2 ? src[t] : 0
                    t += 6
                }
            }
        }
        let count = planes * 6 * q6
        var s = vImage_Buffer(data: scratch, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        var d = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        vImageConvert_PlanarFtoPlanar16F(&s, &d, vImage_Flags(kvImageNoFlags))
    }
}
