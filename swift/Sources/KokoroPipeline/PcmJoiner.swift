import Foundation

public enum PcmJoiner {
    public static let defaultCrossfadeMs = 5.0

    public static func join(
        segments: [[Float]],
        sampleRate: Int = 24_000,
        crossfadeMs: Double = defaultCrossfadeMs
    ) -> [Float] {
        let chunks = segments.filter { !$0.isEmpty }
        guard let first = chunks.first else {
            return []
        }
        let targetFadeSamples = max(0, Int((Double(max(1, sampleRate)) * max(0, crossfadeMs) / 1000.0).rounded(.down)))
        guard chunks.count > 1, targetFadeSamples > 0 else {
            return chunks.flatMap { $0 }
        }

        var output = first
        for chunk in chunks.dropFirst() {
            let fadeSamples = min(targetFadeSamples, output.count, chunk.count)
            if fadeSamples > 0 {
                let fadeStart = output.count - fadeSamples
                for sampleIndex in 0..<fadeSamples {
                    let t = Double(sampleIndex + 1) / Double(fadeSamples + 1)
                    let previousGain = Float(cos(t * Double.pi * 0.5))
                    let nextGain = Float(sin(t * Double.pi * 0.5))
                    output[fadeStart + sampleIndex] = output[fadeStart + sampleIndex] * previousGain + chunk[sampleIndex] * nextGain
                }
            }
            output.append(contentsOf: chunk.dropFirst(fadeSamples))
        }
        return output
    }
}
