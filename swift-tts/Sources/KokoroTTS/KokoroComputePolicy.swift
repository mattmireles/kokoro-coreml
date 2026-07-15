import CoreML
import Foundation

/// Compute-unit policy for Kokoro Core ML stages.
public struct KokoroComputePolicy: Equatable, Sendable {
    /// Duration model compute units.
    public let duration: MLComputeUnits

    /// F0Ntrain model compute units.
    public let f0ntrain: MLComputeUnits

    /// Decoder-pre model compute units.
    public let decoderPre: MLComputeUnits

    /// Generator/HAR-post model compute units.
    public let generator: MLComputeUnits

    /// Gist iPhone default policy.
    public static let gistDefault = KokoroComputePolicy(
        duration: .cpuAndGPU,
        f0ntrain: .cpuAndGPU,
        decoderPre: .cpuAndNeuralEngine,
        generator: .cpuAndGPU
    )

    /// Reliable CPU-only fallback policy.
    public static let cpuOnly = KokoroComputePolicy(
        duration: .cpuOnly,
        f0ntrain: .cpuOnly,
        decoderPre: .cpuOnly,
        generator: .cpuOnly
    )

    /// Creates a compute policy.
    ///
    /// - Parameters:
    ///   - duration: Duration model compute units.
    ///   - f0ntrain: F0Ntrain model compute units.
    ///   - decoderPre: Decoder-pre model compute units.
    ///   - generator: Generator/HAR-post model compute units.
    public init(
        duration: MLComputeUnits,
        f0ntrain: MLComputeUnits,
        decoderPre: MLComputeUnits,
        generator: MLComputeUnits
    ) {
        self.duration = duration
        self.f0ntrain = f0ntrain
        self.decoderPre = decoderPre
        self.generator = generator
    }
}
