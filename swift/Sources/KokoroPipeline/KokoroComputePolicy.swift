import CoreML

/**
 * Per-stage Core ML compute units for the four Kokoro graphs.
 *
 * The placement is measured, not preferential: the generator runs 1535.5 ms on
 * the ANE against 27.2 ms on the GPU, and duration and f0ntrain fail numerical
 * parity on the ANE. Only the duration stage's *host* differs between macOS and
 * iOS, so this type exists to express that one difference rather than to make
 * placement tunable.
 *
 * Called by:
 * - `KokoroPipeline.init(modelsDirectory:buckets:linearWeights:linearBias:policy:)`
 */
public struct KokoroComputePolicy: Equatable, Sendable {
    public let duration: MLComputeUnits
    public let f0ntrain: MLComputeUnits
    public let decoderPre: MLComputeUnits
    public let generator: MLComputeUnits

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

    /// The measured macOS placement, and the pipeline's default so that
    /// existing callers keep the exact configuration they shipped with.
    public static let macOS = KokoroComputePolicy(
        duration: .cpuAndGPU,
        f0ntrain: .cpuAndGPU,
        decoderPre: .cpuAndNeuralEngine,
        generator: .cpuAndGPU
    )

    /// The macOS placement with duration moved to the CPU.
    ///
    /// The padded duration graph can sit in MPSGraph specialization for minutes
    /// on iOS builds. Duration is a small share of end-to-end latency, so the
    /// CPU host costs little and removes the stall; the acoustic stages keep
    /// their measured GPU and ANE placement. This mirrors
    /// `KokoroComputePolicy.gistDefault` in `../kokoro-coreml/swift-tts`.
    public static let iOS = KokoroComputePolicy(
        duration: .cpuOnly,
        f0ntrain: .cpuAndGPU,
        decoderPre: .cpuAndNeuralEngine,
        generator: .cpuAndGPU
    )
}
