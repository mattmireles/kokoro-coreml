import KokoroPipeline

/// What the last synthesis ran and how long each stage took, one entry per
/// chunk. Read it from ``KokoroTTS/lastReport`` after a synthesis call, e.g. to
/// confirm a bundle runs the split/folded/hybrid fast path on a device.
public struct KokoroSynthesisReport: Sendable {
    /// One synthesized chunk.
    public struct Chunk: Sendable {
        /// Acoustic bucket the chunk ran in (seconds).
        public let bucketSeconds: Int
        /// Audio produced, in seconds.
        public let audioSeconds: Double
        /// Duration program: `split` (ANE front + CPU back) or `whole`.
        public let durationPath: String
        /// F0Ntrain program: `split` (CPU LSTM + ANE branches) or `whole`.
        public let f0Path: String
        /// Generator: `hybrid` (ANE convs + Metal), `folded` (ANE), `flexible` or `bucket` (GPU).
        public let generatorPath: String
        /// Stage wall times in milliseconds, in pipeline order.
        public let stageMilliseconds: [(name: String, ms: Double)]
        /// Wall time of the whole chunk in milliseconds.
        public let totalMilliseconds: Double

        init(_ result: SynthesisResult) {
            let t = result.timings
            bucketSeconds = result.bucketSeconds
            audioSeconds = result.audioDurationSeconds
            durationPath = result.durationPath
            f0Path = result.f0Path
            generatorPath = result.generatorPath
            stageMilliseconds = [
                ("duration", t.durationCoreML), ("align", t.alignment + t.matrixOps), ("f0n", t.f0ntrainCoreML),
                ("pad", t.padding), ("decoder_pre", t.decoderPre), ("hnsf", t.hnsfSwift), ("generator", t.generatorCoreML),
            ].map { ($0.0, $0.1 * 1000) }
            totalMilliseconds = result.wallTimeSeconds * 1000
        }
    }

    /// Chunks in order.
    public let chunks: [Chunk]
}
