/// Kokoro TTS pipeline orchestrator: chains CoreML models + Swift DSP.
///
/// Replaces the Python ``extract_vocoder_inputs()`` + ``build_decoder_har_post_inputs_np()``
/// + ``decoder_har_post_bucket_impl()`` chain with native Swift + CoreML.
///
/// ## Pipeline stages
///
/// 1. Duration CoreML → pred_dur, d, t_en, s, ref_s
/// 2. Alignment (Swift) → one-hot matrix from pred_dur
/// 3. Matrix ops (Accelerate) → en = d @ alignment, asr = t_en @ alignment
/// 4. F0Ntrain CoreML → F0_pred, N_pred
/// 5. Pad to bucket geometry (Swift)
/// 6. DecoderPre (bridge / pre-computed / CoreML Phase 4)
/// 7. hn-nsf (Swift/Accelerate, Double precision phase) → har
/// 8. GeneratorFromHar CoreML → waveform
/// 9. Trim to natural utterance length
///
/// ## Stage timing
///
/// Every stage is timed with ``ContinuousClock`` and reported in ``SynthesisResult``.

import CoreML
import Foundation
import Accelerate

// MARK: - Configuration

/// Pipeline configuration matching Kokoro model constants.
public enum PipelineConstants {
    /// Audio sample rate (Hz).
    public static let sampleRate: Int = 24000
    /// F0 frame rate (Hz). Converts F0 frame count to seconds.
    public static let f0FrameRate: Double = 80.0
    /// Samples per duration-model frame (40 fps). Matches Python ``AudioConstants.HOP_LENGTH``.
    public static let samplesPerDurationFrame: Int = sampleRate * 2 / Int(f0FrameRate)
    /// Fade length (~5 ms @ 24 kHz) applied around punctuation-owned spans.
    public static let punctuationFadeSamples: Int = 120
    /// Legacy duration model fixed token length.
    public static let durationTokenLength: Int = 128
    /// Voice embedding total dimension.
    public static let voiceEmbeddingDim: Int = 256
    /// Voice style dimension (ref_s[:, 128:]).
    public static let styleDim: Int = 128
    /// Voice baseline dimension (ref_s[:, :128]).
    public static let baselineDim: Int = 128
    /// Hidden dimension from duration encoder output.
    public static let hiddenDim: Int = 640
    /// Text encoder output dimension.
    public static let textEncoderDim: Int = 512

    /// F0Ntrain input T dimension for each bucket (seconds → T_frames).
    /// Derived from bucket geometry: full_f0_len = bucket_sec * 24000 / 300,
    /// then F0Ntrain's 2× upsample means T_frames = full_f0_len / 2.
    /// Single source of truth — used by both init() and synthesize().
    public static let tFramesForBucket: [Int: Int] = [
        3: 120, 7: 280, 10: 400, 15: 600, 30: 1200, 45: 1800,
    ]

    /// Default bucket seconds used by the bakeoff and runtime package set.
    public static let defaultBuckets: [Int] = [3, 7, 10, 15, 30]

    /// Duration model enumerated token sizes. Caller pads to nearest.
    public static let durationTokenSizes: [Int] = [32, 64, 128, 256, 320, 384, 512]

    /// Largest duration-token bucket shipped in the Core ML bundle.
    public static var maxDurationTokenLength: Int {
        durationTokenSizes.max() ?? 512
    }

    /// Caller-side chunk token cap from `packages/contracts` (`MAX_TTS_CHUNK_TOKENS`).
    public static let maxCallerChunkTokens = 450
}

// MARK: - Stage Timing

/// Timing breakdown for a single synthesis call.
public struct StageTimings {
    public var durationCoreML: Double = 0
    public var alignment: Double = 0
    public var matrixOps: Double = 0
    public var f0ntrainCoreML: Double = 0
    public var padding: Double = 0
    public var decoderPre: Double = 0
    public var hnsfSwift: Double = 0
    public var generatorCoreML: Double = 0
    public var trim: Double = 0

    /// Total pipeline wall time.
    public var total: Double {
        durationCoreML + alignment + matrixOps + f0ntrainCoreML +
        padding + decoderPre + hnsfSwift + generatorCoreML + trim
    }

    /// Pre-decoder overhead (everything before GeneratorFromHar predict).
    public var preDecoder: Double {
        total - generatorCoreML - trim
    }
}

/// Result of a synthesis call including audio and timing.
public struct SynthesisResult {
    /// Raw audio waveform at 24 kHz, trimmed to natural utterance length.
    public let audio: [Float]
    /// Per-stage timing breakdown.
    public let timings: StageTimings
    /// Selected bucket in seconds.
    public let bucketSeconds: Int
    /// Audio duration in seconds (from F0 frame count).
    public let audioDurationSeconds: Double
    /// Timed wall-clock span for the synthesis executor.
    public let wallTimeSeconds: Double
    /// Sum of positive duration frames used for frame-domain expansion.
    public let predictedDurationFrames: Int
    /// Number of valid duration tokens read from the model output.
    public let predictedDurationTokens: Int
    /// Cache key of the selected Duration model package.
    public let durationModelCacheKey: String
    /// Whether the selected Duration package permits padding.
    public let durationModelAllowsPadding: Bool
    /// Static token length of the selected Duration package.
    public let durationTokenLength: Int
    /// F0Ntrain static frame count for the selected bucket.
    public let tFrames: Int
    /// Full bucket F0 length after 300 Hz upsampling geometry.
    public let fullF0Length: Int
    /// DecoderPre ASR frame count for the selected bucket.
    public let decoderFrameCount: Int
    /// Static `x_pre` time dimension expected by the generator model.
    public let xPreExpectedTime: Int
    /// Static harmonic source time dimension expected by the generator model.
    public let harExpectedTime: Int
    /// Number of audio samples retained after trimming.
    public let trimSampleCount: Int
    /// Per-input-token duration frame counts (BOS + phoneme ids + EOS), aligned with the caller's ``inputIds`` prefix.
    public let tokenDurationFrames: [Int]
}

public struct DurationModelChoice {
    public let cacheKey: String
    public let tokenLength: Int
    public let packageURL: URL
    public let requiresAttentionMask: Bool
    public let allowsPadding: Bool
}

// MARK: - Pipeline

/// Main TTS pipeline orchestrator.
///
/// Loads CoreML models and provides ``synthesize()`` for text-to-audio.
///
/// ## Model loading
///
/// ``init(modelsDirectory:buckets:linearWeights:linearBias:)`` calls
/// ``MLModel.compileModel`` synchronously. On first run this can take
/// hundreds of milliseconds per model. For app integration, call init
/// on a background thread or use pre-compiled ``.mlmodelc`` bundles.
public class KokoroPipeline: KokoroModelProvider {
    private let durationModels: [String: MLModel] // keyed by DurationModelChoice.cacheKey
    private let durationChoices: [DurationModelChoice]
    private let f0ntrainModels: [Int: MLModel]  // keyed by T_frames
    private let decoderPreModels: [Int: MLModel] // keyed by bucket seconds
    private let generatorModels: [Int: MLModel]  // keyed by bucket seconds

    /// Learned weights from SourceModuleHnNSF.l_linear.
    private let linearWeights: [Float]
    private let linearBias: Float

    /// Available bucket durations in seconds.
    private let availableBuckets: [Int]

    /// Load all models from a directory.
    ///
    /// Expected files:
    /// - ``kokoro_duration_t{T}.mlpackage`` for each token size, or legacy ``kokoro_duration.mlpackage``
    /// - ``kokoro_f0ntrain_t{T}.mlpackage`` for each bucket's T_frames
    /// - ``kokoro_decoder_pre_{N}s.mlpackage`` for each bucket
    /// - ``kokoro_decoder_har_post_{N}s.mlpackage`` for each bucket
    ///
    /// Note: ``MLModel.compileModel`` is called synchronously. For app
    /// integration, call init on a background queue or use pre-compiled
    /// ``.mlmodelc`` bundles to avoid blocking the main thread.
    public init(
        modelsDirectory: URL,
        buckets: [Int] = PipelineConstants.defaultBuckets,
        linearWeights: [Float],
        linearBias: Float
    ) throws {
        // Duration models. Use padded mask-aware packages for production by
        // default; exact native packages are an opt-in benchmark path.
        let durationChoices = Self.discoverDurationChoices(modelsDirectory: modelsDirectory)
        var durModels: [String: MLModel] = [:]
        for choice in durationChoices {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            durModels[choice.cacheKey] = try MLModel(
                contentsOf: MLModel.compileModel(at: choice.packageURL),
                configuration: config
            )
        }
        guard !durModels.isEmpty else {
            throw PipelineError.modelNotLoaded("duration")
        }
        self.durationModels = durModels
        self.durationChoices = durationChoices

        // F0Ntrain models (one per bucket's T_frames)
        var f0Models: [Int: MLModel] = [:]
        for sec in buckets {
            if let t = PipelineConstants.tFramesForBucket[sec] {
                let url = modelsDirectory.appendingPathComponent("kokoro_f0ntrain_t\(t).mlpackage")
                if FileManager.default.fileExists(atPath: url.path) {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndGPU
                    f0Models[t] = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
                }
            }
        }
        self.f0ntrainModels = f0Models

        // DecoderPre models (Phase 4: CoreML, no longer bridge)
        var decPreModels: [Int: MLModel] = [:]
        for sec in buckets {
            let url = modelsDirectory.appendingPathComponent("kokoro_decoder_pre_\(sec)s.mlpackage")
            if FileManager.default.fileExists(atPath: url.path) {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine
                decPreModels[sec] = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
            }
        }
        self.decoderPreModels = decPreModels

        // Generator (HAR-post) models
        var genModels: [Int: MLModel] = [:]
        for sec in buckets {
            let url = modelsDirectory.appendingPathComponent("kokoro_decoder_har_post_\(sec)s.mlpackage")
            if FileManager.default.fileExists(atPath: url.path) {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndGPU
                genModels[sec] = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
            }
        }
        self.generatorModels = genModels
        self.availableBuckets = Array(genModels.keys.sorted())

        self.linearWeights = linearWeights
        self.linearBias = linearBias
    }

    /// Synthesize audio from pre-tokenized input.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs, optionally padded to one of ``PipelineConstants.durationTokenSizes``.
    ///   - attentionMask: Mask for actual tokens (1) vs padding (0).
    ///   - refS: Voice embedding, shape (256,).
    ///   - speed: Speech rate multiplier.
    /// - Returns: SynthesisResult with audio and timing breakdown.
    public func synthesize(
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float = 1.0
    ) throws -> SynthesisResult {
        var tensorDump: TensorDumpWriter? = nil
        return try executeKokoroSynthesis(
            request: KokoroSynthesisRequest(
                inputIds: inputIds,
                attentionMask: attentionMask,
                refS: refS,
                speed: speed
            ),
            modelProvider: self,
            linearWeights: linearWeights,
            linearBias: linearBias,
            tensorDump: &tensorDump
        )
    }

    // MARK: - Private Helpers

    public static func discoverDurationChoices(
        modelsDirectory: URL,
        useExactDurationModels: Bool = ProcessInfo.processInfo.environment["KOKORO_USE_EXACT_DURATION_MODELS"] == "1",
        maxDurationTokenLength: Int? = nil
    ) -> [DurationModelChoice] {
        var choices: [DurationModelChoice] = []
        let fm = FileManager.default
        let resolvedModelsDirectory = modelsDirectory.resolvingSymlinksInPath()
        func accepts(_ tokenLength: Int) -> Bool {
            guard let maxDurationTokenLength else { return true }
            return tokenLength <= maxDurationTokenLength
        }

        if useExactDurationModels, let urls = try? fm.contentsOfDirectory(
            at: resolvedModelsDirectory,
            includingPropertiesForKeys: nil
        ) {
            for url in urls {
                let name = url.lastPathComponent
                guard name.hasPrefix("kokoro_duration_exact_t"),
                      name.hasSuffix(".mlpackage") else {
                    continue
                }
                let raw = name
                    .replacingOccurrences(of: "kokoro_duration_exact_t", with: "")
                    .replacingOccurrences(of: ".mlpackage", with: "")
                guard let tokenLength = Int(raw) else { continue }
                guard accepts(tokenLength) else { continue }
                choices.append(DurationModelChoice(
                    cacheKey: "exact_t\(tokenLength)",
                    tokenLength: tokenLength,
                    packageURL: url,
                    requiresAttentionMask: false,
                    allowsPadding: false
                ))
            }
        }

        for tokenLength in PipelineConstants.durationTokenSizes {
            guard accepts(tokenLength) else { continue }
            let url = resolvedModelsDirectory.appendingPathComponent("kokoro_duration_t\(tokenLength).mlpackage")
            if fm.fileExists(atPath: url.path) {
                choices.append(DurationModelChoice(
                    cacheKey: "padded_t\(tokenLength)",
                    tokenLength: tokenLength,
                    packageURL: url,
                    requiresAttentionMask: true,
                    allowsPadding: true
                ))
            }
        }

        let legacyURL = resolvedModelsDirectory.appendingPathComponent("kokoro_duration.mlpackage")
        if fm.fileExists(atPath: legacyURL.path),
           !choices.contains(where: { $0.cacheKey == "padded_t128" }) {
            choices.append(DurationModelChoice(
                cacheKey: "padded_t128",
                tokenLength: PipelineConstants.durationTokenLength,
                packageURL: legacyURL,
                requiresAttentionMask: true,
                allowsPadding: true
            ))
        }

        return choices.sorted {
            if $0.tokenLength != $1.tokenLength {
                return $0.tokenLength < $1.tokenLength
            }
            return !$0.allowsPadding && $1.allowsPadding
        }
    }

    public static func selectDurationChoice(
        _ choices: [DurationModelChoice],
        actualTokens: Int
    ) throws -> DurationModelChoice {
        if let exact = choices.first(where: {
            !$0.allowsPadding && $0.tokenLength == actualTokens
        }) {
            return exact
        }

        if let padded = choices.first(where: {
            $0.allowsPadding && actualTokens <= $0.tokenLength
        }) {
            return padded
        }

        throw PipelineError.inputTooLong(
            tokens: actualTokens,
            maxTokens: choices.map { $0.tokenLength }.max() ?? 0
        )
    }

    public func durationModelChoices() -> [DurationModelChoice] {
        durationChoices
    }

    public func availableBucketSeconds() -> [Int] {
        availableBuckets
    }

    public func durationModel(choice: DurationModelChoice) throws -> MLModel {
        guard let model = durationModels[choice.cacheKey] else {
            throw PipelineError.modelNotLoaded(choice.cacheKey)
        }
        return model
    }

    public func f0ntrainModel(tFrames: Int) throws -> MLModel {
        guard let model = f0ntrainModels[tFrames] else {
            throw PipelineError.modelNotLoaded("f0ntrain_t\(tFrames)")
        }
        return model
    }

    public func decoderPreModel(bucketSec: Int) throws -> MLModel {
        guard let model = decoderPreModels[bucketSec] else {
            throw PipelineError.modelNotLoaded("decoder_pre_\(bucketSec)s")
        }
        return model
    }

    public func generatorModel(bucketSec: Int) throws -> MLModel {
        guard let model = generatorModels[bucketSec] else {
            throw PipelineError.modelNotLoaded("decoder_har_post_\(bucketSec)s")
        }
        return model
    }

}

// MARK: - Errors

public enum PipelineError: Error, LocalizedError {
    case noBucketAvailable
    case modelNotLoaded(String)
    case inputTooLong(tokens: Int, maxTokens: Int)

    public var errorDescription: String? {
        switch self {
        case .noBucketAvailable:
            return "No bucket available for the requested duration"
        case .modelNotLoaded(let name):
            return "Model not loaded: \(name)"
        case .inputTooLong(let tokens, let maxTokens):
            return "Input has \(tokens) tokens, but the largest loaded duration model supports \(maxTokens)"
        }
    }
}
