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
    /// Compute units for every decoder-pre package. Decoder-pre stays bucketed
    /// on the Neural Engine at every size: an M1's ANE beats its GPU at every
    /// bucket, and one placement for every machine keeps the runtime free of a
    /// per-machine policy. An M3 Max's GPU would be about 30 ms faster at 30 s.
    public static let decoderPreComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    /// Duration model enumerated token sizes. Caller pads to nearest.
    public static let durationTokenSizes: [Int] = [32, 64, 128, 256, 320, 384, 512]

    /// Largest duration-token bucket shipped in the Core ML bundle.
    public static var maxDurationTokenLength: Int {
        durationTokenSizes.max() ?? 512
    }

    /// Caller-side chunk token cap from `packages/contracts` (`MAX_TTS_CHUNK_TOKENS`).
    public static let maxCallerChunkTokens = 450

    /// Flexible (RangeDim) generator program: one file for every length up to
    /// its trace bucket, masked at each internal resolution, GPU only, macOS 15 /
    /// iOS 18. When present it replaces the per-bucket generator packages; the
    /// bucket then only selects the f0ntrain and decoder-pre packages.
    public static let flexibleGeneratorPackage = "kokoro_decoder_har_post_range.mlpackage"

    /// Row-folded Neural Engine generator for one bucket (see FoldedGenerator.swift).
    /// When present it replaces the GPU generators for that bucket. Two variants:
    /// `kokoro_generator_fold_{N}s` uses the ANE's native `sin` (A17 Pro: 700 ms
    /// at 30 s); `kokoro_generator_fold_poly_{N}s` uses a polynomial Snake that
    /// every ANE runs (M1/M2/A14 have no ANE `sin`; there the native package
    /// falls off the ANE). ``foldedGeneratorCandidates`` orders them per device.
    public static func foldedGeneratorPackage(bucketSec: Int, poly: Bool = false) -> String {
        poly ? "kokoro_generator_fold_poly_\(bucketSec)s.mlpackage" : "kokoro_generator_fold_\(bucketSec)s.mlpackage"
    }

    /// Whether this device's ANE runs `sin` natively. Known only for the chips
    /// measured (README/Notes/ane-folded-generator.md): yes on A17 Pro, no on
    /// M1, M2 and A14. Unmeasured chips get the polynomial variant, which is
    /// correct everywhere.
    public static var aneHasNativeSin: Bool {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        let id = String(cString: machine)                 // e.g. "iPhone16,2"
        guard id.hasPrefix("iPhone"), let major = Int(id.dropFirst(6).prefix { $0 != "," }) else { return false }
        return major >= 16                                // iPhone16,x = A17 Pro
    }

    /// Folded package names to try for a bucket, preferred first.
    public static func foldedGeneratorCandidates(bucketSec: Int) -> [String] {
        let native = foldedGeneratorPackage(bucketSec: bucketSec)
        let poly = foldedGeneratorPackage(bucketSec: bucketSec, poly: true)
        return aneHasNativeSin ? [native, poly] : [poly, native]
    }

    /// Duration-model compute units by padded token length.
    ///
    /// M1 Mini, warm, per bucket (cpuOnly / cpuAndNeuralEngine / cpuAndGPU):
    /// t64 15.9 / 83.1 / 25.5 ms, t128 28.8 / 18.7 / 38.1, t256 56.2 / 42.8 / 105.7,
    /// t512 114.9 / 100.2 / 230.4 (README/Notes/ane-folded-generator.md). On iOS
    /// the duration graph has failed ANECCompile and stalled in MPSGraph
    /// specialization on older phones, so iPhones before the A17 stay on the
    /// CPU like the SDK's gistDefault. On an A17 Pro the padded duration graph
    /// compiles for the ANE and runs 9.2 ms (t64) and 61 ms (t512) vs 14-31
    /// and 104-181 ms on CPU+GPU (ios-bench, 2026-10-21).
    public static func durationComputeUnits(tokenLength: Int) -> MLComputeUnits {
        #if os(iOS)
        return aneHasNativeSin ? .cpuAndNeuralEngine : .cpuOnly
        #else
        return tokenLength <= 64 ? .cpuOnly : .cpuAndNeuralEngine
        #endif
    }

    /// F0Ntrain compute units: the CPU at every size. M2 Air, warm
    /// (cpuOnly / cpuAndNeuralEngine / cpuAndGPU): t120 3.6 / 3.6 / 26.4 ms,
    /// t400 11.3 / 11.5 / 21.0, t1200 32.0 / 40.0 / 65.4
    /// (README/Notes/ane-folded-generator.md). The CPU also has no
    /// compile-time risk on any device.
    public static let f0ntrainComputeUnits: MLComputeUnits = .cpuOnly

    /// Environment override: `0` never loads the folded generator.
    public static let foldedGeneratorEnvKey = "KOKORO_FOLDED_GENERATOR"

    public static var foldedGeneratorAllowed: Bool {
        ProcessInfo.processInfo.environment[foldedGeneratorEnvKey] != "0"
    }

    /// First OS major version whose MPSGraph can abort the process on the
    /// flexible generator's first predict ('mps.expand_dims' op invalid axis,
    /// then `MPSRuntime.mm:1519: failed assertion 'shape for TensorData is not
    /// static'`). Seen intermittently on macOS 27.0.1 (26A434); macOS 26.7.1
    /// runs the same package. The abort cannot be caught, so on every 27.x,
    /// until a release is re-verified, the loaders use the bucketed
    /// generators instead wherever they ship. See
    /// README/Notes/debug-notes.md "Flexible generator aborts on OS 27".
    public static let flexibleGeneratorBrokenOSMajor = 27

    /// Environment override: `0` never loads the flexible generator, `1`
    /// loads it even on an OS the gate blocks (to re-verify a new release).
    public static let flexibleGeneratorEnvKey = "KOKORO_FLEXIBLE_GENERATOR"

    /// Whether this OS can run the flexible generator without aborting.
    /// Callers fall back to the per-bucket generator packages when false.
    public static func flexibleGeneratorAllowed(
        osVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        switch environment[flexibleGeneratorEnvKey] {
        case "0": return false
        case "1": return true
        default: return osVersion.majorVersion < flexibleGeneratorBrokenOSMajor
        }
    }

    /// The one rule both runtimes (KokoroPipeline, the SDK provider) use to
    /// pick a generator. Where the gate blocks the flexible program, any
    /// bucket generator in the bundle wins: buckets without one become
    /// unavailable (or fail validation) rather than risk the abort. A bundle
    /// with no bucket generator at all keeps the flexible one, because it is
    /// the only generator there is.
    public static func usesFlexibleGenerator(
        hasFlexibleGenerator: Bool,
        hasAnyBucketGenerator: Bool,
        allowed: Bool = flexibleGeneratorAllowed()
    ) -> Bool {
        hasFlexibleGenerator && (allowed || !hasAnyBucketGenerator)
    }

    /// Lengths fed to the flexible generator are rounded up to this granule
    /// (0.5 s: 40 x_pre frames at 80 Hz). The GPU
    /// runtime keeps a compiled executable per distinct input shape (about
    /// 6 MB each for the generator, evicted only slowly), so unrounded lengths
    /// grew a process past 1 GB over 100 utterances; at most 60 shapes keeps it
    /// within 60 MB of the fixed packages. The padded tail is masked out of
    /// the AdaIN statistics and trimmed from the audio; it costs at most 0.5 s
    /// of generator compute (about 4 ms on an M3 Max).
    public static let flexibleGranuleSeconds: Double = 0.5
    public static var flexibleXPreGranuleFrames: Int { Int(flexibleGranuleSeconds * f0FrameRate) }
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
    public var decoderPreHnsfOverlap: Double = 0
    public var generatorCoreML: Double = 0
    public var trim: Double = 0

    /// Total pipeline wall time.
    public var total: Double {
        durationCoreML + alignment + matrixOps + f0ntrainCoreML +
        padding + decoderPre + hnsfSwift - decoderPreHnsfOverlap +
        generatorCoreML + trim
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
    /// Which programs ran: duration and F0Ntrain `split` or `whole`; generator
    /// `hybrid`, `folded`, `flexible` or `bucket` (the per-bucket GPU package).
    public let durationPath: String
    public let f0Path: String
    public let generatorPath: String
}

public struct DurationModelChoice {
    public let cacheKey: String
    public let tokenLength: Int
    public let packageURL: URL
    public let requiresAttentionMask: Bool
    public let allowsPadding: Bool
    /// Function inside a multifunction package; nil for a separate per-size package.
    public let functionName: String?

    /// Public memberwise init so app integrators (e.g. ios-bench) can build
    /// choices for precompiled .mlmodelc bundles instead of discovering
    /// .mlpackage files on disk via ``discoverDurationChoices``.
    public init(
        cacheKey: String,
        tokenLength: Int,
        packageURL: URL,
        requiresAttentionMask: Bool,
        allowsPadding: Bool,
        functionName: String? = nil
    ) {
        self.cacheKey = cacheKey
        self.tokenLength = tokenLength
        self.packageURL = packageURL
        self.requiresAttentionMask = requiresAttentionMask
        self.allowsPadding = allowsPadding
        self.functionName = functionName
    }
}

// MARK: - Pipeline

/// Main TTS pipeline orchestrator.
///
/// Loads CoreML models and provides ``synthesize()`` for text-to-audio.
///
/// ## Model loading
///
/// ``init(modelsDirectory:buckets:linearWeights:linearBias:compiledModelCache:)``
/// only discovers which packages exist; each Core ML model is compiled and
/// opened on first use (or by ``prepareForBucket(bucketSec:tFrames:)``) and
/// stays open. The first synthesis for a bucket therefore pays that bucket's
/// compile, which can take seconds per model on first run. For app
/// integration, prewarm on a background thread and pass a durable
/// `compiledModelCache` so the compile is paid once per model set.
public class KokoroPipeline: KokoroModelProvider {
    // Model selection is cheap; opening a Core ML bundle is not. The iPad
    // worker (botnet/apps/ios-worker) loads only the duration shape and the
    // bucket its claimed job needs.
    private let modelsDirectory: URL
    private let compiledModelCache: URL?
    private let durationChoices: [DurationModelChoice]
    private let f0ntrainMultifunction: MultifunctionPackage?
    private let decoderPreMultifunction: MultifunctionPackage?
    private let durationMultifunction: MultifunctionPackage?
    private let usesFlexibleGenerator: Bool
    /// Opened models keyed by stage and shape, e.g. `f0ntrain.t600`. Guarded by `lock`.
    private var openModels: [String: MLModel] = [:]
    private let lock = NSRecursiveLock()

    /// Learned weights from SourceModuleHnNSF.l_linear.
    private let linearWeights: [Float]
    private let linearBias: Float

    /// Available bucket durations in seconds.
    private let availableBuckets: [Int]

    /// Discover the model set in a directory. Models open on first use.
    ///
    /// Expected files:
    /// - ``kokoro_duration_t{T}.mlpackage`` for each token size, or legacy ``kokoro_duration.mlpackage``
    /// - ``kokoro_f0ntrain_t{T}.mlpackage`` for each bucket's T_frames
    /// - ``kokoro_decoder_pre_{N}s.mlpackage`` for each bucket
    /// - ``kokoro_decoder_har_post_{N}s.mlpackage`` for each bucket, or the
    ///   flexible ``kokoro_decoder_har_post_range.mlpackage`` (macOS 15 / iOS 18)
    /// - optionally the multifunction duration, f0ntrain and decoder-pre
    ///   packages, which are compiled here to list their functions
    ///
    /// - Parameter compiledModelCache: Directory to keep compiled `.mlmodelc`
    ///   bundles in across launches. `nil` compiles into a temporary directory
    ///   every time, which is the historical behaviour. The caller owns the
    ///   directory's identity: point it at a path derived from the model set's
    ///   digest so a changed model set cannot read a stale bundle.
    public init(
        modelsDirectory: URL,
        buckets: [Int] = PipelineConstants.defaultBuckets,
        linearWeights: [Float],
        linearBias: Float,
        compiledModelCache: URL? = nil
    ) throws {
        let directory = modelsDirectory.resolvingSymlinksInPath()
        func exists(_ name: String) -> Bool {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
        }

        // Duration: padded mask-aware packages for production by default;
        // exact native packages are an opt-in benchmark path.
        let durationChoices = Self.discoverDurationChoices(modelsDirectory: directory, compiledModelCache: compiledModelCache)
        guard !durationChoices.isEmpty else {
            throw PipelineError.modelNotLoaded("duration")
        }

        // Multifunction packages serve the shapes they have a function for;
        // separate packages fill the rest.
        let f0Multi = try MultifunctionPackage.open(
            at: directory.appendingPathComponent(PipelineConstants.f0ntrainMultifunctionPackage), cache: compiledModelCache)
        let preMulti = try MultifunctionPackage.open(
            at: directory.appendingPathComponent(PipelineConstants.decoderPreMultifunctionPackage), cache: compiledModelCache)
        let durationMulti = durationChoices.contains { $0.functionName != nil }
            ? try MultifunctionPackage.open(
                at: directory.appendingPathComponent(PipelineConstants.durationMultifunctionPackage), cache: compiledModelCache)
            : nil

        // Flexible generator (optional, macOS 15 / iOS 18). On an older OS it
        // is skipped and the bucketed generator packages serve every length.
        // On an OS the gate blocks it is skipped too, unless no bucketed
        // generator ships (PipelineConstants.usesFlexibleGenerator).
        var flexibleSupported = false
        if #available(macOS 15.0, iOS 18.0, *) { flexibleSupported = true }
        let usesFlexibleGenerator = flexibleSupported && PipelineConstants.usesFlexibleGenerator(
            hasFlexibleGenerator: exists(PipelineConstants.flexibleGeneratorPackage),
            hasAnyBucketGenerator: buckets.contains { exists("kokoro_decoder_har_post_\($0)s.mlpackage") }
        )

        // A bucket is available when every stage it needs has a package.
        let availableBuckets = buckets.filter { sec in
            guard let tFrames = PipelineConstants.tFramesForBucket[sec] else { return false }
            let split = SplitF0N.packageNames(tFrames: tFrames)
            let hasF0 = f0Multi?.functionNames.contains(PipelineConstants.f0ntrainFunctionName(tFrames: tFrames)) == true
                || exists("kokoro_f0ntrain_t\(tFrames).mlpackage")
                || (SplitF0N.allowed && exists(split.front) && exists(split.back))
            let hasPre = preMulti?.functionNames.contains(PipelineConstants.decoderPreFunctionName(bucketSec: sec)) == true
                || exists("kokoro_decoder_pre_\(sec)s.mlpackage")
            let hasFolded = PipelineConstants.foldedGeneratorAllowed
                && PipelineConstants.foldedGeneratorCandidates(bucketSec: sec).contains(where: exists)
            let hasGenerator = hasFolded || usesFlexibleGenerator || exists("kokoro_decoder_har_post_\(sec)s.mlpackage")
            return hasF0 && hasPre && hasGenerator
        }.sorted()
        if availableBuckets.isEmpty && !usesFlexibleGenerator && exists(PipelineConstants.flexibleGeneratorPackage) {
            throw PipelineError.modelNotLoaded(
                "\(PipelineConstants.flexibleGeneratorPackage) needs macOS 15 / iOS 18 and no kokoro_decoder_har_post_{N}s packages are present"
            )
        }

        self.modelsDirectory = directory
        self.compiledModelCache = compiledModelCache
        self.durationChoices = durationChoices
        self.f0ntrainMultifunction = f0Multi
        self.decoderPreMultifunction = preMulti
        self.durationMultifunction = durationMulti
        self.usesFlexibleGenerator = usesFlexibleGenerator
        self.availableBuckets = availableBuckets
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
        // Core ML hands back autoreleased Objective-C objects (feature
        // providers, MLMultiArrays). Callers on Swift-concurrency or other
        // pool-less threads never drain them, so without this pool every call
        // leaks its intermediate tensors: a 40-sentence run grew a process
        // from ~960 MB to ~1.4 GB on an M2 Ultra; with it, memory stays flat.
        try autoreleasepool {
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
    }

    // MARK: - Private Helpers

    /// Loads a flexible (RangeDim) program on the GPU, or returns nil when the
    /// package is absent or the OS predates flexible-shape GPU programs.
    /// Callers apply ``PipelineConstants/flexibleGeneratorAllowed(osVersion:environment:)``.
    public static func loadFlexibleProgram(at url: URL, compiledModelCache: URL? = nil) throws -> MLModel? {
        guard #available(macOS 15.0, iOS 18.0, *),
              FileManager.default.fileExists(atPath: url.path) else { return nil }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        return try CompiledModelCache.load(package: url, configuration: config, cache: compiledModelCache)
    }

    public static func discoverDurationChoices(
        modelsDirectory: URL,
        useExactDurationModels: Bool = ProcessInfo.processInfo.environment["KOKORO_USE_EXACT_DURATION_MODELS"] == "1",
        maxDurationTokenLength: Int? = nil,
        compiledModelCache: URL? = nil
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

        // A multifunction duration package serves every token size it has a
        // function for; separate packages fill the rest.
        let multiURL = resolvedModelsDirectory.appendingPathComponent(PipelineConstants.durationMultifunctionPackage)
        let multi = try? MultifunctionPackage.open(at: multiURL, cache: compiledModelCache)
        for tokenLength in PipelineConstants.durationTokenSizes {
            guard accepts(tokenLength) else { continue }
            let functionName = PipelineConstants.durationFunctionName(tokenLength: tokenLength)
            if let multi, multi.functionNames.contains(functionName) {
                choices.append(DurationModelChoice(
                    cacheKey: "padded_t\(tokenLength)",
                    tokenLength: tokenLength,
                    packageURL: multiURL,
                    requiresAttentionMask: true,
                    allowsPadding: true,
                    functionName: functionName
                ))
                continue
            }
            let url = resolvedModelsDirectory.appendingPathComponent("kokoro_duration_t\(tokenLength).mlpackage")
            let split = SplitDuration.packageNames(tokenLength: tokenLength)
            let hasSplit = SplitDuration.allowed
                && fm.fileExists(atPath: resolvedModelsDirectory.appendingPathComponent(split.front).path)
                && fm.fileExists(atPath: resolvedModelsDirectory.appendingPathComponent(split.back).path)
            // A split pair alone (SDK schema-2 bundles ship no whole model)
            // also makes the size available; `packageURL` still names the
            // whole package, and providers serve the choice with the split.
            if fm.fileExists(atPath: url.path) || hasSplit {
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
        try open("duration.\(choice.cacheKey)") {
            let units = PipelineConstants.durationComputeUnits(tokenLength: choice.tokenLength)
            if let function = choice.functionName, let durationMultifunction {
                return try durationMultifunction.load(function: function, computeUnits: units)
            }
            return try load(choice.packageURL, units: units)
        }
    }

    public func f0ntrainModel(tFrames: Int) throws -> MLModel {
        try open("f0ntrain.t\(tFrames)") {
            let function = PipelineConstants.f0ntrainFunctionName(tFrames: tFrames)
            if let f0ntrainMultifunction, f0ntrainMultifunction.functionNames.contains(function) {
                return try f0ntrainMultifunction.load(function: function, computeUnits: PipelineConstants.f0ntrainComputeUnits)
            }
            return try load(package("kokoro_f0ntrain_t\(tFrames).mlpackage"), units: PipelineConstants.f0ntrainComputeUnits)
        }
    }

    public func decoderPreModel(bucketSec: Int) throws -> MLModel {
        try open("decoder_pre.\(bucketSec)s") {
            let units = PipelineConstants.decoderPreComputeUnits
            let function = PipelineConstants.decoderPreFunctionName(bucketSec: bucketSec)
            if let decoderPreMultifunction, decoderPreMultifunction.functionNames.contains(function) {
                return try decoderPreMultifunction.load(function: function, computeUnits: units)
            }
            return try load(package("kokoro_decoder_pre_\(bucketSec)s.mlpackage"), units: units)
        }
    }

    public func generatorModel(bucketSec: Int) throws -> MLModel {
        try open("generator.\(bucketSec)s") {
            try load(package("kokoro_decoder_har_post_\(bucketSec)s.mlpackage"), units: .cpuAndGPU)
        }
    }

    private var hybridGenerators: [Int: AnyObject] = [:]
    /// The hybrid generator when `hybrid/` sits next to the other packages
    /// (on unless KOKORO_HYBRID_GENERATOR=0 or it failed earlier this process).
    public func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? {
        guard #available(macOS 15.0, iOS 18.0, *) else { return nil }
        lock.lock()
        defer { lock.unlock() }
        // After a failure the folded generator serves; free the hybrid instead of holding both.
        guard HybridGenerator.allowed else { hybridGenerators.removeAll(); return nil }
        let dir = modelsDirectory.appendingPathComponent("hybrid")
        guard let rows = HybridGenerator.rows(bucketSec: bucketSec, xPreFrames: xPreFrames, directory: dir) else { return nil }
        if let g = hybridGenerators[rows] { return g }
        // Keep one resident: several loaded at once failed on an M1 (README/Notes/ane-folded-generator.md).
        hybridGenerators.removeAll()
        let package = dir.appendingPathComponent("hybrid_generator_r\(rows).mlpackage")
        guard FileManager.default.fileExists(atPath: package.path), compiledModelCache != nil else {
            let g = try HybridGenerator(directory: dir, rows: rows)
            hybridGenerators[rows] = g
            return g
        }
        // A cached compile Core ML can no longer load (an interrupted move, an
        // OS upgrade) is evicted and recompiled once. Other errors (plan, Metal,
        // KOKORO_HYBRID_E5) are not the cache's fault and are not retried.
        func make(reuseCache: Bool) throws -> HybridGenerator {
            let compiled = try CompiledModelCache.compiledURL(for: package, cache: compiledModelCache) { _ in reuseCache }
            return try HybridGenerator(directory: dir, rows: rows, compiledModel: compiled)
        }
        let g: HybridGenerator
        do {
            g = try make(reuseCache: true)
        } catch HybridGeneratorError.modelLoad {
            // Not testable: needs a real hybrid package and a corrupted compile in a
            // KokoroPipeline model dir; the SDK twin is covered by
            // SchemaTwoManifestTests.testCorruptHybridCompileCacheIsEvictedAndRebuilt.
            g = try make(reuseCache: false)
        }
        hybridGenerators[rows] = g
        return g
    }

    /// Two-part duration model when its packages sit next to the others
    /// (SplitDuration.swift); padded choices only.
    public func splitDuration(choice: DurationModelChoice) throws -> SplitDuration? {
        guard SplitDuration.allowed, choice.allowsPadding, choice.functionName == nil else { return nil }
        let names = SplitDuration.packageNames(tokenLength: choice.tokenLength)
        let frontURL = modelsDirectory.appendingPathComponent(names.front)
        let backURL = modelsDirectory.appendingPathComponent(names.back)
        guard FileManager.default.fileExists(atPath: frontURL.path),
              FileManager.default.fileExists(atPath: backURL.path) else { return nil }
        let front = try open("duration.front.t\(choice.tokenLength)") { try load(frontURL, units: SplitDuration.frontComputeUnits) }
        let back = try open("duration.back.t\(choice.tokenLength)") { try load(backURL, units: SplitDuration.backComputeUnits) }
        return SplitDuration(front: front, back: back)
    }

    /// Two-part F0Ntrain when its packages sit next to the others (SplitDuration.swift).
    public func splitF0N(tFrames: Int) throws -> SplitF0N? {
        guard SplitF0N.allowed else { return nil }
        let names = SplitF0N.packageNames(tFrames: tFrames)
        let frontURL = modelsDirectory.appendingPathComponent(names.front)
        let backURL = modelsDirectory.appendingPathComponent(names.back)
        guard FileManager.default.fileExists(atPath: frontURL.path),
              FileManager.default.fileExists(atPath: backURL.path) else { return nil }
        let front = try open("f0ntrain.front.t\(tFrames)") { try load(frontURL, units: SplitF0N.frontComputeUnits) }
        let back = try open("f0ntrain.back.t\(tFrames)") { try load(backURL, units: SplitF0N.backComputeUnits) }
        return SplitF0N(front: front, back: back)
    }

    public func foldedGeneratorModel(bucketSec: Int) throws -> MLModel? {
        guard PipelineConstants.foldedGeneratorAllowed,
              let name = PipelineConstants.foldedGeneratorCandidates(bucketSec: bucketSec)
                .first(where: { FileManager.default.fileExists(atPath: package($0).path) }) else { return nil }
        return try open("generator.fold.\(bucketSec)s") {
            try load(package(name), units: .cpuAndNeuralEngine)
        }
    }

    public func flexibleGeneratorModel() throws -> MLModel? {
        guard usesFlexibleGenerator else { return nil }
        return try open("generator.range") {
            let url = package(PipelineConstants.flexibleGeneratorPackage)
            guard let model = try Self.loadFlexibleProgram(at: url, compiledModelCache: compiledModelCache) else {
                throw PipelineError.modelNotLoaded(PipelineConstants.flexibleGeneratorPackage)
            }
            return model
        }
    }

    /// Opens the models one bucket needs, so the first synthesis in it is not
    /// the one that pays the compile.
    public func prepareForBucket(bucketSec: Int, tFrames: Int) throws {
        // The program the executor will run: the split pair when it ships.
        if try splitF0N(tFrames: tFrames) == nil {
            _ = try f0ntrainModel(tFrames: tFrames)
        }
        _ = try decoderPreModel(bucketSec: bucketSec)
        // The executor prefers the hybrid; opening the folded or GPU generator
        // too would only hold memory.
        if try resolveHybridGenerator(self, bucketSec: bucketSec, xPreFrames: bucketSec * Int(PipelineConstants.f0FrameRate)) != nil { return }
        if try foldedGeneratorModel(bucketSec: bucketSec) != nil { return }
        if try flexibleGeneratorModel() == nil {
            _ = try generatorModel(bucketSec: bucketSec)
        }
    }

    /// Returns the open model for `key`, opening it with `make` on first use.
    private func open(_ key: String, _ make: () throws -> MLModel) throws -> MLModel {
        lock.lock()
        defer { lock.unlock() }
        if let model = openModels[key] { return model }
        let model = try make()
        openModels[key] = model
        return model
    }

    private func package(_ name: String) -> URL {
        modelsDirectory.appendingPathComponent(name)
    }

    private func load(_ url: URL, units: MLComputeUnits) throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PipelineError.modelNotLoaded(url.deletingPathExtension().lastPathComponent)
        }
        let config = MLModelConfiguration()
        config.computeUnits = units
        return try CompiledModelCache.load(package: url, configuration: config, cache: compiledModelCache)
    }
}

// MARK: - Errors

public enum PipelineError: Error, LocalizedError {
    case noBucketAvailable
    case modelNotLoaded(String)
    case modelContractMismatch(String)
    case inputTooLong(tokens: Int, maxTokens: Int)

    public var errorDescription: String? {
        switch self {
        case .noBucketAvailable:
            return "No bucket available for the requested duration"
        case .modelNotLoaded(let name):
            return "Model not loaded: \(name)"
        case .modelContractMismatch(let message):
            return "Model contract mismatch: \(message)"
        case .inputTooLong(let tokens, let maxTokens):
            return "Input has \(tokens) tokens, but the largest loaded duration model supports \(maxTokens)"
        }
    }
}
