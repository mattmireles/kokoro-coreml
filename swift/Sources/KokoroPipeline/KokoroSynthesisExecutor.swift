import CoreML
import Foundation

/// Supplies Core ML models to the shared synthesis executor.
///
/// The runtime pipeline uses already-loaded model dictionaries. The benchmark
/// uses a lazy cache that can evict bucket models before loading a new bucket.
public protocol KokoroModelProvider {
    func durationModelChoices() -> [DurationModelChoice]
    func availableBucketSeconds() -> [Int]
    func durationModel(choice: DurationModelChoice) throws -> MLModel
    func f0ntrainModel(tFrames: Int) throws -> MLModel
    func decoderPreModel(bucketSec: Int) throws -> MLModel
    func generatorModel(bucketSec: Int) throws -> MLModel
    func prepareForBucket(bucketSec: Int, tFrames: Int) throws
    /// Flexible (RangeDim) generator program, or nil to use the bucket packages.
    func flexibleGeneratorModel() throws -> MLModel?
    /// Row-folded Neural Engine generator for this bucket (see
    /// FoldedGenerator.swift), or nil to use the GPU generators.
    func foldedGeneratorModel(bucketSec: Int) throws -> MLModel?
    /// Hybrid ANE-conv + Metal generator for this bucket (HybridGenerator.swift),
    /// or nil. Preferred over the folded and GPU generators when present. Its
    /// row count follows `xPreFrames` (80 Hz frames to synthesize), not the bucket.
    func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject?
    /// Two-part duration model (ANE front + CPU back) for this choice, or nil
    /// to use `durationModel(choice:)`. See SplitDuration.swift.
    func splitDuration(choice: DurationModelChoice) throws -> SplitDuration?
    /// Two-part F0Ntrain (CPU LSTM + ANE branches) for this size, or nil to use
    /// `f0ntrainModel(tFrames:)`. See SplitDuration.swift.
    func splitF0N(tFrames: Int) throws -> SplitF0N?
}

public extension KokoroModelProvider {
    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {}
    func flexibleGeneratorModel() -> MLModel? { nil }
    func foldedGeneratorModel(bucketSec: Int) -> MLModel? { nil }
    func hybridGenerator(bucketSec: Int, xPreFrames: Int) -> AnyObject? { nil }
    func splitDuration(choice: DurationModelChoice) -> SplitDuration? { nil }
    func splitF0N(tFrames: Int) -> SplitF0N? { nil }
}

/// Pre-tokenized synthesis request for the shared Swift/Core ML pipeline.
public struct KokoroSynthesisRequest {
    public let inputIds: [Int32]
    public let attentionMask: [Int32]
    public let refS: [Float]
    public let speed: Float
    public let seed: UInt64
    public let warmModelsBeforeTiming: Bool
    public let bucketDurationOverrideSeconds: Double?

    public init(
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float = 1.0,
        seed: UInt64 = 42,
        warmModelsBeforeTiming: Bool = false,
        bucketDurationOverrideSeconds: Double? = nil
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.refS = refS
        self.speed = speed
        self.seed = seed
        self.warmModelsBeforeTiming = warmModelsBeforeTiming
        self.bucketDurationOverrideSeconds = bucketDurationOverrideSeconds
    }
}

private struct DurationInputBundle {
    let provider: MLDictionaryFeatureProvider
    let idsArray: MLMultiArray
    let maskArray: MLMultiArray?
    let refSArray: MLMultiArray
    let speedArray: MLMultiArray
}

/// Hands the pre-made harmonic noise to the hn-NSF thread without a copy.
private final class NoiseBox {
    var value: [Float]?
    func take() -> [Float]? {
        let v = value
        value = nil
        return v
    }
}

private struct DurationProbe {
    let bucketSec: Int
    let tFrames: Int
    let fullF0Len: Int
    let validFrames: Int
}

/// Run the Core ML Kokoro pipeline once.
///
/// This is the single orchestration path shared by `KokoroPipeline.synthesize`
/// and the `kokoro-bench` executable. Benchmark-only behavior is injected via
/// `KokoroModelProvider.prepareForBucket(...)`, `warmModelsBeforeTiming`, and
/// the optional tensor dump writer.
/// Diagnostic: KOKORO_FOOTPRINT_LOG=1 prints the process phys_footprint (MB)
/// at each stage boundary to stderr, to attribute peak RAM.
func logFootprint(_ stage: String) {
    guard ProcessInfo.processInfo.environment["KOKORO_FOOTPRINT_LOG"] == "1" else { return }
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    if kr == KERN_SUCCESS {
        fputs(String(format: "FOOTPRINT %@ %.0f MB (peak %.0f MB)\n", stage, Double(info.phys_footprint) / 1048576,
                     Double(info.ledger_phys_footprint_peak) / 1048576), stderr)
    }
}

/// The hybrid generator for one clip, or nil. Any error getting it (plan,
/// digest, compile, ANE load, no Metal) goes to `hybridFailed`.
func resolveHybridGenerator(_ provider: KokoroModelProvider, bucketSec: Int, xPreFrames: Int,
                            requested: Bool = hybridRequested()) throws -> AnyObject? {
    // No failure check here: every provider gates on HybridGenerator.allowed /
    // .enabled itself, and must be called after a failure so it can release
    // the dead hybrid (its ANE functions and IOSurfaces) instead of holding it
    // next to the folded generator.
    guard #available(macOS 15.0, iOS 18.0, *) else { return nil }
    do {
        return try provider.hybridGenerator(bucketSec: bucketSec, xPreFrames: xPreFrames)
    } catch {
        try hybridFailed(error, requested: requested)
        return nil
    }
}

private func hybridRequested() -> Bool {
    guard #available(macOS 15.0, iOS 18.0, *) else { return false }
    return HybridGenerator.requested
}

/// A hybrid error. When a benchmark requested the hybrid it is rethrown, so a
/// folded run is never timed under a hybrid label. Otherwise it is recorded
/// with `HybridGenerator.recordFailure`, which turns the hybrid off for the
/// process; the folded generator serves instead.
func hybridFailed(_ error: Error, requested: Bool = hybridRequested()) throws {
    if requested { throw error }
    if #available(macOS 15.0, iOS 18.0, *) { HybridGenerator.recordFailure(error) }
}

/// x_pre length the flexible generator runs at: the real length rounded up to
/// the granule and clamped to the program's bounds; nil for a fixed shape.
private func flexibleXPreTime(_ model: MLModel, realFrames: Int) throws -> Int? {
    guard let accepted = flexibleTimeRange(of: model, input: "x_pre") else { return nil }
    return try flexibleTimeAxis(realFrames: realFrames, accepted: accepted,
                                granule: PipelineConstants.flexibleXPreGranuleFrames, stage: "generator")
}

/// HybridGenerator owns fixed IOSurface tensors and an arena, so two concurrent
/// `synthesize` calls on one instance corrupt each other (non-finite audio).
/// The SDK never overlaps them; this guards other callers of the public executor.
private let hybridSynthesisLock = NSLock()

public func executeKokoroSynthesis(
    request: KokoroSynthesisRequest,
    modelProvider: KokoroModelProvider,
    linearWeights: [Float],
    linearBias: Float,
    tensorDump: inout TensorDumpWriter?,
    beforeBucketStages: (() -> Void)? = nil
) throws -> SynthesisResult {
    // Drain Core ML's autoreleased outputs per call: without this a caller that
    // loops off the main run loop (CLI, background queue) grew ~50-120 MB per
    // 30 s synthesis on an M1 (README/Notes/ane-folded-generator.md).
    var dump = tensorDump
    defer { tensorDump = dump }
    return try autoreleasepool {
        try executeKokoroSynthesisBody(request: request, modelProvider: modelProvider,
                                       linearWeights: linearWeights, linearBias: linearBias, tensorDump: &dump,
                                       beforeBucketStages: beforeBucketStages)
    }
}

private func executeKokoroSynthesisBody(
    request: KokoroSynthesisRequest,
    modelProvider: KokoroModelProvider,
    linearWeights: [Float],
    linearBias: Float,
    tensorDump: inout TensorDumpWriter?,
    beforeBucketStages: (() -> Void)?
) throws -> SynthesisResult {
    let durationChoices = modelProvider.durationModelChoices()
    let durationChoice = try KokoroPipeline.selectDurationChoice(
        durationChoices,
        actualTokens: requestedTokenCount(
            inputIds: request.inputIds,
            attentionMask: request.attentionMask
        )
    )
    let durationInput = try buildDurationInput(
        inputIds: request.inputIds,
        attentionMask: request.attentionMask,
        refS: request.refS,
        speed: request.speed,
        choice: durationChoice
    )
    let durationModel: StagePredictor = try modelProvider.splitDuration(choice: durationChoice)
        ?? modelProvider.durationModel(choice: durationChoice)
    let durationPath = durationModel is SplitDuration ? "split" : "whole"

    try writeDurationInputs(durationInput, tensorDump: &tensorDump)

    if request.warmModelsBeforeTiming {
        let probe = try probeDurationAndBucket(
            input: durationInput,
            durationModel: durationModel,
            modelProvider: modelProvider,
            validTokenLimit: validTokenCount(
                predDurTokenCount: durationChoice.tokenLength,
                attentionMask: request.attentionMask
            ),
            bucketDurationOverrideSeconds: request.bucketDurationOverrideSeconds
        )
        try warmModels(
            probe: probe,
            durationModel: durationModel,
            durationInput: durationInput.provider,
            modelProvider: modelProvider
        )
    }

    var timings = StageTimings()

    // Stage 1: Duration Core ML.
    logFootprint("start")
    let t0 = CFAbsoluteTimeGetCurrent()
    let durOutput = try durationModel.prediction(from: durationInput.provider)
    let t1 = CFAbsoluteTimeGetCurrent()
    timings.durationCoreML = t1 - t0
    logFootprint("duration")

    let predDurArray = durOutput.featureValue(for: "pred_dur")!.multiArrayValue!
    let dArray = durOutput.featureValue(for: "d")!.multiArrayValue!
    let tEnArray = durOutput.featureValue(for: "t_en")!.multiArrayValue!
    let tokenCount = predDurArray.shape.last!.intValue
    let validTokens = validTokenCount(
        predDurTokenCount: tokenCount,
        attentionMask: request.attentionMask
    )
    let predDur = try readDurationFrames(from: predDurArray, validCount: validTokens)
    let frames = predDur.reduce(0, +)
    let totalSeconds = Double(frames * 2) / PipelineConstants.f0FrameRate
    let bucketSelectionSeconds = request.bucketDurationOverrideSeconds ?? totalSeconds

    try writeDurationOutputs(
        predDurArray: predDurArray,
        predDur: predDur,
        dArray: dArray,
        tEnArray: tEnArray,
        tensorDump: &tensorDump
    )

    // Stage 2: alignment metadata. Tensor dumps keep the old sparse matrix as
    // debug data; the hot path expands token vectors directly in Stage 3.
    let t2 = CFAbsoluteTimeGetCurrent()
    let alignment: [Float]? = tensorDump == nil
        ? nil
        : buildAlignmentMatrix(predDur: predDur, traceLength: tokenCount, frameCount: frames)
    let t3 = CFAbsoluteTimeGetCurrent()
    timings.alignment = t3 - t2

    if let alignment {
        try tensorDump?.writeFloatArray(
            name: "alignment",
            values: alignment,
            shape: [1, tokenCount, frames]
        )
    }

    // Stage 3: direct token-vector expansion.
    let t4 = CFAbsoluteTimeGetCurrent()
    let en = try alignTokenMajorToFrames(
        source: dArray,
        predDur: predDur,
        channels: PipelineConstants.hiddenDim,
        frameCount: frames
    )
    let asr = try alignChannelMajorToFrames(
        source: tEnArray,
        predDur: predDur,
        channels: PipelineConstants.textEncoderDim,
        frameCount: frames
    )
    let t5 = CFAbsoluteTimeGetCurrent()
    timings.matrixOps = t5 - t4

    try tensorDump?.writeMLMultiArray(name: "en", array: en)
    try tensorDump?.writeMLMultiArray(name: "asr", array: asr)

    // Stage 4: F0Ntrain Core ML.
    let t6 = CFAbsoluteTimeGetCurrent()
    guard let bucketSec = selectBucket(
        totalSeconds: bucketSelectionSeconds,
        availableBuckets: modelProvider.availableBucketSeconds()
    ) else {
        throw PipelineError.noBucketAvailable
    }
    guard let tFrames = PipelineConstants.tFramesForBucket[bucketSec] else {
        throw PipelineError.modelNotLoaded("f0ntrain bucket \(bucketSec)")
    }
    // The harmonic source's Gaussian noise depends only on the seed and the
    // bucket length: make it now, while F0Ntrain runs (~28 ms off the critical
    // path at 30 s on an M1). buildHar ignores it if the length differs.
    let noiseFrames = Int(round(Double(bucketSec * PipelineConstants.sampleRate) / Double(HarmonicConstants.upsampleScale)))
    let noiseBox = NoiseBox()
    let noiseGroup = DispatchGroup()
    DispatchQueue.global(qos: .userInitiated).async(group: noiseGroup) {
        noiseBox.value = harmonicNoise(f0FrameCount: noiseFrames, seed: request.seed)
    }
    defer { noiseGroup.wait() }
    // Everything from here on uses bucket models (and may load or evict them):
    // executeKokoroSynthesisPipelined holds the next chunk here until the
    // previous chunk has finished.
    beforeBucketStages?()
    try modelProvider.prepareForBucket(bucketSec: bucketSec, tFrames: tFrames)
    let f0nModel: StagePredictor = try modelProvider.splitF0N(tFrames: tFrames)
        ?? modelProvider.f0ntrainModel(tFrames: tFrames)
    let f0Path = f0nModel is SplitF0N ? "split" : "whole"
    let enPadded = try zeroPad3D(
        source: en,
        channels: PipelineConstants.hiddenDim,
        targetTime: tFrames
    )
    let sArray = try makeZeroArray2D(dim: PipelineConstants.styleDim)
    let sPtr = sArray.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<PipelineConstants.styleDim {
        sPtr[i] = request.refS[PipelineConstants.baselineDim + i]
    }

    try tensorDump?.writeMLMultiArray(name: "en_padded", array: enPadded)
    try tensorDump?.writeMLMultiArray(name: "s", array: sArray)

    let f0nInput = try stageInputs(for: f0nModel, [
        "en": MLFeatureValue(multiArray: enPadded),
        "s": MLFeatureValue(multiArray: sArray),
    ], validFrames: frames, totalFrames: tFrames)
    let f0nOutput = try f0nModel.prediction(from: f0nInput)
    let f0PredArray = f0nOutput.featureValue(for: "F0_pred")!.multiArrayValue!
    let nPredArray = f0nOutput.featureValue(for: "N_pred")!.multiArrayValue!
    let t7 = CFAbsoluteTimeGetCurrent()
    timings.f0ntrainCoreML = t7 - t6
    logFootprint("f0ntrain")

    let f0Curve = floatValues(from: f0PredArray)
    let nCurve = floatValues(from: nPredArray)

    try tensorDump?.writeFloatArray(name: "f0", values: f0Curve, shape: [1, f0Curve.count])
    try tensorDump?.writeFloatArray(name: "n", values: nCurve, shape: [1, nCurve.count])

    // Stage 5: geometry. Decoder-pre stays bucketed on the Neural Engine, so
    // its inputs are zero-padded to the bucket and masked; the flexible
    // generator below runs on the real length. Frame counts: `frames` at 40 Hz
    // (asr, decoder-pre input), `2 * frames` at 80 Hz (f0, n, x_pre).
    let t8 = CFAbsoluteTimeGetCurrent()
    // A folded ANE generator replaces the GPU ones: do not open the flexible
    // program (or any GPU generator) when the bucket has one.
    let hybridGen = try resolveHybridGenerator(modelProvider, bucketSec: bucketSec, xPreFrames: 2 * frames)
    var foldedGen = hybridGen == nil ? try modelProvider.foldedGeneratorModel(bucketSec: bucketSec) : nil
    let flexibleGen = hybridGen == nil && foldedGen == nil ? try modelProvider.flexibleGeneratorModel() : nil
    let bucketSamples = bucketSec * PipelineConstants.sampleRate
    let fullF0Len = Int(round(Double(bucketSamples) / Double(HarmonicConstants.upsampleScale)))
    let frameCount = decoderPreFrameCount(fullF0Len: fullF0Len)
    let f0Padded = zeroPad1D(source: f0Curve, targetLength: fullF0Len)
    let nPadded = zeroPad1D(source: nCurve, targetLength: fullF0Len)
    let asrPadded = try zeroPad3D(
        source: asr,
        channels: PipelineConstants.textEncoderDim,
        targetTime: frameCount
    )
    let t9 = CFAbsoluteTimeGetCurrent()
    timings.padding = t9 - t8

    try tensorDump?.writeFloatArray(name: "f0_padded", values: f0Padded, shape: [1, fullF0Len])
    try tensorDump?.writeFloatArray(name: "n_padded", values: nPadded, shape: [1, fullF0Len])
    try tensorDump?.writeMLMultiArray(name: "asr_padded", array: asrPadded)

    // The flexible generator takes x_pre and har at the real length rounded up
    // to the granule (and the program's bounds); har is then built from the F0
    // curve at that length, and stageInputs masks the padded tail.
    let genXPreTime = try flexibleGen.flatMap { try flexibleXPreTime($0, realFrames: 2 * frames) }
    let harF0 = genXPreTime.map { zeroPad1D(source: f0Curve, targetLength: $0) } ?? f0Padded

    // Stage 7 starts here: the hn-nsf harmonic source needs only the F0 curve,
    // so it runs on a background thread while decoder-pre (stage 6) predicts on
    // the Neural Engine. `decoderPreHnsfOverlap` records the shared span so the stage sum
    // still matches the wall time.
    let wantDebug = tensorDump != nil
    let harGroup = DispatchGroup()
    var harResult: (har: [Float], nFrames: Int, debug: HarDebugComponents?, start: Double, end: Double) = ([], 0, nil, 0, 0)
    let harQueue = DispatchQueue(label: "kokoro.hnsf", qos: .userInitiated)
    harQueue.async(group: harGroup) {
        let start = CFAbsoluteTimeGetCurrent()
        noiseGroup.wait()
        if wantDebug {
            let components = buildHarComponents(f0Padded: harF0, linearWeights: linearWeights, linearBias: linearBias,
                                                seed: request.seed, noise: noiseBox.take())
            harResult = (components.har, components.nFrames, components, start, CFAbsoluteTimeGetCurrent())
        } else {
            let built = buildHar(f0Padded: harF0, linearWeights: linearWeights, linearBias: linearBias,
                                 seed: request.seed, noise: noiseBox.take())
            harResult = (built.har, built.nFrames, nil, start, CFAbsoluteTimeGetCurrent())
        }
    }
    // A throw below must not return while the harmonic source still writes
    // `harResult`; waiting twice on a finished group returns at once.
    defer { harGroup.wait() }

    // Stage 6: DecoderPre Core ML.
    let t10 = CFAbsoluteTimeGetCurrent()
    let decPreModel = try modelProvider.decoderPreModel(bucketSec: bucketSec)
    let f0Array3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
    copyInto(array: f0Array3D, from: f0Padded)
    let nArray3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
    copyInto(array: nArray3D, from: nPadded)
    let decRefS = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    copyInto(array: decRefS, from: request.refS)

    let decPreInput = try stageInputs(for: decPreModel, [
        "asr": MLFeatureValue(multiArray: asrPadded),
        "f0": MLFeatureValue(multiArray: f0Array3D),
        "n_input": MLFeatureValue(multiArray: nArray3D),
        "ref_s": MLFeatureValue(multiArray: decRefS),
    ], validFrames: frames, totalFrames: frameCount)
    let decPreOutput = try decPreModel.prediction(from: decPreInput)
    let xPre = decPreOutput.featureValue(for: "x_pre")!.multiArrayValue!
    let t11 = CFAbsoluteTimeGetCurrent()
    timings.decoderPre = t11 - t10
    logFootprint("decoder_pre")

    try tensorDump?.writeMLMultiArray(name: "x_pre", array: xPre)

    // Stage 7: join the hn-nsf harmonic source.
    harGroup.wait()
    let harFlat = harResult.har
    let harFrames = harResult.nFrames
    let harDebug = harResult.debug
    timings.hnsfSwift = harResult.end - harResult.start
    timings.decoderPreHnsfOverlap = max(0, min(t11, harResult.end) - max(t10, harResult.start))

    if let harDebug {
        try tensorDump?.writeFloatArray(
            name: "har_source",
            values: harDebug.harSource,
            shape: [1, harDebug.harSource.count]
        )
        try tensorDump?.writeFloatArray(
            name: "har_magnitude",
            values: harDebug.magnitude,
            shape: [1, 11, harDebug.nFrames]
        )
        try tensorDump?.writeFloatArray(
            name: "har_phase",
            values: harDebug.phase,
            shape: [1, 11, harDebug.nFrames]
        )
    }
    try tensorDump?.writeFloatArray(name: "har", values: harFlat, shape: [1, 22, harFrames])

    // Stage 8 (Neural Engine): the row-folded generator, when the bundle has one.
    let originalF0Len = frames * 2
    let targetLen = Int(
        round(Double(originalF0Len) / PipelineConstants.f0FrameRate * Double(PipelineConstants.sampleRate))
    )
    var hybridAudio: [Float]?
    var hybridStart = CFAbsoluteTimeGetCurrent()
    if #available(macOS 15.0, iOS 18.0, *), let hybrid = hybridGen as? HybridGenerator {
        hybridStart = CFAbsoluteTimeGetCurrent()
        do {
            hybridSynthesisLock.lock()
            defer { hybridSynthesisLock.unlock() }
            hybridAudio = try hybrid.synthesize(xPre: xPre, validXPreFrames: originalF0Len, har: harFlat,
                                                harFrames: harFrames, refS: request.refS)
        } catch {
            // Any hybrid failure (no GPU in the iOS background, an ANE predict
            // error): this clip and the rest of the process use the folded
            // generator. The folded generator is the hybrid's only fallback (the
            // SDK requires it whenever hybrid/ ships); without one the error stands.
            try hybridFailed(error)
            guard let folded = try modelProvider.foldedGeneratorModel(bucketSec: bucketSec) else { throw error }
            foldedGen = folded
        }
    }
    if #available(macOS 15.0, iOS 18.0, *), var audio = hybridAudio, let hybrid = hybridGen as? HybridGenerator {
        let t14 = hybridStart
        timings.generatorCoreML = CFAbsoluteTimeGetCurrent() - t14
        logFootprint("generator_hybrid")
        if audio.count > targetLen { audio.removeLast(audio.count - targetLen) }
        try tensorDump?.writeFloatArray(name: "waveform", values: audio, shape: [audio.count])
        return SynthesisResult(
            audio: audio,
            timings: timings,
            bucketSeconds: bucketSec,
            audioDurationSeconds: Double(originalF0Len) / PipelineConstants.f0FrameRate,
            wallTimeSeconds: CFAbsoluteTimeGetCurrent() - t0,
            predictedDurationFrames: frames,
            predictedDurationTokens: predDur.count,
            durationModelCacheKey: durationChoice.cacheKey,
            durationModelAllowsPadding: durationChoice.allowsPadding,
            durationTokenLength: durationChoice.tokenLength,
            tFrames: tFrames,
            fullF0Length: fullF0Len,
            decoderFrameCount: frameCount,
            xPreExpectedTime: hybrid.geometry.w0,
            harExpectedTime: hybrid.geometry.w0 * FoldGeometry.stage2Rate + hybrid.plan.extra2,
            trimSampleCount: audio.count,
            tokenDurationFrames: predDur,
            durationPath: durationPath,
            f0Path: f0Path,
            generatorPath: "hybrid"
        )
    }
    if let foldedGen, let geometry = FoldGeometry(model: foldedGen) {
        let t14 = CFAbsoluteTimeGetCurrent()
        let input = try foldedGeneratorInputs(
            xPre: xPre, validXPreFrames: originalF0Len, har: harFlat, harFrames: harFrames,
            refS: request.refS, geometry: geometry
        )
        let output = try foldedGen.prediction(from: input)
        let logits = output.featureValue(for: "logits")!.multiArrayValue!
        let t15 = CFAbsoluteTimeGetCurrent()
        timings.generatorCoreML = t15 - t14
        logFootprint("generator_folded")
        let t16 = CFAbsoluteTimeGetCurrent()
        var audio = foldedGeneratorAudio(logits: logits, validXPreFrames: originalF0Len, geometry: geometry)
        if audio.count > targetLen { audio.removeLast(audio.count - targetLen) }
        timings.trim = CFAbsoluteTimeGetCurrent() - t16
        try tensorDump?.writeFloatArray(name: "waveform", values: audio, shape: [audio.count])
        return SynthesisResult(
            audio: audio,
            timings: timings,
            bucketSeconds: bucketSec,
            audioDurationSeconds: Double(originalF0Len) / PipelineConstants.f0FrameRate,
            wallTimeSeconds: CFAbsoluteTimeGetCurrent() - t0,
            predictedDurationFrames: frames,
            predictedDurationTokens: predDur.count,
            durationModelCacheKey: durationChoice.cacheKey,
            durationModelAllowsPadding: durationChoice.allowsPadding,
            durationTokenLength: durationChoice.tokenLength,
            tFrames: tFrames,
            fullF0Length: fullF0Len,
            decoderFrameCount: frameCount,
            xPreExpectedTime: geometry.w0,
            harExpectedTime: geometry.w2,
            trimSampleCount: audio.count,
            tokenDurationFrames: predDur,
            durationPath: durationPath,
            f0Path: f0Path,
            generatorPath: "folded"
        )
    }

    // Stage 8: GeneratorFromHar Core ML.
    let t14 = CFAbsoluteTimeGetCurrent()
    let genModel = try flexibleGen ?? modelProvider.generatorModel(bucketSec: bucketSec)
    let genRefS = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    copyInto(array: genRefS, from: request.refS)

    let genShapes = inputShapes(from: genModel)
    let xPreExpectedTime = genXPreTime ?? genShapes["x_pre"]?.last ?? xPre.shape.last!.intValue
    let harExpectedTime = genXPreTime == nil ? (genShapes["har"]?.last ?? harFrames) : harFrames
    if let genModel = flexibleGen, let harRange = flexibleTimeRange(of: genModel, input: "har"),
       !harRange.contains(harFrames) {
        throw PipelineError.modelContractMismatch(
            "generator: har has \(harFrames) frames, the flexible program accepts \(harRange)"
        )
    }
    let xPrePadded = try zeroPad3D(
        source: xPre,
        channels: xPre.shape[1].intValue,
        targetTime: xPreExpectedTime
    )
    let harPadded = try zeroPad3D(
        sourceValues: harFlat,
        channels: HarmonicConstants.harChannels,
        sourceTime: harFrames,
        targetTime: harExpectedTime
    )

    try tensorDump?.writeMLMultiArray(name: "x_pre_padded", array: xPrePadded)
    try tensorDump?.writeMLMultiArray(name: "har_padded", array: harPadded)

    // `x_pre` is at twice the decoder frame rate (decode's last block upsamples
    // 2x), so the valid region doubles with it.
    let genInput = try stageInputs(for: genModel, [
        "x_pre": MLFeatureValue(multiArray: xPrePadded),
        "ref_s": MLFeatureValue(multiArray: genRefS),
        "har": MLFeatureValue(multiArray: harPadded),
    ].merging(try flexibleMaskInputs(for: genModel, xPreTime: xPreExpectedTime, validXPreFrames: frames * 2)) { $1 },
    validFrames: frames * 2, totalFrames: xPreExpectedTime)
    let genOutput = try genModel.prediction(from: genInput)
    let t15 = CFAbsoluteTimeGetCurrent()
    timings.generatorCoreML = t15 - t14
    logFootprint("generator")

    // Stage 9: trim waveform.
    let t16 = CFAbsoluteTimeGetCurrent()
    let waveformKey = genOutput.featureNames.contains("waveform") ? "waveform" : genOutput.featureNames.first!
    let waveformArray = genOutput.featureValue(for: waveformKey)!.multiArrayValue!
    let trimLen = min(waveformArray.count, targetLen)
    let audio = floatValues(from: waveformArray, limit: trimLen)
    let t17 = CFAbsoluteTimeGetCurrent()
    timings.trim = t17 - t16

    if tensorDump != nil {
        let waveformValues = floatValues(from: waveformArray)
        try tensorDump?.writeFloatArray(
            name: "waveform_full",
            values: waveformValues,
            shape: waveformArray.shape.map { $0.intValue }
        )
        try tensorDump?.writeFloatArray(name: "waveform", values: audio, shape: [trimLen])
    }

    return SynthesisResult(
        audio: audio,
        timings: timings,
        bucketSeconds: bucketSec,
        audioDurationSeconds: Double(originalF0Len) / PipelineConstants.f0FrameRate,
        wallTimeSeconds: t17 - t0,
        predictedDurationFrames: frames,
        predictedDurationTokens: predDur.count,
        durationModelCacheKey: durationChoice.cacheKey,
        durationModelAllowsPadding: durationChoice.allowsPadding,
        durationTokenLength: durationChoice.tokenLength,
        tFrames: tFrames,
        fullF0Length: fullF0Len,
        decoderFrameCount: frameCount,
        xPreExpectedTime: xPreExpectedTime,
        harExpectedTime: harExpectedTime,
        trimSampleCount: trimLen,
        tokenDurationFrames: predDur,
        durationPath: durationPath,
        f0Path: f0Path,
        generatorPath: flexibleGen == nil ? "bucket" : "flexible"
    )
}

private func requestedTokenCount(inputIds: [Int32], attentionMask: [Int32]) -> Int {
    let maskedTokenCount = attentionMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
    return maskedTokenCount > 0 ? maskedTokenCount : inputIds.count
}

private func validTokenCount(predDurTokenCount: Int, attentionMask: [Int32]) -> Int {
    min(predDurTokenCount, attentionMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) })
}

/// Test hook: the duration model's input features, padded to `choice.tokenLength`.
public func makeDurationInputFeatures(
    inputIds: [Int32],
    attentionMask: [Int32],
    refS: [Float],
    speed: Float,
    choice: DurationModelChoice
) throws -> MLFeatureProvider {
    try buildDurationInput(inputIds: inputIds, attentionMask: attentionMask, refS: refS,
                           speed: speed, choice: choice).provider
}

private func buildDurationInput(
    inputIds: [Int32],
    attentionMask: [Int32],
    refS: [Float],
    speed: Float,
    choice: DurationModelChoice
) throws -> DurationInputBundle {
    let tokenLength = choice.tokenLength
    let idsArray = try MLMultiArray(shape: [1, NSNumber(value: tokenLength)], dataType: .int32)
    let maskArray = choice.requiresAttentionMask
        ? try MLMultiArray(shape: [1, NSNumber(value: tokenLength)], dataType: .int32)
        : nil
    let refSArray = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    let speedArray = try MLMultiArray(shape: [1], dataType: .float32)

    // MLMultiArray(shape:dataType:) does not zero its buffer. When the real
    // token count is shorter than the padded length, the tail would hold stale
    // heap (often an earlier request's ids, mask 1) and shift every duration.
    let idsPtr = idsArray.dataPointer.assumingMemoryBound(to: Int32.self)
    idsPtr.initialize(repeating: 0, count: tokenLength)
    let refSPtr = refSArray.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<min(inputIds.count, tokenLength) {
        idsPtr[i] = inputIds[i]
    }
    if let maskArray {
        let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        maskPtr.initialize(repeating: 0, count: tokenLength)
        for i in 0..<min(attentionMask.count, tokenLength) {
            maskPtr[i] = attentionMask[i]
        }
    }
    for i in 0..<min(refS.count, PipelineConstants.voiceEmbeddingDim) {
        refSPtr[i] = refS[i]
    }
    speedArray[0] = NSNumber(value: speed)

    var features: [String: MLFeatureValue] = [
        "input_ids": MLFeatureValue(multiArray: idsArray),
        "ref_s": MLFeatureValue(multiArray: refSArray),
        "speed": MLFeatureValue(multiArray: speedArray),
    ]
    if let maskArray {
        features["attention_mask"] = MLFeatureValue(multiArray: maskArray)
    }

    return DurationInputBundle(
        provider: try MLDictionaryFeatureProvider(dictionary: features),
        idsArray: idsArray,
        maskArray: maskArray,
        refSArray: refSArray,
        speedArray: speedArray
    )
}

private func writeDurationInputs(
    _ input: DurationInputBundle,
    tensorDump: inout TensorDumpWriter?
) throws {
    try tensorDump?.writeMLMultiArray(name: "tokens", array: input.idsArray)
    if let maskArray = input.maskArray {
        try tensorDump?.writeMLMultiArray(name: "attention_mask", array: maskArray)
    }
    try tensorDump?.writeMLMultiArray(name: "ref_s", array: input.refSArray)
    try tensorDump?.writeMLMultiArray(name: "speed", array: input.speedArray)
}

private func writeDurationOutputs(
    predDurArray: MLMultiArray,
    predDur: [Int],
    dArray: MLMultiArray,
    tEnArray: MLMultiArray,
    tensorDump: inout TensorDumpWriter?
) throws {
    try tensorDump?.writeMLMultiArray(name: "pred_dur", array: predDurArray)
    try tensorDump?.writeInt32Array(
        name: "pred_dur_valid",
        values: predDur.map { Int32($0) },
        shape: [1, predDur.count]
    )
    try tensorDump?.writeMLMultiArray(name: "duration_d", array: dArray)
    try tensorDump?.writeMLMultiArray(name: "duration_t_en", array: tEnArray)
}

private func probeDurationAndBucket(
    input: DurationInputBundle,
    durationModel: StagePredictor,
    modelProvider: KokoroModelProvider,
    validTokenLimit: Int,
    bucketDurationOverrideSeconds: Double?
) throws -> DurationProbe {
    let output = try durationModel.prediction(from: input.provider)
    let predDurArray = output.featureValue(for: "pred_dur")!.multiArrayValue!
    let predDur = try readDurationFrames(from: predDurArray, validCount: validTokenLimit)
    let totalFrames = predDur.reduce(0, +)
    let totalSeconds = Double(totalFrames * 2) / PipelineConstants.f0FrameRate
    guard let bucketSec = selectBucket(
        totalSeconds: bucketDurationOverrideSeconds ?? totalSeconds,
        availableBuckets: modelProvider.availableBucketSeconds()
    ) else {
        throw PipelineError.noBucketAvailable
    }
    guard let tFrames = PipelineConstants.tFramesForBucket[bucketSec] else {
        throw PipelineError.modelNotLoaded("f0ntrain bucket \(bucketSec)")
    }
    try modelProvider.prepareForBucket(bucketSec: bucketSec, tFrames: tFrames)
    let bucketSamples = bucketSec * PipelineConstants.sampleRate
    let fullF0Len = Int(round(Double(bucketSamples) / Double(HarmonicConstants.upsampleScale)))
    return DurationProbe(
        bucketSec: bucketSec,
        tFrames: tFrames,
        fullF0Len: fullF0Len,
        validFrames: totalFrames
    )
}

private func warmModels(
    probe: DurationProbe,
    durationModel: StagePredictor,
    durationInput: MLDictionaryFeatureProvider,
    modelProvider: KokoroModelProvider
) throws {
    _ = try durationModel.prediction(from: durationInput)

    let f0nModel: StagePredictor = try modelProvider.splitF0N(tFrames: probe.tFrames)
        ?? modelProvider.f0ntrainModel(tFrames: probe.tFrames)
    let warmEnArr = try makeZeroArray3D(
        channels: PipelineConstants.hiddenDim,
        time: probe.tFrames
    )
    let warmSArr = try makeZeroArray2D(dim: PipelineConstants.styleDim)
    let warmF0nIn = try stageInputs(for: f0nModel, [
        "en": MLFeatureValue(multiArray: warmEnArr),
        "s": MLFeatureValue(multiArray: warmSArr),
    ], validFrames: probe.validFrames, totalFrames: probe.tFrames)
    _ = try f0nModel.prediction(from: warmF0nIn)

    // Same program choice and lengths as the timed run.
    let decPreModel = try modelProvider.decoderPreModel(bucketSec: probe.bucketSec)
    let warmFrameCount = decoderPreFrameCount(fullF0Len: probe.fullF0Len)
    let warmF0Len = probe.fullF0Len
    let warmAsr = try makeZeroArray3D(
        channels: PipelineConstants.textEncoderDim,
        time: warmFrameCount
    )
    let warmF0 = try makeZeroArray3D(channels: 1, time: warmF0Len)
    let warmN = try makeZeroArray3D(channels: 1, time: warmF0Len)
    let warmRefS = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    let warmDecIn = try stageInputs(for: decPreModel, [
        "asr": MLFeatureValue(multiArray: warmAsr),
        "f0": MLFeatureValue(multiArray: warmF0),
        "n_input": MLFeatureValue(multiArray: warmN),
        "ref_s": MLFeatureValue(multiArray: warmRefS),
    ], validFrames: probe.validFrames, totalFrames: warmFrameCount)
    _ = try decPreModel.prediction(from: warmDecIn)

    // With a folded ANE generator, warm only that: loading the GPU generator as
    // well costs ~1 GB of footprint at 30 s and it would never run.
    if try resolveHybridGenerator(modelProvider, bucketSec: probe.bucketSec, xPreFrames: 2 * probe.validFrames) != nil {
        return  // HybridGenerator compiles and loads in its init; nothing else to warm
    }
    if let folded = try modelProvider.foldedGeneratorModel(bucketSec: probe.bucketSec) {
        var feats: [String: MLFeatureValue] = [:]
        for (name, desc) in folded.modelDescription.inputDescriptionsByName {
            guard let c = desc.multiArrayConstraint else { continue }
            let arr = try MLMultiArray(shape: c.shape, dataType: .float32)
            // Masks of ones keep the masked statistics finite; data stays zero.
            let fill: Float = name.hasPrefix("sm") || name.hasPrefix("zm") ? 1 : 0
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
            ptr.update(repeating: fill, count: arr.count)
            feats[name] = MLFeatureValue(multiArray: arr)
        }
        _ = try folded.prediction(from: MLDictionaryFeatureProvider(dictionary: feats))
        return
    }

    let flexibleGen = try modelProvider.flexibleGeneratorModel()
    let genModel = try flexibleGen ?? modelProvider.generatorModel(bucketSec: probe.bucketSec)
    let genShapes = inputShapes(from: genModel)
    var warmXPreTime: Int? = nil
    if let flexibleGen, let accepted = flexibleTimeRange(of: flexibleGen, input: "x_pre") {
        warmXPreTime = try flexibleTimeAxis(realFrames: 2 * probe.validFrames, accepted: accepted, granule: PipelineConstants.flexibleXPreGranuleFrames, stage: "generator")
    }
    var warmGenInputs: [String: MLFeatureValue] = [:]
    for (name, shape) in genShapes {
        if name.hasPrefix("mask") {
            continue
        }
        if shape.count == 3 {
            // A flexible program reports its default (largest) shape here; warm
            // it at the length the timed run will use instead.
            let time: Int
            switch (name, warmXPreTime) {
            case ("x_pre", let t?): time = t
            case ("har", let t?): time = HarmonicConstants.stftFramesPerXPreFrame * t + 1
            default: time = shape[2]
            }
            warmGenInputs[name] = MLFeatureValue(
                multiArray: try makeZeroArray3D(channels: shape[1], time: time)
            )
        } else if shape.count == 2 {
            warmGenInputs[name] = MLFeatureValue(
                multiArray: try makeZeroArray2D(dim: shape[1])
            )
        }
    }
    let warmGenFrameCount = warmXPreTime ?? genShapes["x_pre"]?.last ?? probe.validFrames * 2
    let warmGenIn = try stageInputs(
        for: genModel,
        warmGenInputs.merging(try flexibleMaskInputs(for: genModel, xPreTime: warmGenFrameCount, validXPreFrames: probe.validFrames * 2)) { $1 },
        validFrames: probe.validFrames * 2,
        totalFrames: warmGenFrameCount
    )
    _ = try genModel.prediction(from: warmGenIn)
}

private func decoderPreFrameCount(fullF0Len: Int) -> Int {
    (fullF0Len - 1) / 2 + 1
}
