/// Kokoro Swift pipeline benchmark CLI.
///
/// Called by ``scripts/bakeoff_harness.py`` as a subprocess for Config F.
///
/// **Single-shot mode** (original):
///
///     kokoro-bench --models-dir DIR --inputs-dir DIR \
///                  --hnsf-weights FILE --input-key KEY --seed 42 \
///                  [--output metrics.json] [--wav out.wav] [--dump-tensors DIR]
///
/// **Generator isolation mode** (feed previously dumped x_pre/ref_s/har):
///
///     kokoro-bench --models-dir DIR --inputs-dir DIR --hnsf-weights FILE \
///                  --generator-input-dump DIR [--warmup 3] [--iterations 10] \
///                  [--output metrics.json]
///
/// **Batch mode** (persistent subprocess — models compiled once):
///
///     kokoro-bench --models-dir DIR --inputs-dir DIR \
///                  --hnsf-weights FILE --batch
///
///     Then send JSON commands on stdin, one per line:
///       {"input_key":"3s","seed":42,"output":"/tmp/result.json","wav":"/tmp/out.wav"}
///     The binary prints "READY\n" to stdout after loading weights,
///     and "DONE\n" after each command completes.
///     Close stdin (EOF) to exit.

import Foundation
import CoreML
@_exported import KokoroPipeline

/// Human-readable label for Core ML's single-policy compute unit modes.
func computeUnitLabel(_ computeUnits: MLComputeUnits) -> String {
    switch computeUnits {
    case .all:
        return "all"
    case .cpuAndNeuralEngine:
        return "cpuAndNeuralEngine"
    case .cpuAndGPU:
        return "cpuAndGPU"
    case .cpuOnly:
        return "cpuOnly"
    @unknown default:
        return "unknown"
    }
}

/// Median of a non-empty timing sample set.
func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count % 2 == 0 {
        return (sorted[middle - 1] + sorted[middle]) / 2.0
    }
    return sorted[middle]
}

// MARK: - JSON Input

struct BenchInput: Decodable {
    let key: String
    let text: String
    let voice: String
    let speed: Float
    let input_ids: [Int32]
    let attention_mask: [Int32]
    let ref_s: [Float]
    let num_tokens: Int
    /// Canonical audio duration from the bakeoff manifest (seconds).
    /// Computed by the Python pipeline's extract_vocoder_inputs as T_f0 / 80.0.
    let canonical_duration_s: Double?
}

struct HnsfWeights: Decodable {
    let linear_weights: [Float]
    let linear_bias: Float
}

/// JSON command received on stdin in batch mode.
struct BatchCommand: Decodable {
    let input_key: String
    let seed: UInt64?
    let output: String
    let wav: String?
    let warmup: Bool?
}

// MARK: - Model Cache

/// Two-layer cache: compiled model URLs (.mlmodelc) persist across the whole
/// session so we never pay compilation twice. Loaded MLModel instances are
/// evicted when switching buckets to keep memory footprint low — critical on
/// 24GB machines where keeping all bucket models resident causes the ANE
/// scheduler to fall back to CPU.
class ModelCache: KokoroModelProvider {
    let modelsDir: URL
    let durationConfig: MLModelConfiguration
    let f0nConfig: MLModelConfiguration
    let computeUnits: MLComputeUnits
    let stagedComputeUnits: Bool
    let generatorConfig: MLModelConfiguration
    private let durationChoices: [DurationModelChoice]

    // Layer 1: Compiled URLs (persist forever — compilation is expensive)
    private var compiledDuration: [String: URL] = [:]
    private var compiledF0n: [Int: URL] = [:]
    private var compiledDecPre: [Int: URL] = [:]
    private var compiledGen: [Int: URL] = [:]
    private var compiledFolded: [URL: URL] = [:]

    // Layer 2: Loaded MLModel instances (evicted on bucket switch)
    var durationModels: [String: MLModel] = [:]  // kept across buckets (small)
    var splitDurations: [Int: SplitDuration] = [:]
    var splitF0Ns: [Int: SplitF0N] = [:]   // kept across buckets (small)
    var f0nModels: [Int: MLModel] = [:]
    var decPreModels: [Int: MLModel] = [:]
    var genModels: [Int: MLModel] = [:]
    var foldedGenModels: [Int: MLModel] = [:]
    // The flexible generator is one file; loaded once, never evicted.
    private var flexibleGen: MLModel??
    // Multifunction packages for the bucketed stages, opened (compiled) once.
    private var f0nMulti: MultifunctionPackage??
    private var preMulti: MultifunctionPackage??

    init(modelsDir: URL, computeUnits: MLComputeUnits = .all, stagedComputeUnits: Bool = false) {
        self.modelsDir = modelsDir
        self.computeUnits = computeUnits
        self.stagedComputeUnits = stagedComputeUnits
        if stagedComputeUnits {
            self.durationConfig = Self.makeConfig(.cpuAndGPU)
            self.f0nConfig = Self.makeConfig(PipelineConstants.f0ntrainComputeUnits)
            self.generatorConfig = Self.makeConfig(.cpuAndGPU)
        } else {
            self.durationConfig = Self.makeConfig(computeUnits)
            self.f0nConfig = Self.makeConfig(computeUnits)
            self.generatorConfig = Self.makeConfig(computeUnits)
        }
        self.durationChoices = KokoroPipeline.discoverDurationChoices(modelsDirectory: modelsDir)
        if stagedComputeUnits {
            fputs("  Compute units: staged (duration by size, f0n=cpuOnly, generator=folded ANE or cpuAndGPU, decoderPre=cpuAndNeuralEngine)\n", stderr)
        } else {
            fputs("  Compute units: \(computeUnitLabel(computeUnits))\n", stderr)
        }
        fputs("  Duration choices: \(durationChoices.map { $0.cacheKey }.joined(separator: ", "))\n", stderr)
    }

    /// Decoder-pre packages run on the Neural Engine under the staged
    /// policy (`PipelineConstants.decoderPreComputeUnits`); other policies use one unit.
    private func decoderPreConfig() -> MLModelConfiguration {
        Self.makeConfig(stagedComputeUnits ? PipelineConstants.decoderPreComputeUnits : computeUnits)
    }

    private static func makeConfig(_ computeUnits: MLComputeUnits) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        return config
    }

    // -- Compiled URL helpers (compile once, cache the .mlmodelc path) --

    func selectDurationModel(actualTokens: Int) throws -> DurationModelChoice {
        try KokoroPipeline.selectDurationChoice(durationChoices, actualTokens: actualTokens)
    }

    func durationModelChoices() -> [DurationModelChoice] {
        durationChoices
    }

    func availableBucketSeconds() -> [Int] {
        PipelineConstants.defaultBuckets
    }

    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {
        evictExcept(bucket: bucketSec, tFrames: tFrames)
    }

    func f0ntrainModel(tFrames: Int) throws -> MLModel {
        try f0nModel(tFrames: tFrames)
    }

    func decoderPreModel(bucketSec: Int) throws -> MLModel {
        try decoderPreModel(bucket: bucketSec)
    }

    func generatorModel(bucketSec: Int) throws -> MLModel {
        try generatorModel(bucket: bucketSec)
    }

    private var hybridGens: [Int: AnyObject] = [:]
    func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? {
        guard #available(macOS 15.0, *), HybridGenerator.enabled else { return nil }
        let dir = modelsDir.appendingPathComponent("hybrid")
        guard let rows = HybridGenerator.rows(bucketSec: bucketSec, xPreFrames: xPreFrames, directory: dir) else { return nil }
        if let g = hybridGens[rows] { return g }
        // One resident hybrid generator: with three loaded (159 function
        // models) the next load failed on an M1 with a misleading
        // "functionName must be nil" error.
        hybridGens.removeAll()
        fputs("  Loading hybrid generator rows=\(rows)...\n", stderr)
        let g = try HybridGenerator(directory: dir, rows: rows)
        hybridGens[rows] = g
        return g
    }

    func foldedGeneratorModel(bucketSec: Int) throws -> MLModel? {
        if let cached = foldedGenModels[bucketSec] { return cached }
        guard PipelineConstants.foldedGeneratorAllowed,
              let pkgURL = PipelineConstants.foldedGeneratorCandidates(bucketSec: bucketSec)
                .map({ modelsDir.appendingPathComponent($0) })
                .first(where: { FileManager.default.fileExists(atPath: $0.path) }) else { return nil }
        // Keep the compiled URL across bucket evictions: recompiling took ~10 s per switch.
        let compiled: URL
        if let url = compiledFolded[pkgURL] {
            compiled = url
        } else {
            fputs("  Compiling folded ANE generator \(bucketSec)s...\n", stderr)
            compiled = try MLModel.compileModel(at: pkgURL)
            compiledFolded[pkgURL] = compiled
        }
        fputs("  Loading folded ANE generator \(bucketSec)s...\n", stderr)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: compiled, configuration: config)
        foldedGenModels[bucketSec] = model
        return model
    }

    func flexibleGeneratorModel() -> MLModel? {
        if let loaded = flexibleGen { return loaded }
        let model = loadFlexibleGenerator()
        flexibleGen = .some(model)
        return model
    }

    /// The flexible generator runs on the GPU only (the ANE runtime rejects
    /// flexible shapes); a compute-unit policy that excludes the GPU, an OS
    /// before macOS 15, or the OS 27 gate leaves it unused so the bucket
    /// packages keep serving.
    private func loadFlexibleGenerator() -> MLModel? {
        let pkgURL = modelsDir.appendingPathComponent(PipelineConstants.flexibleGeneratorPackage)
        guard FileManager.default.fileExists(atPath: pkgURL.path) else { return nil }
        guard stagedComputeUnits || computeUnits == .cpuAndGPU || computeUnits == .all else {
            fputs("  Flexible generator present but compute units exclude the GPU; using bucket packages\n", stderr)
            return nil
        }
        let hasAnyBucketGenerator = PipelineConstants.tFramesForBucket.keys.contains {
            FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent("kokoro_decoder_har_post_\($0)s.mlpackage").path)
        }
        guard PipelineConstants.usesFlexibleGenerator(hasFlexibleGenerator: true, hasAnyBucketGenerator: hasAnyBucketGenerator) else {
            fputs("  Flexible generator gated off on OS \(PipelineConstants.flexibleGeneratorBrokenOSMajor)+ (\(PipelineConstants.flexibleGeneratorEnvKey)=1 overrides); using bucket packages\n", stderr)
            return nil
        }
        do {
            fputs("  Compiling flexible generator...\n", stderr)
            guard let model = try KokoroPipeline.loadFlexibleProgram(at: pkgURL) else {
                fputs("  Flexible generator needs macOS 15; using bucket packages\n", stderr)
                return nil
            }
            let range = flexibleTimeRange(of: model, input: "x_pre")
            fputs("  Loaded flexible generator: frames \(range.map { "\($0.lowerBound)...\($0.upperBound)" } ?? "static")\n", stderr)
            return model
        } catch {
            fputs("  Flexible generator failed to load (\(error)); using bucket packages\n", stderr)
            return nil
        }
    }

    private func compiledDurationURL(choice: DurationModelChoice) throws -> URL {
        // Keyed by package path: the functions of one multifunction package share a compile.
        if let url = compiledDuration[choice.packageURL.path] { return url }
        fputs("  Compiling duration \(choice.packageURL.lastPathComponent)...\n", stderr)
        let compiled = try MLModel.compileModel(at: choice.packageURL)
        compiledDuration[choice.packageURL.path] = compiled
        return compiled
    }

    private func f0nMultifunction() -> MultifunctionPackage? {
        if let opened = f0nMulti { return opened }
        let pkg = try? MultifunctionPackage.open(at: modelsDir.appendingPathComponent(PipelineConstants.f0ntrainMultifunctionPackage))
        if let pkg { fputs("  Opened f0ntrain multifunction package: \(pkg.functionNames.sorted())\n", stderr) }
        f0nMulti = .some(pkg)
        return pkg
    }

    private func preMultifunction() -> MultifunctionPackage? {
        if let opened = preMulti { return opened }
        let pkg = try? MultifunctionPackage.open(at: modelsDir.appendingPathComponent(PipelineConstants.decoderPreMultifunctionPackage))
        if let pkg { fputs("  Opened decoder_pre multifunction package: \(pkg.functionNames.sorted())\n", stderr) }
        preMulti = .some(pkg)
        return pkg
    }

    private func compiledF0nURL(tFrames: Int) throws -> URL {
        if let url = compiledF0n[tFrames] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_f0ntrain_t\(tFrames).mlpackage")
        fputs("  Compiling f0ntrain tFrames=\(tFrames)...\n", stderr)
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledF0n[tFrames] = compiled
        return compiled
    }

    private func compiledDecPreURL(bucket: Int) throws -> URL {
        if let url = compiledDecPre[bucket] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_decoder_pre_\(bucket)s.mlpackage")
        fputs("  Compiling decoder_pre \(bucket)s...\n", stderr)
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledDecPre[bucket] = compiled
        return compiled
    }

    private func compiledGenURL(bucket: Int) throws -> URL {
        if let url = compiledGen[bucket] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_decoder_har_post_\(bucket)s.mlpackage")
        fputs("  Compiling generator \(bucket)s...\n", stderr)
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledGen[bucket] = compiled
        return compiled
    }

    // -- Model accessors (load from cached compiled URL) --

    func durationModel(choice: DurationModelChoice) throws -> MLModel {
        if let cached = durationModels[choice.cacheKey] { return cached }
        let compiled = try compiledDurationURL(choice: choice)
        fputs("  Loading duration \(choice.cacheKey)...\n", stderr)
        // Staged mirrors the production runtime, including its per-size duration units.
        let config = Self.makeConfig(stagedComputeUnits
            ? PipelineConstants.durationComputeUnits(tokenLength: choice.tokenLength)
            : durationConfig.computeUnits)
        if let functionName = choice.functionName {
            guard #available(macOS 15.0, *) else { throw PipelineError.modelNotLoaded("\(choice.cacheKey): function \(functionName) needs macOS 15") }
            config.functionName = functionName
        }
        let model = try MLModel(contentsOf: compiled, configuration: config)
        durationModels[choice.cacheKey] = model
        return model
    }

    /// Staged mode only: the two-part duration model when its packages are in the
    /// models dir (SplitDuration.swift). Kept across buckets, like the whole model.
    func splitDuration(choice: DurationModelChoice) throws -> SplitDuration? {
        guard stagedComputeUnits, SplitDuration.allowed, choice.functionName == nil else { return nil }
        if let cached = splitDurations[choice.tokenLength] { return cached }
        let names = SplitDuration.packageNames(tokenLength: choice.tokenLength)
        let frontURL = modelsDir.appendingPathComponent(names.front), backURL = modelsDir.appendingPathComponent(names.back)
        guard FileManager.default.fileExists(atPath: frontURL.path), FileManager.default.fileExists(atPath: backURL.path) else { return nil }
        fputs("  Loading split duration t\(choice.tokenLength)...\n", stderr)
        let front = try MLModel(contentsOf: try MLModel.compileModel(at: frontURL), configuration: Self.makeConfig(SplitDuration.frontComputeUnits))
        let back = try MLModel(contentsOf: try MLModel.compileModel(at: backURL), configuration: Self.makeConfig(SplitDuration.backComputeUnits))
        let split = SplitDuration(front: front, back: back)
        splitDurations[choice.tokenLength] = split
        return split
    }

    /// Staged mode only: the two-part F0Ntrain when its packages are in the models dir.
    func splitF0N(tFrames: Int) throws -> SplitF0N? {
        guard stagedComputeUnits, SplitF0N.allowed else { return nil }
        if let cached = splitF0Ns[tFrames] { return cached }
        let names = SplitF0N.packageNames(tFrames: tFrames)
        let frontURL = modelsDir.appendingPathComponent(names.front), backURL = modelsDir.appendingPathComponent(names.back)
        guard FileManager.default.fileExists(atPath: frontURL.path), FileManager.default.fileExists(atPath: backURL.path) else { return nil }
        fputs("  Loading split f0ntrain t\(tFrames)...\n", stderr)
        let front = try MLModel(contentsOf: try MLModel.compileModel(at: frontURL), configuration: Self.makeConfig(SplitF0N.frontComputeUnits))
        let back = try MLModel(contentsOf: try MLModel.compileModel(at: backURL), configuration: Self.makeConfig(SplitF0N.backComputeUnits))
        let split = SplitF0N(front: front, back: back)
        splitF0Ns[tFrames] = split
        return split
    }

    func f0nModel(tFrames: Int) throws -> MLModel {
        if let cached = f0nModels[tFrames] { return cached }
        if let multi = f0nMultifunction(), multi.functionNames.contains(PipelineConstants.f0ntrainFunctionName(tFrames: tFrames)) {
            fputs("  Loading f0ntrain function t\(tFrames)...\n", stderr)
            let model = try multi.load(function: PipelineConstants.f0ntrainFunctionName(tFrames: tFrames), computeUnits: f0nConfig.computeUnits)
            f0nModels[tFrames] = model
            return model
        }
        let compiled = try compiledF0nURL(tFrames: tFrames)
        fputs("  Loading f0ntrain tFrames=\(tFrames)...\n", stderr)
        let model = try MLModel(contentsOf: compiled, configuration: f0nConfig)
        f0nModels[tFrames] = model
        return model
    }

    func decoderPreModel(bucket: Int) throws -> MLModel {
        if let cached = decPreModels[bucket] { return cached }
        if let multi = preMultifunction(), multi.functionNames.contains(PipelineConstants.decoderPreFunctionName(bucketSec: bucket)) {
            fputs("  Loading decoder_pre function bucket_\(bucket)s...\n", stderr)
            let model = try multi.load(function: PipelineConstants.decoderPreFunctionName(bucketSec: bucket), computeUnits: decoderPreConfig().computeUnits)
            decPreModels[bucket] = model
            return model
        }
        let compiled = try compiledDecPreURL(bucket: bucket)
        fputs("  Loading decoder_pre \(bucket)s...\n", stderr)
        let model = try MLModel(contentsOf: compiled, configuration: decoderPreConfig())
        decPreModels[bucket] = model
        return model
    }

    func generatorModel(bucket: Int) throws -> MLModel {
        if let cached = genModels[bucket] { return cached }
        let compiled = try compiledGenURL(bucket: bucket)
        fputs("  Loading generator \(bucket)s...\n", stderr)
        let model = try MLModel(contentsOf: compiled, configuration: generatorConfig)
        genModels[bucket] = model
        return model
    }

    /// Evict loaded MLModel instances for all buckets except the given one.
    /// Duration models are kept (small). Compiled URLs are preserved so
    /// reloading is just `MLModel(contentsOf:)` — no recompilation.
    func evictExcept(bucket: Int, tFrames: Int) {
        let f0nBefore = f0nModels.count
        let decBefore = decPreModels.count
        let genBefore = genModels.count

        f0nModels = f0nModels.filter { $0.key == tFrames }
        decPreModels = decPreModels.filter { $0.key == bucket }
        genModels = genModels.filter { $0.key == bucket }
        foldedGenModels = foldedGenModels.filter { $0.key == bucket }

        let evicted = (f0nBefore - f0nModels.count) + (decBefore - decPreModels.count) + (genBefore - genModels.count)
        if evicted > 0 {
            fputs("  Evicted \(evicted) loaded model(s) from other buckets (compiled URLs retained)\n", stderr)
        }
    }
}

// MARK: - WAV export (Config F listening checks)

/// Writes mono 16-bit PCM WAV at `sampleRate` Hz, peak-normalized like ``examples/example_synthesis.py``.
func writeWavMono16(path: String, samples: [Float], sampleRate: UInt32) throws {
    let n = samples.count
    var peak: Float = 1e-7
    for s in samples {
        let a = abs(s)
        if a > peak { peak = a }
    }
    var pcm = [Int16](repeating: 0, count: n)
    for i in 0..<n {
        let x = max(-1.0, min(1.0, samples[i] / peak))
        pcm[i] = Int16((x * 32767.0).rounded())
    }
    let dataSize = UInt32(n * 2)
    let byteRate = sampleRate * 2

    var d = Data()
    d.append(contentsOf: "RIFF".utf8)
    let riffChunkSize: UInt32 = 36 + dataSize
    withUnsafeBytes(of: riffChunkSize.littleEndian) { d.append(contentsOf: $0) }
    d.append(contentsOf: "WAVE".utf8)
    d.append(contentsOf: "fmt ".utf8)
    let subchunk1Size: UInt32 = 16
    withUnsafeBytes(of: subchunk1Size.littleEndian) { d.append(contentsOf: $0) }
    let audioFormat: UInt16 = 1
    withUnsafeBytes(of: audioFormat.littleEndian) { d.append(contentsOf: $0) }
    let numChannels: UInt16 = 1
    withUnsafeBytes(of: numChannels.littleEndian) { d.append(contentsOf: $0) }
    withUnsafeBytes(of: sampleRate.littleEndian) { d.append(contentsOf: $0) }
    withUnsafeBytes(of: byteRate.littleEndian) { d.append(contentsOf: $0) }
    let blockAlign: UInt16 = 2
    withUnsafeBytes(of: blockAlign.littleEndian) { d.append(contentsOf: $0) }
    let bitsPerSample: UInt16 = 16
    withUnsafeBytes(of: bitsPerSample.littleEndian) { d.append(contentsOf: $0) }
    d.append(contentsOf: "data".utf8)
    withUnsafeBytes(of: dataSize.littleEndian) { d.append(contentsOf: $0) }
    pcm.withUnsafeBytes { d.append(contentsOf: $0) }

    try d.write(to: URL(fileURLWithPath: path))
}

// MARK: - Pipeline run (shared between single-shot and batch)

/// Runs the full timed pipeline for one input. Returns the result dictionary as JSON Data.
func runPipeline(
    benchInput: BenchInput,
    inputKey: String,
    seed: UInt64,
    weights: HnsfWeights,
    cache: ModelCache,
    wavOutputPath: String? = nil,
    tensorDumpPath: String? = nil,
    durationTolerance: Double? = 0.02
) throws -> Data {
    var tensorDump: TensorDumpWriter? = nil
    if let tensorDumpPath {
        tensorDump = try TensorDumpWriter(directory: URL(fileURLWithPath: tensorDumpPath))
    }

    let result = try executeKokoroSynthesis(
        request: KokoroSynthesisRequest(
            inputIds: benchInput.input_ids,
            attentionMask: benchInput.attention_mask,
            refS: benchInput.ref_s,
            speed: benchInput.speed,
            seed: seed,
            warmModelsBeforeTiming: true,
            bucketDurationOverrideSeconds: benchInput.canonical_duration_s
        ),
        modelProvider: cache,
        linearWeights: weights.linear_weights,
        linearBias: weights.linear_bias,
        tensorDump: &tensorDump
    )

    let canonicalDur = benchInput.canonical_duration_s ?? result.audioDurationSeconds
    let observedDur = Double(result.trimSampleCount) / Double(PipelineConstants.sampleRate)

    fputs(
        "  Input: \(inputKey), tokens: \(benchInput.num_tokens)/\(result.durationTokenLength), duration=\(result.durationModelCacheKey), canonical=\(String(format: "%.1f", canonicalDur))s, bucket: \(result.bucketSeconds)s\n",
        stderr
    )

    if let wavPath = wavOutputPath {
        try writeWavMono16(
            path: wavPath,
            samples: result.audio,
            sampleRate: UInt32(PipelineConstants.sampleRate)
        )
        fputs("  WAV written: \(wavPath)\n", stderr)
    }

    let timings = result.timings
    let resultRecord: [String: Any] = [
        "config": "f",
        "input_key": inputKey,
        "status": "ok",
        "error": NSNull(),
        "wall_time_s": round(result.wallTimeSeconds * 1e6) / 1e6,
        "canonical_audio_duration_s": round(canonicalDur * 1e6) / 1e6,
        "observed_audio_duration_s": round(observedDur * 1e6) / 1e6,
        "predicted_duration_frames": result.predictedDurationFrames,
        "predicted_duration_tokens": result.predictedDurationTokens,
        "rtf_canonical": canonicalDur > 0 ? round((result.wallTimeSeconds / canonicalDur) * 1e6) / 1e6 : NSNull(),
        "rtf_observed": observedDur > 0 ? round((result.wallTimeSeconds / observedDur) * 1e6) / 1e6 : NSNull(),
        "speed_vs_realtime_canonical": result.wallTimeSeconds > 0 ? round((canonicalDur / result.wallTimeSeconds) * 100) / 100 : NSNull(),
        "bucket_used": "\(result.bucketSeconds)s",
        "t_duration_coreml_s": round(timings.durationCoreML * 1e6) / 1e6,
        "t_alignment_s": round(timings.alignment * 1e6) / 1e6,
        "t_matrix_ops_s": round(timings.matrixOps * 1e6) / 1e6,
        "t_f0ntrain_coreml_s": round(timings.f0ntrainCoreML * 1e6) / 1e6,
        "t_padding_s": round(timings.padding * 1e6) / 1e6,
        "t_decoder_pre_coreml_s": round(timings.decoderPre * 1e6) / 1e6,
        "t_hnsf_swift_s": round(timings.hnsfSwift * 1e6) / 1e6,
        "t_decoder_pre_hnsf_overlap_s": round(timings.decoderPreHnsfOverlap * 1e6) / 1e6,
        "t_coreml_predict_s": round(timings.generatorCoreML * 1e6) / 1e6,
        "t_trim_s": round(timings.trim * 1e6) / 1e6,
        "t_prefix_extract_s": NSNull(),
        "t_decoder_pre_cpu_s": NSNull(),
        "t_har_builder_cpu_s": NSNull(),
        "t_orchestration_s": NSNull(),
        "duration_model": result.durationModelCacheKey,
        "duration_model_allows_padding": result.durationModelAllowsPadding,
    ]

    if var writer = tensorDump {
        try writer.writeManifest(metadata: [
            "producer": "swift",
            "executable": "kokoro-bench",
            "input_key": inputKey,
            "text": benchInput.text,
            "voice": benchInput.voice,
            "speed": benchInput.speed,
            "seed": seed,
            "bucket_seconds": result.bucketSeconds,
            "duration_token_length": result.durationTokenLength,
            "duration_model": result.durationModelCacheKey,
            "duration_model_allows_padding": result.durationModelAllowsPadding,
            "num_tokens": benchInput.num_tokens,
            "natural_frames": result.predictedDurationFrames,
            "canonical_duration_s": benchInput.canonical_duration_s ?? NSNull(),
            "observed_audio_duration_s": observedDur,
            "t_frames": result.tFrames,
            "full_f0_len": result.fullF0Length,
            "decoder_frame_count": result.decoderFrameCount,
            "x_pre_expected_time": result.xPreExpectedTime,
            "har_expected_time": result.harExpectedTime,
            "trim_len": result.trimSampleCount,
            "hnsf_reference_command": "uv run --no-sync python scripts/validate_hnsf_swift.py generate",
        ])
        tensorDump = writer
    }

    try validateDurationAgreement(
        inputKey: inputKey,
        canonical: benchInput.canonical_duration_s,
        observed: observedDur,
        toleranceFraction: durationTolerance
    )

    return try JSONSerialization.data(withJSONObject: resultRecord, options: [.sortedKeys])
}

// MARK: - Single-shot mode

func runSingleShot(modelsDir: String, inputsDir: String, hnsfWeightsPath: String,
                    inputKey: String, seed: UInt64, outputPath: String?, wavPath: String?,
                    tensorDumpPath: String?,
                    warmupCount: Int,
                    computeUnits: MLComputeUnits = .all,
                    stagedComputeUnits: Bool = false,
                    durationTolerance: Double? = 0.02) throws {
    let weightsData = try Data(contentsOf: URL(fileURLWithPath: hnsfWeightsPath))
    let weights = try JSONDecoder().decode(HnsfWeights.self, from: weightsData)

    let inputPath = URL(fileURLWithPath: inputsDir).appendingPathComponent("\(inputKey).json")
    let inputData = try Data(contentsOf: inputPath)
    let benchInput = try JSONDecoder().decode(BenchInput.self, from: inputData)

    let cache = ModelCache(
        modelsDir: URL(fileURLWithPath: modelsDir),
        computeUnits: computeUnits,
        stagedComputeUnits: stagedComputeUnits
    )

    fputs("Loading models...\n", stderr)
    let jsonData = try runPipeline(
        benchInput: benchInput,
        inputKey: inputKey,
        seed: seed,
        weights: weights,
        cache: cache,
        wavOutputPath: wavPath,
        tensorDumpPath: tensorDumpPath,
        durationTolerance: durationTolerance
    )

    let jsonString = String(data: jsonData, encoding: .utf8)!
    if let outputPath = outputPath {
        try jsonString.write(toFile: outputPath, atomically: true, encoding: .utf8)
        fputs("Result written to: \(outputPath)\n", stderr)
    } else {
        print(jsonString)
    }
}

func runGeneratorInputDump(modelsDir: String, tensorInputDumpPath: String,
                           outputPath: String?, tensorDumpPath: String?,
                           warmupCount: Int, iterations: Int,
                           computeUnits: MLComputeUnits = .all,
                           stagedComputeUnits: Bool = false) throws {
    let measuredIterations = max(1, iterations)
    let discardedWarmups = max(0, warmupCount)
    let reader = try TensorDumpReader(directory: URL(fileURLWithPath: tensorInputDumpPath))
    let bucketSec = reader.metadata["bucket_seconds"] as? Int ?? 3
    let trimLen = reader.metadata["trim_len"] as? Int
    let cache = ModelCache(
        modelsDir: URL(fileURLWithPath: modelsDir),
        computeUnits: computeUnits,
        stagedComputeUnits: stagedComputeUnits
    )
    let genModel = try cache.generatorModel(bucket: bucketSec)

    let xPre = try reader.readFloatArray(name: "x_pre_padded")
    let refS = try reader.readFloatArray(name: "ref_s")
    let har = try reader.readFloatArray(name: "har_padded")
    let xPreArray = try makeFloatArray(shape: xPre.shape, values: xPre.values)
    let refSArray = try makeFloatArray(shape: refS.shape, values: refS.values)
    let harArray = try makeFloatArray(shape: har.shape, values: har.values)

    let naturalFrames = reader.metadata["natural_frames"] as? Int
    guard let naturalFrames else {
        throw PipelineError.modelContractMismatch("tensor dump has no natural_frames for generator mask")
    }
    let inputProvider = try stageInputs(for: genModel, [
        "x_pre": MLFeatureValue(multiArray: xPreArray),
        "ref_s": MLFeatureValue(multiArray: refSArray),
        "har": MLFeatureValue(multiArray: harArray),
    ], validFrames: naturalFrames * 2, totalFrames: xPre.shape.last ?? naturalFrames * 2)

    for _ in 0..<discardedWarmups {
        _ = try genModel.prediction(from: inputProvider)
    }

    var predictTimes: [Double] = []
    predictTimes.reserveCapacity(measuredIterations)
    var output: MLFeatureProvider?
    for _ in 0..<measuredIterations {
        let t0 = CFAbsoluteTimeGetCurrent()
        output = try genModel.prediction(from: inputProvider)
        let t1 = CFAbsoluteTimeGetCurrent()
        predictTimes.append(t1 - t0)
    }

    guard let output else {
        throw PipelineError.modelNotLoaded("generator output")
    }
    let waveformKey = output.featureNames.contains("waveform") ? "waveform" : output.featureNames.first!
    let waveformArray = output.featureValue(for: waveformKey)!.multiArrayValue!
    let outputTrimLen = min(waveformArray.count, trimLen ?? waveformArray.count)

    if let tensorDumpPath {
        var writer = try TensorDumpWriter(directory: URL(fileURLWithPath: tensorDumpPath))
        try writer.writeMLMultiArray(name: "x_pre_padded", array: xPreArray)
        try writer.writeMLMultiArray(name: "ref_s", array: refSArray)
        try writer.writeMLMultiArray(name: "har_padded", array: harArray)
        try writer.writeMLMultiArray(name: "waveform_full", array: waveformArray)
        let waveformValues = floatValues(from: waveformArray)
        let waveform = Array(waveformValues.prefix(outputTrimLen))
        try writer.writeFloatArray(name: "waveform", values: waveform, shape: [outputTrimLen])
        try writer.writeManifest(metadata: [
            "producer": "swift",
            "executable": "kokoro-bench",
            "mode": "generator-input-dump",
            "source_tensor_dump": tensorInputDumpPath,
            "bucket_seconds": bucketSec,
            "trim_len": outputTrimLen,
            "prediction_key": waveformKey,
            "warmup_count": discardedWarmups,
            "iterations": measuredIterations,
        ])
    }

    let result: [String: Any] = [
        "config": "f",
        "mode": "generator-input-dump",
        "status": "ok",
        "bucket_used": "\(bucketSec)s",
        "source_tensor_dump": tensorInputDumpPath,
        "observed_audio_duration_s": round((Double(outputTrimLen) / 24000.0) * 1e6) / 1e6,
        "warmup_count": discardedWarmups,
        "iterations": measuredIterations,
        "warm_coreml_predict_times_s": predictTimes.map { round($0 * 1e6) / 1e6 },
        "t_coreml_predict_s": round(median(predictTimes) * 1e6) / 1e6,
    ]
    let jsonString = String(
        data: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]),
        encoding: .utf8
    )!
    if let outputPath {
        try jsonString.write(toFile: outputPath, atomically: true, encoding: .utf8)
        fputs("Result written to: \(outputPath)\n", stderr)
    } else {
        print(jsonString)
    }
}

// MARK: - Batch mode

func runBatch(modelsDir: String, inputsDir: String, hnsfWeightsPath: String,
              computeUnits: MLComputeUnits = .all,
              stagedComputeUnits: Bool = false,
              durationTolerance: Double? = 0.02) throws {
    let weightsData = try Data(contentsOf: URL(fileURLWithPath: hnsfWeightsPath))
    let weights = try JSONDecoder().decode(HnsfWeights.self, from: weightsData)
    let cache = ModelCache(
        modelsDir: URL(fileURLWithPath: modelsDir),
        computeUnits: computeUnits,
        stagedComputeUnits: stagedComputeUnits
    )
    let inputsDirURL = URL(fileURLWithPath: inputsDir)

    fputs("Batch mode ready. Waiting for commands on stdin...\n", stderr)
    // Signal to parent process that we're ready to receive commands
    print("READY")
    fflush(stdout)

    while let line = readLine() {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }

        guard let cmdData = trimmed.data(using: .utf8) else {
            fputs("  Invalid UTF-8 in command\n", stderr)
            print("ERROR")
            fflush(stdout)
            continue
        }

        do {
            let cmd = try JSONDecoder().decode(BatchCommand.self, from: cmdData)
            let seed = cmd.seed ?? 42
            let isWarmup = cmd.warmup ?? false

            // Load input
            let inputPath = inputsDirURL.appendingPathComponent("\(cmd.input_key).json")
            let inputFileData = try Data(contentsOf: inputPath)
            let benchInput = try JSONDecoder().decode(BenchInput.self, from: inputFileData)

            fputs("  Processing: \(cmd.input_key) (warmup=\(isWarmup))\n", stderr)

            let jsonData = try runPipeline(
                benchInput: benchInput,
                inputKey: cmd.input_key,
                seed: seed,
                weights: weights,
                cache: cache,
                wavOutputPath: cmd.wav,
                durationTolerance: durationTolerance
            )

            let jsonString = String(data: jsonData, encoding: .utf8)!
            try jsonString.write(toFile: cmd.output, atomically: true, encoding: .utf8)

            print("DONE")
            fflush(stdout)
        } catch {
            fputs("  Error processing command: \(error)\n", stderr)
            // Write error result to output file if possible
            if let cmdData = trimmed.data(using: .utf8),
               let cmd = try? JSONDecoder().decode(BatchCommand.self, from: cmdData) {
                let errorResult: [String: Any] = [
                    "config": "f",
                    "input_key": cmd.input_key,
                    "status": "swift_error",
                    "error": String(describing: error),
                ]
                if let errData = try? JSONSerialization.data(withJSONObject: errorResult),
                   let errStr = String(data: errData, encoding: .utf8) {
                    try? errStr.write(toFile: cmd.output, atomically: true, encoding: .utf8)
                }
            }
            print("DONE")
            fflush(stdout)
        }
    }

    fputs("Batch mode: stdin closed, exiting.\n", stderr)
}

// MARK: - Hybrid generator check

/// `--hybrid-check DIR`: runs HybridGenerator on the fixed inputs written by
/// `scripts/export_hybrid_generator.py --check` (DIR/in/*.bin), prints warm
/// timings and writes DIR/hybrid_logits.bin (float32) for the parity script.
@available(macOS 15.0, *)
func runHybridCheck(dir: String, iterations: Int) throws {
    let url = URL(fileURLWithPath: dir)
    func bin(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: url.appendingPathComponent("in/\(name).bin"))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    let plan = try JSONSerialization.jsonObject(with: Data(contentsOf: url.appendingPathComponent("plan.json"))) as! [String: Any]
    let xPre = try bin("x_pre"), w0 = plan["w0"] as! Int
    let rows = xPre.count / (512 * w0)
    let t0 = Date()
    let gen = try HybridGenerator(directory: url, rows: rows)
    fputs(String(format: "  hybrid load %.1f s, rows %d\n", Date().timeIntervalSince(t0), rows), stderr)
    let har = try bin("har"), style = try bin("style")
    let sm1 = try bin("sm1"), zm1 = try bin("zm1"), sm2 = try bin("sm2"), zm2 = try bin("zm2")
    var times: [Double] = []
    var logits: MLMultiArray?
    // KOKORO_CHECK_GAP_MS / KOKORO_CHECK_CPU_MS: idle time or a 2-thread CPU busy
    // loop before each run (the generator is slower after a break from its own
    // work; README/Notes/ane-folded-generator.md, "Cold generator after a break").
    let env = ProcessInfo.processInfo.environment
    let gapMs = Double(env["KOKORO_CHECK_GAP_MS"] ?? "") ?? 0, cpuMs = Double(env["KOKORO_CHECK_CPU_MS"] ?? "") ?? 0
    for i in 0..<(iterations + 2) {
        if cpuMs > 0 {
            let end = Date().addingTimeInterval(cpuMs / 1000)
            DispatchQueue.concurrentPerform(iterations: 2) { _ in
                var x = 1.0
                while Date() < end { for _ in 0..<1000 { x = x * 1.0000001 + 1e-9 } }
                if x < 0 { print(x) }
            }
        }
        if gapMs > 0 { usleep(useconds_t(gapMs * 1000)) }
        let t = Date()
        logits = try gen.run(xPre: xPre, har: har, style: style, sm1: sm1, zm1: zm1, sm2: sm2, zm2: zm2)
        if i >= 2 { times.append(Date().timeIntervalSince(t) * 1000) }
    }
    if !gen.profile.isEmpty {
        for (k, v) in gen.profile.sorted(by: { $0.value > $1.value }) { print(String(format: "  %-24@ %7.1f ms", k as NSString, v * 1000)) }
    }
    times.sort()
    print(String(format: "hybrid generator: median %.1f ms, min %.1f ms (n=%d)", times[times.count / 2], times[0], times.count))
    let out = floatValues(from: logits!)
    try out.withUnsafeBytes { try Data($0).write(to: url.appendingPathComponent("hybrid_logits.bin")) }
}

// MARK: - Main

func main() throws {
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: "--hybrid-check"), i + 1 < args.count {
        guard #available(macOS 15.0, *) else { throw PipelineError.modelContractMismatch("--hybrid-check needs macOS 15") }
        let n = args.firstIndex(of: "--iterations").map { Int(args[$0 + 1]) ?? 5 } ?? 5
        try runHybridCheck(dir: args[i + 1], iterations: n)
        return
    }
    var modelsDir: String?
    var inputsDir: String?
    var hnsfWeightsPath: String?
    var inputKey: String?
    var outputPath: String?
    var wavPath: String?
    var tensorDumpPath: String?
    var generatorInputDumpPath: String?
    var warmupCount = 1
    var iterations = 1
    var seed: UInt64 = 42
    var batchMode = false
    var computeUnitsStr = "all"
    var durationTolerance: Double? = 0.02

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--models-dir":
            i += 1; modelsDir = args[i]
        case "--inputs-dir":
            i += 1; inputsDir = args[i]
        case "--hnsf-weights":
            i += 1; hnsfWeightsPath = args[i]
        case "--input-key":
            i += 1; inputKey = args[i]
        case "--warmup":
            i += 1; warmupCount = Int(args[i]) ?? 1
        case "--iterations":
            i += 1; iterations = Int(args[i]) ?? 1
        case "--seed":
            i += 1; seed = UInt64(args[i]) ?? 42
        case "--output":
            i += 1; outputPath = args[i]
        case "--wav":
            i += 1; wavPath = args[i]
        case "--dump-tensors":
            i += 1; tensorDumpPath = args[i]
        case "--generator-input-dump":
            i += 1; generatorInputDumpPath = args[i]
        case "--batch":
            batchMode = true
        case "--compute-units":
            i += 1; computeUnitsStr = args[i]
        case "--no-duration-check":
            // Forced-bucket sweeps render an utterance in a bucket its
            // canonical duration would not select; the length is still right.
            durationTolerance = nil
        default:
            break
        }
        i += 1
    }

    guard let modelsDir = modelsDir,
          let inputsDir = inputsDir,
          let hnsfWeightsPath = hnsfWeightsPath else {
        fputs("Usage: kokoro-bench --models-dir DIR --inputs-dir DIR --hnsf-weights FILE [--input-key KEY | --batch] [--no-duration-check]\n", stderr)
        exit(1)
    }

    // Parse compute units. `staged` mirrors the production Swift pipeline:
    // duration/F0/generator stay on CPU+GPU, while decoder-pre uses CPU+ANE.
    let computeUnits: MLComputeUnits
    let stagedComputeUnits: Bool
    switch computeUnitsStr.lowercased() {
    case "staged", "production":
        computeUnits = .all
        stagedComputeUnits = true
    case "all":
        computeUnits = .all
        stagedComputeUnits = false
    case "cpuandneuralengine":
        computeUnits = .cpuAndNeuralEngine
        stagedComputeUnits = false
    case "cpuandgpu":
        computeUnits = .cpuAndGPU
        stagedComputeUnits = false
    case "cpuonly":
        computeUnits = .cpuOnly
        stagedComputeUnits = false
    default:
        fputs("Unknown compute units: \(computeUnitsStr). Use: staged, all, cpuAndNeuralEngine, cpuAndGPU, cpuOnly\n", stderr)
        exit(1)
    }

    if batchMode {
        try runBatch(modelsDir: modelsDir, inputsDir: inputsDir, hnsfWeightsPath: hnsfWeightsPath,
                     computeUnits: computeUnits,
                     stagedComputeUnits: stagedComputeUnits,
                     durationTolerance: durationTolerance)
    } else if let generatorInputDumpPath {
        try runGeneratorInputDump(
            modelsDir: modelsDir,
            tensorInputDumpPath: generatorInputDumpPath,
            outputPath: outputPath,
            tensorDumpPath: tensorDumpPath,
            warmupCount: warmupCount,
            iterations: iterations,
            computeUnits: computeUnits,
            stagedComputeUnits: stagedComputeUnits
        )
    } else {
        guard let inputKey = inputKey else {
            fputs("Error: --input-key required in single-shot mode (or use --batch)\n", stderr)
            exit(1)
        }
        try runSingleShot(modelsDir: modelsDir, inputsDir: inputsDir, hnsfWeightsPath: hnsfWeightsPath,
                          inputKey: inputKey, seed: seed, outputPath: outputPath, wavPath: wavPath,
                          tensorDumpPath: tensorDumpPath,
                          warmupCount: warmupCount,
                          computeUnits: computeUnits,
                          stagedComputeUnits: stagedComputeUnits,
                          durationTolerance: durationTolerance)
    }
}

try main()
