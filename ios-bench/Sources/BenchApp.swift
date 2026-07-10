/// Headless iPhone benchmark app: Core ML/ANE pipeline vs MLX Swift.
///
/// On launch this app runs the bundled bakeoff inputs (3s/7s/15s/30s,
/// voice af_heart, speed 1.0 — identical token IDs and ref_s to the Mac
/// bakeoff, produced by ``scripts/prepare_swift_bench_inputs.py``) through:
///
///   - Arm "coreml": ``executeKokoroSynthesis`` from this repo's
///     KokoroPipeline package, models precompiled to .mlmodelc by Xcode.
///     Timing boundary: token IDs in → 24 kHz PCM out (same as Mac bakeoff).
///   - Arm "mlx": ``KokoroTTS.generateAudio`` from mlalma/kokoro-ios
///     (MLX Swift). Its public API takes raw text, so this arm ALSO includes
///     Misaki G2P + tokenization. Disclosed wherever results are published.
///
/// Launch arguments (all optional):
///   --arms coreml,mlx       which arms to run (default: both; ladder mode only)
///   --keys 7s,15s,30s       which buckets (default: 3s,7s,15s,30s)
///   --out results.json      output filename in Documents
///   --mode ladder|matrix    ladder (default): walk the compute-policy
///                           fallback ladder per bucket. matrix: single-stage
///                           compute-unit flips for ANE-rejection attribution
///                           (coreml arm only; see ``BenchRunner/matrixCells``).
///   --exact-duration 1      use exact-native-LSTM duration packages
///                           (kokoro_duration_exact_tN, 780 ops) instead of
///                           the padded unrolled ones (17k-134k ops); mirrors
///                           the Mac frontier rows' exact-duration path.
///
/// Results are appended to Documents/<out> after every (arm, key) pair so a
/// jetsam kill mid-run still leaves partial data. Per-stage timings come from
/// ``SynthesisResult/timings`` (StageTimings in KokoroPipeline.swift), which
/// the executor populates unconditionally on every call. Console lines
/// prefixed "BENCH:" mirror progress; "BENCHDONE" marks completion.
import SwiftUI
import CoreML
import KokoroPipeline
import KokoroSwift
import MLX
import MLXUtilsLibrary

// MARK: - Bench input (same JSON schema as swift/Sources/KokoroBenchmark)

struct BenchInput: Decodable {
    let key: String
    let text: String
    let voice: String
    let speed: Float
    let input_ids: [Int32]
    let attention_mask: [Int32]
    let ref_s: [Float]
    let num_tokens: Int
    let canonical_duration_s: Double?
}

struct HnsfWeights: Decodable {
    let linear_weights: [Float]
    let linear_bias: Float
}

// MARK: - Compute policy

/// Per-stage compute-unit policy. The published Mac Config F rows run the
/// staged policy (StageComputeUnitPolicy.staged in
/// swift/Sources/KokoroBenchmark/main.swift); `.all` is the maximal policy
/// Macs accept but both test iPhones (A14 and A17 Pro) reject at first
/// predict with ANECCompile error -9, so the ladder runner walks a fallback
/// ladder per bucket and records which policy actually produced the timings.
/// Matrix mode instead flips one stage at a time to attribute the rejection
/// (see README/Plans/kokoro-iphone-performance-v1.md, Phase 1-2).
struct StagePolicy {
    let name: String
    let duration: MLComputeUnits
    let f0n: MLComputeUnits
    let decoderPre: MLComputeUnits
    let generator: MLComputeUnits

    /// Production-shaped staged policy (decoder-pre on the ANE, everything
    /// else CPU+GPU) — the policy behind the published Mac Config F rows.
    static let staged = StagePolicy(
        name: "staged",
        duration: .cpuAndGPU, f0n: .cpuAndGPU,
        decoderPre: .cpuAndNeuralEngine, generator: .cpuAndGPU
    )

    /// Ladder order: maximal `.all` first, then production-staged (the
    /// policy the published Mac Config F rows use), then no-ANE, then
    /// CPU-only as the last resort.
    static let ladder: [StagePolicy] = [
        StagePolicy(name: "all", duration: .all, f0n: .all, decoderPre: .all, generator: .all),
        staged,
        StagePolicy(name: "cpuAndGPU", duration: .cpuAndGPU, f0n: .cpuAndGPU, decoderPre: .cpuAndGPU, generator: .cpuAndGPU),
        StagePolicy(name: "cpuOnly", duration: .cpuOnly, f0n: .cpuOnly, decoderPre: .cpuOnly, generator: .cpuOnly),
    ]
}

// MARK: - Bundle-backed model provider

/// Serves Xcode-precompiled .mlmodelc bundles to the synthesis executor.
///
/// Mirrors ModelCache in swift/Sources/KokoroBenchmark/main.swift, except
/// models are already compiled (Xcode runs coremlc at build time), so
/// loading is a plain `MLModel(contentsOf:)`. Loaded instances for other
/// buckets are evicted on bucket switch to keep the footprint small on a
/// 4 GB phone.
final class BundleModelCache: KokoroModelProvider {
    /// Buckets bundled in this app (10s omitted — no 10s bakeoff input).
    static let buckets = [3, 7, 15, 30]
    /// Padded duration token sizes bundled (cover 44/105/219/476-token inputs).
    static let durationSizes = [64, 128, 256, 512]
    /// Exact-native-LSTM duration sizes bundled, one per bench input's true
    /// token count (3s/7s/15s/30s inputs are 44/105/219/476 tokens). Opt-in
    /// via --exact-duration; semantics match
    /// KokoroPipeline.discoverDurationChoices (no attention mask, no padding).
    static let exactDurationSizes = [44, 105, 219, 476]

    let policy: StagePolicy
    /// Stage family of the most recently vended model. When an ANEF compile
    /// fails at first predict, this is the best in-process hint for which
    /// stage threw; definitive attribution comes from --mode matrix.
    private(set) var lastVendedStage: String?
    private let durationConfig: MLModelConfiguration
    private let f0nConfig: MLModelConfiguration
    private let decPreConfig: MLModelConfiguration
    private let genConfig: MLModelConfiguration
    private let choices: [DurationModelChoice]
    private var durationModels: [String: MLModel] = [:]
    private var f0nModels: [Int: MLModel] = [:]
    private var decPreModels: [Int: MLModel] = [:]
    private var genModels: [Int: MLModel] = [:]

    init(policy: StagePolicy, useExactDuration: Bool) {
        self.policy = policy
        func cfg(_ u: MLComputeUnits) -> MLModelConfiguration {
            let c = MLModelConfiguration(); c.computeUnits = u; return c
        }
        durationConfig = cfg(policy.duration)
        f0nConfig = cfg(policy.f0n)
        decPreConfig = cfg(policy.decoderPre)
        genConfig = cfg(policy.generator)
        if useExactDuration {
            choices = Self.exactDurationSizes.map { n in
                DurationModelChoice(
                    cacheKey: "exact_t\(n)",
                    tokenLength: n,
                    packageURL: Self.compiledURL("kokoro_duration_exact_t\(n)"),
                    requiresAttentionMask: false,
                    allowsPadding: false
                )
            }
        } else {
            choices = Self.durationSizes.map { n in
                DurationModelChoice(
                    cacheKey: "padded_t\(n)",
                    tokenLength: n,
                    packageURL: Self.compiledURL("kokoro_duration_t\(n)"),
                    requiresAttentionMask: true,
                    allowsPadding: true
                )
            }
        }
    }

    private static func compiledURL(_ name: String) -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            fatalError("Missing compiled model in bundle: \(name).mlmodelc — run ios-bench/prepare_resources.sh and re-run xcodegen generate")
        }
        return url
    }

    func durationModelChoices() -> [DurationModelChoice] { choices }
    func availableBucketSeconds() -> [Int] { Self.buckets }

    func durationModel(choice: DurationModelChoice) throws -> MLModel {
        lastVendedStage = "duration"
        if let m = durationModels[choice.cacheKey] { return m }
        let m = try MLModel(contentsOf: choice.packageURL, configuration: durationConfig)
        durationModels[choice.cacheKey] = m
        return m
    }

    func f0ntrainModel(tFrames: Int) throws -> MLModel {
        lastVendedStage = "f0n"
        if let m = f0nModels[tFrames] { return m }
        let m = try MLModel(contentsOf: Self.compiledURL("kokoro_f0ntrain_t\(tFrames)"), configuration: f0nConfig)
        f0nModels[tFrames] = m
        return m
    }

    func decoderPreModel(bucketSec: Int) throws -> MLModel {
        lastVendedStage = "decoderPre"
        if let m = decPreModels[bucketSec] { return m }
        let m = try MLModel(contentsOf: Self.compiledURL("kokoro_decoder_pre_\(bucketSec)s"), configuration: decPreConfig)
        decPreModels[bucketSec] = m
        return m
    }

    func generatorModel(bucketSec: Int) throws -> MLModel {
        lastVendedStage = "generator"
        if let m = genModels[bucketSec] { return m }
        let m = try MLModel(contentsOf: Self.compiledURL("kokoro_decoder_har_post_\(bucketSec)s"), configuration: genConfig)
        genModels[bucketSec] = m
        return m
    }

    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {
        f0nModels = f0nModels.filter { $0.key == tFrames }
        decPreModels = decPreModels.filter { $0.key == bucketSec }
        genModels = genModels.filter { $0.key == bucketSec }
    }
}

// MARK: - Stage timing serialization

/// One warm series over a (policy, bucket) pair: everything the results JSON
/// needs beyond raw wall times.
private struct WarmSeries {
    var wallTimes: [Double] = []
    var stageRows: [StageTimings] = []
    /// Thermal state name per call (including warmups) — fanless phones
    /// throttle, and a `serious`/`critical` row must not be promoted.
    var thermalStates: [String] = []
    var bucketSeconds: Int = 0
    var durationCacheKey: String = ""
}

/// StageTimings → JSON dict (seconds). Keys mirror the StageTimings field
/// names so iPhone rows diff directly against Mac kokoro-bench output.
private func stageDict(_ t: StageTimings) -> [String: Double] {
    [
        "duration_coreml": t.durationCoreML,
        "alignment": t.alignment,
        "matrix_ops": t.matrixOps,
        "f0ntrain_coreml": t.f0ntrainCoreML,
        "padding": t.padding,
        "decoder_pre": t.decoderPre,
        "hnsf_swift": t.hnsfSwift,
        "decoder_pre_hnsf_overlap": t.decoderPreHnsfOverlap,
        "generator_coreml": t.generatorCoreML,
        "trim": t.trim,
        "total": t.total,
    ]
}

private func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return 0 }
    let s = xs.sorted()
    return s.count % 2 == 0 ? (s[s.count / 2 - 1] + s[s.count / 2]) / 2 : s[s.count / 2]
}

/// Per-stage arrays and medians across the warm rows of a series.
private func stageSummaries(_ rows: [StageTimings]) -> (arrays: [String: [Double]], medians: [String: Double]) {
    var arrays: [String: [Double]] = [:]
    for row in rows {
        for (k, v) in stageDict(row) {
            arrays[k, default: []].append(v)
        }
    }
    return (arrays, arrays.mapValues { median($0) })
}

private func thermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

// MARK: - Runner

@MainActor
final class BenchRunner: ObservableObject {
    @Published var status = "starting…"

    static let warmups = 2
    static let iterations = 5

    /// Launch-argument access. Arms were split into separate processes after
    /// the iPhone 12 Pro (4 GB) jetsammed with both pipelines resident
    /// (signal 9 during the MLX 7s generation).
    static func argValue(_ flag: String) -> String? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
    static let arms = (argValue("--arms") ?? "coreml,mlx").split(separator: ",").map(String.init)
    static let keys = (argValue("--keys") ?? "3s,7s,15s,30s").split(separator: ",").map(String.init)
    static let outName = argValue("--out") ?? "results.json"
    static let mode = argValue("--mode") ?? "ladder"
    static let exactDuration = (argValue("--exact-duration") ?? "0") == "1"

    private var records: [[String: Any]] = []

    func log(_ s: String) {
        print("BENCH: \(s)")
        status = s
    }

    private var resultsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Self.outName)
    }

    /// Persist all records so far — called after every record so a mid-run
    /// jetsam still leaves usable partial data on disk.
    private func flush() {
        var uts = utsname(); uname(&uts)
        let hw = withUnsafeBytes(of: &uts.machine) { raw in
            String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
        }
        let payload: [String: Any] = [
            "device": UIDevice.current.model,
            "hardware": hw,  // e.g. iPhone13,3 = iPhone 12 Pro, iPhone16,2 = 15 Pro Max
            "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            "warmups": Self.warmups,
            "iterations": Self.iterations,
            "mode": Self.mode,
            "exact_duration": Self.exactDuration,
            "records": records,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .prettyPrinted]) {
            try? data.write(to: resultsURL)
        }
    }

    private func loadInput(_ key: String) throws -> BenchInput {
        let url = Bundle.main.url(forResource: key, withExtension: "json")!
        return try JSONDecoder().decode(BenchInput.self, from: Data(contentsOf: url))
    }

    func run() {
        Task.detached(priority: .userInitiated) {
            if Self.mode == "matrix" {
                do {
                    try await self.runCoreMLMatrix()
                } catch {
                    await self.log("matrix run failed: \(error)")
                    await self.record(["arm": "coreml", "key": "matrix-level", "error": String(describing: error)])
                }
            } else {
                // Arms are isolated: a Core ML failure must not block the MLX
                // arm, and vice versa. Per-bucket failures are recorded inside
                // each arm and the run continues.
                if Self.arms.contains("coreml") {
                    do {
                        try await self.runCoreMLArm()
                    } catch {
                        await self.log("coreml arm failed: \(error)")
                        await self.record(["arm": "coreml", "key": "arm-level", "error": String(describing: error)])
                    }
                }
                if Self.arms.contains("mlx") {
                    do {
                        try await self.runMLXArm()
                    } catch {
                        await self.log("mlx arm failed: \(error)")
                        await self.record(["arm": "mlx", "key": "arm-level", "error": String(describing: error)])
                    }
                }
            }
            await self.log("BENCHDONE")
            print("BENCHDONE")
        }
    }

    nonisolated private func record(_ rec: [String: Any]) async {
        await MainActor.run {
            self.records.append(rec)
            self.flush()
        }
    }

    nonisolated private func loadHnsfWeights() throws -> HnsfWeights {
        let wURL = Bundle.main.url(forResource: "hnsf_weights", withExtension: "json")!
        return try JSONDecoder().decode(HnsfWeights.self, from: Data(contentsOf: wURL))
    }

    /// One synthesis call against the given cache. Throws on Core ML failure.
    nonisolated private func synthesizeOnce(
        _ input: BenchInput, weights: HnsfWeights, cache: BundleModelCache
    ) throws -> SynthesisResult {
        var dump: TensorDumpWriter? = nil
        return try executeKokoroSynthesis(
            request: KokoroSynthesisRequest(
                inputIds: input.input_ids,
                attentionMask: input.attention_mask,
                refS: input.ref_s,
                speed: input.speed,
                seed: 42,
                warmModelsBeforeTiming: true,
                bucketDurationOverrideSeconds: input.canonical_duration_s
            ),
            modelProvider: cache,
            linearWeights: weights.linear_weights,
            linearBias: weights.linear_bias,
            tensorDump: &dump
        )
    }

    /// Warmups + recorded iterations for one (policy, bucket) pair. Throws on
    /// the first failed call; the caller decides whether to ladder-step
    /// (ladder mode) or record the cell as failed (matrix mode).
    nonisolated private func runWarmSeries(
        _ input: BenchInput, weights: HnsfWeights, cache: BundleModelCache,
        key: String, policyName: String
    ) async throws -> WarmSeries {
        var series = WarmSeries()
        for i in 0..<(Self.warmups + Self.iterations) {
            let result = try synthesizeOnce(input, weights: weights, cache: cache)
            series.thermalStates.append(thermalStateName())
            series.bucketSeconds = result.bucketSeconds
            series.durationCacheKey = result.durationModelCacheKey
            if i >= Self.warmups {
                series.wallTimes.append(result.wallTimeSeconds)
                series.stageRows.append(result.timings)
            }
            await log("coreml \(key) policy=\(policyName) iter \(i): \(String(format: "%.3f", result.wallTimeSeconds))s bucket=\(result.bucketSeconds)s gen=\(String(format: "%.3f", result.timings.generatorCoreML))s")
        }
        return series
    }

    /// Failure record shared by ladder and matrix paths: NSError fields plus
    /// the last vended stage as the in-process attribution hint.
    nonisolated private func recordFailure(
        key: String, policyName: String, cache: BundleModelCache, error: Error
    ) async {
        let ns = error as NSError
        await record([
            "arm": "coreml",
            "key": key,
            "event": "policy_failure",
            "policy": policyName,
            "last_vended_stage": cache.lastVendedStage ?? "unknown",
            "error_domain": ns.domain,
            "error_code": ns.code,
            "error_description": ns.localizedDescription,
            "error_full": String(describing: error),
            "thermal_state": thermalStateName(),
        ] as [String: Any])
    }

    /// Success record shared by ladder and matrix paths.
    nonisolated private func recordSuccess(
        key: String, policyName: String, series: WarmSeries, canonicalDuration: Double
    ) async {
        let med = median(series.wallTimes)
        let stages = stageSummaries(series.stageRows)
        await record([
            "arm": "coreml",
            "key": key,
            "compute_policy": policyName,
            "warm_times_s": series.wallTimes,
            "median_s": med,
            "stage_seconds": stages.arrays,
            "stage_medians_s": stages.medians,
            "thermal_states": series.thermalStates,
            "bucket_seconds": series.bucketSeconds,
            "duration_cache_key": series.durationCacheKey,
            "canonical_duration_s": canonicalDuration,
            "rtf": canonicalDuration > 0 && med > 0 ? med / canonicalDuration : 0,
            "error": NSNull(),
        ] as [String: Any])
    }

    // MARK: Ladder mode (default)

    nonisolated private func runCoreMLArm() async throws {
        await log("coreml: loading hnsf weights")
        let weights = try loadHnsfWeights()
        // Cache persists across buckets while a policy works; a bucket that
        // fails restarts lower on the ladder with a fresh cache.
        var ladderIndex = 0
        var cache = BundleModelCache(policy: StagePolicy.ladder[ladderIndex], useExactDuration: Self.exactDuration)

        for key in Self.keys {
            let input = try await MainActor.run { try self.loadInput(key) }
            var series: WarmSeries? = nil
            var failure: String? = nil

            while series == nil {
                let policy = StagePolicy.ladder[ladderIndex]
                // First iteration triggers Core ML's on-device E5/ANE AOT
                // specialization. On the Mac bakeoff the 30s bucket spent
                // ~20 min here (README/Notes/external-bakeoff-phase2-run-log.md);
                // expect longer on A14. Silence after this line = compiler.
                await log("coreml \(key) policy=\(policy.name): loading (first load can take many minutes — ANE AOT compile)")
                do {
                    series = try await runWarmSeries(input, weights: weights, cache: cache, key: key, policyName: policy.name)
                } catch {
                    await log("coreml \(key) policy=\(policy.name) FAILED: \(error)")
                    await recordFailure(key: key, policyName: policy.name, cache: cache, error: error)
                    if ladderIndex + 1 < StagePolicy.ladder.count {
                        ladderIndex += 1
                        cache = BundleModelCache(policy: StagePolicy.ladder[ladderIndex], useExactDuration: Self.exactDuration)
                    } else {
                        failure = String(describing: error)
                        break
                    }
                }
            }

            let dur = input.canonical_duration_s ?? 0
            if let series {
                await recordSuccess(key: key, policyName: StagePolicy.ladder[ladderIndex].name, series: series, canonicalDuration: dur)
            } else {
                await record([
                    "arm": "coreml",
                    "key": key,
                    "compute_policy": StagePolicy.ladder[ladderIndex].name,
                    "canonical_duration_s": dur,
                    "error": failure ?? "unknown",
                ] as [String: Any])
            }
        }
    }

    // MARK: Matrix mode (--mode matrix)

    /// Single-stage compute-unit flips against the staged baseline, used to
    /// attribute the iOS `.all` ANEF rejection to a specific stage and to
    /// test whether decoder-pre's ANE pin is real on the phone:
    ///   staged            — baseline
    ///   duration=ne       — duration → CPU+NE (padded or exact per flag)
    ///   f0n=ne            — F0Ntrain → CPU+NE
    ///   decoderPre=gpu    — decoder-pre ANE pin REMOVED (inverse probe)
    ///   generator=ne      — generator → CPU+NE (3s body axis 14,401 fits the
    ///                       16,384 ANE cap; 30s does not — 3s vs 30s failure
    ///                       parity tests enforcement granularity)
    ///   cpuOnly           — floor reference
    /// No ladder fallback: a failed cell IS the data point.
    nonisolated static func matrixCells(allowedKeys: [String]) -> [(policy: StagePolicy, keys: [String])] {
        let base = allowedKeys.contains("3s") ? ["3s"] : Array(allowedKeys.prefix(1))
        let generatorKeys = base + (allowedKeys.contains("30s") ? ["30s"] : [])
        func flip(_ name: String, d: MLComputeUnits = .cpuAndGPU, f: MLComputeUnits = .cpuAndGPU,
                  p: MLComputeUnits = .cpuAndNeuralEngine, g: MLComputeUnits = .cpuAndGPU) -> StagePolicy {
            StagePolicy(name: name, duration: d, f0n: f, decoderPre: p, generator: g)
        }
        return [
            (StagePolicy.staged, base),
            (flip("duration=ne", d: .cpuAndNeuralEngine), base),
            (flip("f0n=ne", f: .cpuAndNeuralEngine), base),
            (flip("decoderPre=gpu", p: .cpuAndGPU), base),
            (flip("generator=ne", g: .cpuAndNeuralEngine), generatorKeys),
            (flip("cpuOnly", d: .cpuOnly, f: .cpuOnly, p: .cpuOnly, g: .cpuOnly), base),
        ]
    }

    nonisolated private func runCoreMLMatrix() async throws {
        await log("matrix: loading hnsf weights")
        let weights = try loadHnsfWeights()
        for cell in Self.matrixCells(allowedKeys: Self.keys) {
            for key in cell.keys {
                let input = try await MainActor.run { try self.loadInput(key) }
                // Fresh cache per cell+bucket: no state leaks across cells,
                // and only one bucket's models are resident on the 4 GB phone.
                let cache = BundleModelCache(policy: cell.policy, useExactDuration: Self.exactDuration)
                await log("matrix \(key) policy=\(cell.policy.name): loading (first load can take many minutes — ANE AOT compile)")
                do {
                    let series = try await runWarmSeries(input, weights: weights, cache: cache, key: key, policyName: cell.policy.name)
                    await recordSuccess(key: key, policyName: cell.policy.name, series: series, canonicalDuration: input.canonical_duration_s ?? 0)
                } catch {
                    await log("matrix \(key) policy=\(cell.policy.name) FAILED: \(error)")
                    await recordFailure(key: key, policyName: cell.policy.name, cache: cache, error: error)
                }
            }
        }
    }

    // MARK: MLX arm

    nonisolated private func runMLXArm() async throws {
        // Cap MLX's GPU buffer cache so warm iterations don't accumulate
        // freed buffers — on the 4 GB iPhone 12 Pro the uncapped cache plus
        // resident Core ML models jetsammed the process (signal 9).
        MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)
        await log("mlx: loading kokoro-v1_0.safetensors")
        let modelURL = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")!
        let voicesURL = Bundle.main.url(forResource: "voices", withExtension: "npz")!
        let tts = KokoroTTS(modelPath: modelURL)
        let voices = NpyzReader.read(fileFromPath: voicesURL) ?? [:]
        guard let voice = voices["af_heart.npy"] else {
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "af_heart voice missing from voices.npz"])
        }

        for key in Self.keys {
            let input = try await MainActor.run { try self.loadInput(key) }
            var times: [Double] = []
            var audioSeconds = 0.0
            var failure: String? = nil
            for i in 0..<(Self.warmups + Self.iterations) {
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let (audio, _) = try tts.generateAudio(
                        voice: voice, language: .enUS, text: input.text, speed: input.speed
                    )
                    let t1 = CFAbsoluteTimeGetCurrent()
                    audioSeconds = Double(audio.count) / 24000.0
                    if i >= Self.warmups { times.append(t1 - t0) }
                    await log("mlx \(key) iter \(i): \(String(format: "%.3f", t1 - t0))s audio=\(String(format: "%.1f", audioSeconds))s")
                } catch {
                    failure = String(describing: error)
                    await log("mlx \(key) iter \(i) FAILED: \(error)")
                    break
                }
            }
            let med = median(times)
            await record([
                "arm": "mlx",
                "key": key,
                "warm_times_s": times,
                "median_s": med,
                "observed_audio_s": audioSeconds,
                "rtf": audioSeconds > 0 && med > 0 ? med / audioSeconds : 0,
                "error": failure ?? NSNull(),
            ] as [String: Any])
        }
    }
}

// MARK: - App shell

@main
struct KokoroIPhoneBenchApp: App {
    @StateObject private var runner = BenchRunner()

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 16) {
                Text("Kokoro iPhone Bench").font(.headline)
                Text(runner.status).font(.caption).multilineTextAlignment(.center)
            }
            .padding()
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true
                runner.run()
            }
        }
    }
}
