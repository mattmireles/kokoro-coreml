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
///   --mode ladder|matrix|g2p  ladder (default): walk the compute-policy
///                           fallback ladder per bucket. matrix: single-stage
///                           compute-unit flips for ANE-rejection attribution
///                           (coreml arm only; see ``BenchRunner/matrixCells``).
///                           g2p: time ONLY the Misaki G2P pass per input
///                           (via ``KokoroTTS/phonemizeOnlyForBench``) to
///                           bound the raw-text-vs-pretokenized boundary
///                           asymmetry between the two arms numerically.
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
/// (see README/Plans/007-kokoro-iphone-performance-plan.md, Phase 1-2).
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
    static let fullLadder: [StagePolicy] = [
        StagePolicy(name: "all", duration: .all, f0n: .all, decoderPre: .all, generator: .all),
        staged,
        StagePolicy(name: "cpuAndGPU", duration: .cpuAndGPU, f0n: .cpuAndGPU, decoderPre: .cpuAndGPU, generator: .cpuAndGPU),
        StagePolicy(name: "cpuOnly", duration: .cpuOnly, f0n: .cpuOnly, decoderPre: .cpuOnly, generator: .cpuOnly),
    ]

    /// `--policy NAME` pins the ladder to one policy: any fullLadder name,
    /// `staged-dne` (staged with duration on CPU+ANE), or `runtime` (the
    /// placement KokoroPipeline ships: PipelineConstants duration and F0Ntrain units).
    static let ladder: [StagePolicy] = {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--policy"), i + 1 < args.count else { return fullLadder }
        let name = args[i + 1]
        if name == "runtime" {
            return [StagePolicy(name: name, duration: PipelineConstants.durationComputeUnits(tokenLength: 512),
                                f0n: PipelineConstants.f0ntrainComputeUnits,
                                decoderPre: .cpuAndNeuralEngine, generator: .cpuAndGPU)]
        }
        if name == "staged-dne" {
            return [StagePolicy(name: name, duration: .cpuAndNeuralEngine, f0n: .cpuAndGPU,
                                decoderPre: .cpuAndNeuralEngine, generator: .cpuAndGPU)]
        }
        return fullLadder.filter { $0.name == name }
    }()
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
    /// Buckets bundled in this app (the five frozen runtime buckets).
    static let buckets = [3, 7, 10, 15, 30]
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
    /// Shipped default: one flexible (RangeDim) generator on CPU+GPU, loaded
    /// once (KokoroPipeline.loadFlexibleProgram semantics). `--bucket-generator 1`
    /// restores the per-bucket generator packages.
    private var flexibleGen: MLModel??

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

    /// Two-part duration model (SplitDuration.swift), opt-in with
    /// `--split-duration`, when the bundle carries its packages.
    private var splitDurations: [Int: SplitDuration] = [:]
    func splitDuration(choice: DurationModelChoice) throws -> SplitDuration? {
        guard ProcessInfo.processInfo.arguments.contains("--split-duration"), SplitDuration.allowed,
              choice.allowsPadding else { return nil }
        lastVendedStage = "duration"
        if let d = splitDurations[choice.tokenLength] { return d }
        let names = SplitDuration.packageNames(tokenLength: choice.tokenLength)
        func compiled(_ pkg: String) -> URL? {
            Bundle.main.url(forResource: (pkg as NSString).deletingPathExtension, withExtension: "mlmodelc")
        }
        guard let f = compiled(names.front), let b = compiled(names.back) else { return nil }
        let fc = MLModelConfiguration(); fc.computeUnits = SplitDuration.frontComputeUnits
        let bc = MLModelConfiguration(); bc.computeUnits = SplitDuration.backComputeUnits
        let d = SplitDuration(front: try MLModel(contentsOf: f, configuration: fc), back: try MLModel(contentsOf: b, configuration: bc))
        splitDurations[choice.tokenLength] = d
        print("BENCH: split duration t\(choice.tokenLength) loaded")
        return d
    }

    /// Two-part F0Ntrain (SplitDuration.swift), opt-in with `--split-f0n`.
    private var splitF0Ns: [Int: SplitF0N] = [:]
    func splitF0N(tFrames: Int) throws -> SplitF0N? {
        guard ProcessInfo.processInfo.arguments.contains("--split-f0n"), SplitF0N.allowed else { return nil }
        lastVendedStage = "f0n"
        if let d = splitF0Ns[tFrames] { return d }
        let names = SplitF0N.packageNames(tFrames: tFrames)
        func compiled(_ pkg: String) -> URL? {
            Bundle.main.url(forResource: (pkg as NSString).deletingPathExtension, withExtension: "mlmodelc")
        }
        guard let f = compiled(names.front), let b = compiled(names.back) else { return nil }
        let fc = MLModelConfiguration(); fc.computeUnits = SplitF0N.frontComputeUnits
        let bc = MLModelConfiguration(); bc.computeUnits = SplitF0N.backComputeUnits
        let d = SplitF0N(front: try MLModel(contentsOf: f, configuration: fc), back: try MLModel(contentsOf: b, configuration: bc))
        splitF0Ns[tFrames] = d
        print("BENCH: split f0ntrain t\(tFrames) loaded")
        return d
    }

    /// Hybrid ANE-conv + Metal generator, opt-in with `--hybrid-generator`.
    private var hybridGens: [Int: AnyObject] = [:]
    func hybridGenerator(bucketSec: Int, xPreFrames: Int) throws -> AnyObject? {
        guard ProcessInfo.processInfo.arguments.contains("--hybrid-generator"),
              let rows = HybridGenerator.rows(bucketSec: bucketSec, xPreFrames: xPreFrames, directory: Bundle.main.bundleURL) else { return nil }
        lastVendedStage = "generator"
        if let g = hybridGens[rows] { return g }
        hybridGens.removeAll()  // one resident: several at once failed to load on an M1
        let g = try HybridGenerator(directory: Bundle.main.bundleURL, rows: rows)
        hybridGens[rows] = g
        return g
    }

    /// Row-folded Neural Engine generator, opt-in with `--folded-generator`
    /// and only when the bundle carries it (see prepare_resources.sh). The
    /// native-sin or polynomial variant is chosen like the runtime does
    /// (PipelineConstants.foldedGeneratorCandidates).
    private var foldedGens: [Int: MLModel] = [:]
    func foldedGeneratorModel(bucketSec: Int) throws -> MLModel? {
        guard ProcessInfo.processInfo.arguments.contains("--folded-generator"),
              let url = PipelineConstants.foldedGeneratorCandidates(bucketSec: bucketSec).lazy
                .compactMap({ Bundle.main.url(forResource: ($0 as NSString).deletingPathExtension, withExtension: "mlmodelc") })
                .first else { return nil }
        lastVendedStage = "generator"
        if let m = foldedGens[bucketSec] { return m }
        let c = MLModelConfiguration(); c.computeUnits = .cpuAndNeuralEngine
        let m = try MLModel(contentsOf: url, configuration: c)
        foldedGens[bucketSec] = m
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

    func flexibleGeneratorModel() throws -> MLModel? {
        if let loaded = flexibleGen { return loaded }
        guard !BenchRunner.bucketGenerator else { flexibleGen = .some(nil); return nil }
        lastVendedStage = "generator"
        let c = MLModelConfiguration(); c.computeUnits = .cpuAndGPU
        let m = try MLModel(contentsOf: Self.compiledURL("kokoro_decoder_har_post_range"), configuration: c)
        flexibleGen = .some(m)
        return m
    }

    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {
        // KOKORO_BENCH_KEEP_MODELS=1: keep every bucket's models, like the SDK provider.
        if ProcessInfo.processInfo.environment["KOKORO_BENCH_KEEP_MODELS"] == "1" { return }
        f0nModels = f0nModels.filter { $0.key == tFrames }
        decPreModels = decPreModels.filter { $0.key == bucketSec }
        genModels = genModels.filter { $0.key == bucketSec }
        foldedGens = foldedGens.filter { $0.key == bucketSec }
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
    /// First call after the discarded preflight calls (the frozen cold marker).
    var coldTime: Double = 0
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

    /// Frozen protocol: 3 discarded preflight calls, 1 cold marker, 10 warm.
    static let warmups = 3
    static let iterations = 10

    /// Launch-argument access. Arms were split into separate processes after
    /// the iPhone 12 Pro (4 GB) jetsammed with both pipelines resident
    /// (signal 9 during the MLX 7s generation).
    static func argValue(_ flag: String) -> String? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
    static let arms = (argValue("--arms") ?? "coreml,mlx").split(separator: ",").map(String.init)
    static let keys = (argValue("--keys") ?? "3s,7s,10s,15s,30s").split(separator: ",").map(String.init)
    static let outName = argValue("--out") ?? "results.json"
    static let mode = argValue("--mode") ?? "ladder"
    static let exactDuration = (argValue("--exact-duration") ?? "0") == "1"
    static let bucketGenerator = (argValue("--bucket-generator") ?? "0") == "1"

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
            "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            "warmups": Self.warmups,
            "iterations": Self.iterations,
            "mode": Self.mode,
            "exact_duration": Self.exactDuration,
            "generator": Self.bucketGenerator ? "bucket" : "flexible",
            "records": records,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .prettyPrinted]) {
            try? data.write(to: resultsURL)
        }
    }

    /// `--mode hybrid-check`: runs the hybrid generator on Documents/hc/in/*.bin
    /// (written by scripts/export_hybrid_generator.py --check) and writes
    /// Documents/hc/hybrid_logits.bin (float32) for the parity script; the
    /// on-device counterpart of `kokoro-bench --hybrid-check`.
    nonisolated private func runHybridCheck() async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("hc")
        func bin(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: docs.appendingPathComponent("in/\(name).bin"))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let plan = try JSONSerialization.jsonObject(with: Data(contentsOf: Bundle.main.bundleURL.appendingPathComponent("plan.json"))) as! [String: Any]
        let xPre = try bin("x_pre"), rows = xPre.count / (512 * (plan["w0"] as! Int))
        let gen = try HybridGenerator(directory: Bundle.main.bundleURL, rows: rows)
        let har = try bin("har"), style = try bin("style")
        let sm1 = try bin("sm1"), zm1 = try bin("zm1"), sm2 = try bin("sm2"), zm2 = try bin("zm2")
        var times: [Double] = []
        var logits: MLMultiArray?
        // KOKORO_CHECK_GAP_MS: idle time before each run, like the pipeline's
        // earlier stages (does an idle GPU/ANE slow the generator down?).
        let gapMs = Double(ProcessInfo.processInfo.environment["KOKORO_CHECK_GAP_MS"] ?? "") ?? 0
        // KOKORO_CHECK_CPU_MS: busy CPU work on 2 threads before each run, like the
        // pipeline's LSTM/hn-NSF stages. KOKORO_CHECK_ALLOC_MB: fresh Float arrays
        // written before each run, like the pipeline's host-side folds.
        let cpuMs = Double(ProcessInfo.processInfo.environment["KOKORO_CHECK_CPU_MS"] ?? "") ?? 0
        let memMs = Double(ProcessInfo.processInfo.environment["KOKORO_CHECK_MEMCPY_MS"] ?? "") ?? 0
        let memBufs: [UnsafeMutableRawPointer] = memMs > 0 ? (0..<4).map { _ in
            let p = UnsafeMutableRawPointer.allocate(byteCount: 64 << 20, alignment: 16384); memset(p, 1, 64 << 20); return p } : []
        let cpuThreads = Int(ProcessInfo.processInfo.environment["KOKORO_CHECK_CPU_THREADS"] ?? "") ?? 2
        let allocMB = Int(ProcessInfo.processInfo.environment["KOKORO_CHECK_ALLOC_MB"] ?? "") ?? 0
        // KOKORO_CHECK_RUNS: runs (default 5, the last 3 kept). Longer series show
        // the A14 drifting from boost to sustained speed (334 → ~385 ms over 25 runs).
        let runs = Int(ProcessInfo.processInfo.environment["KOKORO_CHECK_RUNS"] ?? "") ?? 5
        for i in 0..<runs {
            if cpuMs > 0 {
                let end = Date().addingTimeInterval(cpuMs / 1000)
                DispatchQueue.concurrentPerform(iterations: cpuThreads) { _ in
                    var x: Double = 1
                    while Date() < end { for _ in 0..<1000 { x = x * 1.0000001 + 1e-9 } }
                    if x < 0 { print(x) }
                }
            }
            if allocMB > 0 {
                var junk = [Float](repeating: 0, count: allocMB * 262_144)
                for k in stride(from: 0, to: junk.count, by: 16) { junk[k] = Float(k) }
                if junk[16] < 0 { print(junk[0]) }
            }
            if gapMs > 0 { usleep(useconds_t(gapMs * 1000)) }   // after any CPU load
            // KOKORO_CHECK_MEMCPY_MS: memory-bandwidth-heavy CPU work (64 MB copies on
            // 2 threads) right before the run: does DRAM demand keep the generator fast?
            if memMs > 0 {
                let end = Date().addingTimeInterval(memMs / 1000)
                DispatchQueue.concurrentPerform(iterations: 2) { k in
                    while Date() < end { memcpy(memBufs[k], memBufs[2 + k], 64 << 20) }
                }
            }
            let t = Date()
            logits = try gen.run(xPre: xPre, har: har, style: style, sm1: sm1, zm1: zm1, sm2: sm2, zm2: zm2)
            if i >= 2 { times.append(Date().timeIntervalSince(t) * 1000) }
            if ProcessInfo.processInfo.environment["KOKORO_HYBRID_TIMELINE"] == "1" { print(gen.timelineLine(), terminator: "") }
        }
        if !gen.profile.isEmpty {   // KOKORO_HYBRID_PROFILE=1: last run's per-step totals
            for (k, v) in gen.profile.sorted(by: { $0.value > $1.value }) { print(String(format: "PROFILE %@ %.2f ms", k, v * 1000)) }
        }
        let out = floatValues(from: logits!)
        try out.withUnsafeBytes { try Data($0).write(to: docs.appendingPathComponent("hybrid_logits.bin")) }
        await log("hybrid check rows \(rows): \(times.sorted().map { Int($0) }) ms, peak footprint \(Int(footprintMB().peak)) MB")
        await record(["arm": "hybrid-check", "key": "rows\(rows)", "warm_ms": times, "error": NSNull()])
    }

    /// `--mode document --keys 10s,7s,10s`: the keys as consecutive chunks of one
    /// document, run sequentially and with executeKokoroSynthesisPipelined
    /// (next chunk's duration during this chunk's generator), interleaved.
    /// Records total and first-chunk time and the largest sample difference.
    nonisolated private func runDocumentMode() async throws {
        let weights = try loadHnsfWeights()
        let cache = BundleModelCache(policy: StagePolicy.ladder[0], useExactDuration: Self.exactDuration)
        var requests: [KokoroSynthesisRequest] = []
        for key in Self.keys {
            let input = try await MainActor.run { try self.loadInput(key) }
            requests.append(KokoroSynthesisRequest(inputIds: input.input_ids, attentionMask: input.attention_mask,
                                                   refS: input.ref_s, speed: input.speed, seed: 42,
                                                   bucketDurationOverrideSeconds: input.canonical_duration_s))
        }
        func sequential() throws -> (total: Double, first: Double, audio: [[Float]]) {
            let t = Date(); var first = 0.0; var audio: [[Float]] = []
            for r in requests {
                var dump: TensorDumpWriter? = nil
                let res = try executeKokoroSynthesis(request: r, modelProvider: cache, linearWeights: weights.linear_weights,
                                                     linearBias: weights.linear_bias, tensorDump: &dump)
                if audio.isEmpty { first = Date().timeIntervalSince(t) }
                audio.append(res.audio)
            }
            return (Date().timeIntervalSince(t), first, audio)
        }
        func pipelined() throws -> (total: Double, first: Double, audio: [[Float]]) {
            let t = Date(); var first = 0.0; var audio: [[Float]] = []
            try executeKokoroSynthesisPipelined(requests: requests, modelProvider: cache, linearWeights: weights.linear_weights,
                                                linearBias: weights.linear_bias) { _, res in
                if audio.isEmpty { first = Date().timeIntervalSince(t) }
                audio.append(res.audio)
            }
            return (Date().timeIntervalSince(t), first, audio)
        }
        for _ in 0..<2 { _ = try sequential(); _ = try pipelined() }   // load + ANE compile
        var seq: [Double] = [], pipe: [Double] = [], seqFirst: [Double] = [], pipeFirst: [Double] = []
        var maxDiff: Float = 0
        for _ in 0..<5 {
            let a = try sequential(), b = try pipelined()
            seq.append(a.total * 1000); seqFirst.append(a.first * 1000)
            pipe.append(b.total * 1000); pipeFirst.append(b.first * 1000)
            for (x, y) in zip(a.audio, b.audio) {
                if x.count != y.count { maxDiff = .infinity; continue }
                for k in 0..<x.count { maxDiff = max(maxDiff, abs(x[k] - y[k])) }
            }
        }
        await log("document \(Self.keys.joined(separator: ",")): sequential \(seq.sorted().map { Int($0) }) ms (first \(seqFirst.sorted().map { Int($0) })), pipelined \(pipe.sorted().map { Int($0) }) ms (first \(pipeFirst.sorted().map { Int($0) })), max diff \(maxDiff), peak footprint \(Int(footprintMB().peak)) MB")
        await record(["arm": "document", "key": Self.keys.joined(separator: ","), "sequential_ms": seq, "pipelined_ms": pipe,
                      "sequential_first_ms": seqFirst, "pipelined_first_ms": pipeFirst, "max_abs_diff": Double(maxDiff), "error": NSNull()])
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
            } else if Self.mode == "hybrid-check" {
                do {
                    try await self.runHybridCheck()
                } catch {
                    await self.log("hybrid check failed: \(error)")
                    await self.record(["arm": "hybrid-check", "key": "mode-level", "error": String(describing: error)])
                }
            } else if Self.mode == "document" {
                do {
                    try await self.runDocumentMode()
                } catch {
                    await self.log("document run failed: \(error)")
                    await self.record(["arm": "document", "key": "mode-level", "error": String(describing: error)])
                }
            } else if Self.mode == "g2p" {
                do {
                    try await self.runG2PMode()
                } catch {
                    await self.log("g2p run failed: \(error)")
                    await self.record(["arm": "g2p", "key": "mode-level", "error": String(describing: error)])
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
        for i in 0..<(Self.warmups + 1 + Self.iterations) {
            let result = try synthesizeOnce(input, weights: weights, cache: cache)
            series.thermalStates.append(thermalStateName())
            series.bucketSeconds = result.bucketSeconds
            series.durationCacheKey = result.durationModelCacheKey
            if i == Self.warmups { series.coldTime = result.wallTimeSeconds }
            if i > Self.warmups {
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

    nonisolated private func footprintMB() -> (now: Double, peak: Double) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (0, 0) }
        return (Double(info.phys_footprint) / 1048576, Double(info.ledger_phys_footprint_peak) / 1048576)
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
            "cold_s": series.coldTime,
            "median_s": med,
            "stage_seconds": stages.arrays,
            "stage_medians_s": stages.medians,
            "thermal_states": series.thermalStates,
            // Process phys_footprint now and its lifetime peak (MB). Run one
            // key per launch for a clean per-key peak.
            "footprint_mb": footprintMB().now,
            "peak_footprint_mb": footprintMB().peak,
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

    // MARK: G2P-only mode (--mode g2p)

    /// Times ONLY the Misaki G2P pass of the MLX arm for each bench input.
    ///
    /// Purpose: the MLX arm's `generateAudio` timings include G2P (raw-text
    /// API) while the Core ML arm starts from pre-tokenized IDs. This mode
    /// measures that boundary asymmetry per input so published comparisons
    /// can subtract it numerically instead of calling it "small but nonzero".
    /// Uses the same warmup/iteration discipline as the arms; first warmup
    /// absorbs Misaki dictionary loading.
    nonisolated private func runG2PMode() async throws {
        // KokoroTTS init loads the full 327 MB weight set even though only
        // g2pProcessor is used — acceptable: this mode reuses the vendored
        // init as-is rather than forking it (SIMPLER IS BETTER).
        await log("g2p: constructing KokoroTTS (loads full weights; G2P init is what we time)")
        let modelURL = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")!
        let tts = KokoroTTS(modelPath: modelURL)
        for key in Self.keys {
            let input = try await MainActor.run { try self.loadInput(key) }
            var times: [Double] = []
            var phonemeCount = 0
            var failure: String? = nil
            for i in 0..<(Self.warmups + Self.iterations) {
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let phonemes = try tts.phonemizeOnlyForBench(language: .enUS, text: input.text)
                    let t1 = CFAbsoluteTimeGetCurrent()
                    phonemeCount = phonemes.count
                    if i >= Self.warmups { times.append(t1 - t0) }
                    await log("g2p \(key) iter \(i): \(String(format: "%.4f", t1 - t0))s phonemes=\(phonemes.count)")
                } catch {
                    failure = String(describing: error)
                    await log("g2p \(key) iter \(i) FAILED: \(error)")
                    break
                }
            }
            await record([
                "arm": "g2p",
                "key": key,
                "warm_times_s": times,
                "median_s": median(times),
                "phoneme_count": phonemeCount,
                "text_chars": input.text.count,
                "thermal_state": thermalStateName(),
                "error": failure ?? NSNull(),
            ] as [String: Any])
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
            var cold = 0.0
            var thermal: [String] = []
            var audioSeconds = 0.0
            var failure: String? = nil
            for i in 0..<(Self.warmups + 1 + Self.iterations) {
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let (audio, _) = try tts.generateAudio(
                        voice: voice, language: .enUS, text: input.text, speed: input.speed
                    )
                    let t1 = CFAbsoluteTimeGetCurrent()
                    audioSeconds = Double(audio.count) / 24000.0
                    thermal.append(thermalStateName())
                    if i == Self.warmups { cold = t1 - t0 }
                    if i > Self.warmups { times.append(t1 - t0) }
                    // Footprint after every call (same metric as the Core ML cells), so a
                    // jetsam kill still leaves the last value in the console log.
                    let fp = footprintMB()
                    await log("mlx \(key) iter \(i): \(String(format: "%.3f", t1 - t0))s audio=\(String(format: "%.1f", audioSeconds))s footprint \(Int(fp.now)) MB peak \(Int(fp.peak)) MB")
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
                "cold_s": cold,
                "thermal_states": thermal,
                "median_s": med,
                "observed_audio_s": audioSeconds,
                "rtf": audioSeconds > 0 && med > 0 ? med / audioSeconds : 0,
                "footprint_mb": footprintMB().now,
                "peak_footprint_mb": footprintMB().peak,
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
