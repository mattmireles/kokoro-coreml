import CoreML
import SwiftUI
import UIKit

struct BenchInput: Codable, Sendable {
    let key: String
    let text: String
    let voice: String
    let speed: Float
    let inputIds: [Int32]
    let attentionMask: [Int32]
    let refS: [Float]
    let numTokens: Int
    let canonicalDurationS: Double?
    let textSHA256: String?

    enum CodingKeys: String, CodingKey {
        case key
        case text
        case voice
        case speed
        case inputIds = "input_ids"
        case attentionMask = "attention_mask"
        case refS = "ref_s"
        case numTokens = "num_tokens"
        case canonicalDurationS = "canonical_duration_s"
        case textSHA256 = "text_sha256"
    }
}

struct HnsfWeights: Codable, Sendable {
    let linearWeights: [Float]
    let linearBias: Float

    enum CodingKeys: String, CodingKey {
        case linearWeights = "linear_weights"
        case linearBias = "linear_bias"
    }
}

struct ConfigFBenchmarkPayload: Codable, Sendable {
    let impl: String
    let framework: String
    let hardwareTarget: String
    let computeUnits: String
    let preflightDiscardedRuns: Int
    let warmIterations: Int
    let records: [ConfigFBenchmarkRecord]

    enum CodingKeys: String, CodingKey {
        case impl
        case framework
        case hardwareTarget = "hardware_target"
        case computeUnits = "compute_units"
        case preflightDiscardedRuns = "preflight_discarded_runs"
        case warmIterations = "warm_iterations"
        case records
    }
}

struct ConfigFBenchmarkRecord: Codable, Sendable {
    let inputKey: String
    let textSHA256: String
    let voice: String
    let canonicalAudioDurationS: Double
    let expectedBucketS: Int
    let postPreflightColdWallTimeS: Double
    let warmWallTimesS: [Double]
    let observedAudioDurationS: Double
    let sampleCount: Int
    let bucketUsed: String
    let durationModel: String
    let stageMediansS: StageMedianRecord
    let rawWarmStageTimingsS: [StageTimingRecord]

    enum CodingKeys: String, CodingKey {
        case inputKey = "input_key"
        case textSHA256 = "text_sha256"
        case voice
        case canonicalAudioDurationS = "canonical_audio_duration_s"
        case expectedBucketS = "expected_bucket_s"
        case postPreflightColdWallTimeS = "post_preflight_cold_wall_time_s"
        case warmWallTimesS = "warm_wall_times_s"
        case observedAudioDurationS = "observed_audio_duration_s"
        case sampleCount = "sample_count"
        case bucketUsed = "bucket_used"
        case durationModel = "duration_model"
        case stageMediansS = "stage_medians_s"
        case rawWarmStageTimingsS = "raw_warm_stage_timings_s"
    }
}

struct StageTimingRecord: Codable, Sendable {
    let durationCoreML: Double
    let alignment: Double
    let matrixOps: Double
    let f0ntrainCoreML: Double
    let padding: Double
    let decoderPre: Double
    let hnsfSwift: Double
    let decoderPreHnsfOverlap: Double
    let generatorCoreML: Double
    let trim: Double
    let total: Double

    enum CodingKeys: String, CodingKey {
        case durationCoreML = "duration_coreml"
        case alignment
        case matrixOps = "matrix_ops"
        case f0ntrainCoreML = "f0ntrain_coreml"
        case padding
        case decoderPre = "decoder_pre"
        case hnsfSwift = "hnsf_swift"
        case decoderPreHnsfOverlap = "decoder_pre_hnsf_overlap"
        case generatorCoreML = "generator_coreml"
        case trim
        case total
    }
}

struct StageMedianRecord: Codable, Sendable {
    let durationCoreML: Double
    let alignment: Double
    let matrixOps: Double
    let f0ntrainCoreML: Double
    let padding: Double
    let decoderPre: Double
    let hnsfSwift: Double
    let decoderPreHnsfOverlap: Double
    let generatorCoreML: Double
    let trim: Double
    let total: Double

    enum CodingKeys: String, CodingKey {
        case durationCoreML = "duration_coreml"
        case alignment
        case matrixOps = "matrix_ops"
        case f0ntrainCoreML = "f0ntrain_coreml"
        case padding
        case decoderPre = "decoder_pre"
        case hnsfSwift = "hnsf_swift"
        case decoderPreHnsfOverlap = "decoder_pre_hnsf_overlap"
        case generatorCoreML = "generator_coreml"
        case trim
        case total
    }
}

@main
struct ConfigFIOSRunnerApp: App {
    var body: some Scene {
        WindowGroup {
            BenchmarkView()
        }
    }
}

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var status = "Ready"
    @Published var result = ""

    private let preflightRuns = 3
    private let warmIterations = 5
    private let inputKeys = ["3s", "7s", "10s", "15s", "30s"]

    func run() {
        status = "Running"
        result = ""
        let preflightRuns = self.preflightRuns
        let warmIterations = self.warmIterations
        let inputKeys = self.inputKeys

        Task {
            do {
                let payload = try await Task.detached(priority: .userInitiated) {
                    try runConfigFBenchmark(
                        inputKeys: inputKeys,
                        preflightRuns: preflightRuns,
                        warmIterations: warmIterations
                    )
                }.value
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(payload)
                let rendered = String(data: data, encoding: .utf8) ?? ""
                result = rendered
                try writeResultFile(data: data)
                UIPasteboard.general.string = rendered
                status = "Done; JSON copied and saved"
            } catch {
                status = "Failed"
                result = String(describing: error)
                UIPasteboard.general.string = result
            }
        }
    }

    private func writeResultFile(data: Data) throws {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        try data.write(to: documents.appendingPathComponent("config_f_ios_result.json"), options: .atomic)
    }
}

struct BenchmarkView: View {
    @StateObject private var model = BenchmarkViewModel()
    @State private var hasAutoStarted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Config F Kokoro")
                .font(.title2)
                .bold()
            Text(model.status)
                .font(.headline)
            HStack {
                Button("Run") {
                    model.run()
                }
                .buttonStyle(.borderedProminent)
                Button("Copy JSON") {
                    UIPasteboard.general.string = model.result
                }
                .disabled(model.result.isEmpty)
            }
            List(["3s", "7s", "10s", "15s", "30s"], id: \.self) { key in
                Text(key)
                    .font(.headline)
            }
            .frame(height: 180)
            ScrollView {
                Text(model.result)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .onAppear {
            guard !hasAutoStarted else { return }
            hasAutoStarted = true
            model.run()
        }
    }
}

final class IOSModelCache: KokoroModelProvider {
    private let bundle: Bundle
    private let durationConfig = IOSModelCache.makeConfig(.cpuAndGPU)
    private let f0nConfig = IOSModelCache.makeConfig(.cpuAndGPU)
    private let decoderPreConfig = IOSModelCache.makeConfig(.cpuAndNeuralEngine)
    private let generatorConfig = IOSModelCache.makeConfig(.cpuAndGPU)
    private let durationChoices: [DurationModelChoice]

    private var compiledDuration: [String: URL] = [:]
    private var compiledF0n: [Int: URL] = [:]
    private var compiledDecPre: [Int: URL] = [:]
    private var compiledGen: [Int: URL] = [:]

    private var durationModels: [String: MLModel] = [:]
    private var f0nModels: [Int: MLModel] = [:]
    private var decPreModels: [Int: MLModel] = [:]
    private var genModels: [Int: MLModel] = [:]

    init(bundle: Bundle = .main) throws {
        self.bundle = bundle
        guard let resourceURL = bundle.resourceURL else {
            throw RunnerError.missingResource("bundle resourceURL")
        }
        self.durationChoices = KokoroPipeline.discoverDurationChoices(
            modelsDirectory: resourceURL,
            useExactDurationModels: true
        )
        guard !durationChoices.isEmpty else {
            throw RunnerError.missingResource("exact duration mlpackages")
        }
    }

    private static func makeConfig(_ computeUnits: MLComputeUnits) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        return config
    }

    func durationModelChoices() -> [DurationModelChoice] {
        durationChoices
    }

    func availableBucketSeconds() -> [Int] {
        PipelineConstants.defaultBuckets
    }

    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {
        f0nModels = f0nModels.filter { $0.key == tFrames }
        decPreModels = decPreModels.filter { $0.key == bucketSec }
        genModels = genModels.filter { $0.key == bucketSec }
    }

    func durationModel(choice: DurationModelChoice) throws -> MLModel {
        if let cached = durationModels[choice.cacheKey] { return cached }
        let compiled = try compiledDurationURL(choice: choice)
        let model = try MLModel(contentsOf: compiled, configuration: durationConfig)
        durationModels[choice.cacheKey] = model
        return model
    }

    func f0ntrainModel(tFrames: Int) throws -> MLModel {
        if let cached = f0nModels[tFrames] { return cached }
        let compiled = try compiledF0nURL(tFrames: tFrames)
        let model = try MLModel(contentsOf: compiled, configuration: f0nConfig)
        f0nModels[tFrames] = model
        return model
    }

    func decoderPreModel(bucketSec: Int) throws -> MLModel {
        if let cached = decPreModels[bucketSec] { return cached }
        let compiled = try compiledBucketURL(prefix: "kokoro_decoder_pre", bucketSec: bucketSec, cache: &compiledDecPre)
        let model = try MLModel(contentsOf: compiled, configuration: decoderPreConfig)
        decPreModels[bucketSec] = model
        return model
    }

    func generatorModel(bucketSec: Int) throws -> MLModel {
        if let cached = genModels[bucketSec] { return cached }
        let compiled = try compiledBucketURL(prefix: "kokoro_decoder_har_post", bucketSec: bucketSec, cache: &compiledGen)
        let model = try MLModel(contentsOf: compiled, configuration: generatorConfig)
        genModels[bucketSec] = model
        return model
    }

    private func compiledDurationURL(choice: DurationModelChoice) throws -> URL {
        if let cached = compiledDuration[choice.cacheKey] { return cached }
        let compiled = try MLModel.compileModel(at: choice.packageURL)
        compiledDuration[choice.cacheKey] = compiled
        return compiled
    }

    private func compiledF0nURL(tFrames: Int) throws -> URL {
        if let cached = compiledF0n[tFrames] { return cached }
        let package = try resourceURL(name: "kokoro_f0ntrain_t\(tFrames)", withExtension: "mlpackage")
        let compiled = try MLModel.compileModel(at: package)
        compiledF0n[tFrames] = compiled
        return compiled
    }

    private func compiledBucketURL(prefix: String, bucketSec: Int, cache: inout [Int: URL]) throws -> URL {
        if let cached = cache[bucketSec] { return cached }
        let package = try resourceURL(name: "\(prefix)_\(bucketSec)s", withExtension: "mlpackage")
        let compiled = try MLModel.compileModel(at: package)
        cache[bucketSec] = compiled
        return compiled
    }

    private func resourceURL(name: String, withExtension ext: String) throws -> URL {
        if let url = bundle.url(forResource: name, withExtension: ext) {
            return url
        }
        throw RunnerError.missingResource("\(name).\(ext)")
    }
}

enum RunnerError: Error, CustomStringConvertible {
    case missingResource(String)

    var description: String {
        switch self {
        case .missingResource(let name):
            return "missing resource: \(name)"
        }
    }
}

func runConfigFBenchmark(
    inputKeys: [String],
    preflightRuns: Int,
    warmIterations: Int
) throws -> ConfigFBenchmarkPayload {
    let bundle = Bundle.main
    let cache = try IOSModelCache(bundle: bundle)
    let weights = try loadWeights(bundle: bundle)
    var records: [ConfigFBenchmarkRecord] = []

    for key in inputKeys {
        let input = try loadInput(key: key, bundle: bundle)
        for _ in 0..<preflightRuns {
            _ = try runOne(input: input, weights: weights, cache: cache)
        }

        let coldStart = CFAbsoluteTimeGetCurrent()
        _ = try runOne(input: input, weights: weights, cache: cache)
        let postPreflightCold = CFAbsoluteTimeGetCurrent() - coldStart

        var warmResults: [SynthesisResult] = []
        warmResults.reserveCapacity(warmIterations)
        for _ in 0..<warmIterations {
            warmResults.append(try runOne(input: input, weights: weights, cache: cache))
        }
        guard let last = warmResults.last else {
            throw RunnerError.missingResource("warm results")
        }
        let warmTimes = warmResults.map { $0.wallTimeSeconds }
        let canonical = input.canonicalDurationS ?? last.audioDurationSeconds
        records.append(ConfigFBenchmarkRecord(
            inputKey: key,
            textSHA256: input.textSHA256 ?? "",
            voice: input.voice,
            canonicalAudioDurationS: canonical,
            expectedBucketS: last.bucketSeconds,
            postPreflightColdWallTimeS: rounded(postPreflightCold),
            warmWallTimesS: warmTimes.map(rounded),
            observedAudioDurationS: rounded(Double(last.trimSampleCount) / Double(PipelineConstants.sampleRate)),
            sampleCount: last.trimSampleCount,
            bucketUsed: "\(last.bucketSeconds)s",
            durationModel: last.durationModelCacheKey,
            stageMediansS: stageMedians(warmResults.map(\.timings)),
            rawWarmStageTimingsS: warmResults.map { stageRecord($0.timings) }
        ))
    }

    return ConfigFBenchmarkPayload(
        impl: "config-f-reference-ios",
        framework: "Swift + Core ML",
        hardwareTarget: "ANE/Core ML",
        computeUnits: "staged(duration/f0n/generator=cpuAndGPU,decoderPre=cpuAndNeuralEngine)",
        preflightDiscardedRuns: preflightRuns,
        warmIterations: warmIterations,
        records: records
    )
}

func runOne(input: BenchInput, weights: HnsfWeights, cache: IOSModelCache) throws -> SynthesisResult {
    var tensorDump: TensorDumpWriter? = nil
    return try executeKokoroSynthesis(
        request: KokoroSynthesisRequest(
            inputIds: input.inputIds,
            attentionMask: input.attentionMask,
            refS: input.refS,
            speed: input.speed,
            seed: 42,
            warmModelsBeforeTiming: true,
            bucketDurationOverrideSeconds: input.canonicalDurationS
        ),
        modelProvider: cache,
        linearWeights: weights.linearWeights,
        linearBias: weights.linearBias,
        tensorDump: &tensorDump
    )
}

func loadInput(key: String, bundle: Bundle) throws -> BenchInput {
    guard let url = bundle.url(forResource: key, withExtension: "json") else {
        throw RunnerError.missingResource("\(key).json")
    }
    return try JSONDecoder().decode(BenchInput.self, from: Data(contentsOf: url))
}

func loadWeights(bundle: Bundle) throws -> HnsfWeights {
    guard let url = bundle.url(forResource: "hnsf_weights", withExtension: "json") else {
        throw RunnerError.missingResource("hnsf_weights.json")
    }
    return try JSONDecoder().decode(HnsfWeights.self, from: Data(contentsOf: url))
}

func stageRecord(_ timings: StageTimings) -> StageTimingRecord {
    StageTimingRecord(
        durationCoreML: rounded(timings.durationCoreML),
        alignment: rounded(timings.alignment),
        matrixOps: rounded(timings.matrixOps),
        f0ntrainCoreML: rounded(timings.f0ntrainCoreML),
        padding: rounded(timings.padding),
        decoderPre: rounded(timings.decoderPre),
        hnsfSwift: rounded(timings.hnsfSwift),
        decoderPreHnsfOverlap: rounded(timings.decoderPreHnsfOverlap),
        generatorCoreML: rounded(timings.generatorCoreML),
        trim: rounded(timings.trim),
        total: rounded(timings.total)
    )
}

func stageMedians(_ timings: [StageTimings]) -> StageMedianRecord {
    StageMedianRecord(
        durationCoreML: rounded(median(timings.map(\.durationCoreML))),
        alignment: rounded(median(timings.map(\.alignment))),
        matrixOps: rounded(median(timings.map(\.matrixOps))),
        f0ntrainCoreML: rounded(median(timings.map(\.f0ntrainCoreML))),
        padding: rounded(median(timings.map(\.padding))),
        decoderPre: rounded(median(timings.map(\.decoderPre))),
        hnsfSwift: rounded(median(timings.map(\.hnsfSwift))),
        decoderPreHnsfOverlap: rounded(median(timings.map(\.decoderPreHnsfOverlap))),
        generatorCoreML: rounded(median(timings.map(\.generatorCoreML))),
        trim: rounded(median(timings.map(\.trim))),
        total: rounded(median(timings.map(\.total)))
    )
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count % 2 == 0 {
        return (sorted[middle - 1] + sorted[middle]) / 2.0
    }
    return sorted[middle]
}

func rounded(_ value: Double) -> Double {
    (value * 1_000_000).rounded() / 1_000_000
}
