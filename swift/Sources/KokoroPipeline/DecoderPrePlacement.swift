/// Where the bucketed stages run, decided per machine.
///
/// Three thresholds: decoder-pre buckets up to `neuralEngineMaxBucketSeconds`,
/// duration sizes up to `durationNeuralEngineMaxTokens` and f0ntrain sizes up
/// to `f0ntrainNeuralEngineMaxFrames` run on the Neural Engine, larger ones on
/// the GPU. Measured once per machine from the largest size down (the Neural
/// Engine's advantage grows as sizes shrink on both machines measured): M3 Max
/// duration to t256 by 2-4 ms and f0ntrain to t280 by a tie's width; M1
/// duration from t128 by 10-16 ms and f0ntrain to t600 by 3-6 ms. Duration
/// t512 stays on the GPU everywhere: the Neural Engine rounds a sigmoid sum
/// on a half-frame boundary the other way once in 1,241 tokens there (one
/// frame at 30 s), and t64 to t256 agree with PyTorch token for token.
///
/// Every decoder-pre op is Neural Engine eligible at every bucket, but the two
/// engines scale differently with the frame axis: the ANE's cost per frame
/// rises with length (its per-block reductions over the whole axis tile
/// badly), the GPU's falls (launch overhead amortised). Where they cross
/// depends on the GPU: on an M3 Max the GPU wins from the 15 s bucket
/// (47 vs 17 ms at 30 s), on an M1 the ANE stays ahead at every length
/// (a GPU placement cost 29 / 65 ms at 15 / 30 s). So the threshold is not a
/// constant: an explicit override wins, else the cached answer for this
/// machine, else a one-time measurement of the 30 s decoder-pre on both
/// engines, else the safe default of the ANE throughout.

import CoreML
import Foundation
import Metal

public struct DecoderPrePlacement: Codable, Equatable {
    /// Buckets up to this many seconds run on the Neural Engine (fixed
    /// functions); longer ones on the GPU, through the flexible program when
    /// one is loaded, else the bucket's GPU package.
    public var neuralEngineMaxBucketSeconds: Int
    /// Duration packages up to this token size run on the Neural Engine (0: none).
    public var durationNeuralEngineMaxTokens: Int
    /// f0ntrain packages up to this frame count run on the Neural Engine (0: none).
    public var f0ntrainNeuralEngineMaxFrames: Int
    /// "override", "cache", "measured" or "default"; for logs and tests.
    public var source: String
    public var deviceName: String

    public init(neuralEngineMaxBucketSeconds: Int, durationNeuralEngineMaxTokens: Int = 0, f0ntrainNeuralEngineMaxFrames: Int = 0, source: String, deviceName: String) {
        self.neuralEngineMaxBucketSeconds = neuralEngineMaxBucketSeconds
        self.durationNeuralEngineMaxTokens = durationNeuralEngineMaxTokens
        self.f0ntrainNeuralEngineMaxFrames = f0ntrainNeuralEngineMaxFrames
        self.source = source
        self.deviceName = deviceName
    }

    /// The three thresholds a measurement or a cache entry carries.
    public struct Thresholds: Codable, Equatable {
        public var decoderPreSeconds: Int
        public var durationTokens: Int
        public var f0ntrainFrames: Int
        public init(decoderPreSeconds: Int, durationTokens: Int = 0, f0ntrainFrames: Int = 0) {
            self.decoderPreSeconds = decoderPreSeconds; self.durationTokens = durationTokens; self.f0ntrainFrames = f0ntrainFrames
        }
    }
    public var thresholds: Thresholds { Thresholds(decoderPreSeconds: neuralEngineMaxBucketSeconds, durationTokens: durationNeuralEngineMaxTokens, f0ntrainFrames: f0ntrainNeuralEngineMaxFrames) }
    static func from(_ t: Thresholds, source: String, deviceName: String) -> DecoderPrePlacement {
        DecoderPrePlacement(neuralEngineMaxBucketSeconds: t.decoderPreSeconds, durationNeuralEngineMaxTokens: t.durationTokens, f0ntrainNeuralEngineMaxFrames: t.f0ntrainFrames, source: source, deviceName: deviceName)
    }

    /// Duration sizes measured, largest first; sizes below the smallest measured
    /// follow it. t512 is not measured: it stays on the GPU (see the header).
    public static let durationSizesToMeasure = [256]
    /// f0ntrain sizes measured, largest first; t120's 1 ms is not worth a load.
    public static let f0ntrainSizesToMeasure = [1200, 600, 280]
    public static let durationOverrideEnvironmentVariable = "KOKORO_DURATION_ANE_MAX_TOKENS"
    public static let f0ntrainOverrideEnvironmentVariable = "KOKORO_F0NTRAIN_ANE_MAX_FRAMES"

    public func computeUnits(durationTokens: Int) -> MLComputeUnits {
        durationTokens <= durationNeuralEngineMaxTokens ? .cpuAndNeuralEngine : .cpuAndGPU
    }

    public func computeUnits(f0ntrainFrames: Int) -> MLComputeUnits {
        f0ntrainFrames <= f0ntrainNeuralEngineMaxFrames ? .cpuAndNeuralEngine : .cpuAndGPU
    }

    /// The largest size at which the Neural Engine wins, searching from the
    /// largest down and stopping at the first win (smaller sizes follow), or 0.
    /// `neuralEngineWins(size)` returns nil when that size could not be compared.
    public static func largestNeuralEngineWin(sizes: [Int], neuralEngineWins: (Int) throws -> Bool?) rethrows -> Int {
        for size in sizes.sorted(by: >) {
            if let wins = try neuralEngineWins(size), wins { return size }
        }
        return 0
    }

    /// The threshold applied when the GPU wins the 30 s measurement (the M3
    /// Max crossover sits between the 10 s and 15 s buckets).
    public static let gpuWinsThresholdSeconds = 10
    /// Neural Engine throughout: the default before any measurement, and the
    /// M1 answer.
    public static let neuralEngineThroughoutSeconds = PipelineConstants.defaultBuckets.max() ?? 30
    public static let overrideEnvironmentVariable = "KOKORO_DECODER_PRE_ANE_MAX_SECONDS"

    public func computeUnits(bucketSec: Int) -> MLComputeUnits {
        bucketSec <= neuralEngineMaxBucketSeconds ? .cpuAndNeuralEngine : .cpuAndGPU
    }

    public static var currentDeviceName: String {
        MTLCreateSystemDefaultDevice()?.name ?? "unknown"
    }

    /// The cache entry key. A measured answer is only valid for the machine,
    /// OS and package it was taken on: a macOS update changes the Core ML,
    /// ANE and Metal compilers, and a re-export changes the graph, and a
    /// stale answer costs 30-65 ms on every long utterance without any sign.
    /// `packageFingerprint` is the decoder-pre package's size and newest file
    /// date (`packageFingerprint(of:)`); `schema` bumps when the measurement
    /// itself changes.
    public static let cacheSchema = 3
    public static func cacheKey(deviceName: String, osVersion: String = currentOSVersion, packageFingerprint: String) -> String {
        "v\(cacheSchema)|\(deviceName)|macOS \(osVersion)|\(packageFingerprint)"
    }

    public static var currentOSVersion: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    /// Total byte size and newest modification date of every file under a
    /// package: cheap, and different for any re-export. `Manifest.json` is
    /// skipped: coremltools rewrites it whenever it opens the package, which
    /// would otherwise re-measure after every Python session.
    public static func packageFingerprint(of url: URL) -> String {
        let fm = FileManager.default
        let root = url.resolvingSymlinksInPath().path  // arms link their packages
        guard let paths = try? fm.subpathsOfDirectory(atPath: root) else { return "missing" }
        var bytes = 0
        var newest = Date.distantPast
        for sub in paths where (sub as NSString).lastPathComponent != "Manifest.json" {
            guard let attrs = try? fm.attributesOfItem(atPath: (root as NSString).appendingPathComponent(sub)) else { continue }
            bytes += (attrs[.size] as? Int) ?? 0
            if let d = attrs[.modificationDate] as? Date, d > newest { newest = d }
        }
        return "\(bytes)b@\(Int(newest.timeIntervalSince1970))"
    }

    /// The cache file: one entry per Metal device name.
    public static var cacheURL: URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("kokoro-coreml", isDirectory: true).appendingPathComponent("decoder-pre-placement.json")
    }

    /// Resolution order: `override`, the environment variables, the cache for
    /// this device, `measure` (whose result is cached), the default. A partial
    /// override (only some variables set) fills the rest from the cache or the
    /// measurement. `measure` returns nil when it could not compare; nothing is
    /// cached then.
    public static func resolve(
        override: Thresholds? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        cacheURL: URL = cacheURL,
        deviceName: String = currentDeviceName,
        cacheKey: String? = nil,
        measure: (() throws -> Thresholds?)? = nil
    ) -> DecoderPrePlacement {
        let key = cacheKey ?? Self.cacheKey(deviceName: deviceName, packageFingerprint: "unknown")
        if let override { return from(override, source: "override", deviceName: deviceName) }
        let env = Thresholds(
            decoderPreSeconds: environment[overrideEnvironmentVariable].flatMap(Int.init) ?? -1,
            durationTokens: environment[durationOverrideEnvironmentVariable].flatMap(Int.init) ?? -1,
            f0ntrainFrames: environment[f0ntrainOverrideEnvironmentVariable].flatMap(Int.init) ?? -1)
        func apply(_ base: Thresholds, source: String) -> DecoderPrePlacement {
            let merged = Thresholds(decoderPreSeconds: env.decoderPreSeconds >= 0 ? env.decoderPreSeconds : base.decoderPreSeconds,
                                    durationTokens: env.durationTokens >= 0 ? env.durationTokens : base.durationTokens,
                                    f0ntrainFrames: env.f0ntrainFrames >= 0 ? env.f0ntrainFrames : base.f0ntrainFrames)
            let anyEnv = env.decoderPreSeconds >= 0 || env.durationTokens >= 0 || env.f0ntrainFrames >= 0
            return from(merged, source: anyEnv ? "override+" + source : source, deviceName: deviceName)
        }
        if env.decoderPreSeconds >= 0, env.durationTokens >= 0, env.f0ntrainFrames >= 0 { return from(env, source: "override", deviceName: deviceName) }
        var cache = (try? Data(contentsOf: cacheURL)).flatMap { try? JSONDecoder().decode([String: Thresholds].self, from: $0) } ?? [:]
        if let cached = cache[key] { return apply(cached, source: "cache") }
        if let measure, let measured = (try? measure()) ?? nil {
            cache[key] = measured
            if let data = try? JSONEncoder().encode(cache) {
                try? FileManager.default.createDirectory(at: cacheURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? data.write(to: cacheURL)
            }
            return apply(measured, source: "measured")
        }
        return apply(Thresholds(decoderPreSeconds: neuralEngineThroughoutSeconds), source: "default")
    }

    /// Warm median prediction time of a model on zero inputs at its largest shape.
    public static func warmTime(_ model: MLModel, iterations: Int = 3) throws -> Double {
        let input = try zeroInputs(for: model)
        _ = try model.prediction(from: input)
        var times: [Double] = []
        for _ in 0..<max(iterations, 1) {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try model.prediction(from: input)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        return times.sorted()[times.count / 2]
    }

    /// The Neural Engine must be this much faster than the GPU to be chosen:
    /// a tie flapped between runs (f0ntrain t280 on the M3 Max), and the
    /// Neural Engine program costs seconds to compile on a fresh path where
    /// the GPU one costs a fraction of that.
    public static let neuralEngineWinMargin = 0.95

    /// Whether the Neural Engine model beats the GPU one; nil when either is missing.
    public static func neuralEngineWins(neuralEngine: MLModel?, gpu: MLModel?, iterations: Int = 3) throws -> Bool? {
        guard let neuralEngine, let gpu else { return nil }
        return neuralEngineWins(neuralEngineTime: try warmTime(neuralEngine, iterations: iterations), gpuTime: try warmTime(gpu, iterations: iterations))
    }

    public static func neuralEngineWins(neuralEngineTime: Double, gpuTime: Double) -> Bool {
        neuralEngineTime <= gpuTime * neuralEngineWinMargin
    }

    /// Threshold from one timed comparison at the 30 s bucket; nil when either
    /// side is missing.
    public static func measuredThreshold(neuralEngine: MLModel?, gpu: MLModel?, iterations: Int = 3) throws -> Int? {
        guard let wins = try neuralEngineWins(neuralEngine: neuralEngine, gpu: gpu, iterations: iterations) else { return nil }
        return wins ? neuralEngineThroughoutSeconds : gpuWinsThresholdSeconds
    }

    /// All three thresholds from `load(stage, size, units)`, which returns nil
    /// for a model the set does not have. Decoder-pre is measured at the
    /// largest bucket (a set without that fixed model keeps the Neural Engine
    /// for the buckets it has; the executor serves the rest from the flexible
    /// program); duration and f0ntrain from the largest size down. The cache
    /// key carries the set's fingerprints, so the answer is specific to it.
    public static func measuredThresholds(
        largestBucketSeconds: Int,
        load: (_ stage: String, _ size: Int, _ units: MLComputeUnits) throws -> MLModel?
    ) throws -> Thresholds? {
        let pre = try measuredThreshold(neuralEngine: try load("decoder-pre", largestBucketSeconds, .cpuAndNeuralEngine), gpu: try load("decoder-pre", largestBucketSeconds, .cpuAndGPU)) ?? neuralEngineThroughoutSeconds
        let duration = try largestNeuralEngineWin(sizes: durationSizesToMeasure) { size in
            try neuralEngineWins(neuralEngine: try load("duration", size, .cpuAndNeuralEngine), gpu: try load("duration", size, .cpuAndGPU))
        }
        let f0ntrain = try largestNeuralEngineWin(sizes: f0ntrainSizesToMeasure) { size in
            try neuralEngineWins(neuralEngine: try load("f0ntrain", size, .cpuAndNeuralEngine), gpu: try load("f0ntrain", size, .cpuAndGPU))
        }
        return Thresholds(decoderPreSeconds: pre, durationTokens: duration, f0ntrainFrames: f0ntrain)
    }

    /// Zero inputs at each input's declared (or largest) shape, masks all ones.
    static func zeroInputs(for model: MLModel) throws -> MLDictionaryFeatureProvider {
        var features: [String: MLFeatureValue] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            guard let c = desc.multiArrayConstraint else { continue }
            var shape = c.shape.map { $0.intValue }
            if let range = flexibleTimeRange(of: model, input: name) { shape[shape.count - 1] = range.upperBound }
            let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: c.dataType == .int32 ? .int32 : .float32)
            // masks and attention masks all ones, speed 1.0, token ids 1, activations zero
            let ones = name.hasPrefix("mask") || name == "attention_mask" || name == "speed" || name == "input_ids"
            if c.dataType == .int32 {
                let ptr = array.dataPointer.assumingMemoryBound(to: Int32.self)
                for i in 0..<array.count { ptr[i] = ones ? 1 : 0 }
            } else {
                let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<array.count { ptr[i] = ones ? 1 : 0 }
            }
            features[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }
}
