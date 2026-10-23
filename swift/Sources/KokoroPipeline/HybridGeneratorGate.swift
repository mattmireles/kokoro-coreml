/// When the hybrid generator may run, and what happens when it fails.
///
/// The SDK and app path treat the hybrid as an optimization: the first hybrid
/// error in a process is remembered and the folded generator serves until
/// relaunch. A benchmark that asked for the hybrid (``HybridGenerator/requested``)
/// gets the error instead, so a folded run is never timed under a hybrid label.
/// Callers: `resolveHybridGenerator` and the generator stage in
/// KokoroSynthesisExecutor.swift, the providers in KokoroPipeline.swift,
/// KokoroBenchmark/main.swift, KokoroSDKModelProvider.swift and ios-bench.
import CoreML
import Foundation

@available(macOS 15.0, iOS 18.0, *)
extension HybridGenerator {
    /// Environment switch, same polarity as the other fast paths
    /// (KOKORO_SPLIT_DURATION, KOKORO_SPLIT_F0N, KOKORO_FOLDED_GENERATOR):
    /// `0` turns the hybrid off; anything else leaves it on where the models
    /// ship. The benchmark's own provider still needs an explicit `1`.
    public static let envKey = "KOKORO_HYBRID_GENERATOR"
    /// ios-bench's opt-in launch argument (ios-bench/Sources/BenchApp.swift).
    static let benchArgument = "--hybrid-generator"
    /// Benchmark opt-in: `KOKORO_HYBRID_GENERATOR=1`.
    public static var enabled: Bool { ProcessInfo.processInfo.environment[envKey] == "1" && !failedThisProcess }
    /// Whether providers may use a shipped hybrid generator: not switched off
    /// by the environment, and no hybrid failure yet in this process.
    public static var allowed: Bool { ProcessInfo.processInfo.environment[envKey] != "0" && !failedThisProcess }
    /// Whether a benchmark explicitly asked for the hybrid: kokoro-bench's
    /// `KOKORO_HYBRID_GENERATOR=1` or ios-bench's `--hybrid-generator`. Then any
    /// hybrid error fails the run instead of falling back to folded. An app that
    /// sets `KOKORO_HYBRID_GENERATOR=1` gets the same fail-hard behavior; apps
    /// should leave it unset (default on, with fallback).
    public static var requested: Bool {
        ProcessInfo.processInfo.environment[envKey] == "1" || CommandLine.arguments.contains(benchArgument)
    }

    // MARK: - Failure memory

    /// The first hybrid failure (init, load or predict) in this process. Once
    /// set the hybrid stays off until relaunch and the folded generator serves:
    /// a failure that happened once (no Metal in the iOS background, an ANE
    /// load error) would likely happen again and cost a full failed run each time.
    private static var failure: String?
    private static let failureMemoryLock = NSLock()
    /// The recorded failure's description, or nil.
    public static var recordedFailure: String? {
        failureMemoryLock.lock(); defer { failureMemoryLock.unlock() }
        return failure
    }
    public static var failedThisProcess: Bool { recordedFailure != nil }
    /// Records `error` as this process's hybrid failure (the first one wins).
    public static func recordFailure(_ error: Error) {
        failureMemoryLock.lock()
        let first = failure == nil
        if first { failure = "\(error)" }
        failureMemoryLock.unlock()
        if first { fputs("kokoro: hybrid generator failed (\(error)); using the folded generator until relaunch\n", stderr) }
    }
    /// Clears the recorded failure. Tests only (also from KokoroTTSTests via
    /// `@testable import KokoroPipeline`).
    static func resetFailureForTesting() {
        failureMemoryLock.lock(); failure = nil; failureMemoryLock.unlock()
    }

    // MARK: - Row selection

    /// Shortest bucket the hybrid serves. Below it the folded generator wins:
    /// the hybrid's ~100 Core ML/Metal round trips are a fixed cost (3 s,
    /// wall: M1 211 vs 135 ms folded, M2 Air 200 vs 103 ms), while from 7 s up
    /// the hybrid is 1.5-2x faster than folded.
    public static let minBucketSec = 7

    /// Row count for `xPreFrames` (80 Hz) in a bucket: the smallest packaged row
    /// count that holds them. Ship few packages that match the chunker (e.g. r4
    /// for 10 s phone chunks, r6 for 15 s, r11 for 30 s): each row-count switch
    /// reloads 53 function models (~0.35-0.5 s on an M1).
    public static func rows(bucketSec: Int, xPreFrames: Int, directory: URL) -> Int? {
        guard bucketSec >= minBucketSec else { return nil }
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("plan.json")),
              let plan = try? JSONDecoder().decode(HybridPlan.self, from: data) else { return nil }
        let needed = (xPreFrames + plan.core - 1) / plan.core
        let fm = FileManager.default
        func hasPackage(_ r: Int) -> Bool {
            fm.fileExists(atPath: directory.appendingPathComponent("hybrid_generator_r\(r).mlpackage").path)
                || fm.fileExists(atPath: directory.appendingPathComponent("hybrid_generator_r\(r).mlmodelc").path)
        }
        // KOKORO_HYBRID_ROWS=4,11: only these packaged row counts (benchmarks that
        // mimic an app shipping fewer packages).
        let only = ProcessInfo.processInfo.environment["KOKORO_HYBRID_ROWS"].map { Set($0.split(separator: ",").compactMap { Int($0) }) }
        return plan.rows.sorted().first { $0 >= needed && hasPackage($0) && (only?.contains($0) ?? true) }
    }

    // MARK: - Load checks

    /// Rejects an adain.bin too short for plan.json's offsets: `run` reads
    /// weight [2C x 128], bias [2C] and alpha [C] at each entry's offsets
    /// without bounds checks, so a mismatched pair would read past the table.
    static func checkAdaINOffsets(_ entries: [HybridPlan.AdaIN], tableCount: Int, styleDim: Int = PipelineConstants.styleDim) throws {
        for e in entries {
            let spans = [(e.weight, 2 * e.channels * styleDim), (e.bias, 2 * e.channels), (e.alpha, e.channels)]
            guard e.channels > 0, spans.allSatisfy({ $0.0 >= 0 && $0.0 + $0.1 <= tableCount }) else {
                throw HybridGeneratorError.badPlan("adain \(e.name): offsets outside adain.bin (\(tableCount) floats)")
            }
        }
    }

    // MARK: - Compile memo

    /// Compiled .mlmodelc per package, once per process (compiling takes ~7 s).
    /// Only for callers without a durable cache (the benchmark); the SDK and
    /// KokoroPipeline pass their compiled-model cache's URL to `init` instead.
    private static var compiledURLs: [URL: URL] = [:]
    private static let compileLock = NSLock()
    static func compiled(_ package: URL) throws -> URL {
        compileLock.lock()
        defer { compileLock.unlock() }
        if let url = compiledURLs[package] { return url }
        let url = try MLModel.compileModel(at: package)
        compiledURLs[package] = url
        return url
    }
}
