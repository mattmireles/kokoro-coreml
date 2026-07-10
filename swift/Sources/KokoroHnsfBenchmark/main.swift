/// Standalone warmed timing probe for Kokoro's Swift HnSF source/STFT boundary.
///
/// The external bakeoff optimizes whether the HAR source and STFT work should
/// remain in Swift or move behind a Core ML boundary. This executable measures
/// the shipping Swift implementation by bucket, separating:
/// - source generation still needed by a `har_source -> Core ML` package
/// - STFT work that such a package would remove from Swift
/// - full `buildHar` latency as the current shipping baseline

import Foundation
import KokoroPipeline

private let defaultBuckets = [3, 7, 10, 15, 30]
private let defaultLinearWeights: [Float] = [
    -0.08154187, -0.18519667, -0.18263398, -0.17837206, -0.09873895,
     0.08264039,  0.08743999, -0.39068547, -0.54774433,
]
private let defaultLinearBias: Float = -0.02945026

private struct BucketTiming: Encodable {
    let bucket_s: Int
    let f0_frames: Int
    let source_samples: Int
    let stft_frames: Int
    let warmup: Int
    let iterations: Int
    let source_median_ms: Double
    let stft_median_ms: Double
    let build_har_median_ms: Double
    let source_p10_ms: Double
    let source_p90_ms: Double
    let stft_p10_ms: Double
    let stft_p90_ms: Double
    let build_har_p10_ms: Double
    let build_har_p90_ms: Double
    let checksum: Double
}

private struct Report: Encodable {
    let generated_at: String
    let timing_boundary: String
    let buckets: [BucketTiming]
}

private struct Options {
    var buckets = defaultBuckets
    var warmup = 5
    var iterations = 30
    var output: String?
    var seed: UInt64 = 42
    var f0Hz: Float = 200.0
}

private enum CliError: Error, CustomStringConvertible {
    case missingValue(String)
    case invalidValue(String, String)
    case unknownArgument(String)

    var description: String {
        switch self {
        case let .missingValue(flag):
            return "Missing value for \(flag)"
        case let .invalidValue(flag, value):
            return "Invalid value for \(flag): \(value)"
        case let .unknownArgument(arg):
            return "Unknown argument: \(arg)"
        }
    }
}

private func parseOptions() throws -> Options {
    var options = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    while !args.isEmpty {
        let arg = args.removeFirst()
        switch arg {
        case "--buckets":
            guard !args.isEmpty else { throw CliError.missingValue(arg) }
            let value = args.removeFirst()
            let parsed = value.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            guard !parsed.isEmpty else { throw CliError.invalidValue(arg, value) }
            options.buckets = parsed
        case "--warmup":
            guard !args.isEmpty else { throw CliError.missingValue(arg) }
            let value = args.removeFirst()
            guard let parsed = Int(value), parsed >= 0 else { throw CliError.invalidValue(arg, value) }
            options.warmup = parsed
        case "--iterations":
            guard !args.isEmpty else { throw CliError.missingValue(arg) }
            let value = args.removeFirst()
            guard let parsed = Int(value), parsed > 0 else { throw CliError.invalidValue(arg, value) }
            options.iterations = parsed
        case "--output":
            guard !args.isEmpty else { throw CliError.missingValue(arg) }
            options.output = args.removeFirst()
        case "--seed":
            guard !args.isEmpty else { throw CliError.missingValue(arg) }
            let value = args.removeFirst()
            guard let parsed = UInt64(value) else { throw CliError.invalidValue(arg, value) }
            options.seed = parsed
        case "--f0-hz":
            guard !args.isEmpty else { throw CliError.missingValue(arg) }
            let value = args.removeFirst()
            guard let parsed = Float(value), parsed >= 0 else { throw CliError.invalidValue(arg, value) }
            options.f0Hz = parsed
        case "--help", "-h":
            printUsage()
            exit(0)
        default:
            throw CliError.unknownArgument(arg)
        }
    }
    return options
}

private func printUsage() {
    print("""
    Usage: kokoro-hnsf-bench [--buckets 3,7,10,15,30] [--warmup 5] [--iterations 30] [--output report.json]

    Measures the warmed Swift HnSF source/STFT boundary by runtime bucket.
    """)
}

private func median(_ values: [Double]) -> Double {
    percentile(values, 0.5)
}

private func percentile(_ values: [Double], _ q: Double) -> Double {
    precondition(!values.isEmpty)
    let sorted = values.sorted()
    let position = q * Double(sorted.count - 1)
    let lower = Int(floor(position))
    let upper = Int(ceil(position))
    if lower == upper {
        return sorted[lower]
    }
    let weight = position - Double(lower)
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

private func elapsedMilliseconds(_ body: () -> Double) -> (Double, Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let checksum = body()
    let end = DispatchTime.now().uptimeNanoseconds
    return (Double(end - start) / 1_000_000.0, checksum)
}

private func checksum(_ values: [Float]) -> Double {
    guard !values.isEmpty else { return 0.0 }
    return Double(values[0]) + Double(values[values.count / 2]) + Double(values[values.count - 1])
}

private func checksum(_ values: (magnitude: [Float], phase: [Float])) -> Double {
    checksum(values.magnitude) + checksum(values.phase)
}

private func checksum(_ values: (har: [Float], nFrames: Int)) -> Double {
    checksum(values.har) + Double(values.nFrames)
}

private func measureBucket(bucket: Int, options: Options) -> BucketTiming {
    let f0Frames = bucket * 24_000 / HarmonicConstants.upsampleScale
    let f0 = [Float](repeating: options.f0Hz, count: f0Frames)
    let source = sineGenFromF0Frames(
        f0Frames: f0,
        linearWeights: defaultLinearWeights,
        linearBias: defaultLinearBias,
        seed: options.seed
    )
    let stft = stftTransform(source)

    var sourceTimes: [Double] = []
    var stftTimes: [Double] = []
    var buildHarTimes: [Double] = []
    var accumulatedChecksum = checksum(source) + checksum(stft)

    for iteration in 0..<(options.warmup + options.iterations) {
        let keep = iteration >= options.warmup

        let sourceRun = elapsedMilliseconds {
            checksum(sineGenFromF0Frames(
                f0Frames: f0,
                linearWeights: defaultLinearWeights,
                linearBias: defaultLinearBias,
                seed: options.seed
            ))
        }
        accumulatedChecksum += sourceRun.1
        if keep { sourceTimes.append(sourceRun.0) }

        let sourceForStft = sineGenFromF0Frames(
            f0Frames: f0,
            linearWeights: defaultLinearWeights,
            linearBias: defaultLinearBias,
            seed: options.seed
        )
        let stftRun = elapsedMilliseconds {
            checksum(stftTransform(sourceForStft))
        }
        accumulatedChecksum += stftRun.1
        if keep { stftTimes.append(stftRun.0) }

        let buildHarRun = elapsedMilliseconds {
            checksum(buildHar(
                f0Padded: f0,
                linearWeights: defaultLinearWeights,
                linearBias: defaultLinearBias,
                seed: options.seed
            ))
        }
        accumulatedChecksum += buildHarRun.1
        if keep { buildHarTimes.append(buildHarRun.0) }
    }

    return BucketTiming(
        bucket_s: bucket,
        f0_frames: f0Frames,
        source_samples: source.count,
        stft_frames: stft.magnitude.count / HarmonicConstants.stftFreqBins,
        warmup: options.warmup,
        iterations: options.iterations,
        source_median_ms: median(sourceTimes),
        stft_median_ms: median(stftTimes),
        build_har_median_ms: median(buildHarTimes),
        source_p10_ms: percentile(sourceTimes, 0.1),
        source_p90_ms: percentile(sourceTimes, 0.9),
        stft_p10_ms: percentile(stftTimes, 0.1),
        stft_p90_ms: percentile(stftTimes, 0.9),
        build_har_p10_ms: percentile(buildHarTimes, 0.1),
        build_har_p90_ms: percentile(buildHarTimes, 0.9),
        checksum: accumulatedChecksum
    )
}

private func main() throws {
    let options = try parseOptions()
    let report = Report(
        generated_at: ISO8601DateFormatter().string(from: Date()),
        timing_boundary: "Shipping Swift HnSF; source_median_ms remains required by har_source Core ML candidates, stft_median_ms is the removable Swift work.",
        buckets: options.buckets.map { measureBucket(bucket: $0, options: options) }
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    if let output = options.output {
        try data.write(to: URL(fileURLWithPath: output))
    } else {
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}

do {
    try main()
} catch {
    fputs("\(error)\n", stderr)
    printUsage()
    exit(2)
}
