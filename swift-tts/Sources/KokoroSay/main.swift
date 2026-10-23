import Foundation
import KokoroTTS

/// `kokoro-say "Hello"`: speech to ./kokoro.wav with zero setup.
///
/// With no source flag it downloads the pinned starter bundle
/// (``KokoroHostedModels``) into ~/Library/Caches/kokoro-coreml, then prints
/// per-chunk stage timings and which programs ran (split/whole duration and
/// F0, hybrid/folded generator). The release checklist (README/SDK.md) runs it
/// with `--bundle <bundle-root>`.
func run() async throws {
    let options = try SayOptions(arguments: Array(CommandLine.arguments.dropFirst()))
    let loadStart = Date()
    let resources: KokoroResourceProvider
    let printer = ProgressPrinter()
    if let bundleRoot = options.bundleRoot {
        resources = .directory(bundleRoot)
    } else if let manifestURL = options.manifestURL {
        resources = try await KokoroDownloadedModelStore(
            manifestURL: manifestURL,
            expectedManifestSHA256: options.manifestSHA256,
            cacheDirectory: options.cacheDirectory
        ).hydrate(phaseProgress: { printer.report($0) })
    } else {
        resources = try await .huggingFace(cacheDirectory: options.cacheDirectory, progress: { printer.report($0) })
    }
    let tts = try await KokoroTTS.load(resources: resources, computePolicy: options.cpuOnly ? .cpuOnly : .gistDefault)
    print(String(format: "load: %.0f ms", Date().timeIntervalSince(loadStart) * 1000))

    var audio = KokoroAudio(samples: [])
    for run in 1...options.runs {
        let start = Date()
        audio = try await tts.synthesize(options.text, voice: options.voice)
        // The first run also compiles (first launch only) and specializes Core ML models.
        print(String(format: "run %d: %.0f ms for %.2f s of audio", run, Date().timeIntervalSince(start) * 1000, audio.durationSeconds))
        for (index, chunk) in (await tts.lastReport?.chunks ?? []).enumerated() {
            let stages = chunk.stageMilliseconds.map { String(format: "%@ %.1f", $0.name, $0.ms) }.joined(separator: ", ")
            print("  chunk \(index): bucket \(chunk.bucketSeconds)s, duration=\(chunk.durationPath) f0=\(chunk.f0Path) "
                + "generator=\(chunk.generatorPath) | \(stages) | total \(String(format: "%.1f", chunk.totalMilliseconds)) ms")
        }
    }
    try writeWAV(audio: audio, to: options.outputURL)
    let rms = (audio.samples.reduce(0) { $0 + $1 * $1 } / Float(max(audio.samples.count, 1))).squareRoot()
    print(String(format: "wrote %@ (%.2f s, %d Hz, rms %.4f)", options.outputURL.path, audio.durationSeconds, audio.sampleRate, rms))
}

/// Command-line options.
struct SayOptions {
    static let usage = """
    usage: kokoro-say [text | --text <text>] [--voice <voice>] [--out <wav>] [--runs <n>] [--cpu-only]
                      [--bundle <bundle-root> | --manifest-url <url> --manifest-sha256 <sha256>] [--cache-dir <dir>]
           kokoro-say <bundle-root> [text]

    Defaults: text "Hello world.", voice af_heart, out ./kokoro.wav, models from the
    pinned Hugging Face starter bundle cached in ~/Library/Caches/kokoro-coreml.

    """

    var cpuOnly = false
    var bundleRoot: URL?
    var manifestURL: URL?
    var manifestSHA256 = ""
    var cacheDirectory = KokoroHostedModels.defaultCacheDirectory
    var text = "Hello world."
    var voice = KokoroVoiceID.afHeart
    var outputURL = URL(fileURLWithPath: "kokoro.wav")
    var runs = 1

    init(arguments: [String]) throws {
        var positional: [String] = []
        var index = 0
        while index < arguments.count {
            let key = arguments[index]
            index += 1
            if key == "--cpu-only" { cpuOnly = true; continue }
            if key == "--help" || key == "-h" { throw SayError.help }
            guard key.hasPrefix("--") else { positional.append(key); continue }
            guard index < arguments.count else { throw SayError.usage }
            let value = arguments[index]
            index += 1
            switch key {
            case "--bundle": bundleRoot = URL(fileURLWithPath: value, isDirectory: true)
            case "--manifest-url": manifestURL = try URL(string: value).orUsage()
            case "--manifest-sha256": manifestSHA256 = value
            case "--cache-dir": cacheDirectory = URL(fileURLWithPath: value, isDirectory: true)
            case "--text": text = value
            case "--voice": voice = KokoroVoiceID(value)
            case "--out": outputURL = URL(fileURLWithPath: value)
            case "--runs": runs = try Int(value).flatMap { $0 > 0 ? $0 : nil }.orUsage()
            default: throw SayError.usage
            }
        }
        // `<bundle-root> [text]`: a first argument that looks like a path (an
        // existing directory, or a "/" and no spaces) is a bundle root, never text.
        if let first = positional.first, Self.looksLikePath(first) {
            guard bundleRoot == nil else { throw SayError.invalid("\(first): give the bundle root once") }
            let root = URL(fileURLWithPath: first, isDirectory: true)
            guard FileManager.default.fileExists(atPath: root.appendingPathComponent("KokoroRuntimeManifest.json").path) else {
                throw SayError.invalid("\(first) is not a bundle root: it has no KokoroRuntimeManifest.json")
            }
            bundleRoot = root
            positional.removeFirst()
        }
        if bundleRoot != nil, manifestURL != nil {
            throw SayError.invalid("use either a bundle root (--bundle) or --manifest-url, not both")
        }
        guard positional.count <= 1, (manifestURL == nil) == manifestSHA256.isEmpty else { throw SayError.usage }
        if let first = positional.first { text = first }
    }

    /// Whether a positional argument names a path rather than text to speak.
    static func looksLikePath(_ argument: String) -> Bool {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: argument, isDirectory: &isDirectory), isDirectory.boolValue { return true }
        return argument.contains("/") && !argument.contains(" ")
    }
}

enum SayError: Error {
    case usage
    case help
    /// A usage error with its reason.
    case invalid(String)
}

/// Prints first-run download progress to stderr, one line per whole percent.
final class ProgressPrinter: @unchecked Sendable {
    private let lock = NSLock()
    private var last: (KokoroHydrationProgress.Phase, Int)?

    func report(_ progress: KokoroHydrationProgress) {
        let percent = Int(progress.fraction * 100)
        lock.lock()
        defer { lock.unlock() }
        guard last.map({ $0.0 != progress.phase || $0.1 != percent }) ?? true else { return }
        last = (progress.phase, percent)
        let word = progress.phase == .downloading ? "downloading" : "verifying"
        FileHandle.standardError.write(Data("kokoro-say: \(word) models \(percent)%\n".utf8))
    }
}

extension Optional {
    func orUsage() throws -> Wrapped {
        guard let self else { throw SayError.usage }
        return self
    }
}

/// Writes mono Float PCM audio as 16-bit little-endian WAV.
func writeWAV(audio: KokoroAudio, to url: URL) throws {
    var payload = Data()
    for sample in audio.samples {
        let pcm = Int16(max(-1.0, min(1.0, sample)) * Float(Int16.max)).littleEndian
        withUnsafeBytes(of: pcm) { payload.append(contentsOf: $0) }
    }
    var data = Data()
    data.append(contentsOf: "RIFF".utf8)
    data.append(UInt32(36 + payload.count).littleEndianData)
    data.append(contentsOf: "WAVEfmt ".utf8)
    data.append(UInt32(16).littleEndianData)
    data.append(UInt16(1).littleEndianData)   // PCM
    data.append(UInt16(1).littleEndianData)   // mono
    data.append(UInt32(audio.sampleRate).littleEndianData)
    data.append(UInt32(audio.sampleRate * 2).littleEndianData)
    data.append(UInt16(2).littleEndianData)
    data.append(UInt16(16).littleEndianData)
    data.append(contentsOf: "data".utf8)
    data.append(UInt32(payload.count).littleEndianData)
    data.append(payload)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url, options: .atomic)
}

extension FixedWidthInteger {
    /// This integer as little-endian bytes.
    var littleEndianData: Data {
        var value = self.littleEndian
        return withUnsafeBytes(of: &value) { Data($0) }
    }
}

do {
    try await run()
} catch SayError.help {
    print(SayOptions.usage, terminator: "")
} catch SayError.usage {
    FileHandle.standardError.write(Data(SayOptions.usage.utf8))
    exit(2)
} catch SayError.invalid(let reason) {
    FileHandle.standardError.write(Data("kokoro-say: \(reason)\n\n\(SayOptions.usage)".utf8))
    exit(2)
} catch {
    FileHandle.standardError.write(Data("kokoro-say: \(error.localizedDescription)\n".utf8))
    exit(1)
}
