import Foundation
import KokoroTTS

/// Runs raw-text local synthesis against a generated SDK bundle.
func runSmoke() async throws {
    let options = try SmokeOptions(arguments: Array(CommandLine.arguments.dropFirst()))
    let resources: KokoroResourceProvider
    if let manifestURL = options.manifestURL {
        guard let cacheDirectory = options.cacheDirectory else {
            throw SmokeError.usage
        }
        resources = try await KokoroDownloadedModelStore(
            manifestURL: manifestURL,
            cacheDirectory: cacheDirectory
        ).hydrate()
    } else if let bundleRoot = options.bundleRoot {
        resources = .directory(bundleRoot)
    } else {
        FileHandle.standardError.write(Data(SmokeOptions.usage.utf8))
        throw ExitCode.failure
    }
    let tts = try await KokoroTTS.load(
        resources: resources,
        computePolicy: options.cpuOnly ? .cpuOnly : .gistDefault
    )
    let audio = try await tts.synthesize(options.text, voice: options.voice)
    if let outputURL = options.outputURL {
        try writeWAV(audio: audio, to: outputURL)
    }
    print("samples=\(audio.samples.count) sampleRate=\(audio.sampleRate) duration=\(audio.durationSeconds)")
}

/// Command-line options accepted by the SDK smoke executable.
struct SmokeOptions {
    /// Usage text shown for invalid invocations.
    static let usage = """
    usage: kokoro-sdk-smoke [--cpu-only] [--bundle <bundle-root> | --manifest-url <url> --cache-dir <dir>] [--text <text>] [--voice <voice>] [--out <wav>]
           kokoro-sdk-smoke [--cpu-only] <bundle-root> [text]
    """

    /// Whether to force CPU-only Core ML loading.
    let cpuOnly: Bool

    /// Explicit generated bundle root.
    let bundleRoot: URL?

    /// Hosted manifest URL for downloaded-resource mode.
    let manifestURL: URL?

    /// Cache directory for downloaded-resource mode.
    let cacheDirectory: URL?

    /// Raw text to synthesize.
    let text: String

    /// Voice to synthesize.
    let voice: KokoroVoiceID

    /// Optional WAV output URL.
    let outputURL: URL?

    /// Parses command-line arguments while preserving the Phase 5 positional form.
    ///
    /// - Parameter arguments: Process arguments after executable name.
    init(arguments: [String]) throws {
        var values = arguments
        var cpuOnly = false
        var bundleRoot: URL?
        var manifestURL: URL?
        var cacheDirectory: URL?
        var text = "Hello world."
        var voice = KokoroVoiceID.afHeart
        var outputURL: URL?

        if values.contains("--cpu-only") {
            cpuOnly = true
            values.removeAll { $0 == "--cpu-only" }
        }

        if let first = values.first, !first.hasPrefix("--") {
            bundleRoot = URL(fileURLWithPath: first)
            if values.count > 1 {
                text = values[1]
            }
            self.cpuOnly = cpuOnly
            self.bundleRoot = bundleRoot
            self.manifestURL = nil
            self.cacheDirectory = nil
            self.text = text
            self.voice = voice
            self.outputURL = nil
            return
        }

        var index = 0
        while index < values.count {
            let key = values[index]
            guard key.hasPrefix("--"), index + 1 < values.count else {
                throw SmokeError.usage
            }
            let value = values[index + 1]
            switch key {
            case "--bundle":
                bundleRoot = URL(fileURLWithPath: value)
            case "--manifest-url":
                guard let url = URL(string: value) else {
                    throw SmokeError.usage
                }
                manifestURL = url
            case "--cache-dir":
                cacheDirectory = URL(fileURLWithPath: value, isDirectory: true)
            case "--text":
                text = value
            case "--voice":
                voice = KokoroVoiceID(value)
            case "--out":
                outputURL = URL(fileURLWithPath: value)
            default:
                throw SmokeError.usage
            }
            index += 2
        }

        self.cpuOnly = cpuOnly
        self.bundleRoot = bundleRoot
        self.manifestURL = manifestURL
        self.cacheDirectory = cacheDirectory
        self.text = text
        self.voice = voice
        self.outputURL = outputURL
    }
}

/// Errors raised by smoke command-line parsing.
enum SmokeError: Error {
    /// Invalid command-line invocation.
    case usage
}

/// Writes mono Float PCM audio as 16-bit little-endian WAV.
///
/// - Parameters:
///   - audio: Audio returned by the SDK.
///   - url: Destination WAV file URL.
func writeWAV(audio: KokoroAudio, to url: URL) throws {
    var payload = Data()
    for sample in audio.samples {
        let clamped = max(-1.0, min(1.0, sample))
        let pcm = Int16(clamped * Float(Int16.max)).littleEndian
        withUnsafeBytes(of: pcm) { payload.append(contentsOf: $0) }
    }
    var data = Data()
    data.append(contentsOf: "RIFF".utf8)
    data.append(UInt32(36 + payload.count).littleEndianData)
    data.append(contentsOf: "WAVEfmt ".utf8)
    data.append(UInt32(16).littleEndianData)
    data.append(UInt16(1).littleEndianData)
    data.append(UInt16(1).littleEndianData)
    data.append(UInt32(audio.sampleRate).littleEndianData)
    data.append(UInt32(audio.sampleRate * 2).littleEndianData)
    data.append(UInt16(2).littleEndianData)
    data.append(UInt16(16).littleEndianData)
    data.append(contentsOf: "data".utf8)
    data.append(UInt32(payload.count).littleEndianData)
    data.append(payload)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try data.write(to: url, options: .atomic)
}

/// Little-endian byte helpers for WAV headers.
extension FixedWidthInteger {
    /// Returns this integer as little-endian bytes.
    var littleEndianData: Data {
        var value = self.littleEndian
        return withUnsafeBytes(of: &value) { Data($0) }
    }
}

/// Process exit sentinel for invalid CLI usage.
enum ExitCode: Error {
    /// Invalid command-line invocation.
    case failure
}

do {
    try await runSmoke()
} catch SmokeError.usage {
    FileHandle.standardError.write(Data(SmokeOptions.usage.utf8))
    throw ExitCode.failure
}
