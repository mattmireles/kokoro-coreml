import Foundation
import KokoroTTS

/// Runs a minimal consumer-package synthesis smoke using only public SDK APIs.
func runConsumerFixture() async throws {
    let options = try ConsumerOptions(arguments: Array(CommandLine.arguments.dropFirst()))
    let tts = try await KokoroTTS.load(resources: .directory(options.bundleRoot))
    let audio = try await tts.synthesize(options.text, voice: options.voice)
    if let outputURL = options.outputURL {
        try writeWAV(audio: audio, to: outputURL)
    }
    print("samples=\(audio.samples.count) sampleRate=\(audio.sampleRate) duration=\(audio.durationSeconds)")
}

/// Command-line options for the public consumer fixture.
struct ConsumerOptions {
    /// Usage text for invalid invocations.
    static let usage = """
    usage: kokoro-consumer-fixture --bundle <bundle-root> [--text <text>] [--voice <voice>] [--out <wav>]
    """

    /// Generated SDK bundle root.
    let bundleRoot: URL

    /// Raw text to synthesize.
    let text: String

    /// Voice to synthesize.
    let voice: KokoroVoiceID

    /// Optional WAV output URL.
    let outputURL: URL?

    /// Parses command-line arguments.
    ///
    /// - Parameter arguments: Process arguments after executable name.
    init(arguments: [String]) throws {
        var bundleRoot: URL?
        var text = "Hello world."
        var voice = KokoroVoiceID.afHeart
        var outputURL: URL?
        var index = 0
        while index < arguments.count {
            let key = arguments[index]
            guard key.hasPrefix("--"), index + 1 < arguments.count else {
                throw ConsumerError.usage
            }
            let value = arguments[index + 1]
            switch key {
            case "--bundle":
                bundleRoot = URL(fileURLWithPath: value, isDirectory: true)
            case "--text":
                text = value
            case "--voice":
                voice = KokoroVoiceID(value)
            case "--out":
                outputURL = URL(fileURLWithPath: value)
            default:
                throw ConsumerError.usage
            }
            index += 2
        }
        guard let bundleRoot else {
            throw ConsumerError.usage
        }
        self.bundleRoot = bundleRoot
        self.text = text
        self.voice = voice
        self.outputURL = outputURL
    }
}

/// Command-line errors for the consumer fixture.
enum ConsumerError: Error {
    /// Invalid command-line invocation.
    case usage
}

/// Writes mono Float PCM audio as 16-bit little-endian WAV.
///
/// - Parameters:
///   - audio: SDK audio value.
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

do {
    try await runConsumerFixture()
} catch ConsumerError.usage {
    FileHandle.standardError.write(Data(ConsumerOptions.usage.utf8))
    throw ConsumerError.usage
}
