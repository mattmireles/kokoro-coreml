import Foundation
import KokoroTTS

/// Single probe row emitted as JSON.
private struct ProbeRow: Encodable {
    /// Input text supplied to the probe.
    let text: String

    /// Whether the British English Misaki path was used.
    let british: Bool

    /// Phoneme output returned by the SDK phonemizer.
    let phonemes: String

    /// UTF-16 code-unit length of the phoneme string.
    let utf16Count: Int
}

/// Probe failure row emitted as JSON.
private struct ProbeErrorRow: Encodable {
    /// Input text supplied to the probe.
    let text: String

    /// Whether the British English Misaki path was used.
    let british: Bool

    /// Error string returned by the SDK phonemizer.
    let error: String
}

/// Parses process arguments into a British flag and text cases.
///
/// - Parameter arguments: Raw command-line arguments including executable name.
/// - Returns: Parsed mode plus input texts.
private func parseArguments(_ arguments: [String]) -> (british: Bool, texts: [String]) {
    var british = false
    var texts: [String] = []
    for argument in arguments.dropFirst() {
        if argument == "--british" {
            british = true
        } else {
            texts.append(argument)
        }
    }
    if texts.isEmpty {
        texts = [
            "Hello world.",
            "Dr. Smith paid $12.50 for apples.",
            "Visit https://example.com, then email me@example.com.",
            "I live in Reading.",
        ]
    }
    return (british, texts)
}

/// Encodes a probe row to standard output.
///
/// - Parameter row: Encodable row to write.
private func emitJSONLine<T: Encodable>(_ row: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(row)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

let parsed = parseArguments(CommandLine.arguments)
let phonemizer = KokoroMisakiPhonemizer(british: parsed.british)
for text in parsed.texts {
    do {
        let result = try phonemizer.phonemize(text)
        try emitJSONLine(ProbeRow(
            text: text,
            british: parsed.british,
            phonemes: result.phonemes,
            utf16Count: result.utf16Count
        ))
    } catch {
        try emitJSONLine(ProbeErrorRow(
            text: text,
            british: parsed.british,
            error: String(describing: error)
        ))
    }
}
