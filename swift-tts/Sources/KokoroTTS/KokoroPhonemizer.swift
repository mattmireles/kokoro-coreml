import Foundation

/// Errors raised by raw-text phonemizers before tokenization.
public enum KokoroPhonemizerError: Error, Equatable, LocalizedError {
    /// The phonemizer returned no phonemes for a non-empty or empty input.
    case emptyOutput

    /// Human-readable explanation for logs and command-line probes.
    public var errorDescription: String? {
        switch self {
        case .emptyOutput:
            return "The phonemizer returned no phonemes."
        }
    }
}

/// Raw phoneme output from an English text phonemizer.
public struct KokoroPhonemeResult: Equatable, Codable {
    /// Phoneme string consumed by Kokoro vocabulary tokenization.
    public let phonemes: String

    /// UTF-16 code-unit length of ``phonemes``.
    ///
    /// Botnet's JavaScript prep uses `phonemes.length`, so Swift must preserve
    /// UTF-16 length for later voice-row parity.
    public let utf16Count: Int

    /// Creates a phoneme result.
    ///
    /// - Parameter phonemes: Phoneme string returned by the backend.
    public init(phonemes: String) {
        self.phonemes = phonemes
        self.utf16Count = phonemes.utf16.count
    }
}

/// Protocol boundary for the V1 raw-text phonemizer.
///
/// The SDK depends on this abstraction so a future pure-Swift or server-proven
/// backend can replace MisakiSwift without changing the public prep pipeline.
public protocol KokoroPhonemizer {
    /// Converts natural-language text into Kokoro-compatible phonemes.
    ///
    /// - Parameter text: Raw user text supplied by the app.
    /// - Returns: Phonemes plus the UTF-16 count needed for voice-row lookup.
    func phonemize(_ text: String) throws -> KokoroPhonemeResult
}
