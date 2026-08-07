import Foundation
import MisakiSwift

/// MisakiSwift-backed English phonemizer used by the raw-text SDK.
///
/// This matches the Gist iOS app pattern: use the mattmireles MisakiSwift fork
/// for on-device English G2P and keep it out of the lower-floor
/// ``KokoroPipeline`` package.
public final class KokoroMisakiPhonemizer: KokoroPhonemizer {
    /// Underlying Misaki English grapheme-to-phoneme engine.
    ///
    /// MisakiSwift initializes MLX resources when `EnglishG2P` is created.
    /// Keep this cached instance lazy so `KokoroTTS.load` can validate
    /// resources and return from app startup without touching MLX; the first
    /// `prepare` or `synthesize` call pays the G2P setup cost.
    private var cachedG2P: EnglishG2P?

    /// Serializes access to ``cachedG2P`` and its mutable NLP state.
    ///
    /// Swift `lazy var` is not a synchronization primitive, and Misaki's
    /// `EnglishG2P` owns mutable tagger state. This class is public, so callers
    /// may share one phonemizer across tasks even though the `KokoroTTS` facade
    /// itself is actor-isolated.
    private let g2pLock = NSLock()

    /// Whether this phonemizer uses the British English Misaki path.
    public let british: Bool

    /// Creates a MisakiSwift-backed phonemizer.
    ///
    /// - Parameter british: When true, asks MisakiSwift for British English
    ///   phonemes. The default is the U.S. English path used by `af_*` voices.
    public init(british: Bool = false) {
        self.british = british
    }

    /// Marker Misaki substitutes for a word it could not resolve.
    ///
    /// Matches `EnglishG2P(unk:)`'s default. The marker has no Kokoro vocab
    /// entry, so it evaporates during tokenization: without counting it here the
    /// loss leaves no trace at all.
    static let unknownMarker: Character = "❓"

    /// Converts raw text to Kokoro-compatible phonemes with MisakiSwift.
    ///
    /// - Parameter text: Raw English text.
    /// - Returns: Non-empty phonemes, their UTF-16 length, and the dropped-word count.
    public func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        let result = phonemizeLocked(text)
        guard !result.phonemes.isEmpty else {
            throw KokoroPhonemizerError.emptyOutput
        }
        return KokoroPhonemeResult(
            phonemes: result.phonemes,
            droppedTokens: result.droppedTokens
        )
    }

    /// Runs Misaki under lock because `EnglishG2P` owns mutable NLP state.
    ///
    /// Misaki's per-word `MToken` type is vended by a transitive dependency that
    /// this package does not import, so the token array is consumed inline where
    /// type inference supplies its element type. Only the two fields that matter
    /// cross into ``isDroppedToken(text:phonemes:)``, which stays testable.
    ///
    /// - Parameter text: Raw English text.
    /// - Returns: Phoneme string and the count of words that lost their sound.
    private func phonemizeLocked(_ text: String) -> (phonemes: String, droppedTokens: Int) {
        g2pLock.lock()
        defer { g2pLock.unlock() }
        let g2p: EnglishG2P
        if let cachedG2P {
            g2p = cachedG2P
        } else {
            g2p = EnglishG2P(british: british)
            cachedG2P = g2p
        }
        let result = g2p.phonemize(text: text)
        let droppedTokens = result.1.reduce(into: 0) { total, token in
            if Self.isDroppedToken(text: token.text, phonemes: token.phonemes) {
                total += 1
            }
        }
        return (result.0, droppedTokens)
    }

    /// Returns whether one Misaki token lost the word it stood for.
    ///
    /// Only words carrying letters or digits count. Punctuation tokens
    /// legitimately carry no phonemes and must never be reported as loss.
    ///
    /// - Parameters:
    ///   - text: Source word the token covers.
    ///   - phonemes: Phonemes Misaki resolved for it, if any.
    /// - Returns: True when a speech-bearing word produced no usable phonemes.
    static func isDroppedToken(text: String, phonemes: String?) -> Bool {
        guard text.contains(where: { $0.isLetter || $0.isNumber }) else {
            return false
        }
        let resolved = phonemes ?? ""
        return resolved.allSatisfy(\.isWhitespace) || resolved.contains(unknownMarker)
    }
}
