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

    /// Converts raw text to Kokoro-compatible phonemes with MisakiSwift.
    ///
    /// - Parameter text: Raw English text.
    /// - Returns: Non-empty phonemes and their UTF-16 length.
    public func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        let phonemes = phonemizeLocked(text)
        guard !phonemes.isEmpty else {
            throw KokoroPhonemizerError.emptyOutput
        }
        return KokoroPhonemeResult(phonemes: phonemes)
    }

    /// Runs Misaki under lock because `EnglishG2P` owns mutable NLP state.
    private func phonemizeLocked(_ text: String) -> String {
        g2pLock.lock()
        defer { g2pLock.unlock() }
        if let cachedG2P {
            return cachedG2P.phonemize(text: text).0
        }
        let created = EnglishG2P(british: british)
        cachedG2P = created
        return created.phonemize(text: text).0
    }
}
