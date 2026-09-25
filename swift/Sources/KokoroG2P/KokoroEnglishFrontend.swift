// Ported from FluidAudio (Apache-2.0), github.com/FluidInference/FluidAudio v0.17.4
// (KokoroAneManager English frontend wiring + LexiconAssetCache).

import Foundation

/// English text → Kokoro IPA phonemes, with no FluidAudio dependency.
///
/// Pipeline: NeMo text normalization (`$42.50` → `forty two dollars fifty
/// cents`) → Misaki lexicon lookup → BART G2P Core ML fallback for
/// out-of-vocabulary words. Output uses the Kokoro v1.0 vocab alphabet.
public actor KokoroEnglishFrontend {
    private var phonemizer: EnglishPhonemizer
    private let g2p: G2PModel

    /// Load all assets up front.
    ///
    /// `assetsDirectory` must contain:
    /// - `us_lexicon_cache.json` — Misaki US lexicon (`{lower, caseSensitive}`)
    /// - `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc` — BART G2P for OOV words
    /// - `g2p_vocab.json` — BART G2P grapheme/phoneme tables
    public init(assetsDirectory: URL) throws {
        let (lower, caseSensitive) = try Self.loadLexicon(
            assetsDirectory.appendingPathComponent("us_lexicon_cache.json"))
        // Stress/length marks (ˈ ˌ ː) are Unicode modifier letters, so
        // `isLetter` keeps them out of the punctuation set.
        let punctuation = Set(
            KokoroSymbols.all.map(Character.init).filter { !$0.isLetter && !$0.isNumber && !$0.isWhitespace })
        phonemizer = EnglishPhonemizer(
            wordToPhonemes: lower,
            caseSensitiveWordToPhonemes: caseSensitive,
            allowedPunctuation: punctuation)
        g2p = try G2PModel(directory: assetsDirectory)
    }

    /// Text → Kokoro IPA phoneme string (same alphabet as the Kokoro vocab).
    public func phonemes(for text: String) async throws -> String {
        let normalized = EnglishTextNormalizer.normalizeForFrontend(text)
        return try phonemizer.phonemize(normalized) { [g2p] word in
            try g2p.phonemize(word: word)
        }
    }

    /// Optional overrides: word → IPA. Applied before lexicon lookup.
    public func setCustomLexicon(_ entries: [String: String]) {
        phonemizer.customLexicon = entries
    }

    // MARK: - Lexicon

    private struct LexiconPayload: Decodable {
        let lower: [String: [String]]
        let caseSensitive: [String: [String]]
    }

    /// Load `us_lexicon_cache.json`, dropping phoneme tokens the Kokoro vocab
    /// cannot encode and joining each entry into one string. The file stores
    /// every phoneme as its own JSON string; kept as `[String]` in memory that
    /// cost ~127 MB (one heap String per phoneme per word). Every lookup
    /// joined them anyway.
    private static func loadLexicon(_ url: URL) throws -> ([String: String], [String: String]) {
        let payload = try JSONDecoder().decode(LexiconPayload.self, from: Data(contentsOf: url))
        let allowed = Set(KokoroSymbols.all.map(String.init))
        let lower = payload.lower.mapValues { $0.filter { allowed.contains($0) }.joined() }
        let caseSensitive = payload.caseSensitive.mapValues { $0.filter { allowed.contains($0) }.joined() }
        guard !lower.isEmpty else { throw KokoroG2PError("us_lexicon_cache.json has no usable entries") }
        return (lower, caseSensitive)
    }
}

/// Frontend failure with a human-readable reason.
public struct KokoroG2PError: Error, CustomStringConvertible {
    public let description: String
    init(_ description: String) { self.description = description }
}

/// The 114 symbols of the Kokoro v1.0 vocab (hexgrad/Kokoro-82M `config.json`),
/// one Unicode scalar each, in token-id order.
enum KokoroSymbols {
    static let all: [Unicode.Scalar] = Array(
        (";:,.!?\u{2014}\u{2026}\u{0022}()\u{201C}\u{201D} \u{0303}\u{02A3}\u{02A5}\u{02A6}\u{02A8}\u{1D5D}\u{AB67}"
            + "AIOQSTWY\u{1D4A}abcdefhijklmnopqrstuvwxyz"
            + "\u{0251}\u{0250}\u{0252}\u{00E6}\u{03B2}\u{0254}\u{0255}\u{00E7}\u{0256}\u{00F0}\u{02A4}\u{0259}"
            + "\u{025A}\u{025B}\u{025C}\u{025F}\u{0261}\u{0265}\u{0268}\u{026A}\u{029D}\u{026F}\u{0270}\u{014B}"
            + "\u{0273}\u{0272}\u{0274}\u{00F8}\u{0278}\u{03B8}\u{0153}\u{0279}\u{027E}\u{027B}\u{0281}\u{027D}"
            + "\u{0282}\u{0283}\u{0288}\u{02A7}\u{028A}\u{028B}\u{028C}\u{0263}\u{0264}\u{03C7}\u{028E}\u{0292}"
            + "\u{0294}\u{02C8}\u{02CC}\u{02D0}\u{02B0}\u{02B2}\u{2193}\u{2192}\u{2197}\u{2198}\u{1D7B}").unicodeScalars)
}
