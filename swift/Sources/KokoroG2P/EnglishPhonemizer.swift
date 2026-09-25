// Ported from FluidAudio (Apache-2.0), github.com/FluidInference/FluidAudio v0.17.4
// (TTS/KokoroAne/G2P/English/KokoroAneEnglishPhonemizer.swift + TTS/Shared/EnglishInitialisms.swift).

import Foundation
import os

/// English words → Misaki-style IPA.
///
/// Word resolution order (mirrors Kokoro's Misaki frontend):
///   1. caller-supplied custom lexicon (case-sensitive, then lower-cased)
///   2. letter-name overrides for `AI` / `US` spelled from per-letter entries
///   3. case-sensitive Misaki lexicon hit on the original spelling
///   4. case-sensitive hit on the normalized lower-case form
///   5. lower-cased Misaki lexicon hit (weak function-word forms: `to` → `tu`)
///   6. strict ASCII all-caps initialisms (`FBI`) spelled as letter names
///   7. whole-compound possessive stem lookup (`C-section's`)
///   8. hyphenated-compound split (`land-use's` → `land` + `use's`)
///   9. `-'s` stem + clitic for known stems (`today's` → `today` + /z/)
///   10. BART G2P fallback for OOV words (injected by the caller)
///
/// Vocab-supported punctuation is kept and attached to the preceding word,
/// matching upstream `KPipeline.g2p` output. Misaki diphthong shorthand
/// (`A O I Y W`) is kept: the Kokoro vocab carries those tokens directly.
struct EnglishPhonemizer {
    private static let logger = Logger(subsystem: "KokoroG2P", category: "EnglishPhonemizer")

    /// Lower-cased word → phoneme tokens (pre-filtered against the vocab).
    let wordToPhonemes: [String: [String]]
    /// Original-case word → phoneme tokens (`"AI"`, `"iPhone"`, …).
    let caseSensitiveWordToPhonemes: [String: [String]]
    /// Caller overrides (word → IPA). Exact spelling wins over lower-cased.
    var customLexicon: [String: String] = [:]
    /// Punctuation characters the Kokoro vocab can encode; others are dropped.
    let allowedPunctuation: Set<Character>

    /// Convert text to IPA. Words are joined with single spaces; punctuation
    /// attaches to the preceding word (`"Hello, world!"` → `"həlˈO, wˈɜɹld!"`).
    ///
    /// - Parameter fallback: per-word G2P for lexicon misses; receives the
    ///   normalized (lower-cased) spelling. `nil` skips the word.
    func phonemize(_ text: String, fallback: (String) throws -> [String]?) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw KokoroG2PError("empty input") }

        // Fold typographic apostrophes so `we’re` stays one token.
        let prepared = Self.normalizeApostrophes(trimmed)

        var parts: [String] = []
        for token in Self.splitWords(prepared) {
            if token.isEmpty { continue }

            // Punctuation token (single non-word char from the splitter).
            if token.count == 1, let ch = token.first, !ch.isLetter, !ch.isNumber {
                guard allowedPunctuation.contains(ch) else { continue }
                if parts.isEmpty {
                    parts.append(String(ch))
                } else {
                    parts[parts.count - 1].append(ch)
                }
                continue
            }

            if let ipa = try resolveWord(token, fallback: fallback) {
                parts.append(ipa)
            }
        }

        let joined = parts.joined(separator: " ")
        if joined.isEmpty { throw KokoroG2PError("produced no phonemes for input '\(trimmed)'") }
        return joined
    }

    // MARK: - Word resolution

    /// - Parameter allowFallback: `false` skips BART G2P so an OOV stem in the
    ///   possessive rule resolves to `nil` (Misaki's `stem_s` semantics).
    private func resolveWord(
        _ word: String,
        allowFallback: Bool = true,
        fallback: (String) throws -> [String]?
    ) throws -> String? {
        let normalized = Self.normalizeKey(word)
        // Lower-cased with hyphens intact: the only form that reaches the
        // lexicon's hyphenated keys (`twenty-one`).
        let lowered = word.lowercased()

        if let custom = customLexicon[word] ?? customLexicon[normalized] {
            return custom
        }

        // `AI` / `US` have bundled entries that don't read as letter names.
        if Self.letterNameOverrides.contains(word) {
            if let spelled = spellAsLetterNames(word) {
                return spelled
            }
            Self.logger.warning(
                "Letter-name override '\(word, privacy: .public)' unspellable; using bundled pronunciation")
        }

        if let phonemes = lookupMisakiWord(word) {
            return phonemes
        }

        if Self.isInitialismCandidate(word), let spelled = spellAsLetterNames(word) {
            return spelled
        }

        if let possessive = resolveWholeCompoundPossessive(word, lowered: lowered) {
            return possessive
        }

        if word.contains("-"),
            let compound = try resolveHyphenatedCompound(word, allowFallback: allowFallback, fallback: fallback)
        {
            return compound
        }

        if let possessive = try resolvePossessive(word, lowered: lowered, fallback: fallback) {
            return possessive
        }

        guard allowFallback, !normalized.isEmpty else { return nil }
        if let phonemes = try fallback(normalized), !phonemes.isEmpty {
            return phonemes.joined()
        }
        Self.logger.warning("G2P returned nil for word '\(normalized, privacy: .public)' — skipping")
        return nil
    }

    /// Direct lexicon lookup: no spelling, splitting, stemming, or G2P.
    private func lookupMisakiWord(_ word: String) -> String? {
        let normalized = Self.normalizeKey(word)
        guard
            let phonemes = caseSensitiveWordToPhonemes[word]
                ?? caseSensitiveWordToPhonemes[normalized]
                ?? wordToPhonemes[word.lowercased()]
                ?? wordToPhonemes[normalized],
            !phonemes.isEmpty
        else {
            return nil
        }
        return phonemes.joined()
    }

    /// Lexicon-only probe of a whole hyphenated possessive stem (`mother-in-law's`).
    private func resolveWholeCompoundPossessive(_ word: String, lowered: String) -> String? {
        guard word.contains("-"), lowered.hasSuffix("'s") else { return nil }
        let stem = String(word.dropLast(2))
        guard !stem.isEmpty, !stem.hasSuffix("'") else { return nil }
        guard
            let stemIPA = customLexicon[stem]
                ?? customLexicon[Self.normalizeKey(stem)]
                ?? lookupMisakiWord(stem),
            !stemIPA.isEmpty
        else {
            return nil
        }
        return stemIPA + Self.clitic(after: stemIPA)
    }

    /// Split a hyphenated compound and resolve each part; `nil` if any part fails.
    private func resolveHyphenatedCompound(
        _ word: String,
        allowFallback: Bool,
        fallback: (String) throws -> [String]?
    ) throws -> String? {
        let parts = word.split(separator: "-", omittingEmptySubsequences: true).map(String.init)
        guard parts.count >= 2 else { return nil }

        var resolved: [String] = []
        for part in parts {
            guard let ipa = try resolveWord(part, allowFallback: allowFallback, fallback: fallback),
                !ipa.isEmpty
            else {
                return nil
            }
            resolved.append(ipa)
        }
        return resolved.joined(separator: " ")
    }

    // MARK: - Possessive / `-'s` clitic

    /// Stem + `-s` clitic when the stem is a known word (Misaki `Lexicon.stem_s`).
    private func resolvePossessive(
        _ word: String,
        lowered: String,
        fallback: (String) throws -> [String]?
    ) throws -> String? {
        guard lowered.count >= 3, lowered.hasSuffix("'s") else { return nil }
        let stem = String(word.dropLast(2))
        guard !stem.isEmpty, !stem.hasSuffix("'") else { return nil }
        guard let stemIPA = try resolveWord(stem, allowFallback: false, fallback: fallback), !stemIPA.isEmpty
        else {
            return nil
        }
        return stemIPA + Self.clitic(after: stemIPA)
    }

    private static let voicelessNonSibilants: Set<Character> = ["p", "t", "k", "f", "θ"]
    private static let sibilants: Set<Character> = ["s", "z", "ʃ", "ʒ", "ʧ", "ʤ"]

    /// US `-s` clitic by English phonology (Misaki `Lexicon._s`).
    static func clitic(after stemIPA: String) -> String {
        guard let last = stemIPA.last else { return "z" }
        if voicelessNonSibilants.contains(last) { return "s" }
        if sibilants.contains(last) { return "ᵻz" }
        return "z"
    }

    // MARK: - Letter-name initialisms

    /// Exact uppercase spellings whose lexicon entry isn't the letter-name reading.
    private static let letterNameOverrides: Set<String> = ["AI", "US"]

    /// Strict ASCII all-caps token of length 2...5 (`FBI`, `ATP`).
    private static func isInitialismCandidate(_ word: String) -> Bool {
        guard (2...5).contains(word.count) else { return false }
        return word.allSatisfy { $0.isASCII && $0.isUppercase && $0.isLetter }
    }

    /// Spell a token as letter names (`FBI` → `ˈɛf bˈi ˈI`); `nil` if any letter is missing.
    private func spellAsLetterNames(_ word: String) -> String? {
        var letters: [String] = []
        for character in word {
            guard let tokens = caseSensitiveWordToPhonemes[String(character)], !tokens.isEmpty else {
                return nil
            }
            letters.append(tokens.joined())
        }
        return letters.isEmpty ? nil : letters.joined(separator: " ")
    }

    // MARK: - Text helpers

    private static let smartApostrophes: Set<Character> = ["\u{2019}", "\u{2018}", "\u{02BC}"]

    /// Fold `’ ‘ ʼ` to ASCII `'`.
    static func normalizeApostrophes(_ text: String) -> String {
        guard text.contains(where: { smartApostrophes.contains($0) }) else { return text }
        return String(text.map { smartApostrophes.contains($0) ? "'" : $0 })
    }

    /// Lowercase + keep only letters, digits, and `'`.
    static func normalizeKey(_ word: String) -> String {
        let allowedSet = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))
        let filtered = word.lowercased().unicodeScalars.filter { allowedSet.contains($0) }
        return String(String.UnicodeScalarView(filtered))
    }

    private static let knownLeadingApostropheWords: Set<String> = [
        "'cause", "'em", "'til", "'tis", "'twas", "'twere",
    ]

    /// Runs of letters/digits (internal `'` and `-` stay inside words),
    /// single punctuation chars as their own tokens, whitespace dropped.
    static func splitWords(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""

        func flushCurrent() {
            if !current.isEmpty {
                out.append(current)
                current.removeAll(keepingCapacity: true)
            }
        }

        for index in text.indices {
            let ch = text[index]
            if ch.isWhitespace {
                flushCurrent()
            } else if ch == "'" {
                let nextIndex = text.index(after: index)
                let nextIsWord =
                    nextIndex < text.endIndex && (text[nextIndex].isLetter || text[nextIndex].isNumber)
                if !current.isEmpty && nextIsWord {
                    current.append(ch)
                } else if current.isEmpty && startsKnownLeadingApostropheWord(in: text, at: index) {
                    current.append(ch)
                } else {
                    flushCurrent()
                    out.append(String(ch))
                }
            } else if ch.isLetter || ch.isNumber || ch == "-" {
                current.append(ch)
            } else {
                flushCurrent()
                out.append(String(ch))
            }
        }
        flushCurrent()
        return out
    }

    private static func startsKnownLeadingApostropheWord(in text: String, at apostropheIndex: String.Index) -> Bool {
        let nextIndex = text.index(after: apostropheIndex)
        guard nextIndex < text.endIndex, text[nextIndex].isLetter else { return false }
        var endIndex = nextIndex
        while endIndex < text.endIndex, text[endIndex].isLetter {
            endIndex = text.index(after: endIndex)
        }
        return knownLeadingApostropheWords.contains("'" + text[nextIndex..<endIndex].lowercased())
    }
}
