import Foundation

/// Boundary-aware splitting for a chunk that overflowed the model's token shape.
///
/// `KokoroTTS.prepare` chunks text for duration first, then re-splits any chunk
/// whose phonemes exceed the runtime duration shape. That retry split used to
/// cut at the nearest whitespace to the midpoint, which produced mid-clause
/// seams and sub-eight-character orphan fragments — about 4% of fragments in a
/// 5,000-input probe. An orphan fragment is voiced as its own utterance with
/// sentence-final intonation, which is what listeners heard as clipped speech.
///
/// The rules here mirror `splitKokoroChunkForRetry` in
/// `packages/protocol/shared/kokoro-chunk-plan.js`, which the web and extension
/// paths already use: prefer sentence and clause boundaries, never break an
/// abbreviation away from the word it modifies, and fold or re-place an
/// undersized tail.
enum KokoroChunkBoundaries {
    /// Minimum characters a retry fragment should carry.
    ///
    /// Mirrors `KOKORO_CHUNK_RETRY_MIN_CHARS`.
    static let retryMinimumCharacters = 32

    /// Splits one oversized chunk at the best available boundary.
    ///
    /// Unlike the JS entry point this never returns the input unchanged: the
    /// caller re-splits recursively until every fragment fits the duration
    /// shape, so a no-op result would not terminate. An empty result means no
    /// usable split exists and the caller must fail the chunk instead.
    ///
    /// - Parameter text: Chunk that exceeded the runtime duration shape.
    /// - Returns: Two or more strictly shorter fragments, or an empty array.
    static func splitForRetry(_ text: String) -> [String] {
        let normalized = KokoroTextProcessor.normalizeWhitespace(text)
        let characters = Array(normalized)
        guard characters.count > 1 else {
            return []
        }
        let maxCharacters = min(characters.count - 1, (characters.count + 1) / 2)
        // The JS planner short-circuits inputs below the minimum instead of
        // clamping. This caller cannot short-circuit, so the floor is clamped to
        // the window it must fit inside.
        let minCharacters = min(retryMinimumCharacters, max(1, maxCharacters))

        // Sentence and clause structure first, exactly like the JS planner's
        // `splitChunkUnit` pass. `TextChunker` is the Swift-side equivalent and
        // already shares Botnet's boundary heuristics.
        let sentenceChunks = TextChunker().chunks(
            for: normalized,
            speed: 1,
            maxCharacters: maxCharacters
        )
        if let usable = usableSplit(
            mergeUndersizedTail(
                sentenceChunks,
                maxCharacters: maxCharacters,
                minCharacters: minCharacters
            ),
            of: normalized
        ) {
            return usable
        }

        let boundary = lastSoftBoundary(
            in: characters,
            maxExclusive: maxCharacters,
            minIndex: Int(Double(maxCharacters) * 0.35)
        )
        let index = boundary > 0 ? boundary : maxCharacters
        let left = KokoroTextProcessor.normalizeWhitespace(String(characters[..<index]))
        let right = KokoroTextProcessor.normalizeWhitespace(String(characters[index...]))
        guard !left.isEmpty, !right.isEmpty else {
            return []
        }
        return usableSplit(
            mergeUndersizedTail(
                [left, right],
                maxCharacters: maxCharacters,
                minCharacters: minCharacters
            ),
            of: normalized
        ) ?? []
    }

    /// Folds or re-places a tail fragment that is too short to voice on its own.
    ///
    /// A sub-minimum tail such as "centers," synthesizes as an isolated
    /// sub-second utterance. It is merged into the previous fragment when the
    /// result still fits, and otherwise the break point moves left so the tail
    /// clears the floor.
    ///
    /// - Parameters:
    ///   - chunks: Candidate fragments in order.
    ///   - maxCharacters: Largest fragment the caller will accept.
    ///   - minCharacters: Smallest tail the caller will voice on its own.
    /// - Returns: Fragments with the undersized tail resolved where possible.
    static func mergeUndersizedTail(
        _ chunks: [String],
        maxCharacters: Int,
        minCharacters: Int
    ) -> [String] {
        var chunks = chunks
        while chunks.count > 1 {
            let tail = chunks[chunks.count - 1]
            if tail.count >= minCharacters {
                break
            }
            let previous = chunks[chunks.count - 2]
            let combined = "\(previous) \(tail)"
            if combined.count <= maxCharacters {
                chunks.replaceSubrange((chunks.count - 2)..., with: [combined])
                continue
            }
            let combinedCharacters = Array(combined)
            let boundary = lastSoftBoundary(
                in: combinedCharacters,
                maxExclusive: min(maxCharacters, combinedCharacters.count - minCharacters),
                minIndex: minCharacters
            )
            guard boundary > 0 else {
                break
            }
            let left = KokoroTextProcessor.normalizeWhitespace(
                String(combinedCharacters[..<boundary])
            )
            let right = KokoroTextProcessor.normalizeWhitespace(
                String(combinedCharacters[boundary...])
            )
            guard !left.isEmpty,
                  !right.isEmpty,
                  right.count >= minCharacters,
                  right.count <= maxCharacters else {
                break
            }
            chunks.replaceSubrange((chunks.count - 2)..., with: [left, right])
            break
        }
        return chunks
    }

    /// Finds the last clause or word boundary inside a window.
    ///
    /// Clause punctuation breaks after the mark; whitespace breaks at the space,
    /// and the caller trims. Numeric commas, decimal points, hyphenated words,
    /// and abbreviation-plus-name pairs are all skipped.
    ///
    /// - Parameters:
    ///   - characters: Full character buffer.
    ///   - maxExclusive: One past the last index the break may occupy.
    ///   - minIndex: Earliest index the break may occupy.
    /// - Returns: Split index, or `-1` when no safe boundary exists.
    static func lastSoftBoundary(in characters: [Character], maxExclusive: Int, minIndex: Int) -> Int {
        let start = min(maxExclusive, characters.count) - 1
        let floor = max(0, minIndex)
        guard start >= floor else {
            return -1
        }
        for index in stride(from: start, through: floor, by: -1) {
            let character = characters[index]
            if character == ";" || character == ":" {
                return index + 1
            }
            if character == ",", !isProtectedSoftBreak(characters, index: index) {
                return index + 1
            }
            if character.isWhitespace {
                if KokoroAbbreviations.isProtectedBoundaryToken(token(in: characters, endingAt: index)) {
                    continue
                }
                return index
            }
        }
        return -1
    }

    /// Protects decimals, comma-grouped numbers, and hyphenated words.
    ///
    /// - Parameters:
    ///   - characters: Full character buffer.
    ///   - index: Candidate separator index.
    /// - Returns: True when the separator must not split the chunk.
    static func isProtectedSoftBreak(_ characters: [Character], index: Int) -> Bool {
        guard index >= 0, index < characters.count else {
            return false
        }
        let character = characters[index]
        let previous = index > 0 ? characters[index - 1] : nil
        let next = index + 1 < characters.count ? characters[index + 1] : nil
        if character == ",", previous?.isNumber == true, next?.isNumber == true {
            return true
        }
        if character == ".", previous?.isNumber == true, next?.isNumber == true {
            return true
        }
        if character == "-",
           previous?.isLetter == true || previous?.isNumber == true,
           next?.isLetter == true || next?.isNumber == true {
            return true
        }
        return false
    }

    /// Returns the whitespace-delimited token ending just before an index.
    ///
    /// - Parameters:
    ///   - characters: Full character buffer.
    ///   - endExclusive: One past the last character of the token.
    /// - Returns: Token text, possibly empty.
    static func token(in characters: [Character], endingAt endExclusive: Int) -> String {
        var start = min(max(0, endExclusive), characters.count)
        while start > 0, !characters[start - 1].isWhitespace {
            start -= 1
        }
        return String(characters[start..<min(max(0, endExclusive), characters.count)])
    }

    /// Accepts a split only when it makes strict progress on the input.
    ///
    /// Returning the input unchanged would make the caller's recursion spin, so
    /// a candidate that collapsed back to the original is rejected outright.
    ///
    /// - Parameters:
    ///   - chunks: Candidate fragments.
    ///   - text: Normalized input the fragments came from.
    /// - Returns: Usable fragments, or `nil` when the split made no progress.
    private static func usableSplit(_ chunks: [String], of text: String) -> [String]? {
        let cleaned = chunks.filter { !$0.isEmpty }
        guard cleaned.count > 1, !cleaned.contains(text) else {
            return nil
        }
        return cleaned
    }
}
