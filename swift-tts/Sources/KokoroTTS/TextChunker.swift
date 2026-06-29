import Foundation

/// Botnet-compatible text chunker with an SDK-specific configurable duration cap.
struct TextChunker: Sendable {
    /// SDK starter-profile chunk cap.
    static let defaultMaxChunkSeconds = KokoroSynthesisOptions.defaultMaxChunkSeconds

    /// Botnet fleet chunk cap preserved for parity tests and overrides.
    static let botnetMaxChunkSeconds = 30.0

    /// Maximum estimated output duration returned by duration estimates.
    static let maxOutputSeconds = 900

    /// Character-rate heuristic copied from Botnet's fleet chunker.
    private static let estimatedCharactersPerSecond = 14.0

    /// Sentence-boundary abbreviations copied from Botnet's fleet chunker.
    private static let abbreviations: Set<String> = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "No.", "vs.", "etc.", "e.g.", "i.e.",
    ]

    /// Maximum estimated seconds allowed in one chunk.
    let maxChunkSeconds: Double

    /// Creates a text chunker.
    ///
    /// - Parameter maxChunkSeconds: Maximum estimated seconds per chunk.
    init(maxChunkSeconds: Double = Self.defaultMaxChunkSeconds) {
        self.maxChunkSeconds = max(0.25, maxChunkSeconds)
    }

    /// Compatibility tokenizer copied from Botnet for recursive fallback tests.
    ///
    /// - Parameter text: Text to map to pseudo-token IDs.
    /// - Returns: Deterministic scalar-derived token IDs.
    func tokenize(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value % 255) + 1 }
    }

    /// Splits text using default speed.
    ///
    /// - Parameter text: Raw text to split.
    /// - Returns: Single-spaced chunks.
    func chunks(for text: String) -> [String] {
        chunks(for: text, speed: 1.0)
    }

    /// Splits text using a speech-speed multiplier.
    ///
    /// - Parameters:
    ///   - text: Raw text to split.
    ///   - speed: Speech speed multiplier.
    /// - Returns: Single-spaced chunks.
    func chunks(for text: String, speed: Double) -> [String] {
        chunks(for: text, speed: speed, maxCharacters: nil)
    }

    /// Splits text with an optional hard character cap.
    ///
    /// - Parameters:
    ///   - text: Raw text to split.
    ///   - speed: Speech speed multiplier.
    ///   - maxCharacters: Optional hard character cap for recursive fallback.
    /// - Returns: Single-spaced chunks.
    func chunks(for text: String, speed: Double, maxCharacters: Int?) -> [String] {
        let normalized = KokoroTextProcessor.normalizeWhitespace(text)
        guard !normalized.isEmpty else {
            return []
        }

        let maxCharacters = maxCharacters.map { max(1, $0) }
        let chunks = sentenceUnits(for: normalized).flatMap { unit in
            isWithinLimit(unit, maxCharacters: maxCharacters, speed: speed)
                ? [unit]
                : splitOversizedUnit(unit, maxCharacters: maxCharacters, speed: speed)
        }
        return chunks.isEmpty ? [normalized] : chunks
    }

    /// Builds sentence-like units without splitting common abbreviations.
    ///
    /// - Parameter text: Normalized text.
    /// - Returns: Sentence-like units.
    private func sentenceUnits(for text: String) -> [String] {
        var units: [String] = []
        var current = ""
        let characters = Array(text)
        var index = 0
        while index < characters.count {
            let character = characters[index]
            current.append(character)
            if ".!?".contains(character), isLikelyBoundary(text, punctuationIndex: index) {
                var closerIndex = index + 1
                while closerIndex < characters.count, ")\"'”’]".contains(characters[closerIndex]) {
                    current.append(characters[closerIndex])
                    closerIndex += 1
                }
                appendTrimmed(current, to: &units)
                current = ""
                index = closerIndex
                continue
            }
            index += 1
        }
        appendTrimmed(current, to: &units)
        return units.isEmpty ? [text] : units
    }

    /// Decides whether punctuation is a likely sentence boundary.
    ///
    /// - Parameters:
    ///   - text: Normalized text.
    ///   - punctuationIndex: Character-array index of punctuation.
    /// - Returns: True when the punctuation should end the current sentence.
    private func isLikelyBoundary(_ text: String, punctuationIndex: Int) -> Bool {
        let characters = Array(text)
        if punctuationIndex > 0,
           punctuationIndex + 1 < characters.count,
           characters[punctuationIndex - 1].isLetterOrNumber,
           characters[punctuationIndex + 1].isLetterOrNumber {
            return false
        }

        var tokenStart = punctuationIndex
        while tokenStart > 0, !characters[tokenStart - 1].isWhitespace {
            tokenStart -= 1
        }
        let token = String(characters[tokenStart...punctuationIndex])
        if Self.abbreviations.contains(token) {
            return false
        }

        var nextIndex = punctuationIndex + 1
        while nextIndex < characters.count, ")\"'”’]".contains(characters[nextIndex]) {
            nextIndex += 1
        }
        while nextIndex < characters.count, characters[nextIndex].isWhitespace {
            nextIndex += 1
        }
        guard nextIndex < characters.count else {
            return true
        }
        let next = characters[nextIndex]
        if token.count == 2,
           token.first?.isUppercase == true,
           token.last == ".",
           next.isUppercase {
            return false
        }
        return next.isUppercase || next.isNumber || "\"“‘([".contains(next)
    }

    /// Splits one oversized sentence-like unit.
    ///
    /// - Parameters:
    ///   - unit: Sentence-like text unit.
    ///   - maxCharacters: Optional hard character cap.
    ///   - speed: Speech speed multiplier.
    /// - Returns: Smaller chunks.
    private func splitOversizedUnit(_ unit: String, maxCharacters: Int?, speed: Double) -> [String] {
        if let maxCharacters {
            return splitByCharacterWindow(unit, maxCharacters: maxCharacters)
        }

        let punctuationParts = split(unit: unit, afterAnyOf: [",", ";", ":"])
        if punctuationParts.count > 1 {
            let packed = pack(parts: punctuationParts, maxCharacters: maxCharacters, speed: speed)
            if packed.allSatisfy({ isWithinLimit($0, maxCharacters: maxCharacters, speed: speed) }) {
                return packed
            }
        }

        return splitByWords(unit, maxCharacters: maxCharacters, speed: speed)
    }

    /// Splits text using a hard character window and safe soft breaks.
    ///
    /// - Parameters:
    ///   - text: Text to split.
    ///   - maxCharacters: Maximum characters per chunk.
    /// - Returns: Windowed chunks.
    private func splitByCharacterWindow(_ text: String, maxCharacters: Int) -> [String] {
        if text.count <= maxCharacters {
            return [text]
        }

        var chunks: [String] = []
        var remaining = text
        while remaining.count > maxCharacters {
            let window = String(remaining.prefix(maxCharacters))
            let breakAt = lastSafeSoftBreakIndex(in: window, separators: [";", ":", ",", " "]) ?? -1
            let index = breakAt > maxCharacters / 2 ? breakAt + 1 : maxCharacters
            let splitIndex = remaining.index(remaining.startIndex, offsetBy: index)
            chunks.append(String(remaining[..<splitIndex]).trimmingCharacters(in: .whitespacesAndNewlines))
            remaining = String(remaining[splitIndex...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if !remaining.isEmpty {
            chunks.append(remaining)
        }
        return chunks
    }

    /// Splits a unit after unprotected soft punctuation.
    ///
    /// - Parameters:
    ///   - unit: Text unit to split.
    ///   - separators: Soft punctuation characters.
    /// - Returns: Split parts.
    private func split(unit: String, afterAnyOf separators: Set<Character>) -> [String] {
        var parts: [String] = []
        var current = ""
        let characters = Array(unit)
        for (index, character) in characters.enumerated() {
            current.append(character)
            if separators.contains(character), !isProtectedSoftBreak(characters, index: index) {
                appendTrimmed(current, to: &parts)
                current = ""
            }
        }
        appendTrimmed(current, to: &parts)
        return parts
    }

    /// Packs candidate parts into chunks under the active limit.
    ///
    /// - Parameters:
    ///   - parts: Candidate chunk parts.
    ///   - maxCharacters: Optional hard character cap.
    ///   - speed: Speech speed multiplier.
    /// - Returns: Packed chunks.
    private func pack(parts: [String], maxCharacters: Int?, speed: Double) -> [String] {
        var chunks: [String] = []
        var current = ""
        for part in parts {
            let candidate = current.isEmpty ? part : "\(current) \(part)"
            if isWithinLimit(candidate, maxCharacters: maxCharacters, speed: speed) {
                current = candidate
            } else {
                if !current.isEmpty {
                    chunks.append(current)
                }
                current = part
            }
        }
        if !current.isEmpty {
            chunks.append(current)
        }
        return chunks
    }

    /// Finds the last unprotected separator index inside a text window.
    ///
    /// - Parameters:
    ///   - text: Candidate text window.
    ///   - separators: Allowed soft break characters.
    /// - Returns: Last safe break index when present.
    private func lastSafeSoftBreakIndex(in text: String, separators: Set<Character>) -> Int? {
        let characters = Array(text)
        for index in stride(from: characters.count - 1, through: 0, by: -1) {
            if separators.contains(characters[index]), !isProtectedSoftBreak(characters, index: index) {
                return index
            }
        }
        return nil
    }

    /// Protects decimals, comma-grouped numbers, and hyphenated words.
    ///
    /// - Parameters:
    ///   - characters: Full character buffer.
    ///   - index: Candidate separator index.
    /// - Returns: True when the separator must not split the chunk.
    private func isProtectedSoftBreak(_ characters: [Character], index: Int) -> Bool {
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
        if character == "-", previous?.isLetterOrNumber == true, next?.isLetterOrNumber == true {
            return true
        }
        return false
    }

    /// Splits by words, falling back to character chunks when needed.
    ///
    /// - Parameters:
    ///   - text: Text to split.
    ///   - maxCharacters: Optional hard character cap.
    ///   - speed: Speech speed multiplier.
    /// - Returns: Split chunks.
    private func splitByWords(_ text: String, maxCharacters: Int?, speed: Double) -> [String] {
        let words = text.split(separator: " ").map(String.init)
        let packed = pack(parts: words, maxCharacters: maxCharacters, speed: speed)
        return packed.flatMap { chunk in
            isWithinLimit(chunk, maxCharacters: maxCharacters, speed: speed)
                ? [chunk]
                : splitByCharacters(chunk, maxCharacters: maxCharacters, speed: speed)
        }
    }

    /// Splits text into fixed character chunks.
    ///
    /// - Parameters:
    ///   - text: Text to split.
    ///   - maxCharacters: Optional hard character cap.
    ///   - speed: Speech speed multiplier.
    /// - Returns: Character chunks.
    private func splitByCharacters(_ text: String, maxCharacters: Int?, speed: Double) -> [String] {
        let maxCharacters = maxCharacters
            ?? max(1, Int(maxChunkSeconds * Self.estimatedCharactersPerSecond * max(0.1, speed)))
        var chunks: [String] = []
        var current = ""
        for character in text {
            if current.count >= maxCharacters {
                chunks.append(current)
                current = ""
            }
            current.append(character)
        }
        if !current.isEmpty {
            chunks.append(current)
        }
        return chunks
    }

    /// Checks whether text fits under the active chunk limit.
    ///
    /// - Parameters:
    ///   - text: Candidate chunk text.
    ///   - maxCharacters: Optional hard character cap.
    ///   - speed: Speech speed multiplier.
    /// - Returns: True when the chunk fits.
    private func isWithinLimit(_ text: String, maxCharacters: Int?, speed: Double) -> Bool {
        if let maxCharacters {
            return text.count <= maxCharacters
        }
        return estimatedDurationSeconds(text: text, speed: speed) <= maxChunkSeconds
    }

    /// Appends a trimmed non-empty string.
    ///
    /// - Parameters:
    ///   - value: Candidate string.
    ///   - values: Destination array.
    private func appendTrimmed(_ value: String, to values: inout [String]) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            values.append(trimmed)
        }
    }

    /// Estimates speech duration using default speed.
    ///
    /// - Parameter text: Text to estimate.
    /// - Returns: Estimated duration in seconds.
    public func estimatedDurationSeconds(text: String) -> Double {
        estimatedDurationSeconds(text: text, speed: 1.0)
    }

    /// Estimates speech duration with a speed multiplier.
    ///
    /// - Parameters:
    ///   - text: Text to estimate.
    ///   - speed: Speech speed multiplier.
    /// - Returns: Estimated duration in seconds.
    public func estimatedDurationSeconds(text: String, speed: Double) -> Double {
        let normalizedSpeed = max(0.1, speed)
        return min(
            Double(Self.maxOutputSeconds),
            max(0.25, Double(text.count) / (Self.estimatedCharactersPerSecond * normalizedSpeed))
        )
    }
}

private extension Character {
    /// Whether every scalar in this character is alphanumeric.
    var isLetterOrNumber: Bool {
        unicodeScalars.allSatisfy { CharacterSet.alphanumerics.contains($0) }
    }
}
