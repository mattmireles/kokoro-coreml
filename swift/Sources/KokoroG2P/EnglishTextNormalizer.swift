// Ported from FluidAudio (Apache-2.0), github.com/FluidInference/FluidAudio v0.17.4
// (TTS/Shared/EnglishTextNormalizer.swift, TTS/Shared/NemoTextNormalizer.swift,
// and the number-spelling subset of TTS/SSML/SayAsInterpreter.swift).

import CNemoTextProcessing
import Foundation

/// Written → spoken English before G2P.
///
/// Primary pass is byte-exact NVIDIA NeMo TN via the prebuilt FST engine
/// (currency, measures, dates, ranges, …). Only if NeMo leaves the text
/// unchanged does the conservative baseline run: standalone cardinals,
/// ordinals (`13th`), leading-zero digits (`007`), decimals (`3.14`),
/// meridiem times (`1:49 PM`), decades (`1770s`, `'90s`), years (1000–2099).
enum EnglishTextNormalizer {

    static func normalizeForFrontend(_ text: String) -> String {
        let fst = nemo(text)
        return fst == text ? normalize(text) : fst
    }

    /// NeMo English TN; returns `text` unchanged if the engine declines it.
    static func nemo(_ text: String) -> String {
        guard let ptr = nemo_tn_fst(text, "en") else { return text }
        defer { nemo_free_string(ptr) }
        return String(cString: ptr)
    }

    /// Conservative baseline. Passes run in priority order so a token is
    /// consumed by the most specific rule.
    static func normalize(_ text: String) -> String {
        var result = text
        result = apply(meridiemTimeRegex, to: result, transform: spellMeridiemTime)
        result = apply(decadeRegex, to: result, transform: spellDecade)
        result = apply(decimalRegex, to: result, transform: spellDecimal)
        result = apply(ordinalRegex, to: result, transform: spellOrdinal)
        result = apply(leadingZeroRegex, to: result, transform: spellLeadingZero)
        result = apply(yearRegex, to: result, transform: spellYear)
        result = apply(cardinalRegex, to: result, transform: spellCardinal)
        return result
    }

    // MARK: - Patterns

    // A standalone number must not be glued to a letter, digit, or a `. , :`
    // separator; a trailing sentence period is allowed.
    private static let leadBoundary = #"(?<![A-Za-z0-9.,:])"#
    private static let trailBoundary = #"(?![A-Za-z0-9])(?![.,:][0-9])"#

    private static let meridiemTimeRegex = regex(
        leadBoundary + #"(1[0-2]|[1-9]):([0-5][0-9])\s*([AaPp])(?:\.[Mm]\.?|[Mm])"# + #"(?![A-Za-z])"#)
    private static let decadeRegex = regex(leadBoundary + #"'?([0-9]{4}|[0-9]{2})s"# + #"(?![A-Za-z0-9])"#)
    private static let decimalRegex = regex(leadBoundary + #"([0-9]+)\.([0-9]+)"# + trailBoundary)
    private static let ordinalRegex = regex(leadBoundary + #"([0-9]+)(st|nd|rd|th)"# + #"(?![A-Za-z])"#)
    private static let leadingZeroRegex = regex(leadBoundary + #"(0[0-9]+)"# + trailBoundary)
    private static let yearRegex = regex(leadBoundary + #"([0-9]{4})"# + trailBoundary)
    private static let cardinalRegex = regex(leadBoundary + #"([0-9]+)"# + trailBoundary)

    // MARK: - Per-match spelling (nil leaves the match unchanged)

    private static func spellMeridiemTime(_ groups: [String]) -> String? {
        let spoken = spaced(clockTime(hours: Int(groups[1])!, minutes: Int(groups[2])!))
        guard !containsDigit(spoken) else { return nil }
        let meridiem = groups[3].lowercased() == "p" ? "p m" : "a m"
        return "\(spoken) \(meridiem)"
    }

    private static func spellDecade(_ groups: [String]) -> String? {
        // Only conventional decades ending in 0; skip all-zero (`'00s`).
        let digits = groups[1]
        guard digits.last == "0", let value = Int(digits), value != 0 else { return nil }
        let base = spaced(digits.count == 4 ? year(value) : cardinal(digits))
        guard !base.isEmpty, !containsDigit(base) else { return nil }
        return pluralizeLastWord(base)
    }

    private static func spellYear(_ groups: [String]) -> String? {
        guard let value = Int(groups[1]), (1000...2099).contains(value) else { return nil }
        let spoken = spaced(year(value))
        return containsDigit(spoken) ? nil : spoken
    }

    private static func spellDecimal(_ groups: [String]) -> String? {
        guard let integerPart = cardinalWords(groups[1]) else { return nil }
        let fractionalPart = digitWords(groups[2])
        guard !containsDigit(fractionalPart) else { return nil }
        return "\(integerPart) point \(fractionalPart)"
    }

    private static func spellOrdinal(_ groups: [String]) -> String? {
        guard let number = Int(groups[1]), expectedOrdinalSuffix(for: number) == groups[2].lowercased()
        else { return nil }
        let spoken = spaced(ordinal(number))
        return containsDigit(spoken) ? nil : spoken
    }

    private static func spellLeadingZero(_ groups: [String]) -> String? {
        let spoken = digitWords(groups[1])
        return containsDigit(spoken) ? nil : spoken
    }

    private static func spellCardinal(_ groups: [String]) -> String? {
        cardinalWords(groups[1])
    }

    // MARK: - Number words (SayAsInterpreter subset)

    private static let spellOutFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .spellOut
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.maximumFractionDigits = 0
        formatter.roundingMode = .down
        return formatter
    }()

    private static let digitNames = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    private static let ordinalWords: [Int: String] = [
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
        16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
    ]

    /// `"123"` → `one hundred twenty-three`; digits returned unchanged on overflow.
    private static func cardinal(_ digits: String) -> String {
        guard let number = Int(digits.filter { $0.isNumber || $0 == "-" }) else { return digits }
        return spellOutFormatter.string(from: NSNumber(value: number)) ?? digits
    }

    /// `"123"` → `one two three`.
    private static func digitWords(_ digits: String) -> String {
        digits.compactMap { Int(String($0)).map { digitNames[$0] } }.joined(separator: " ")
    }

    /// `1985` → `nineteen eighty-five`, `2005` → `two thousand five`, `1905` → `nineteen oh five`.
    private static func year(_ year: Int) -> String {
        guard (1000...9999).contains(year) else { return cardinal(String(year)) }
        let century = year / 100
        let remainder = year % 100
        if remainder == 0 {
            return year == 2000 ? "two thousand" : cardinal(String(century)) + " hundred"
        } else if (2000...2009).contains(year) {
            return "two thousand " + cardinal(String(remainder))
        } else if (1...9).contains(remainder) {
            return "\(cardinal(String(century))) oh \(cardinal(String(remainder)))"
        }
        return "\(cardinal(String(century))) \(cardinal(String(remainder)))"
    }

    /// `2:30` → `two thirty`, `3:05` → `three oh five`, `4:00` → `four o'clock`.
    private static func clockTime(hours: Int, minutes: Int) -> String {
        if minutes == 0 { return "\(cardinal(String(hours))) o'clock" }
        if (1...9).contains(minutes) { return "\(cardinal(String(hours))) oh \(cardinal(String(minutes)))" }
        return "\(cardinal(String(hours))) \(cardinal(String(minutes)))"
    }

    /// `23` → `twenty-third`.
    private static func ordinal(_ number: Int) -> String {
        if let word = ordinalWords[number] { return word }
        guard let spelled = spellOutFormatter.string(from: NSNumber(value: number)) else { return "\(number)th" }

        let lastTwoDigits = number % 100
        if (11...13).contains(lastTwoDigits) {
            if spelled.hasSuffix("one") { return String(spelled.dropLast(3)) + "eleventh" }
            if spelled.hasSuffix("two") { return String(spelled.dropLast(3)) + "twelfth" }
            if spelled.hasSuffix("three") { return String(spelled.dropLast(5)) + "thirteenth" }
        }
        let replacements: [Int: (suffix: String, ordinal: String)] = [
            1: ("one", "first"), 2: ("two", "second"), 3: ("three", "third"),
            5: ("five", "fifth"), 8: ("eight", "eighth"), 9: ("nine", "ninth"), 0: ("y", "ieth"),
        ]
        if let rule = replacements[number % 10], spelled.hasSuffix(rule.suffix) {
            return String(spelled.dropLast(rule.suffix.count)) + rule.ordinal
        }
        return spelled + "th"
    }

    // MARK: - Helpers

    private static func cardinalWords(_ digits: String) -> String? {
        let spoken = spaced(cardinal(digits))
        return containsDigit(spoken) ? nil : spoken
    }

    private static func expectedOrdinalSuffix(for number: Int) -> String {
        if (11...13).contains(number % 100) { return "th" }
        switch number % 10 {
        case 1: return "st"
        case 2: return "nd"
        case 3: return "rd"
        default: return "th"
        }
    }

    private static func spaced(_ text: String) -> String {
        text.replacingOccurrences(of: "-", with: " ")
    }

    /// `seventy` → `seventies`, `hundred` → `hundreds`.
    private static func pluralizeLastWord(_ text: String) -> String {
        var words = text.split(separator: " ").map(String.init)
        guard let last = words.last else { return text }
        words[words.count - 1] = last.hasSuffix("y") ? String(last.dropLast()) + "ies" : last + "s"
        return words.joined(separator: " ")
    }

    private static func containsDigit(_ text: String) -> Bool {
        text.contains { $0.isNumber }
    }

    private static func regex(_ pattern: String) -> NSRegularExpression {
        try! NSRegularExpression(pattern: pattern, options: [])
    }

    /// Replace each match with `transform(groups)`, splicing in reverse.
    private static func apply(
        _ regex: NSRegularExpression,
        to text: String,
        transform: ([String]) -> String?
    ) -> String {
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return text }

        let mutable = NSMutableString(string: text)
        for match in matches.reversed() {
            let groups = (0..<match.numberOfRanges).map { index -> String in
                let range = match.range(at: index)
                return range.location == NSNotFound ? "" : ns.substring(with: range)
            }
            if let replacement = transform(groups) {
                mutable.replaceCharacters(in: match.range, with: replacement)
            }
        }
        return mutable as String
    }
}
