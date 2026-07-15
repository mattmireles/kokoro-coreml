import Foundation
import KokoroPipeline

/// Errors raised while turning raw text into prepared Kokoro model inputs.
public enum KokoroTextProcessingError: Error, Equatable, LocalizedError {
    /// Text was empty after whitespace normalization.
    case emptyText

    /// The bundled Kokoro vocabulary could not be loaded.
    case vocabUnavailable

    /// Phonemization succeeded but no phoneme characters survived vocab lookup.
    case emptyTokenization

    /// The requested speed was zero, negative, NaN, or infinite.
    case invalidSpeed(Float)

    /// The prepared chunk exceeds the public caller token budget.
    case tokenBudgetExceeded(actual: Int, maximum: Int)

    /// A voice embedding row did not have the required 256 floats.
    case invalidVoiceEmbedding(actual: Int, expected: Int)

    /// Human-readable explanation for app logs and tests.
    public var errorDescription: String? {
        switch self {
        case .emptyText:
            return "Kokoro text is empty after whitespace normalization."
        case .vocabUnavailable:
            return "Kokoro vocabulary is unavailable."
        case .emptyTokenization:
            return "Kokoro tokenization produced no model tokens."
        case .invalidSpeed(let speed):
            return "Kokoro speed must be positive and finite; observed \(speed)."
        case .tokenBudgetExceeded(let actual, let maximum):
            return "Kokoro chunk has \(actual) tokens, exceeding the \(maximum)-token budget."
        case .invalidVoiceEmbedding(let actual, let expected):
            return "Kokoro voice embedding has \(actual) floats, expected \(expected)."
        }
    }
}

/// Converts raw English text into `KokoroPreparedInput` values.
///
/// The processor owns only deterministic prep: whitespace normalization, Misaki
/// phoneme invocation through ``KokoroPhonemizer``, checked vocab lookup,
/// BOS/EOS framing, duration-shape padding, attention-mask construction, and
/// voice-row selection. Core ML model loading stays in later SDK phases.
public struct KokoroTextProcessor {
    /// BOS/EOS token ID used by Kokoro tokenization.
    public static let boundaryTokenID = KokoroVocabulary.bosEosTokenId

    /// Verified hn-NSF weights digest bundled by Phase 2.
    public static let hnsfWeightsSHA256 = "25a471a6fc81fc9c5ff7c46e4be9d9ec3710dbbfea6e121a99fac75e4a97ad99"

    /// Phonemizer backend used for raw text.
    private let phonemizer: KokoroPhonemizer

    /// Kokoro phoneme-character vocabulary table.
    private let vocab: [String: Int32]

    /// Creates a text processor with the bundled SDK vocab.
    ///
    /// - Parameter phonemizer: Raw-text phonemizer backend.
    /// - Throws: ``KokoroTextProcessingError/vocabUnavailable`` if the bundled
    ///   vocab cannot be read.
    public init(phonemizer: KokoroPhonemizer) throws {
        self.phonemizer = phonemizer
        self.vocab = try Self.loadBundledVocab()
    }

    /// Creates a text processor with an explicit vocab table.
    ///
    /// - Parameters:
    ///   - phonemizer: Raw-text phonemizer backend.
    ///   - vocab: Kokoro phoneme-character vocabulary table.
    public init(phonemizer: KokoroPhonemizer, vocab: [String: Int32]) {
        self.phonemizer = phonemizer
        self.vocab = vocab
    }

    /// Loads the checked Kokoro vocab bundled with the SDK package.
    ///
    /// - Returns: Phoneme-character vocabulary table.
    /// - Throws: ``KokoroTextProcessingError/vocabUnavailable`` if the resource
    ///   is absent or malformed.
    public static func loadBundledVocab() throws -> [String: Int32] {
        let url = try KokoroRuntimeAssets.url(for: .vocab)
        return try loadVocab(from: url)
    }

    /// Loads a Kokoro vocab file from an explicit URL.
    ///
    /// - Parameter url: JSON file with a top-level `vocab` dictionary.
    /// - Returns: Phoneme-character vocabulary table.
    public static func loadVocab(from url: URL) throws -> [String: Int32] {
        let data = try Data(contentsOf: url)
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rawVocab = json["vocab"] as? [String: Any]
        else {
            throw KokoroTextProcessingError.vocabUnavailable
        }
        var table: [String: Int32] = [:]
        for (key, value) in rawVocab {
            if let number = value as? NSNumber {
                table[key] = number.int32Value
            } else if let int = value as? Int {
                table[key] = Int32(int)
            } else {
                throw KokoroTextProcessingError.vocabUnavailable
            }
        }
        return table
    }

    /// Converts one raw text chunk into a model-ready prepared input.
    ///
    /// - Parameters:
    ///   - text: Raw or already chunked English text.
    ///   - voice: Voice identifier used for metadata.
    ///   - refS: Selected 256-float voice embedding row.
    ///   - options: Synthesis prep options.
    ///   - key: Optional fixture or batch key.
    ///   - phonemeResult: Optional precomputed phoneme result. Supplying this
    ///     avoids a second G2P pass when the caller already phonemized to choose
    ///     a voice embedding row.
    /// - Returns: Prepared input with padded token IDs and attention mask.
    public func prepare(
        text: String,
        voice: KokoroVoiceID,
        refS: [Float],
        options: KokoroSynthesisOptions = KokoroSynthesisOptions(),
        key: String? = nil,
        phonemeResult: KokoroPhonemeResult? = nil
    ) throws -> KokoroPreparedInput {
        let normalized = Self.normalizeWhitespace(text)
        guard !normalized.isEmpty else {
            throw KokoroTextProcessingError.emptyText
        }
        guard options.speed.isFinite, options.speed > 0 else {
            throw KokoroTextProcessingError.invalidSpeed(options.speed)
        }
        guard refS.count == PipelineConstants.voiceEmbeddingDim else {
            throw KokoroTextProcessingError.invalidVoiceEmbedding(
                actual: refS.count,
                expected: PipelineConstants.voiceEmbeddingDim
            )
        }

        let resolvedPhonemeResult: KokoroPhonemeResult
        if let phonemeResult {
            resolvedPhonemeResult = phonemeResult
        } else {
            resolvedPhonemeResult = try phonemizer.phonemize(normalized)
        }
        let modelTokenIDs = tokenIDs(forPhonemes: resolvedPhonemeResult.phonemes)
        guard !modelTokenIDs.isEmpty else {
            throw KokoroTextProcessingError.emptyTokenization
        }

        let framed = [Self.boundaryTokenID] + modelTokenIDs + [Self.boundaryTokenID]
        guard framed.count <= PipelineConstants.maxCallerChunkTokens else {
            throw KokoroTextProcessingError.tokenBudgetExceeded(
                actual: framed.count,
                maximum: PipelineConstants.maxCallerChunkTokens
            )
        }

        let paddedLength = PipelineConstants.durationTokenSizes.first { $0 >= framed.count }
            ?? PipelineConstants.durationTokenSizes.last
            ?? framed.count
        let padded = framed + Array(repeating: Self.boundaryTokenID, count: max(0, paddedLength - framed.count))
        let mask = Array(repeating: Int32(1), count: framed.count)
            + Array(repeating: Int32(0), count: max(0, paddedLength - framed.count))

        return KokoroPreparedInput(
            key: key,
            text: normalized,
            voice: voice.rawValue,
            inputIds: padded,
            attentionMask: mask,
            refS: refS,
            speed: options.speed,
            canonicalDurationSeconds: nil,
            numTokens: framed.count,
            hnsfWeightsSHA256: Self.hnsfWeightsSHA256
        )
    }

    /// Converts phoneme characters to Kokoro model token IDs.
    ///
    /// Characters without vocab entries are dropped to match Gist's iOS path.
    ///
    /// - Parameter phonemes: Phoneme string returned by Misaki or a test stub.
    /// - Returns: Unframed model token IDs.
    public func tokenIDs(forPhonemes phonemes: String) -> [Int32] {
        phonemes.compactMap { vocab[String($0)] }
    }

    /// Converts text to phonemes after the same whitespace normalization used by prep.
    ///
    /// - Parameter text: Raw caller text.
    /// - Returns: Phoneme result used for tokenization and voice-row selection.
    public func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        let normalized = Self.normalizeWhitespace(text)
        guard !normalized.isEmpty else {
            throw KokoroTextProcessingError.emptyText
        }
        return try phonemizer.phonemize(normalized)
    }

    /// Normalizes caller text the same way the Botnet chunker does.
    ///
    /// - Parameter text: Raw caller text.
    /// - Returns: Single-spaced text trimmed of leading and trailing whitespace.
    public static func normalizeWhitespace(_ text: String) -> String {
        text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
