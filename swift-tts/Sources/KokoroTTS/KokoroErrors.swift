import Foundation

/// Public SDK errors for resource loading, preparation, and synthesis.
public enum KokoroError: Error, Equatable, LocalizedError {
    /// A required runtime manifest is missing.
    case missingManifest(URL)

    /// A required model package is missing.
    case missingModel(String)

    /// A required voice file is missing.
    case missingVoice(String)

    /// A runtime asset other than a model or voice file is missing.
    case missingRuntimeAsset(String)

    /// A voice file exists but is malformed.
    case malformedVoice(String)

    /// A resource hash did not match its manifest entry.
    case badHash(path: String)

    /// A resource path escaped its expected root.
    case pathEscape(String)

    /// The runtime manifest schema is not supported by this SDK.
    case unsupportedManifestSchema(Int)

    /// The requested voice or language is not supported by this bundle.
    case unsupportedVoice(String)

    /// Text was empty after whitespace normalization.
    case emptyText

    /// Phonemization or vocabulary lookup produced no model tokens.
    case emptyPhonemizerOutput

    /// The requested speed was zero, negative, NaN, or infinite.
    case invalidSpeed(Float)

    /// The prepared input exceeds the largest available duration model.
    case inputTooLong(tokens: Int, maxTokens: Int)

    /// Core ML could not compile or load a model.
    case coreMLLoadFailed(String)

    /// Core ML loaded but prediction failed during synthesis.
    case coreMLPredictionFailed(String)

    /// The synthesis task was cancelled before completion.
    case synthesisCancelled

    /// Synthesized audio was empty or contained non-finite samples.
    case invalidAudioOutput

    /// Human-readable error text for app logs and tests.
    public var errorDescription: String? {
        switch self {
        case .missingManifest(let url):
            return "Kokoro runtime manifest is missing at \(url.path)."
        case .missingModel(let name):
            return "Kokoro model package is missing: \(name)."
        case .missingVoice(let voice):
            return "Kokoro voice file is missing: \(voice)."
        case .missingRuntimeAsset(let path):
            return "Kokoro runtime asset is missing: \(path)."
        case .malformedVoice(let voice):
            return "Kokoro voice file is malformed: \(voice)."
        case .badHash(let path):
            return "Kokoro resource hash mismatch: \(path)."
        case .pathEscape(let path):
            return "Kokoro resource path escapes its root: \(path)."
        case .unsupportedManifestSchema(let version):
            return "Kokoro runtime manifest schema is not supported: \(version)."
        case .unsupportedVoice(let voice):
            return "Kokoro voice is not supported by this bundle: \(voice)."
        case .emptyText:
            return "Kokoro text is empty after whitespace normalization."
        case .emptyPhonemizerOutput:
            return "Kokoro text produced no model tokens after phonemization."
        case .invalidSpeed(let speed):
            return "Kokoro speed must be positive and finite; observed \(speed)."
        case .inputTooLong(let tokens, let maxTokens):
            return "Kokoro input has \(tokens) tokens, but the largest loaded duration model supports \(maxTokens)."
        case .coreMLLoadFailed(let model):
            return "Core ML could not compile or load Kokoro model: \(model)."
        case .coreMLPredictionFailed(let detail):
            return "Core ML Kokoro prediction failed: \(detail)."
        case .synthesisCancelled:
            return "Kokoro synthesis was cancelled."
        case .invalidAudioOutput:
            return "Kokoro synthesis produced invalid audio."
        }
    }
}
