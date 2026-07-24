import Foundation

/// Privacy-safe progress emitted while Kokoro proves its runtime path.
public enum KokoroPrewarmStage: Equatable, Sendable {
    /// Raw text is being phonemized and converted into model inputs.
    case textPreparation
    /// A named Core ML model is being loaded.
    case modelLoad(String)
    /// The complete duration-to-audio prediction is running.
    case prediction
    /// The discarded proof prediction completed successfully.
    case complete

    /// Stable label suitable for diagnostics and watchdog checkpoints.
    public var diagnosticName: String {
        switch self {
        case .textPreparation:
            return "text_preparation"
        case .modelLoad(let model):
            return "model_load.\(model)"
        case .prediction:
            return "prediction"
        case .complete:
            return "complete"
        }
    }
}
