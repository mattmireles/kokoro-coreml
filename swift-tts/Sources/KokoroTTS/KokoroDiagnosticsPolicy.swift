import Foundation

/// Privacy policy for SDK diagnostics.
///
/// The default policy allows counters, hashes, timings, model identifiers, and
/// typed error codes. It does not allow raw text or phoneme strings, and the SDK
/// must not persist those payloads even when a caller opts into exposing them
/// for an interactive debug session.
struct KokoroDiagnosticsPolicy: Equatable, Sendable {
    /// Whether diagnostics may include the raw caller-provided text.
    let includesRawText: Bool

    /// Whether diagnostics may include raw phoneme strings.
    let includesPhonemes: Bool

    /// Whether the SDK may persist raw text or phoneme payloads.
    let persistsRawPayloads: Bool

    /// Default privacy-safe diagnostics policy.
    static let privacySafeDefault = KokoroDiagnosticsPolicy(
        includesRawText: false,
        includesPhonemes: false
    )

    /// Debug-only policy for caller-controlled interactive probes.
    ///
    /// The SDK still refuses persistence; this is only for explicit caller
    /// inspection of text-prep drift while developing an app.
    static let interactiveDebugPayloads = KokoroDiagnosticsPolicy(
        includesRawText: true,
        includesPhonemes: true
    )

    /// Creates a diagnostics policy.
    ///
    /// - Parameters:
    ///   - includesRawText: Whether raw text may appear in diagnostics.
    ///   - includesPhonemes: Whether phonemes may appear in diagnostics.
    init(
        includesRawText: Bool,
        includesPhonemes: Bool
    ) {
        self.includesRawText = includesRawText
        self.includesPhonemes = includesPhonemes
        self.persistsRawPayloads = false
    }
}
