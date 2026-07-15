import Foundation

/// Caller-controlled options for raw-text Kokoro synthesis preparation.
public struct KokoroSynthesisOptions: Equatable, Sendable {
    /// Default SDK starter-profile chunk cap.
    public static let defaultMaxChunkSeconds = 15.0

    /// Duration-model speed multiplier.
    public let speed: Float

    /// Optional hard character window used by recursive token-budget fallback.
    public let maxCharacters: Int?

    /// Maximum estimated duration per prepared text chunk.
    public let maxChunkSeconds: Double

    /// Creates synthesis preparation options.
    ///
    /// - Parameters:
    ///   - speed: Duration-model speed multiplier. Must be positive and finite
    ///     when used by ``KokoroTextProcessor``.
    ///   - maxCharacters: Optional hard character window for fallback splits.
    ///   - maxChunkSeconds: Estimated chunk cap. Defaults to the SDK starter
    ///     profile's 15-second cap.
    public init(
        speed: Float = 1.0,
        maxCharacters: Int? = nil,
        maxChunkSeconds: Double = Self.defaultMaxChunkSeconds
    ) {
        self.speed = speed
        self.maxCharacters = maxCharacters
        self.maxChunkSeconds = maxChunkSeconds
    }
}
