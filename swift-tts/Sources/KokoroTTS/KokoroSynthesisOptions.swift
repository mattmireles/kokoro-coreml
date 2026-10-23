import Foundation

/// Caller-controlled options for raw-text Kokoro synthesis preparation.
public struct KokoroSynthesisOptions: Equatable, Sendable {
    /// Default SDK starter-profile chunk cap. iOS uses 10 s: the fast ANE+Metal
    /// generator peaks at ~350 MB for a 10 s chunk on an iPhone 12 Pro vs ~500 MB
    /// at 15 s (shipping GPU path ~400-440 MB), with the same throughput per audio
    /// second and earlier first audio (README/Notes/ane-folded-generator.md).
    #if os(iOS)
    public static let defaultMaxChunkSeconds = 10.0
    #else
    public static let defaultMaxChunkSeconds = 15.0
    #endif

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
    ///     profile cap (10 s on iOS, 15 s elsewhere).
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
