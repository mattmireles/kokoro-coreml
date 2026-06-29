import Foundation

/// Prepared Kokoro input that can be passed directly to the Core ML pipeline.
///
/// This type is intentionally low level. It does not know how text was
/// phonemized, where voice assets came from, or whether the caller used the
/// high-level `KokoroTTS` SDK. It only captures the model-ready tensors and the
/// small metadata fields used by existing fixtures and bundle manifests.
public struct KokoroPreparedInput: Equatable, Sendable {
    /// Optional stable key used by generated fixtures and batch jobs.
    public let key: String?

    /// Optional source text preserved for fixture/debug parity.
    public let text: String?

    /// Optional voice identifier used to select the `refS` row.
    public let voice: String?

    /// Padded Kokoro token IDs including BOS/EOS boundary token `0`.
    public let inputIds: [Int32]

    /// Attention mask aligned with ``inputIds``.
    public let attentionMask: [Int32]

    /// Selected 256-float Kokoro voice embedding row.
    public let refS: [Float]

    /// Synthesis speed multiplier passed to the duration model.
    public let speed: Float

    /// Optional canonical duration from fixture generation.
    public let canonicalDurationSeconds: Double?

    /// Optional unpadded token count, including BOS/EOS.
    public let numTokens: Int?

    /// Optional verified hn-NSF weights digest used by generated manifests.
    public let hnsfWeightsSHA256: String?

    /// Creates a prepared Kokoro input value.
    ///
    /// - Parameters:
    ///   - key: Optional fixture or batch key.
    ///   - text: Optional source text.
    ///   - voice: Optional Kokoro voice identifier.
    ///   - inputIds: Padded token IDs including BOS/EOS.
    ///   - attentionMask: Attention mask with the same length as `inputIds`.
    ///   - refS: Selected 256-float voice embedding row.
    ///   - speed: Duration-model speed multiplier.
    ///   - canonicalDurationSeconds: Optional fixture duration.
    ///   - numTokens: Optional unpadded token count.
    ///   - hnsfWeightsSHA256: Optional verified hn-NSF weights digest.
    public init(
        key: String? = nil,
        text: String? = nil,
        voice: String? = nil,
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float = 1.0,
        canonicalDurationSeconds: Double? = nil,
        numTokens: Int? = nil,
        hnsfWeightsSHA256: String? = nil
    ) {
        self.key = key
        self.text = text
        self.voice = voice
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.refS = refS
        self.speed = speed
        self.canonicalDurationSeconds = canonicalDurationSeconds
        self.numTokens = numTokens
        self.hnsfWeightsSHA256 = hnsfWeightsSHA256
    }

    /// Converts this value to the executor request type.
    ///
    /// - Returns: `KokoroSynthesisRequest` preserving tensor fields and speed.
    public func synthesisRequest() -> KokoroSynthesisRequest {
        KokoroSynthesisRequest(
            inputIds: inputIds,
            attentionMask: attentionMask,
            refS: refS,
            speed: speed
        )
    }
}
