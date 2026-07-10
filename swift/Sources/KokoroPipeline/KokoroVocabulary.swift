/// Kokoro-82M token IDs used by the Swift synthesis pipeline.
///
/// Canonical mapping lives in HuggingFace ``hexgrad/Kokoro-82M`` ``config.json``
/// ``vocab`` and in Python ``kokoro/model.py`` via ``KModel.vocab``. Keep
/// punctuation suppression aligned with those sources — do not invent IDs here.

import Foundation

public enum KokoroVocabulary {
    /// BOS / EOS padding token.
    public static let bosEosTokenId: Int32 = 0
    /// Whitespace token (``" "`` in vocab).
    public static let whitespaceTokenId: Int32 = 16

    /// Punctuation tokens whose duration spans should be faded to silence in
    /// Core ML output. Stress markers ``ˈ`` / ``ˌ`` (156/157) are intentionally
    /// excluded — they are phonetic, not pause punctuation.
    public static let silentPunctuationTokenIds: Set<Int32> = [
        1,  // ;
        2,  // :
        3,  // ,
        4,  // .
        5,  // !
        6,  // ?
        9,  // —
        10, // …
        11, // "
        12, // (
        13, // )
        14, // “
        15, // ”
    ]
}
