/// Kokoro-82M token IDs used by the Swift synthesis pipeline.
///
/// Canonical mapping lives in HuggingFace ``hexgrad/Kokoro-82M`` ``config.json``
/// ``vocab`` and in Python ``kokoro/model.py`` via ``KModel.vocab``; do not
/// invent IDs here.

import Foundation

public enum KokoroVocabulary {
    /// BOS / EOS padding token.
    public static let bosEosTokenId: Int32 = 0
    /// Whitespace token (``" "`` in vocab).
    public static let whitespaceTokenId: Int32 = 16
}
