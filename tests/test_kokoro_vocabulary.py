"""Parity checks for Kokoro token IDs shared with the Swift pipeline."""

from __future__ import annotations

# Must stay aligned with swift/Sources/KokoroPipeline/KokoroVocabulary.swift
SWIFT_SILENT_PUNCTUATION_TOKEN_IDS = frozenset({1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15})
SWIFT_WHITESPACE_TOKEN_ID = 16

CANONICAL_PUNCTUATION_CHARS = (
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "—",
    "…",
    '"',
    "(",
    ")",
    "“",
    "”",
)


def test_silent_punctuation_ids_match_swift_constants() -> None:
    assert SWIFT_SILENT_PUNCTUATION_TOKEN_IDS == frozenset(
        {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15}
    )
    assert SWIFT_WHITESPACE_TOKEN_ID == 16


def test_silent_punctuation_ids_match_kokoro_vocab() -> None:
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")

    from kokoro.model import KModel

    model = KModel().to("cpu")
    vocab = model.vocab

    observed = {vocab[char] for char in CANONICAL_PUNCTUATION_CHARS}
    assert observed == SWIFT_SILENT_PUNCTUATION_TOKEN_IDS
    assert vocab[" "] == SWIFT_WHITESPACE_TOKEN_ID

    # Stress markers are phonetic, not pause punctuation — must stay excluded.
    assert vocab["ˈ"] not in SWIFT_SILENT_PUNCTUATION_TOKEN_IDS
    assert vocab["ˌ"] not in SWIFT_SILENT_PUNCTUATION_TOKEN_IDS
