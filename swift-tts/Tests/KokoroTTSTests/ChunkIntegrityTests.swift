import Foundation
import KokoroPipeline
import XCTest
@testable import KokoroTTS

/// Emits one fixed phoneme per input character.
///
/// Makes token count track character count exactly, so a fixture can force the
/// oversized-chunk retry path without a real G2P runtime.
private struct CharacterCountPhonemizer: KokoroPhonemizer {
    /// Phoneme emitted once per normalized input character.
    let phoneme: Character

    func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        let count = KokoroTextProcessor.normalizeWhitespace(text).count
        return KokoroPhonemeResult(phonemes: String(repeating: String(phoneme), count: count))
    }
}

/// Returns a caller-supplied phoneme string regardless of input.
private struct FixedPhonemizer: KokoroPhonemizer {
    /// Phonemes returned for every input.
    let phonemes: String

    /// Source words reported as dropped.
    let droppedTokens: Int

    /// Creates a fixed phonemizer.
    ///
    /// - Parameters:
    ///   - phonemes: Phonemes returned for every input.
    ///   - droppedTokens: Source words reported as dropped.
    init(phonemes: String, droppedTokens: Int = 0) {
        self.phonemes = phonemes
        self.droppedTokens = droppedTokens
    }

    func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        KokoroPhonemeResult(phonemes: phonemes, droppedTokens: droppedTokens)
    }
}

/// Guards that text survives chunking, and that lost speech is never silent.
///
/// A 2026-07-30 audit found the chunker and PCM concatenation lossless but text
/// disappearing at the grapheme-to-phoneme boundary: a chunk whose words all
/// failed to resolve still tokenized, because the spaces between them survive
/// and the space character has a real vocab entry. The model then synthesized
/// near-silence and every guard passed.
final class ChunkIntegrityTests: XCTestCase {
    /// Verifies the chunk plan never loses input text.
    func testPreparedChunksReconstructInput() async throws {
        let root = try KokoroBundleFixture.makeBundleRoot()
        let modelProvider = try KokoroSDKModelProvider(resources: .directory(root))
        let processor = KokoroTextProcessor(
            phonemizer: CharacterCountPhonemizer(phoneme: "h"),
            vocab: try modelProvider.vocab()
        )
        let tts = KokoroTTS(
            chunker: TextChunker(maxChunkSeconds: 10_000),
            americanTextProcessor: processor,
            britishTextProcessor: processor,
            voiceTable: VoiceTable(
                voicesDirectory: root.appendingPathComponent("voices", isDirectory: true)
            ),
            modelProvider: modelProvider,
            hnsf: try modelProvider.hnsfWeights()
        )
        let text = """
        The quarterly review was led by Dr. Hartman, who opened with a summary of \
        revenue across every region; the team then walked through churn, expansion, \
        and the pipeline that closed in March. Nobody expected the numbers from the \
        smaller markets to hold, yet they did, and the forecast for the coming year \
        was revised upward by a comfortable margin. A second session covered staffing, \
        tooling, and the migration that had slipped twice already.
        """
        let normalized = KokoroTextProcessor.normalizeWhitespace(text)

        let prepared = try await tts.prepare(text, voice: .afHeart)

        XCTAssertGreaterThan(prepared.count, 1, "the fixture must exercise the retry split")
        let texts = prepared.compactMap(\.text)
        XCTAssertEqual(texts.count, prepared.count, "every prepared chunk must carry its text")
        // The universal invariant: chunking may re-place whitespace but must
        // never drop, duplicate, or reorder a single non-space character.
        XCTAssertEqual(
            texts.joined().filter { !$0.isWhitespace },
            normalized.filter { !$0.isWhitespace }
        )
        // Every split lands on whitespace or just after clause punctuation, so
        // for prose the join is exact too.
        XCTAssertEqual(texts.joined(separator: " "), normalized)
    }

    /// Verifies a chunk that phonemizes to silence throws instead of synthesizing mute audio.
    func testInaudibleChunkFailsLoudly() throws {
        let processor = KokoroTextProcessor(
            phonemizer: FixedPhonemizer(phonemes: "   ", droppedTokens: 3),
            vocab: try KokoroTextProcessor.loadBundledVocab()
        )

        XCTAssertThrowsError(
            try processor.prepare(
                text: "→ → →",
                voice: .afHeart,
                refS: Array(repeating: 0, count: PipelineConstants.voiceEmbeddingDim)
            )
        ) { error in
            guard case .inaudibleChunk(let characters, let droppedTokens)? = error as? KokoroTextProcessingError else {
                XCTFail("expected inaudibleChunk, got \(error)")
                return
            }
            XCTAssertEqual(characters, 5)
            XCTAssertEqual(droppedTokens, 3)
        }
    }

    /// Verifies the loud-fail guard does not fire on a legitimately short chunk.
    ///
    /// "2024." is the shape most at risk of a false positive: a single spoken
    /// token plus terminal punctuation. It must still synthesize.
    func testShortNumericChunkStillPrepares() throws {
        let processor = KokoroTextProcessor(
            phonemizer: FixedPhonemizer(phonemes: "twˈɛnti twˈɛnti fˈɔːɹ."),
            vocab: try KokoroTextProcessor.loadBundledVocab()
        )

        let prepared = try processor.prepare(
            text: "2024.",
            voice: .afHeart,
            refS: Array(repeating: 0, count: PipelineConstants.voiceEmbeddingDim)
        )

        XCTAssertEqual(prepared.text, "2024.")
        XCTAssertGreaterThan(prepared.numTokens ?? 0, 2)
    }

    /// Verifies characters with no vocab entry are counted rather than silently dropped.
    func testTokenizationReportsDroppedPhonemeCharacters() throws {
        let processor = KokoroTextProcessor(
            phonemizer: FixedPhonemizer(phonemes: ""),
            vocab: ["h": 50, " ": 16]
        )

        let tokenization = processor.tokenization(forPhonemes: "h❓h✗ h")

        XCTAssertEqual(tokenization.tokenIDs, [50, 50, 16, 50])
        XCTAssertEqual(tokenization.droppedCharacters, 2)
        XCTAssertEqual(processor.tokenIDs(forPhonemes: "h❓h✗ h"), tokenization.tokenIDs)
    }

    /// Verifies Misaki's unresolved-word signals are recognized, and punctuation is not.
    func testDroppedTokenDetectionIgnoresPunctuation() {
        XCTAssertTrue(KokoroMisakiPhonemizer.isDroppedToken(text: "kubernetes", phonemes: nil))
        XCTAssertTrue(KokoroMisakiPhonemizer.isDroppedToken(text: "kubernetes", phonemes: "  "))
        XCTAssertTrue(KokoroMisakiPhonemizer.isDroppedToken(text: "kubernetes", phonemes: "❓"))
        XCTAssertFalse(KokoroMisakiPhonemizer.isDroppedToken(text: "kubernetes", phonemes: "kˈubɚnˌɛtiz"))
        XCTAssertFalse(KokoroMisakiPhonemizer.isDroppedToken(text: ",", phonemes: nil))
        XCTAssertFalse(KokoroMisakiPhonemizer.isDroppedToken(text: "—", phonemes: ""))
    }

    /// Verifies a retry split never separates an abbreviation from the word it modifies.
    func testRetrySplitKeepsAbbreviationWithFollowingWord() {
        let characters = Array("Meeting with Dr. Hartman")

        let boundary = KokoroChunkBoundaries.lastSoftBoundary(
            in: characters,
            maxExclusive: characters.count,
            minIndex: 0
        )

        // The last space sits between "Dr." and "Hartman"; the guard must skip
        // it and fall back to the space after "with".
        XCTAssertEqual(boundary, 12)
        XCTAssertEqual(String(characters[boundary...]).trimmingCharacters(in: .whitespaces), "Dr. Hartman")
    }

    /// Verifies enumerated acronyms and single initials are protected too.
    func testAbbreviationGuardCoversAcronymsAndInitials() {
        XCTAssertTrue(KokoroAbbreviations.isProtectedBoundaryToken("F.B.I."))
        XCTAssertTrue(KokoroAbbreviations.isProtectedBoundaryToken("J."))
        XCTAssertTrue(KokoroAbbreviations.isProtectedBoundaryToken("Ph.D."))
        XCTAssertFalse(KokoroAbbreviations.isProtectedBoundaryToken("go."))
        XCTAssertFalse(KokoroAbbreviations.isProtectedBoundaryToken("agreed."))
    }

    /// Verifies a sub-minimum tail is folded into the fragment before it.
    func testMergeUndersizedTailFoldsShortTailIntoPrevious() {
        let chunks = KokoroChunkBoundaries.mergeUndersizedTail(
            ["The first fragment carries most of the sentence", "centers,"],
            maxCharacters: 80,
            minCharacters: 32
        )

        XCTAssertEqual(chunks, ["The first fragment carries most of the sentence centers,"])
    }

    /// Verifies an unfoldable short tail moves the break point instead.
    ///
    /// The fragment lengths are chosen so folding would exceed `maxCharacters`,
    /// forcing the branch that re-places the break far enough left for the tail
    /// to clear `minCharacters`.
    func testMergeUndersizedTailRepositionsBreakWhenFoldWouldOverflow() throws {
        let previous = "delta omega chi kappa upsilon delta iota psi upsilon epsilon delta"
        let tail = "mu,"
        let combined = "\(previous) \(tail)"

        let chunks = KokoroChunkBoundaries.mergeUndersizedTail(
            [previous, tail],
            maxCharacters: 50,
            minCharacters: 32
        )

        XCTAssertEqual(chunks, ["delta omega chi kappa upsilon delta", "iota psi upsilon epsilon delta mu,"])
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(chunks.last).count, 32)
        XCTAssertEqual(chunks.joined(separator: " "), combined)
    }

    /// Verifies the retry split always makes strict progress on its input.
    func testRetrySplitAlwaysShortensEveryFragment() {
        let text = "The migration slipped twice already, and the team agreed to freeze scope."

        let fragments = KokoroChunkBoundaries.splitForRetry(text)

        XCTAssertGreaterThan(fragments.count, 1)
        XCTAssertTrue(fragments.allSatisfy { $0.count < text.count })
        XCTAssertEqual(fragments.joined(separator: " "), text)
    }
}
