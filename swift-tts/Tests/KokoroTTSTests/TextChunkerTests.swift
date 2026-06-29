import XCTest
@testable import KokoroTTS

final class TextChunkerTests: XCTestCase {
    /// Verifies whitespace normalization and sentence-boundary splitting.
    func testChunkerNormalizesWhitespaceAndPreservesAbbreviationsAndDecimals() {
        let chunker = TextChunker(maxChunkSeconds: TextChunker.botnetMaxChunkSeconds)

        let chunks = chunker.chunks(for: "  Dr. Smith paid $12.50. \n Next line!  ")

        XCTAssertEqual(chunks, ["Dr. Smith paid $12.50.", "Next line!"])
    }

    /// Verifies initials followed by capitals stay in the same sentence unit.
    func testChunkerPreservesInitials() {
        let chunker = TextChunker(maxChunkSeconds: TextChunker.botnetMaxChunkSeconds)

        XCTAssertEqual(chunker.chunks(for: "A. B. Testing starts."), ["A. B. Testing starts."])
    }

    /// Verifies soft punctuation can split oversized units without splitting numbers.
    func testChunkerProtectsNumericCommasAndDecimals() {
        let chunker = TextChunker(maxChunkSeconds: 0.6)

        let chunks = chunker.chunks(for: "Revenue was 1,000.50, then it doubled: good news.")

        XCTAssertEqual(chunks, ["Revenue", "was", "1,000.50", ",", "then it", "doubled:", "good", "news."])
    }

    /// Verifies hyphenated words survive character-window fallback when possible.
    func testChunkerCharacterWindowUsesSafeBreaks() {
        let chunker = TextChunker()

        let chunks = chunker.chunks(for: "abcdef ghij klmn", speed: 1.0, maxCharacters: 6)

        XCTAssertEqual(chunks, ["abcdef", "ghij", "klmn"])
    }

    /// Verifies speech speed changes the duration-based chunk count.
    func testChunkerSpeedChangesChunking() {
        let chunker = TextChunker(maxChunkSeconds: 1.0)
        let text = String(repeating: "a", count: 20)

        XCTAssertEqual(chunker.chunks(for: text, speed: 2.0), [text])
        XCTAssertGreaterThan(chunker.chunks(for: text, speed: 0.5).count, 1)
    }

    /// Verifies the SDK default is the documented 15-second cap.
    func testChunkerDefaultCapIsSDKStarterCap() {
        XCTAssertEqual(TextChunker().maxChunkSeconds, 15.0)
    }

    /// Verifies the compatibility tokenizer is deterministic.
    func testChunkerCompatibilityTokenizer() {
        XCTAssertEqual(TextChunker().tokenize("AZ"), [66, 91])
    }
}
