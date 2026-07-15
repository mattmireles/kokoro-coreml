import KokoroPipeline
import XCTest
@testable import KokoroTTS

private struct StubPhonemizer: KokoroPhonemizer {
    /// Phoneme string returned to the text processor.
    let phonemes: String

    /// Returns the configured phoneme string without invoking Misaki/MLX.
    ///
    /// - Parameter text: Ignored raw text.
    /// - Returns: Stub phoneme result.
    func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        KokoroPhonemeResult(phonemes: phonemes)
    }
}

final class KokoroTextProcessorTests: XCTestCase {
    /// Returns the repository root from this test file's absolute path.
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// Verifies bundled vocab lookup drops unknown characters like the Gist path.
    func testTokenizationDropsUnknownPhonemeCharacters() throws {
        let processor = try KokoroTextProcessor(phonemizer: StubPhonemizer(phonemes: "h🙂."))

        XCTAssertEqual(processor.tokenIDs(forPhonemes: "h🙂."), [50, 4])
    }

    /// Verifies prepared inputs use BOS/EOS framing, enum padding, and metadata.
    func testPrepareBuildsPaddedPreparedInput() throws {
        let processor = try KokoroTextProcessor(phonemizer: StubPhonemizer(phonemes: "həlˈO."))
        let refS = (0..<PipelineConstants.voiceEmbeddingDim).map(Float.init)

        let prepared = try processor.prepare(
            text: "  Hello   world. ",
            voice: .afHeart,
            refS: refS,
            options: KokoroSynthesisOptions(speed: 1.25),
            key: "hello"
        )

        XCTAssertEqual(prepared.key, "hello")
        XCTAssertEqual(prepared.text, "Hello world.")
        XCTAssertEqual(prepared.voice, "af_heart")
        XCTAssertEqual(prepared.refS, refS)
        XCTAssertEqual(prepared.speed, 1.25)
        XCTAssertEqual(prepared.hnsfWeightsSHA256, KokoroTextProcessor.hnsfWeightsSHA256)
        XCTAssertEqual(prepared.inputIds.count, 32)
        XCTAssertEqual(prepared.attentionMask.count, 32)
        XCTAssertEqual(prepared.inputIds.first, KokoroTextProcessor.boundaryTokenID)
        let tokenCount = try XCTUnwrap(prepared.numTokens)
        XCTAssertEqual(prepared.inputIds[tokenCount - 1], KokoroTextProcessor.boundaryTokenID)
        XCTAssertEqual(Array(prepared.attentionMask.prefix(tokenCount)), Array(repeating: 1, count: tokenCount))
        XCTAssertTrue(prepared.attentionMask.dropFirst(tokenCount).allSatisfy { $0 == 0 })
    }

    /// Verifies raw-text SDK prep matches the legacy prepared-input pipeline boundary.
    ///
    /// The old JS path emitted `KokoroPreparedInput`-compatible tensors directly.
    /// For matching phonemes and voice rows, the SDK path must produce an
    /// identical `KokoroSynthesisRequest`, so existing `KokoroPipeline`
    /// execution remains a drop-in implementation detail.
    func testPreparedInputBoundaryMatchesLegacyPreparedRequestContract() throws {
        let phonemes = "həlˈoʊ wˈɜːld."
        let expectedTokenIDs: [Int32] = [50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 4]
        let expectedInputIDs: [Int32] = [0] + expectedTokenIDs + [0]
        let expectedPaddedInputIDs = expectedInputIDs + Array(repeating: Int32(0), count: 16)
        let expectedAttentionMask = Array(repeating: Int32(1), count: expectedInputIDs.count)
            + Array(repeating: Int32(0), count: 16)
        let processor = try KokoroTextProcessor(phonemizer: StubPhonemizer(phonemes: phonemes))
        var voiceTable = VoiceTable(voicesDirectory: repoRoot.appendingPathComponent("kokoro.js/voices"))
        let refS = try voiceTable.refS(voiceID: .afHeart, phonemeCount: phonemes.utf16.count)

        let sdkPrepared = try processor.prepare(
            text: "  Hello   world. ",
            voice: .afHeart,
            refS: refS,
            options: KokoroSynthesisOptions(speed: 1.0),
            key: "hello-world",
            phonemeResult: KokoroPhonemeResult(phonemes: phonemes)
        )
        let legacyPrepared = KokoroPreparedInput(
            key: "hello-world",
            text: "Hello world.",
            voice: "af_heart",
            inputIds: expectedPaddedInputIDs,
            attentionMask: expectedAttentionMask,
            refS: refS,
            speed: 1.0,
            canonicalDurationSeconds: nil,
            numTokens: expectedInputIDs.count,
            hnsfWeightsSHA256: KokoroTextProcessor.hnsfWeightsSHA256
        )

        XCTAssertEqual(KokoroTextProcessor.boundaryTokenID, 0)
        XCTAssertEqual(processor.tokenIDs(forPhonemes: phonemes), expectedTokenIDs)
        XCTAssertTrue(expectedPaddedInputIDs.dropFirst(expectedInputIDs.count).allSatisfy { $0 == 0 })
        XCTAssertEqual(sdkPrepared, legacyPrepared)
        XCTAssertEqual(sdkPrepared.synthesisRequest().inputIds, legacyPrepared.synthesisRequest().inputIds)
        XCTAssertEqual(sdkPrepared.synthesisRequest().attentionMask, legacyPrepared.synthesisRequest().attentionMask)
        XCTAssertEqual(sdkPrepared.synthesisRequest().refS, legacyPrepared.synthesisRequest().refS)
        XCTAssertEqual(sdkPrepared.synthesisRequest().speed, legacyPrepared.synthesisRequest().speed)
    }

    /// Verifies all-unknown phonemes fail instead of creating boundary-only inputs.
    func testPrepareRejectsEmptyTokenizationAfterUnknownDrop() throws {
        let processor = try KokoroTextProcessor(phonemizer: StubPhonemizer(phonemes: "🙂"))
        let refS = Array(repeating: Float(0), count: PipelineConstants.voiceEmbeddingDim)

        XCTAssertThrowsError(try processor.prepare(text: "hello", voice: .afHeart, refS: refS)) { error in
            XCTAssertEqual(error as? KokoroTextProcessingError, .emptyTokenization)
        }
    }

    /// Verifies invalid speed values are rejected before phonemization output is used.
    func testPrepareRejectsInvalidSpeed() throws {
        let processor = try KokoroTextProcessor(phonemizer: StubPhonemizer(phonemes: "h"))
        let refS = Array(repeating: Float(0), count: PipelineConstants.voiceEmbeddingDim)

        XCTAssertThrowsError(
            try processor.prepare(
                text: "hello",
                voice: .afHeart,
                refS: refS,
                options: KokoroSynthesisOptions(speed: 0)
            )
        ) { error in
            XCTAssertEqual(error as? KokoroTextProcessingError, .invalidSpeed(0))
        }
    }

    /// Verifies real voice `.bin` files use the fleet phoneme-count row rule.
    func testVoiceTableSelectsRowsByPhonemeUTF16Count() throws {
        var table = VoiceTable(voicesDirectory: repoRoot.appendingPathComponent("kokoro.js/voices"))

        let rowForZero = try table.refS(voiceID: .afHeart, phonemeCount: 0)
        let rowForOne = try table.refS(voiceID: .afHeart, phonemeCount: 1)
        let rowForTwo = try table.refS(voiceID: .afHeart, phonemeCount: 2)

        XCTAssertEqual(rowForZero.count, PipelineConstants.voiceEmbeddingDim)
        XCTAssertEqual(rowForZero, rowForOne)
        XCTAssertNotEqual(rowForZero, rowForTwo)
    }

    /// Verifies missing voices surface a typed loader error.
    func testVoiceTableRejectsMissingVoice() {
        var table = VoiceTable(voicesDirectory: repoRoot.appendingPathComponent("kokoro.js/voices"))

        XCTAssertThrowsError(try table.refS(voiceID: "missing_voice", phonemeCount: 1)) { error in
            XCTAssertEqual(error as? KokoroVoiceTableError, .missingVoice("missing_voice"))
        }
    }

    /// Verifies public voice IDs cannot escape the configured voice directory.
    func testVoiceTableRejectsPathLikeVoiceID() {
        var table = VoiceTable(voicesDirectory: repoRoot.appendingPathComponent("kokoro.js/voices"))

        XCTAssertThrowsError(try table.refS(voiceID: "../af_heart", phonemeCount: 1)) { error in
            XCTAssertEqual(error as? KokoroVoiceTableError, .missingVoice("../af_heart"))
        }
    }
}
