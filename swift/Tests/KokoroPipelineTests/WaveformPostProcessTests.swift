import XCTest
@testable import KokoroPipeline

final class WaveformPostProcessTests: XCTestCase {

    func testPipelineConstantsSamplesPerDurationFrameMatchesHopLength() {
        XCTAssertEqual(PipelineConstants.samplesPerDurationFrame, 600)
    }

    func testSilentPunctuationTokenIdsMatchCanonicalVocab() {
        XCTAssertEqual(
            KokoroVocabulary.silentPunctuationTokenIds,
            Set([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15])
        )
        XCTAssertEqual(KokoroVocabulary.whitespaceTokenId, 16)
    }

    func testSuppressPunctuationTokenAudioSilencesPunctuationSpansWithFades() throws {
        let audio = [Float](repeating: 1.0, count: 20)
        let result = suppressPunctuationTokenAudio(
            audio,
            inputIds: [99, 3, 100, 4],
            predDur: [1, 2, 1, 1],
            samplesPerDurationFrame: 4,
            fadeSamples: 2
        )

        XCTAssertEqual(result[0], 1.0)
        XCTAssertEqual(result[2], 1.0, accuracy: 0.0001)
        XCTAssertEqual(result[3], 0.5, accuracy: 0.0001)
        XCTAssertEqual(Array(result[4..<12]), Array(repeating: 0.0, count: 8))
        XCTAssertEqual(result[12], 0.5, accuracy: 0.0001)
        XCTAssertEqual(result[13], 1.0, accuracy: 0.0001)
        XCTAssertEqual(result[15], 0.5, accuracy: 0.0001)
        XCTAssertEqual(Array(result[16..<20]), Array(repeating: 0.0, count: 4))
    }

    func testSuppressPunctuationTokenAudioSilencesAdjacentPunctuationWhitespace() throws {
        let audio = [Float](repeating: 1.0, count: 16)
        let result = suppressPunctuationTokenAudio(
            audio,
            inputIds: [99, 3, 16, 100],
            predDur: [1, 1, 1, 1],
            samplesPerDurationFrame: 4,
            fadeSamples: 0
        )

        XCTAssertEqual(Array(result[0..<4]), Array(repeating: 1.0, count: 4))
        XCTAssertEqual(Array(result[4..<12]), Array(repeating: 0.0, count: 8))
        XCTAssertEqual(Array(result[12..<16]), Array(repeating: 1.0, count: 4))
    }

    func testSuppressPunctuationTokenAudioUsesDefaultFadeAtProductionConstants() {
        let spanSamples = PipelineConstants.samplesPerDurationFrame
        let audio = [Float](repeating: 1.0, count: spanSamples * 2)
        let result = suppressPunctuationTokenAudio(
            audio,
            inputIds: [99, 4],
            predDur: [1, 1]
        )

        let fade = PipelineConstants.punctuationFadeSamples
        let periodStart = spanSamples
        let fadeMid = periodStart - fade / 2
        XCTAssertEqual(result[0], 1.0)
        XCTAssertEqual(result[periodStart - fade - 1], 1.0, accuracy: 0.0001)
        XCTAssertEqual(result[fadeMid], 0.5, accuracy: 0.02)
        XCTAssertEqual(Array(result[periodStart..<(spanSamples * 2)]), Array(repeating: 0.0, count: spanSamples))
    }

    func testSuppressPunctuationTokenAudioIgnoresTrailingPaddedInputIds() {
        let audio = [Float](repeating: 1.0, count: 8)
        let aligned = suppressPunctuationTokenAudio(
            audio,
            inputIds: [99, 4],
            predDur: [1, 1],
            samplesPerDurationFrame: 4,
            fadeSamples: 0
        )
        let withPadding = suppressPunctuationTokenAudio(
            audio,
            inputIds: [99, 4, 0, 0],
            predDur: [1, 1],
            samplesPerDurationFrame: 4,
            fadeSamples: 0
        )

        XCTAssertEqual(aligned, withPadding)
    }

    func testSuppressPunctuationTokenAudioSilencesRealVocabPunctuationSequence() {
        // Mirrors debug-notes "(2023) —" style punctuation IDs from Kokoro vocab.
        let inputIds: [Int32] = [0, 12, 13, 9, 0]
        let predDur = [1, 2, 2, 3, 1]
        let samplesPerFrame = 4
        let audio = [Float](repeating: 1.0, count: predDur.reduce(0, +) * samplesPerFrame)

        let result = suppressPunctuationTokenAudio(
            audio,
            inputIds: inputIds,
            predDur: predDur,
            samplesPerDurationFrame: samplesPerFrame,
            fadeSamples: 0
        )

        XCTAssertEqual(result[0], 1.0)
        XCTAssertEqual(result[1], 1.0)
        XCTAssertEqual(Array(result[4..<32]), Array(repeating: 0.0, count: 28))
        XCTAssertEqual(result.last, 1.0)
    }
}
