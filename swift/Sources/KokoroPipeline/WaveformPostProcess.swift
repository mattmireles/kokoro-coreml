/// Waveform post-processing for the Swift Core ML Kokoro pipeline.
///
/// Called by ``executeKokoroSynthesis`` after generator trim. Keeps ML tensor
/// helpers in ``MLMultiArrayHelpers`` separate from audio-domain fixes.

import Foundation

/// Suppress Core ML decoder transients on punctuation-owned duration spans.
///
/// Kokoro's duration model assigns real time to punctuation tokens so pauses
/// survive synthesis. The PyTorch reference renders those spans as near-silence
/// (measured -38 to -85 dBFS on the frozen 15s bench input), but the split
/// Core ML decoder path can leave short impulses there. This post-process keeps
/// the predicted timing intact while fading punctuation spans down to silence
/// in the final waveform.
///
/// Scope is punctuation tokens ONLY. Whitespace tokens adjacent to punctuation
/// were suppressed too until 2026-07-14, but those spans carry real speech —
/// phrase-final decays and next-word onsets at -11 to -22 dBFS in the PyTorch
/// reference — and zeroing them cut up to ~125 ms of audible speech at every
/// phrase boundary. The CS1 perceptual evaluation heard that as "abrupt
/// dropouts and hard cuts to silence" and failed the 15s bucket 3/3
/// (see README/Notes/cs1-audio-quality-evaluation-2026-07-14.md).
public func suppressPunctuationTokenAudio(
    _ audio: [Float],
    inputIds: [Int32],
    predDur: [Int],
    samplesPerDurationFrame: Int = PipelineConstants.samplesPerDurationFrame,
    fadeSamples: Int = PipelineConstants.punctuationFadeSamples
) -> [Float] {
    guard !audio.isEmpty, samplesPerDurationFrame > 0 else { return audio }
    let tokenCount = min(inputIds.count, predDur.count)
    guard tokenCount > 0 else { return audio }
    let alignedInputIds = Array(inputIds.prefix(tokenCount))
    let alignedPredDur = Array(predDur.prefix(tokenCount))

    var result = audio
    var frameStart = 0
    for tokenIndex in 0..<tokenCount {
        let durationFrames = max(0, alignedPredDur[tokenIndex])
        defer { frameStart += durationFrames }

        guard durationFrames > 0,
              shouldSuppressPunctuationSpan(inputIds: alignedInputIds, tokenIndex: tokenIndex) else {
            continue
        }

        let rawStart = frameStart * samplesPerDurationFrame
        let rawEnd = (frameStart + durationFrames) * samplesPerDurationFrame
        fadeToSilence(
            audio: &result,
            rawStart: rawStart,
            rawEnd: rawEnd,
            fadeSamples: fadeSamples
        )
    }
    return result
}

private func shouldSuppressPunctuationSpan(inputIds: [Int32], tokenIndex: Int) -> Bool {
    // Punctuation tokens only. Whitespace spans — including whitespace next to
    // punctuation — carry real speech (word onsets and phrase-final decays)
    // and must never be silenced. See the doc comment on
    // suppressPunctuationTokenAudio for the 2026-07-14 measurement.
    KokoroVocabulary.silentPunctuationTokenIds.contains(inputIds[tokenIndex])
}

private func fadeToSilence(
    audio: inout [Float],
    rawStart: Int,
    rawEnd: Int,
    fadeSamples: Int
) {
    let clampedRawStart = max(0, min(rawStart, audio.count))
    let clampedRawEnd = max(clampedRawStart, min(rawEnd, audio.count))
    guard clampedRawStart < clampedRawEnd else { return }

    let fade = max(0, fadeSamples)
    let fadeStart = max(0, clampedRawStart - fade)
    let fadeEnd = min(audio.count, clampedRawEnd + fade)

    if fadeStart < clampedRawStart {
        let count = clampedRawStart - fadeStart
        for index in fadeStart..<clampedRawStart {
            let progress = Float(index - fadeStart) / Float(max(1, count))
            audio[index] *= max(0, 1 - progress)
        }
    }

    for index in clampedRawStart..<clampedRawEnd {
        audio[index] = 0
    }

    if clampedRawEnd < fadeEnd {
        let count = fadeEnd - clampedRawEnd
        for index in clampedRawEnd..<fadeEnd {
            let progress = Float(index - clampedRawEnd + 1) / Float(max(1, count))
            audio[index] *= min(1, progress)
        }
    }
}
