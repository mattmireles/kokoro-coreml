# Chunk-Stitch Listening Evidence

Collected: 2026-06-10. Perceptual gate run with the audio-judge fallback
(`scripts/gemini_audio_judge_direct.py`, FFmpeg worker was failing at
probing); two independent Gemini votes, both with a clean in-lineup control.

## Question

Does the production text-chunking path (gist `kokoro-chunk-plan.js` →
per-chunk Kokoro synthesis → raw concatenation, as used by the gist/botnet
reader-audio pipeline) produce acceptable long-form audio at the seams?

## Method

- ~860-char news-style passage containing one >450-token sentence, chunked
  with the real production chunker
  (`gist/packages/protocol/shared/kokoro-chunk-plan.js`,
  `chunkTextForKokoro`), and the oversized chunk re-split with the real
  `splitKokoroChunkForRetry` after the production prep script
  (`botnet/scripts/kokoro-prepare-input.py`) rejected it at 460 tokens >
  `MAX_TTS_CHUNK_TOKENS` 450 — i.e. the exact production retry path.
- 8 segments synthesized independently via `kokoro-bench` (staged compute
  units, af_heart, speed 1.0, this Mac), concatenated end-to-end with no
  crossfade (production stitches uniform per-slide MP3s by raw byte concat —
  `gist/.../share-audio-builder.ts`).
- Judge lineup: stitched 59 s clip vs one single-sentence clip as control;
  segment boundaries not disclosed; anomaly timestamps requested.
- Clips: `outputs/chunk_listening/` (gitignored). Verdicts:
  `outputs/audio-judge-fallback/chunk-stitch-v{1,2}.json`.

## Verdict (2/2 votes: fail; control passed both votes)

Both votes located the true seams blind:

- `Dr. | Hartman's` — the retry splitter severed the honorific from the name
  ("unnatural pause splitting the title and name").
- `distribution | centers,` — the retry splitter emitted an 8-char orphan
  chunk ("centers,") that synthesized as an isolated utterance.
- Clicks/pops at several joins (raw concat, no crossfade, no zero-crossing
  alignment).
- "Terminal intonation in the middle of sentences" — each fragment is
  synthesized in isolation, so mid-clause fragments get sentence-final
  falling pitch. This is inherent to independent per-chunk synthesis, not a
  splitter bug.

Two of five flagged clicks fell inside single chunks (not at seams) — either
model-generated transients or judge noise; not load-bearing.

## Interpretation

- The COMMON path (sentence-per-chunk, no retry split) was not the failing
  case here; sentence-boundary seams are inherently more benign, but the
  click risk from raw concat applies to every seam.
- The LONG-SENTENCE path is broken: `splitKokoroChunkForRetry` lacks the
  abbreviation/initial guards that `isLikelyBoundary` gives the sentence
  chunker, and it emits tail fragments below any minimum length. Both bugs
  live in `gist/packages/protocol/shared/kokoro-chunk-plan.js`.
- For this repo's planned generator-internal overlap-add chunking
  ([kokoro-iphone-performance-v1.md](../Plans/kokoro-iphone-performance-v1.md)
  Phase 5), this evidence raises the bar explicitly: those seams land
  mid-word at fixed intervals, so receptive-field overlap + crossfade are
  mandatory, and the same blind-judge protocol (boundaries undisclosed,
  timestamped anomalies, in-lineup control) is the acceptance gate.

## Caveats

- Mac-synthesized WAV concat; production concatenates MP3 frames (codec
  framing may slightly alter click audibility in either direction).
- Fallback judge path (no MP3 normalization); clips were gain-matched by the
  fallback script.
- Single passage, single voice; votes were 2/2 with passing controls.
