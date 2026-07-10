---
name: audio-judge
description: >-
  Judge generated Kokoro TTS audio clips with Gemini through llm-workflows. Use
  when the user asks whether synthesized speech sounds good, intelligible,
  whispery, corrupt, or acceptable, or when comparing PyTorch reference clips
  against Core ML / Swift pipeline output.
---

# Audio Judge (Kokoro TTS)

## Purpose

Fast perceptual gate for Kokoro TTS WAV clips: Gemini listens and returns a
structured verdict. Objective waveform probes run first; listening is for
`needs_listening` candidates and A/B comparisons.

## Use When

- The user asks whether TTS output sounds acceptable, intelligible, whispery,
  or corrupt.
- Comparing PyTorch reference clips against Core ML / Swift pipeline output.
- A listening pack or bakeoff clip needs a smoke gate before human review.

## Do Not Use When

- The task is **numeric parity only** — use **coreml-validate** first.
- The task is **waveform health only** with no listening need — use
  `scripts/audio_quality_probe.py` alone.
- The FFmpeg worker is healthy and you only need timings — use **bakeoff**.

## Ground Rules

1. Read `CLAUDE.md`. This skill is for **kokoro-coreml** (Kokoro TTS), not
   Crossfade music generation.
2. Do not judge TTS by correlation/SNR alone. Perceptual listening is the
   gate when metrics and ears disagree.
3. Keep the boundary intact:
   - FFmpeg owns media prep and Gemini file upload on the primary path.
   - `llm-workflows` owns prompt execution and primary artifacts.
   - kokoro-coreml owns generation, context, and the TTS-native fallback
     script.
   The fallback is **only** for worker-down recovery — never replace listening
   with metrics.
4. Kokoro rubric (from `scripts/audio_quality_probe.py` and bakeoff practice):
   intelligible English speech, natural prosody, no whisper/static/clicks/
   dropouts, spoken text matches the input prompt when provided.

**Primary-path caveat:** `audio_judge_v1` was written for Crossfade music.
Compensate with strong TTS framing in `--prompt`, `--expected-style`, and
`--context-file`. Use the **primary path** whenever the worker is healthy.

## Preflight (before the first run)

1. **Objective gate** (cheap rejects):

   ```bash
   uv run --no-sync python scripts/audio_quality_probe.py /path/to/clip.wav
   ```

   If `quality_decision` is `reject_without_listening`, fix generation before
   Gemini. Only send `needs_listening` or reference clips.

2. **Primary path only:** invoke with `node`, not `pnpm run judge:audio --`
   (the `--` is forwarded and breaks the script).

3. **Primary path only:** convert float32 WAVs to 16-bit PCM before upload —
   the FFmpeg worker probing stage rejects float32:

   ```bash
   afconvert -f WAVE -d LEI16 in.wav out.wav
   ```

   The fallback script auto-converts WAVs; step 3 is not required there.

4. **Env:** `WORKFLOW_RUNTIME_TOKEN` and `FFMPEG_CLIENT_TOKEN` in
   `llm-workflows/.env` (or set `LLM_WORKFLOWS_ENV` to that file).
   `GEMINI_API_KEY` is required for the fallback (same `.env` or env var).
   Optional: `WORKFLOW_RUNTIME_BASE_URL`, `FFMPEG_BASE_URL`.

## Typical clip sources

| Label | Role | Common path |
| --- | --- | --- |
| `pytorch` / `pytorch_3s` | Known-good PyTorch reference | `outputs/audio-parity/references/pytorch_{3s,7s,15s,30s}.wav` |
| `coreml` / `config_f_3s` | Swift + Core ML candidate | `outputs/bakeoff/listen/config_f_{3s,7s,15s,30s}.wav` |
| `candidate` | F0-source or parity experiment | `outputs/f0_source_listening/**/wav/*_candidate.wav` |

Render fresh Config F clips:

```bash
uv run --no-sync python scripts/bakeoff_listen.py --keys 3s
```

Bakeoff text: `scripts/bakeoff_harness.py` (`BAKEOFF_INPUTS`). Pass matching
text as `--prompt`.

## Command (primary path)

From the `llm-workflows` checkout (sibling of this repo by default):

```bash
node scripts/run-audio-judge.mjs \
  --clip pytorch=/absolute/path/to/pytorch_3s.wav \
  --clip coreml=/absolute/path/to/config_f_3s.wav \
  --baseline-label pytorch \
  --prompt "The quick brown fox jumps over the dog." \
  --expected-style "clear intelligible English speech, natural prosody, no whispering or static, no clicks or dropouts, 24 kHz mono" \
  --context-file /absolute/path/to/kokoro_context.md
```

Options: `--context-file` (neutral facts only), `--no-wait`, `--resume`.

**Context-file safety:** the entire file is sent to Gemini. Never point it at
`.env`, credentials, or unrelated secrets.

### Context file template

```markdown
# Kokoro TTS audio judge context

- Task: perceptual TTS quality gate (not music).
- Sample rate: 24000 Hz mono 16-bit PCM.
- Voice: af_heart, speed: 1.0.
- Input key: 3s bakeoff sentence.
- Candidate: Config F (Swift + Core ML, compute_units=ALL).
- Baseline: PyTorch eager reference from audio-parity exports.
- Objective probe: needs_listening; rms=4600, active32=78.8%, zcr=9.1%.
- Question: Is Core ML output acceptably close to PyTorch for shipping?
```

## Fallback (worker down only)

```bash
uv run --no-sync python scripts/gemini_audio_judge_direct.py \
  --clip pytorch=/absolute/path/pytorch_3s.wav \
  --clip coreml=/absolute/path/config_f_3s.wav \
  --baseline-label pytorch \
  --prompt "The quick brown fox jumps over the dog." \
  --probe-first \
  --context-file /path/to/kokoro_context.md \
  --output outputs/audio-judge-fallback/<slug>.json
```

Fallback behavior:

- Reads `GEMINI_API_KEY` from env or `LLM_WORKFLOWS_ENV` (default sibling
  `../llm-workflows/.env`).
- Auto-converts WAVs to 16-bit PCM; gain-matches non-baseline clips to
  baseline RMS (disable with `--no-gain-match`).
- `--probe-first` runs `audio_quality_probe` before Gemini.
- Clips must be `.wav`/`.mp3` audio files under size limits — not arbitrary
  local files.
- `--output` must be inside this repo; writes a structured envelope with
  `verdict`, clip paths, and schema hints.

Clips upload directly to Gemini (no FFmpeg/R2). Mark fallback reports in notes.
Switch back to the primary path once the worker is healthy.

## Known failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Unknown argument: --` | `pnpm run judge:audio --` | `node scripts/run-audio-judge.mjs` |
| `FFMPEG_CLIENT_TOKEN is required` | missing from `.env` | Preflight 4 |
| job fails at `probing` | float32 WAV | `afconvert -d LEI16`, retry |
| `409 idempotency_conflict` | poisoned job key | new clip labels (`coreml_v2=...`) |
| `Container start failed with HTTP 500` | worker down | fallback script |

## Output

**Primary:** `llm-workflows/outputs/audio-judge/<slug>/<timestamp>/` —
`result.json`, `report.md`, checkpoints.

**Fallback:** `--output` JSON envelope under `outputs/audio-judge-fallback/`.

Capture conclusions with **write-notes** (link artifacts, clip paths, config,
primary vs fallback).

## Judging protocol

1. Neutral context only — facts, never expected outcome.
2. Always include controls: PyTorch reference and a known-bad clip when
   available. Discard runs that misrank controls.
3. One run = one vote; require 2–3 agreeing lineups for ship/no-ship.
4. Cross-check `audio_quality_probe`, waveform alignment, bakeoff parity JSON.
5. No ASR/WER gate unless the user explicitly asks.

## Interpreting results

- `overallVerdict: "pass"` is a smoke gate, not bakeoff readiness.
- Failed Core ML + passing PyTorch → Core ML/runtime/glue, not input text.
- Failed PyTorch baseline → fix reference path before ANE debugging.

**Primary path** (`result.json`): use
`comparison.iphoneAcceptablyCloseToMlx` as “Core ML acceptably close to
PyTorch” when labels are `coreml` vs `pytorch`.

**Fallback path** (`verdict` envelope): use
`verdict.comparisons["<label>_vs_<baseline>"].same_quality_class` — not
`iphoneAcceptablyCloseToMlx`. The envelope documents both field names.

## Related skills

- [bakeoff](../bakeoff/SKILL.md) — render Config F clips and timings.
- [coreml-validate](../coreml-validate/skill.md) — numeric parity before
  listening.
- [write-notes](../write-notes/SKILL.md) — institutional memory after verdicts.
