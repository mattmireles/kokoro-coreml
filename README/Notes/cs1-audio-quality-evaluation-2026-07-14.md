# Case Study 1 — Perceptual Audio-Quality Evaluation (PyTorch vs Config F)

Date: 2026-07-14
Purpose: Retire the "No formal audio quality evaluation" limitation for Case
Study 1 (Kokoro-82M) in `Scratchpad/surgical-inference.md` §7.5 by producing a
documented, reproducible perceptual comparison of the PyTorch reference against
the shipped Config F (Swift + Core ML "Surgical Inference") pipeline, using the
`audio-judge` skill (Gemini multimodal listening) with a known-bad control.

> **UPDATE (2026-07-14, same day, later session): both long-bucket failures
> below were root-caused and FIXED.** See "Root causes and fixes" at the end
> of this note. Two attributions in the original text are corrected there:
> the 15s pause elongation was NOT the Core ML duration model (it is at
> per-token parity), and the 30s spectral tilt was NOT bucket-specific (it
> was present in every bucket). After the fixes, all four buckets pass blind
> paired lineups (15s 2/2, 30s 2/2, 3s 1/1, 7s 1/1) and the control is still
> rejected. The verdicts and analyses in the body of this note describe the
> PRE-fix pipeline and are retained as the record of what the gate caught.

## TL;DR

- Blind Gemini listening finds Config F **perceptually indistinguishable from
  the PyTorch reference on the 3s and 7s buckets** (same quality class, no
  artifacts, control correctly rejected in the same lineup).
- The **15s bucket consistently fails** the paired comparison (3/3 agreeing
  votes): the judge hears "unnatural silent gaps / dropouts between phrases."
  Signal analysis traces this to the Core ML duration model rendering
  phrase-boundary pauses **40–110 ms longer** than PyTorch at every comma/period
  boundary (same positions, same total duration, envelope corr 0.93). It is a
  prosody deviation at pause boundaries, not corruption: speech content,
  timbre, and intelligibility match the reference per the judge's own text.
- The gate has demonstrated discriminative power: a speech-shaped static
  control passes the objective waveform probe (`needs_listening`) but is
  unambiguously rejected by the perceptual judge in both labeled and blind
  lineups, while real speech passes.
- The **30s bucket fails 2/2 blind votes** on a different axis: the judge hears
  "persistent background static and hiss" over intelligible speech. Signal
  analysis shows a spectral tilt (1–9 kHz energy at 0.35–0.48× of the
  gain-matched reference, extremes intact), not an elevated pause noise floor.
- Net: the perceptual gate certifies parity on short/medium utterances and
  surfaces two concrete, reproducible quality deviations on long buckets that
  per-stage numerical parity (matched inputs) never saw — which is precisely
  the point of running one.

## What was compared

Two arms rendered from the SAME frozen bakeoff text and the SAME tokenized
inputs (`outputs/swift_bench_inputs/{key}.json` — one tokenization, shared by
both arms via `scripts/prepare_swift_bench_inputs.py`), so the arms differ only
in the inference engine:

- **PyTorch reference** — `kokoro.KModel` eager, CPU, float32, single-shot
  `kmodel(phonemes, ref_s, speed)` (identical to bakeoff Config E inference,
  reused via `scripts/bakeoff_harness.PyTorchContext`). Quality target, not a
  speed arm. Script: `scripts/gen_pytorch_reference_wavs.py`. Paths:
  `outputs/audio-parity/references/pytorch_{3s,7s,15s,30s}.wav`.
- **Config F (staged)** — Swift + Core ML decomposed pipeline (duration / F0N /
  decoder-pre / generator submodels), FP16, **shipped staged compute policy**
  (`kokoro-bench --compute-units staged`; recorded in each metrics JSON as
  `duration=cpuAndGPU,f0n=cpuAndGPU,decoderPre=cpuAndNeuralEngine,generator=cpuAndGPU`).
  Script: `scripts/gen_config_f_staged.sh`. Paths:
  `outputs/bakeoff/listen/staged/config_f_staged_{key}.wav`.
- **Config F (`.all`)** — same pipeline, `--compute-units all`, rendered via
  `scripts/bakeoff_listen.py` as a policy cross-check. Paths:
  `outputs/bakeoff/listen/config_f_{key}.wav`.
- **Known-bad control** — deterministic speech-shaped band-limited static with
  a 4 Hz syllable-rate envelope, 3.0 s (`scripts/gen_known_bad_control.py`,
  seed 0). Path: `outputs/audio-parity/references/known_bad_static_3s.wav`.

Frozen inputs (`scripts/bakeoff_harness.py::BAKEOFF_INPUTS`, voice `af_heart`,
speed `1.0`, seed 0 for the Swift arm, `torch.manual_seed(0)` for PyTorch):

| Key | Text (frozen) | Duration (both arms) |
| --- | --- | --- |
| 3s  | "The quick brown fox jumps over the dog." | 2.80 s |
| 7s  | "The morning sun cast long shadows across the garden as birds began their chorus in the ancient oak tree." | 6.75 s |
| 15s | "The ancient lighthouse stood alone on the rocky cliff, ... guiding sailors home." | 13.90 s |
| 30s | "When the last train departed that evening, ... its roof sheltering the pigeons." | 27.38 s (PyTorch) / 27.38 s (Config F staged; 27.40 s `.all`) |

> Note on "five buckets": the pipeline ships five model *buckets*
> (3s/7s/10s/15s/30s `.mlpackage`), but the frozen listening *input set*
> (`BAKEOFF_INPUTS`, `outputs/swift_bench_inputs`) has four texts — there is no
> 10s bakeoff text. This evaluation covers all four frozen inputs; the 10s
> bucket has no canonical listening input to render.

## Build / environment identification

- Repo git HEAD: `05e184895f03c17431e9333d6c84780407f0e2bb`
  (branch `codex/kokoro-drop-in-sdk-v1`).
- Swift: Apple Swift 6.3.3 (swiftlang-6.3.3.1.3), target arm64-apple-macosx26.0.
- Swift build: `swift build -c release --product kokoro-bench` (auto-rebuilt by
  `scripts/bakeoff_listen.py::_ensure_bench` because sources were newer than
  the June binary — clips are from the CURRENT pipeline build).
- Host: Apple M2 Ultra (Mac14,14), 64 GB, macOS 26.5.1 (25F80).
- PyTorch reference: repo-vendored `kokoro` package, `hexgrad/Kokoro-82M`.
- Core ML models: `coreml/kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage`
  plus duration / f0ntrain / decoder-pre / generator stage models.
- Judge model: Gemini via `scripts/gemini_audio_judge_direct.py` (fallback
  path — see "Instrument path" below).

## Exact commands

```bash
# 1. PyTorch reference WAVs (CPU eager, float32)
.venv/bin/python scripts/gen_pytorch_reference_wavs.py

# 2. Known-bad negative control (deterministic static)
.venv/bin/python scripts/gen_known_bad_control.py

# 3. Config F (.all) clips + waveform-health gate (also rebuilds kokoro-bench)
REF=outputs/audio-parity/references
.venv/bin/python scripts/bakeoff_listen.py --keys 3s,7s,15s,30s \
  --reference-wavs $REF/pytorch_3s.wav $REF/pytorch_7s.wav \
                   $REF/pytorch_15s.wav $REF/pytorch_30s.wav

# 4. Config F under the SHIPPED staged policy
bash scripts/gen_config_f_staged.sh            # all four keys, or pass a subset

# 5. Objective probe (references derive thresholds; candidates classified)
.venv/bin/python scripts/audio_quality_probe.py \
  --reference $REF/pytorch_3s.wav $REF/pytorch_7s.wav $REF/pytorch_15s.wav $REF/pytorch_30s.wav \
  --candidate outputs/bakeoff/listen/staged/config_f_staged_3s.wav \
              outputs/bakeoff/listen/staged/config_f_staged_7s.wav \
              outputs/bakeoff/listen/staged/config_f_staged_15s.wav \
              $REF/known_bad_static_3s.wav \
  --out-dir outputs/bakeoff/listen/staged/quality

# 6. Perceptual judge (fallback path; one lineup per command). Labeled example:
.venv/bin/python scripts/gemini_audio_judge_direct.py \
  --clip pytorch=$REF/pytorch_3s.wav \
  --clip coreml=outputs/bakeoff/listen/staged/config_f_staged_3s.wav \
  --baseline-label pytorch \
  --prompt "The quick brown fox jumps over the dog." \
  --expected-style "clear intelligible English speech, natural prosody, no whispering or static, no clicks or dropouts, 24 kHz mono" \
  --probe-first --context-file <neutral context .md> \
  --output outputs/audio-judge-fallback/cs1-staged-3s.json
# Blind lineups use de-identified labels clip_a/clip_b/clip_c and a neutral
# context file that does not reveal which engine produced which clip.
```

## Instrument path

The primary `llm-workflows` path was attempted first. The FFmpeg media-prep
worker was healthy (both clips transcoded), but run submission failed:

```
Error: https://llm-workflows-staging.gist-backend.workers.dev/v1/runs failed with 403:
"Client token does not authorize this workflow request." (code: forbidden)
```

This is a token-scoping issue on the workflow runtime, not a worker outage.
Per the audio-judge skill, the documented fallback
(`scripts/gemini_audio_judge_direct.py`, direct Gemini upload, auto 16-bit
conversion, RMS gain-match to baseline) was used for all verdicts below.
**These are fallback-path reports.** Re-run on the primary path once the token
scope is fixed. *(Resolved later the same day: the staging worker's dedicated
`WORKFLOW_CLIENT_AUDIO_JUDGE_TOKEN` secret now matches the local env and the
primary path is healthy — see "Primary-path replication" at the end of this
note.)*

## Objective waveform-health gate (pre-listening)

`outputs/bakeoff/listen/staged/quality/audio_quality_report.json`:

| Clip | Role | Decision | RMS (pcm) | active>32 | ZCR | Dur |
| --- | --- | --- | --- | --- | --- | --- |
| pytorch_3s.wav | reference | reference_pass | 1546 | 0.643 | 0.120 | 2.80 |
| pytorch_7s.wav | reference | reference_pass | 1605 | 0.725 | 0.126 | 6.75 |
| pytorch_15s.wav | reference | reference_pass | 1705 | 0.752 | 0.138 | 13.90 |
| pytorch_30s.wav | reference | reference_pass | 1665 | 0.763 | 0.143 | 27.38 |
| config_f_staged_3s.wav | candidate | needs_listening | 4532 | 0.781 | 0.089 | 2.80 |
| config_f_staged_7s.wav | candidate | needs_listening | 4897 | 0.852 | 0.101 | 6.75 |
| config_f_staged_15s.wav | candidate | needs_listening | 5300 | 0.824 | 0.100 | 13.90 |
| known_bad_static_3s.wav | candidate | needs_listening | 2122 | 0.892 | 0.291 | 3.00 |

Two lessons the probe teaches on its own:

1. The RMS gap (Config F ≈ 3× PyTorch) is a **WAV-writer convention, not
   pipeline gain**: `kokoro-bench` peak-normalizes its WAV output (documented
   at `swift/Sources/KokoroBenchmark/main.swift:364`), while
   `gen_pytorch_reference_wavs.py` writes raw amplitude (peaks 0.35–0.54).
   The judge script gain-matches candidates to the baseline RMS before upload.
2. The **objective gate cannot reject the known-bad control** — speech-shaped
   static lands in `needs_listening`, not `reject_without_listening`. The
   perceptual stage is what actually discriminates (see control results).

During the `.all` renders, the 7s and 30s runs logged
`MILCompilerForANE error: ... ANECCompile() FAILED` — Core ML fell back per its
policy and the runs completed and validated. Recorded here because it is more
evidence for the paper's "requested compute units are a request" theme.

## Perceptual judge — method

- Instrument: Gemini multimodal listening via
  `scripts/gemini_audio_judge_direct.py` (audio-judge skill fallback path),
  RMS gain-match to baseline, `--probe-first` on labeled runs.
- Per-clip absolute judgments (intelligibility, artifacts, naturalness) plus a
  paired comparison against the PyTorch baseline (`same_quality_class`).
- **Blind lineups**: labels `clip_a`/`clip_b`/`clip_c` with a neutral context
  file that does not reveal which engine produced which clip. Labeled lineups
  use `pytorch`/`coreml` labels and a factual context file.
- **Control**: `known_bad_static_3s.wav` included in the labeled control run
  and the blind 3s lineup.
- Protocol per the skill: one run = one vote; 2–3 agreeing lineups required.

## Per-bucket verdicts (verbatim from result JSONs)

### Control run (`cs1-control-3s.json`, labeled: pytorch vs control)

- pytorch: verdict **pass** — "Clear, intelligible English speech with natural
  prosody. The spoken content accurately matches the input text with no
  noticeable artifacts."
- control: verdict **fail** — artifacts `["loud static", "pulsing noise",
  "metallic buzz", "complete loss of speech"]` — "The clip consists entirely of
  loud, pulsing static and metallic buzz. There is no intelligible speech."
- `control_vs_pytorch.same_quality_class: false`.

### 3s — PASS (2/2 votes)

Labeled (`cs1-staged-3s.json`): overall **pass**.
- coreml: "Clear, natural-sounding speech that accurately matches the input
  text. The audio is clean, intelligible, and free of any static, metallic
  buzz, clicks, or dropouts." verdict pass, artifacts [].
- `coreml_vs_pytorch`: "The Core ML output is perceptually indistinguishable
  from the PyTorch baseline. It successfully retains the same voice
  characteristics, prosody, and audio clarity without introducing any audible
  artifacts from the FP16 conversion or surgical pipeline."
  `same_quality_class: true`.

Blind 3-clip lineup (`cs1-blind-3s.json`; clip_a=pytorch, clip_b=Config F
staged, clip_c=control — identities withheld from the judge):
- clip_b (Config F): pass — "Clear and intelligible speech that matches the
  input text. The delivery and voice quality are excellent." /
  "Clip B is in the exact same quality class as the baseline."
  `same_quality_class: true`.
- clip_c (control): fail — "Completely unintelligible audio... loud, harsh
  static and periodic metallic buzzing. No speech can be heard."
  `same_quality_class: false`.
- Overall lineup verdict is "fail" only because the control is in the lineup —
  the control failing while both speech clips pass is the desired outcome and
  demonstrates the gate's discriminative power under blind labels.

### 7s — PASS (2/3 votes; the single fail is an outlier contradicted by signal analysis)

Labeled (`cs1-staged-7s.json`): overall **fail** — coreml flagged with
"persistent metallic buzz and static noise overlaying the voice throughout"
(`same_quality_class: false`).

Blind repeats (`cs1-blind-7s-v3.json`, `cs1-blind-7s-v4.json`): both **pass**.
- v3 clip_b: "Clear and intelligible English speech, essentially identical to
  the baseline in delivery and tone. No audible corruption, artifacts, or
  anomalies." — "perceptually indistinguishable", `same_quality_class: true`.
- v4 clip_b: "Clear and highly intelligible speech. Prosody matches the
  baseline very closely, with no obvious degradation, static, or distortion."
  `same_quality_class: true`.

Cross-check: the Config F 7s clip has *less* high-frequency (>8 kHz) energy
than the PyTorch reference (0.018 vs 0.025 of total), i.e. no measurable added
static. The labeled fail is treated as single-vote judge noise (possibly primed
by the labeled context mentioning FP16/Core ML); the blind votes and the signal
agree on pass. (Two additional blind 7s attempts, v1/v2, aborted on Gemini
malformed-JSON responses and produced no verdict — logged for completeness.)

### 15s — FAIL on pause prosody (3/3 agreeing votes)

Labeled run (`cs1-staged-15s` attempt): Gemini's raw response (run aborted on a
JSON formatting error after the verdict text) — coreml: "The voice timbre and
pronunciation match the baseline, but there are abrupt dropouts and hard cuts
to silence between phrases (e.g., after 'cliff', 'gone', 'walls'" — verdict
fail. pytorch: pass.

Blind repeats (`cs1-blind-15s-v1.json`, `cs1-blind-15s-v2.json`): both **fail**.
- v1 clip_b: artifacts `["dropouts", "clicks", "gaps"]` — "Speech is
  intelligible and has decent underlying prosody, but suffers from severe audio
  dropouts, sudden gaps, and clicks throughout the recording."
  `same_quality_class: false`.
- v2 clip_b: artifacts `["dropouts", "long silence gaps"]` — "The audio
  contains frequent, unnatural silent gaps or dropouts between phrases,
  completely disrupting the natural flow of speech."
  `same_quality_class: false`.

Root-cause signal analysis (this session):
- Envelope correlation PyTorch vs Config F (10 ms frames): **0.93** — same
  phrase layout, same speech content.
- Pause table (silence runs ≥60 ms, threshold 2% of peak envelope):
  every phrase-boundary pause in Config F is **40–110 ms longer** than the
  PyTorch reference at the same position (e.g. 6660 ms boundary: 740 ms vs
  630 ms; 8550 ms: 370 ms vs 280 ms), while total duration is identical
  (13.90 s both arms).
- The judge's cited positions ("after 'cliff', 'gone', 'walls'") are exactly
  the comma boundaries in the frozen text.
- The pause structure is **identical between the staged and `.all` policies**
  (both come from the same Core ML duration model), so this is a
  duration-model prosody deviation at phrase boundaries, not a compute-policy
  or vocoder artifact. Both arms gate inter-phrase silence to digital zero, so
  the longer pauses read as "dropouts" next to the reference.
- No added HF energy (Config F 0.016 vs PyTorch 0.023 above 8 kHz); hard-clip
  fraction negligible (≤0.0015% of samples, from peak normalization).

### 30s — FAIL on spectral tilt / perceived hiss (2/2 blind votes)

Blind repeats (`cs1-blind-30s-v1.json`, `cs1-blind-30s-v2.json`): both **fail**.
- v1 clip_b: artifacts `["static", "metallic buzz"]` — "The speech is
  intelligible, but there is a prominent background static, hiss, and a
  metallic buzzing quality to the voice, typical of heavy compression or
  vocoder degradation." `same_quality_class: false`.
- v2 clip_b: artifacts `["static", "hiss"]` — "The speech is intelligible and
  the prosody is natural, but the audio quality is degraded by persistent
  background static and hiss throughout the entire clip."
  `same_quality_class: false`.

Root-cause signal analysis (this session, RMS gain-matched before comparison):
- Pause noise floor is NOT elevated (−65.7 dBFS vs −63.7 dBFS reference), so
  this is not literal added hiss during silence.
- Band-energy ratios Config F / PyTorch: 0–1 kHz **1.25×**, 1–3 kHz **0.35×**,
  3–6 kHz **0.37×**, 6–9 kHz **0.48×**, 9–12 kHz **1.00×**. The 30s render has
  markedly less mid/high-mid speech energy while retaining full energy at the
  spectral extremes — relatively elevated >9 kHz content over duller speech,
  which is exactly what "persistent static/hiss over the voice" sounds like.
- Pause structure is close to reference here (total pause 5.08 s vs 5.11 s,
  boundary deltas +30–50 ms; envelope corr 0.92) — the 30s failure mode is
  spectral, distinct from the 15s pause-prosody failure mode.
- Context: the 30s bucket runs the largest tensors (padded_t512 duration,
  476/512 tokens), and the `.all`-policy 30s render logged an
  `ANECCompile() FAILED` fallback; a per-stage numeric audit of the 30s bucket
  end-to-end (not per-stage matched-input parity) is the follow-up.

## Summary table

| Bucket | Objective gate | Perceptual verdict (votes) | same_quality_class vs PyTorch | Judge's key finding |
| --- | --- | --- | --- | --- |
| 3s  | needs_listening | **PASS** (2/2: labeled + blind) | true | "perceptually indistinguishable" |
| 7s  | needs_listening | **PASS** (2/3; blind votes pass, one labeled outlier fail) | true (blind) | "essentially identical to the baseline"; buzz claim not supported by spectrum |
| 15s | needs_listening | **FAIL** (3/3) | false | phrase-boundary pauses 40–110 ms longer than reference, heard as gaps/dropouts |
| 30s | needs_listening | **FAIL** (2/2 blind) | false | mid-band energy 0.35–0.48× of reference (spectral tilt), heard as "static and hiss" over intelligible speech |
| control | needs_listening (probe cannot reject) | **FAIL** (2/2: labeled + blind) | false | "no intelligible speech" — gate discriminates |

## Instrument limits (state plainly)

This is an **automated perceptual judgment by a multimodal model (Gemini), not
a human Mean Opinion Score study.** It is a repeatable smoke/quality gate: it
catches gross intelligibility loss, added noise, and prosody deviations, and it
correctly rejects a known-bad control that the objective waveform probe cannot.
It does not produce a calibrated MOS, has no human listener panel, no ASR/WER
scoring, and exhibits single-run noise (the 7s labeled outlier); the protocol
compensates with repeat blind votes. All verdicts here are from the documented
fallback path (direct Gemini upload) because the primary llm-workflows path
returned a 403 token-scope error; primary-path re-runs should replicate before
the paper's camera-ready. *(Done: post-fix verdicts replicate on the primary
path for all four buckets plus the control — see "Primary-path replication"
at the end of this note.)*

## Follow-ups

1. ~~Fix the llm-workflows client token scope and replicate on the primary
   path.~~ **Done** — token scope resolved server-side (dedicated
   `WORKFLOW_CLIENT_AUDIO_JUDGE_TOKEN` staging secret); all post-fix verdicts
   replicate on the primary path (see "Primary-path replication").
2. ~~Investigate the duration-model pause elongation on the 15s bucket.~~
   **Done** — duration model exonerated (per-token parity); real cause was
   whitespace suppression (see "Root causes and fixes").
3. ~~Investigate the 30s spectral tilt.~~ **Done** — one-sided iSTFT scaling
   bug in `CustomSTFT.inverse`, present on every bucket, policy-independent
   (cpuOnly discrimination run confirmed); fixed and re-exported (see "Root
   causes and fixes").
4. Add a 10s frozen input if the 10s bucket should be listening-covered.
5. ~~Consider crossfade/decay shaping at inter-phrase silence onsets.~~
   Moot — with whitespace suppression removed, phrase-boundary audio matches
   the reference; punctuation spans remain gated below audibility.

## Artifacts

- PyTorch refs: `outputs/audio-parity/references/pytorch_{3s,7s,15s,30s}.wav`
- Known-bad control: `outputs/audio-parity/references/known_bad_static_3s.wav`
- Config F staged: `outputs/bakeoff/listen/staged/config_f_staged_{key}.wav` (+ `.json` metrics with `compute_unit_policy`)
- Config F `.all`: `outputs/bakeoff/listen/config_f_{key}.wav` (+ `.json`)
- Objective gate: `outputs/bakeoff/listen/staged/quality/audio_quality_report.json`
- Judge envelopes: `outputs/audio-judge-fallback/cs1-*.json`
  (`cs1-control-3s`, `cs1-staged-{3s,7s}`, `cs1-blind-3s`,
  `cs1-blind-7s-{v3,v4}`, `cs1-blind-15s-{v1,v2}`, `cs1-blind-30s-{v1,v2}`)
- Generation scripts (new this session):
  `scripts/gen_pytorch_reference_wavs.py`, `scripts/gen_known_bad_control.py`,
  `scripts/gen_config_f_staged.sh`

## Root causes and fixes (2026-07-14, follow-up session)

Both long-bucket failures were investigated to root cause the same day, fixed,
re-rendered, and re-judged. Chain of evidence below; every probe script is
checked in.

### 15s "pause elongation" — the duration model was innocent

`scripts/probe_duration_pause_parity.py` compares per-token `pred_dur` for the
frozen 15s input (219 tokens) across PyTorch eager (exact length, fp32),
Core ML `padded_t256` (CPU_ONLY and ALL), and Core ML `exact_t219`:

- Total frames: 556 in every arm. Largest deviation: ONE token at ±1 frame
  (25 ms) in the CPU arms, zero tokens under ALL. The Core ML duration model
  is at per-token parity; the original attribution was wrong.

The actual cause was Swift's `suppressPunctuationTokenAudio`
(`swift/Sources/KokoroPipeline/WaveformPostProcess.swift`, added 2026-05-26
for punctuation clicks): it hard-zeroed not only punctuation-owned spans but
also **adjacent whitespace spans**. `scripts/analyze_punctuation_span_energy.py`
shows those whitespace spans carry real speech in the PyTorch reference
(word onsets / phrase-final decays at **-11.9 to -22.2 dBFS**, up to 125 ms),
while punctuation spans proper are near-silence (-38 to -85 dBFS). Zeroing
speech-bearing whitespace produced the "abrupt dropouts / hard cuts" the judge
heard and the 40-110 ms measured pause elongation (pause runs extend across
the zeroed whitespace at the 2%-of-peak threshold; total duration unchanged).
`scripts/analyze_raw_punct_spans.py` on a `--dump-tensors` render confirms the
PRE-suppression Core ML waveform already matched the reference in every one of
those spans (no clicks in the current HAR/HNSF pipeline on this input; raw
punctuation-span peaks -24 to -50 dBFS, same order as the reference).

**Fix:** suppression narrowed to punctuation tokens only; whitespace is never
silenced (`WaveformPostProcess.swift`, `KokoroVocabulary.swift`, tests
updated). Punctuation-span zeroing is retained as click protection — the
reference is below the pause-measurement threshold there, so it does not
perturb measured pause structure.

### 30s "spectral tilt" — a one-sided iSTFT scaling bug in every bucket

Discrimination chain (each step one variable):

1. `scripts/analyze_band_energy_ratio.py`: the 1-9 kHz deficit (0.35-0.48x)
   is **identical under staged, `.all`, and `cpuOnly`** policies and present
   on **all four buckets** — including 3s/7s, which the judge had passed. Not
   placement, not FP16-on-GPU, not bucket-size.
2. Python fp32 mirror of the Config F stage decomposition
   (`scripts/capture_audio_parity_tensors.py`, complex-STFT KModel) is at
   spectral parity with the eager reference → the decomposition itself is
   sound.
3. Swift tensor dump vs Python mirror (`compare_audio_parity_tensors.py`):
   everything through `x_pre` matches at corr ≥ 0.999996. At seed 42 the Swift
   `har_source` is spectrally identical to PyTorch (1.00x per band), and the
   Core ML generator reproduces the fp32 `disable_complex` generator at corr
   1.0000 on matched inputs — yet the waveform still shows the tilt.
4. Generator A/B with identical `har` input: `TorchSTFT` (complex) generator
   is clean; **`CustomSTFT` (`disable_complex=True`) generator reproduces the
   tilt exactly** (1.22x / 0.41-0.47x / 0.76x). The forward transform is fine;
   the **inverse** is the bug.

Root cause in `kokoro/custom_stft.py::__init__`: the inverse DFT weights used
uniform `1/n_fft` scaling for all frequency bins of a **one-sided** spectrum.
Interior bins k = 1..n_fft/2-1 represent both +k and -k and need `2/n_fft`;
only DC and Nyquist appear once. With the generator's n_fft=20 @ 24 kHz,
every bin from 1.2-10.8 kHz was reconstructed at half amplitude while DC
(0-600 Hz) and Nyquist (12 kHz) came through at full strength — exactly the
measured signature after gain matching. The inverse also lacked the
overlap-add window-power normalization (`1/1.5` for periodic Hann at hop 5).
The exported `kokoro_decoder_har_post_*` packages baked this in; the trained
generator expects `torch.istft` semantics.

**Fix:** one-sided doubling + OLA normalization folded into the
`weight_backward_*` buffers. The fixed `CustomSTFT.inverse` now matches
`torch.istft` bit-exactly in the interior (max abs diff 8.9e-08 on an
inconsistent spectrogram), and the fp32 `disable_complex` generator matches
the complex generator at max abs diff 4.9e-07. `tests/test_custom_stft.py`
tightened (round-trip SNR > 35 dB; new inconsistent-spectrogram
torch.istft-parity test) so the regression cannot return. All five
`kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage` re-exported
(`python -m export_synth.main --mode decoder-har --buckets 3s,7s,10s,15s,30s
-o coreml`); the new FP16 3s package scores corr 0.99999 / SNR 45.7 dB against
the complex-STFT fp32 reference on the parity dump
(`scripts/check_coreml_generator_from_dump.py`).

### Aggravator: seed-0 RNG degeneracy in the Swift HNSF noise

`SeededRNG` (xorshift64) had an absorbing zero state: `--seed 0` — used for
every CS1 clip — made every draw return 0, turning the Box-Muller "Gaussian"
noise into a deterministic `(5.65, 0, 5.65, 0, ...)` DC + Nyquist impulse
train and removing ALL broadband noise from the harmonic source (har_source
3-6 kHz at 0.01x of PyTorch, 9-12 kHz at 41x). Combined with the iSTFT bug
this made the seed-0 renders duller still. **Fix:** SplitMix64 seed scrambling
in `SeededRNG.init` (`HarmonicSource.swift`); at any seed the Swift
`har_source` now matches the PyTorch source spectrum at 1.00x per band.

### Post-fix verification (this session)

- Band ratios vs reference after fixes (staged policy, seed 0):
  3s `0.99/0.95/0.99/1.20/1.45`, 7s `1.00/0.87/1.05/1.13/1.38`,
  15s `1.00/0.90/1.10/1.22/1.44`, 30s `1.00/0.89/0.97/1.17/1.43`
  (bands 0-1/1-3/3-6/6-9/9-12 kHz; pre-fix mid-band was 0.31-0.49x). The
  small >9 kHz surplus is within the fp32 Python mirror's own
  noise-realization variance (it measures 1.33x there).
- 15s whitespace spans now carry speech at reference level
  (-11.9/-19.6/-20.4/-12.9/-19.9 dBFS vs reference -11.9/-21.0/-22.2/-13.3/
  -19.8); punctuation spans remain gated, reference is below threshold there.
- Blind paired lineups (fallback Gemini path, neutral context,
  `outputs/audio-judge-fallback/cs1fix-*.json`): **15s PASS 2/2**
  ("Both clips are of excellent quality... same high-quality class"),
  **30s PASS 2/2** (no artifacts, `same_quality_class: true`; previously
  "static and hiss" 2/2 fail), 3s PASS 1/1, 7s PASS 1/1.
- Blind 3-clip control lineup (`cs1fix-blind-3s-control.json`): Config F
  "excellent speech quality on par with the baseline"; the speech-shaped
  static control is still unambiguously rejected — the gate's discriminative
  power is unchanged.
- `pytest` (122 passed) and `swift test` (46 tests, 0 failures) are green;
  objective probe re-run: all references `reference_pass`, all candidates
  `needs_listening` as designed.

Files changed: `kokoro/custom_stft.py`,
`swift/Sources/KokoroPipeline/WaveformPostProcess.swift`,
`swift/Sources/KokoroPipeline/KokoroVocabulary.swift`,
`swift/Sources/KokoroPipeline/HarmonicSource.swift`,
`tests/test_custom_stft.py`,
`swift/Tests/KokoroPipelineTests/WaveformPostProcessTests.swift`, plus
re-exported `coreml/kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage`.
New diagnostics: `scripts/probe_duration_pause_parity.py`,
`scripts/analyze_punctuation_span_energy.py`,
`scripts/analyze_raw_punct_spans.py`, `scripts/analyze_band_energy_ratio.py`.

### Primary-path replication (2026-07-14, third session)

The llm-workflows 403 (`Client token does not authorize this workflow
request`) no longer reproduces: the staging worker now carries the dedicated
`WORKFLOW_CLIENT_AUDIO_JUDGE_TOKEN` secret (per the ops-runbook per-client
secret migration) and it matches `llm-workflows/.env`
`WORKFLOW_RUNTIME_TOKEN` — verified by an auth probe
(`GET /v1/clients/audio-judge/runs/<bogus>` returns 404 authorized, not 403)
and by five completed runs. No server or env change was required.

All post-fix verdicts replicate on the PRIMARY path
(`node scripts/run-audio-judge.mjs`, workflow `audio_judge_v1`, artifacts
under `llm-workflows/outputs/audio-judge/`):

| Pair (pytorch vs Config F staged) | overallVerdict | `iphoneAcceptablyCloseToMlx` | clicks/dropouts | noise |
| --- | --- | --- | --- | --- |
| 3s  | pass | true | none | 0 |
| 7s  | pass | true | none | 0 |
| 15s | pass | true | none | 0 — "perceptually indistinguishable from the PyTorch baseline" |
| 30s | pass | true | none | 0 |
| control (pytorch vs known-bad static) | fail | — | — | 100 — "complete static", correctly rejected |

Operational note: re-running the same clip label + same file used to 409
(`idempotency_conflict`) because `run-audio-judge.mjs` derived its
idempotency keys from content only while the request bodies embed
per-invocation values (fresh FFmpeg uploadId; per-run Gemini file URIs).
Fixed 2026-07-14 in llm-workflows (`scripts/run-audio-judge.mjs`): the
FFmpeg job key now includes the uploadId and the run key includes the
invocation stamp, matching the other runner scripts. Verified by running the
previously poisoned pytorch/control 3s lineup twice back-to-back — both
completed, control correctly rejected both times (noise 100, ranked worst).

Residual follow-up: add a 10s frozen listening input (unchanged from the
original Follow-ups list).

## For the paper (§7.5 rewrite or short §5 quality subsection)

The paper draft (`Scratchpad/surgical-inference.md` §7.5 "Audio quality
evaluation is automated, not human — and it caught real defects that
per-stage parity certified as correct") was updated 2026-07-14 to tell the
full arc: initial run passes short/medium and fails both long buckets with
signal-analysis backing; stage-boundary tensor bisection traces both failures
to implementation defects invisible to per-stage matched-input correlation
(speech-bearing whitespace suppression; one-sided iSTFT scaling at half
amplitude on interior bins, present in every bucket); after the fixes all
four inputs pass the blind evaluation (15s 2/2, 30s 2/2) with the control
still rejected. The §5 numerical-validation bullet was updated to match.
