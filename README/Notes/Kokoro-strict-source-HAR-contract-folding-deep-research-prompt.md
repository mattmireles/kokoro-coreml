# Kokoro Strict Source/HAR Contract Folding Deep Research Prompt

June 6, 2026

Create an advanced developer field guide on strict source/HAR contract folding
for Kokoro/ISTFTNet-style vocoders on Apple Silicon.

## Context

We are optimizing `kokoro-coreml`, a Swift + Core ML Kokoro/CoPro-style TTS
pipeline for Apple devices. The target is warmed inference only, excluding Core
ML compile/cache time. Runtime buckets are fixed: `3s`, `7s`, `10s`, `15s`,
and `30s`.

Current lower-end Mac blocker:

- Corrected warmed Config F already beats MLX on valid Mac full-duration rows.
- The remaining Mac blocker is `laishere/kokoro-coreml` on lower-end Macs,
  especially Irvine M1 `3s`, `7s`, and `10s`.
- After two strict candidates, DecoderPre/HnSF runtime overlap and HAR-post
  upsample ConvTranspose rewrite, Irvine still needs another strict save of
  `27.0 ms` / `28.0 ms` / `12.8 ms` on warmed-profile `3s` / `7s` / `10s`.
  Paper frontier needs `45.7 ms` / `77.6 ms` / `63.7 ms` / `64.4 ms` on
  `3s` / `7s` / `10s` / `15s`.

Current source/body evidence:

- The Swift-like sine/source equation is solved. Seeded Swift-like source
  generation matches dumped `har_source` across buckets with minimum SNR
  `138.15 dB`.
- Recomputing HAR/STFT from even the dumped source is not solved. The best
  dumped-source recomputed HAR SNR is only `8.23 dB`, with `2*pi` raw phase
  branch errors.
- The generator consumes raw phase branch values, not phase modulo `2*pi`.
- Exact Swift Float Nyquist real/imag dot products plus `atan2` repairs the
  padded Nyquist contract to about `48-50 dB` waveform SNR, but preserving the
  full strict HAR/STFT representation is too expensive and not a net speed win.
- The body-only counterfactual is compelling: on Irvine M1 `3s`, body-only would
  save `62.4 ms` if `x_source_*` tensors were free, but the strict full
  source/noise split loses `11.5 ms` once source/noise production is included.
- Existing exact HAR-post splits, source/noise splits, broad DecoderPre+Generator
  merges, output backing, palettization, RangeDim/flexible shapes, native
  InstanceNorm/broadcast surface matching, and style specialization have been
  tried and rejected or kept only as small strict candidates.

Current hard constraint:

We need a strict or quality-defensible way to make `x_source_*` cheap inside a
runtime-positive boundary. Another extra Core ML prediction boundary is likely a
trap unless it removes more cost than it adds.

## Primary Research Goal

Find concrete implementation paths that make Kokoro's strict source/HAR contract
cheaper on Apple Silicon without adding losing runtime boundaries. The best
answer should explain how to fold, reparameterize, approximate with strict
recovery, or distill the source/HAR/STFT/noise-conv contract so Core ML can
produce generator body inputs faster than the current full HAR tensor path.

## Questions To Answer

- Can the first `noise_convs` / `noise_res` layers be algebraically folded with
  STFT/HAR computation so the model never materializes the full `[mag, phase]`
  HAR tensor?
- Is there a representation that preserves strict behavior but is cheaper than
  raw magnitude + raw phase, such as real/imag, sin/cos phase, branch-coded
  phase, Nyquist side-channel, or a learned compact source code?
- Can a tiny learned adapter map a cheap source representation to `x_source_*`
  or to early generator activations with strict waveform parity or controlled
  listening-equivalent drift?
- What training/distillation objective would recover the current generator's
  behavior: `x_source_*` MSE, early activation MSE, waveform spectral loss,
  multi-resolution STFT loss, adversarial loss, or hybrid?
- Can the fixed Swift source equation be moved into an MLProgram cheaply, or
  should it stay in Swift/Accelerate/Metal while only a folded tail enters Core
  ML?
- How should exact Swift Nyquist `atan2` and DC branch repair be represented
  without triggering Core ML CPU/GPU fallback or losing fusion?
- Are there Core ML graph patterns for constant DFT/STFT banks, grouped
  `Conv1d`, complex multiply, or fused real/imag projections that schedule
  better than explicit STFT + phase ops?
- Can the first noise convolution over HAR be precomputed into a direct
  convolution/filterbank over `har_source`, avoiding phase materialization?
- When is it better to hand-write a Metal kernel for source/HAR-to-`x_source`
  versus exporting it as Core ML?
- How should fixed buckets (`3s`, `7s`, `10s`, `15s`, `30s`) be exploited for
  precomputed constants, static shapes, and memory layout?
- What layouts are best for this source-contract region on Apple hardware:
  `[B,C,T]`, `[B,C,1,T]`, `[B,T,C]`, packed complex channels, or split real/imag
  buffers?
- How do laishere-like implementations likely avoid this cost, and which parts
  can be replicated while preserving first-party strict quality?
- What profiling evidence proves a folded source contract is genuinely faster:
  Core ML `MLComputePlan`, Instruments Core ML, Metal System Trace, powermetrics,
  stage timings, cache warmup isolation, memory bandwidth counters?
- What objective quality gates should be used before human listening when strict
  waveform parity is impossible and Whisper/ASR is explicitly skipped?
- What are the most promising implementation experiments, ordered by expected
  speed impact, quality risk, and engineering cost?

## Output Format

- Start with an executive summary of the most plausible ways to make
  `x_source_*` cheap.
- Provide a practical developer field guide with implementation-level recipes.
- Include "do this / avoid this" tables.
- Include a section on algebraic folding of STFT/HAR into first noise
  convolutions.
- Include a section on representation alternatives: raw phase, real/imag,
  sin/cos, branch-coded phase, compact learned code, and direct `x_source`
  distillation.
- Include a section on Core ML vs Swift/Accelerate/Metal placement for this
  exact source-contract region.
- Include a section on bucket-specific static-shape optimization.
- Include a section on distillation and parity/listening gates with no ASR.
- Include concrete profiling commands/tools and evidence requirements.
- Include references to Apple/Core ML docs, MLX/docs/source when useful,
  ISTFTNet/Kokoro/vocoder papers or repos, STFT/filterbank implementation
  examples, and relevant Core ML conversion examples.
- Clearly mark speculative ideas separately from evidence-backed
  recommendations.

## Launch Status

Attempted to launch the external `create_guide_v1` workflow on June 6, 2026:

```bash
pnpm run research:create-guide --topic "Kokoro strict source/HAR contract folding for Apple Silicon Core ML vocoders" \
  --context-file /Users/mm/Documents/GitHub/kokoro-coreml/README/Notes/Kokoro-strict-source-HAR-contract-folding-deep-research-prompt.md \
  --target-repo kokoro-coreml \
  --target-guide-path README/Guides/apple-silicon/Kokoro-strict-source-HAR-contract-folding-guide.md \
  --agent-mode max \
  --no-wait
```

The local workflow service rejected the request with `403 forbidden`:
`Client token does not authorize this workflow request.` The local
`llm-workflows/.env` token metadata does not currently expose a `guides` client
authorized for `create_guide_v1`. Resume by fixing the workflow token/client
authorization, then rerun the command above. Do not create the target guide by
hand; `README/Guides/` should only receive the external raw report after it
lands.
