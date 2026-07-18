# Monolithic Core ML Control Experiment (Kokoro-82M)

Provenance: local experiment, 2026-07-14. Produced as the missing scientific
control for `Scratchpad/surgical-inference.md` (the "Surgical Inference" paper).
The paper claims decomposition is the active ingredient; a reviewer will demand
the null hypothesis — convert the *whole* model to a single Core ML package and
run it under one policy. This note is that control.

**Result in one line:** the monolith cannot exist as a valid artifact. The full
token-IDs-to-waveform graph fails Core ML conversion outright on the
data-dependent alignment op (branch a), and even the strongest downstream
partial monolith (aligned features to waveform, one package) converts but is
numerically invalid at correlation ~0.27 because of the hn-nsf SineGen
cumulative-phase path (branch b). Both failures are exactly the boundaries the
shipped pipeline cuts at.

## Machine + environment

- Machine: Apple M2 Ultra, 64 GB unified memory (`hw.memsize` = 68719476736).
- OS: macOS 26.5.1, build 25F80.
- Python venv (`.venv`): Python 3.11, coremltools 8.3.0, torch **2.6.0**,
  numpy 1.26.4.
  - Deviation from paper §4.2: the paper states torch pinned at 2.5.0 for export;
    the on-disk `.venv` has torch 2.6.0. coremltools prints
    `Torch version 2.6.0 has not been tested with coremltools. ... Torch 2.5.0 is
    the most recent version that has been tested.` The failures below are
    structural (data-dependent op; stochastic phase), not version-sensitive, so
    this deviation does not change the branch outcomes. Flagged for honesty.
- Weights: `hexgrad/Kokoro-82M` from the HF cache (`checkpoints/*.pth` symlinks
  on this box are dangling; `KModel(repo_id=...)` auto-loads from cache).
- Frozen input: the 3 s bakeoff bucket, `outputs/swift_bench_inputs/3s.json`
  ("The quick brown fox jumps over the dog."), 44 tokens (BOS + 42 + EOS),
  canonical duration 2.8 s, voice `af_heart`, speed 1.0. Same frozen input the
  bakeoff uses.

## What "the monolith" means here

The true monolithic forward is `KModelForONNX.forward_with_tokens`
(`kokoro/model.py:206`): `input_ids, ref_s, speed -> (audio, pred_dur)`. This is
the entire reference graph: BERT/ALBERT -> prosody LSTM + duration proj ->
duration-dependent expansion -> F0Ntrain -> text encoder -> decoder (AdaIN
encode/decode + hn-nsf SineGen source + iSTFT generator).

Note: the repo's existing `export_synth --mode full` is **not** a true monolith.
`SynthesizerModel.forward` (`export_synth/wrappers.py:377`) takes the
pre-built alignment matrix `pred_aln_trg` as an *input* and even bypasses the
LSTM and F0/N stacks. No prior true whole-model export exists in the repo
(`grep -rn` for `KModelForONNX` / `forward_with_tokens` / `repeat_interleave`
across `README/Notes`, `scripts`, `export_synth` returns nothing). This control
is the first attempt at the actual monolith.

## Branch (a): full monolith FAILS — two independent proofs

Script: `scratchpad/monolith_export.py` (run under `.venv`). Trace the full
`KModelForONNX` at the 3 s input (speed baked to 1.0 so the graph has exactly
two inputs: `input_ids (1,44)`, `ref_s (1,256)`), then convert to mlprogram
FP16, `minimum_deployment_target=iOS16`, `compute_units=ALL`.

### (a.1) Core ML conversion hard-fails at the data-dependent expansion

`torch.jit.trace` does not raise (it emits ~110 `TracerWarning`s about
"Converting a tensor to a Python integer/boolean ... treated as a constant ...
the trace might not generalize to other inputs"). Conversion then fails:

```
ERROR - converting 'repeat_interleave' op (located at: 'indices'):
...
  File ".../mil/frontend/torch/ops.py", line 6758, in repeat_interleave_dim0
    x_tiled = mb.tile(x=x, reps=reps)
  File ".../mil/mil/builder.py", line 122, in _add_const
    raise ValueError(err_msg)
ValueError: Cannot add const [None]
```

Root cause: `indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]),
pred_dur)` (`kokoro/model.py:262`). `pred_dur` is a per-token tensor of
*predicted* frame counts, so the `reps` argument to `tile` is not a
compile-time constant. Core ML has no operation for value-dependent output
extent; the converter tries to materialize a const and gets `None`. This is the
Data-Dependent Logic motif the paper routes to Swift, hitting the compiler wall
head-on. The converter also warns that the prosody LSTM lowers to
`_pack_padded_sequence` / `_pad_packed_sequence` ("not efficient due to the
current limitation of Core ML ... use a fixed batch size model is recommended").

### (a.2) The trace is baked, not a valid model

Even setting conversion aside, the TorchScript itself is not a model — it is a
recording of one utterance's geometry. Proof: run the traced module on input B =
the same 44-token input with the interior phonemes reversed (identical tensor
shape, different BERT context -> different predicted durations). Eager PyTorch
gives A = 112 frames / 67200 samples, B = 109 frames / 65400 samples. The
TorchScript reproduces A exactly (corr 1.0000) but **hard-crashes on B**:

```
RuntimeError: The following operation failed in the TorchScript interpreter.
  ... kokoro/istftnet.py(339): forward   # SineGen
RuntimeError: The size of tensor a (65400) must match the size of tensor b (67200)
              at non-singleton dimension 1
```

A's 67200-sample harmonic geometry is frozen into the graph; B's 65400-sample F0
curve cannot broadcast against it. A monolith traced this way would be correct
only for utterances whose durations happen to match the dummy — i.e. not a
model. This is the subtlety the control was designed to catch: "trace-baked, not
a valid model."

## Branch (b): downstream partial monolith CONVERTS but is NUMERICALLY INVALID

Per mission step 4, the strongest defensible partial monolith is everything
*downstream of alignment* fused into one package: the full `Decoder`
(`kokoro/istftnet.py`) = F0/N convs + AdaIN encode/decode + hn-nsf SineGen
harmonic source + iSTFT generator, taking real aligned features
`asr (1,512,112)`, `F0 (1,224)`, `N (1,224)`, `s (1,128)` captured from the 3 s
reference forward. Script: `scratchpad/partial_monolith.py`.

This one is well-behaved structurally: it traces, and `ct.convert` to mlprogram
succeeds (both FP16 and FP32). But the output does not match PyTorch:

| Export | policy | corr(CoreML, PyTorch decoder) |
|--------|--------|-------------------------------|
| FP16   | CPU+GPU  | **0.2742** |
| FP32   | CPU_ONLY | **0.2546** |

Correlation ~0.25 is a categorical failure (the paper's gate is > 0.99; its
documented hn-nsf-in-Core-ML figure is ~0.00). Attribution — the collapse is
**not** FP16 and **not** RNG:

- FP32 CPU-only is equally broken (0.25), so it is not a float16 precision
  problem.
- Eager PyTorch decoder across different RNG seeds stays at corr 0.996-0.997
  (seed0 vs seed0 = 1.0000, vs seed1 = 0.9969, vs seed7 = 0.9957). So the
  in-graph `torch.rand` initial phase and `torch.randn_like` noise inside
  SineGen do not explain a drop to 0.25 either.

What remains is a genuine structural divergence in how Core ML executes the
hn-nsf SineGen cumulative-phase reconstruction (`_f02sine`,
`kokoro/istftnet.py:261`: downsample -> `torch.cumsum` -> `*2pi` -> upsample ->
`sin`). Small per-step differences in the cumsum/interpolate chain accumulate
over the full audio-rate phase trajectory and globally decorrelate the waveform
while leaving spectral magnitude plausible. This is precisely the
"cumulative-phase sensitivity" that motivated extracting hn-nsf into
deterministic native Swift (paper §3.3, §4.3), and it independently justifies
that cut — the partial monolith reproduces the failure the decomposition was
built to avoid.

## Exact commands

```bash
cd /Users/mm/Documents/GitHub/kokoro-coreml
# Branch (a): full token-IDs -> waveform monolith
.venv/bin/python <scratchpad>/monolith_export.py
# Branch (b): downstream aligned-features -> waveform decoder monolith
.venv/bin/python <scratchpad>/partial_monolith.py
# Attribution checks (FP32 recovery, cross-seed stochasticity) inline in this note
```

Scripts live in the session scratchpad (temporary); the load-bearing evidence is
the verbatim errors and correlations transcribed above. Regenerate by wrapping
`KModelForONNX` (full) or `Decoder` (partial) and re-running `ct.convert` at the
3 s frozen input.

## For the paper (control result, stated neutrally)

We tested the null hypothesis a reviewer will raise against decomposition:
compile Kokoro-82M as a single Core ML package and run it under one compute
policy. It cannot be done as a valid artifact, for two distinct and independently
sufficient reasons, each landing exactly on a boundary the decomposed pipeline
already cuts. First, the complete token-IDs-to-waveform graph fails Core ML
conversion outright: the duration-dependent expansion
(`torch.repeat_interleave` with per-token predicted durations,
`kokoro/model.py:262`) has no static-shape lowering, and coremltools 8.3.0
aborts with `ValueError: Cannot add const [None]` while converting the
`repeat_interleave` op. `torch.jit.trace` appears to "succeed" only by baking one
utterance's alignment into constants — the resulting TorchScript reproduces the
traced input at correlation 1.000 but hard-crashes on a same-shape input with
different predicted durations (`size of tensor a (65400) must match b (67200)`),
confirming it is a recording of a single utterance's geometry rather than a
model. Second, even the strongest partial monolith — every stage downstream of
alignment fused into one package (aligned features to waveform, including the
hn-nsf harmonic source and iSTFT) — converts cleanly but is numerically invalid,
reaching only correlation 0.27 (FP16) / 0.25 (FP32) against the PyTorch
reference. This is not a precision artifact (FP32 is equally wrong) and not RNG
mismatch (eager PyTorch across seeds holds at 0.996); it is a structural
divergence in Core ML's execution of the SineGen cumulative-phase path, the same
failure that forced hn-nsf into deterministic native Swift.

The control therefore strengthens rather than threatens the paper's central
claim. The monolith is blocked at precisely the two seams the methodology
identifies — the data-dependent alignment (routed to Swift) and the hn-nsf
harmonic source (reimplemented natively) — so the decomposition is not one design
among several but the enabling move: it is what makes any correct, compilable
on-device Core ML artifact for this model possible at all. There is no
single-package baseline to benchmark against, because a correct single package
does not exist.
