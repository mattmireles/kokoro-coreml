# LFM2.5 Selective Six-Piece Prefill Experiment Plan

**Date:** 2026-07-20  
**Status:** Ready for execution

> This is a new, independently gated follow-up to
> [plan 010](./010-lfm2-surgical-prefill-plan.md). It does not reopen, amend,
> or weaken plan 010's terminal G1a result. Plan 010 correctly killed the
> 13-segment experiment after direct Swift measurement showed that 12 added
> boundaries consumed 37.43% of segmented prefill. This plan tests one frozen
> lower-boundary partition and accepts another negative result as success.

## Executive Summary

Test exactly one selective partition of the real LFM2.5-350M layer stack:

```text
[C0-1] ANE | [A2] GPU | [C3-4] ANE | [A5] GPU | [C6-7] ANE | [A8-C9-A10-C11-A12-C13-A14-C15] GPU
```

The partition isolates only the three two-convolution runs that have the best
chance of amortizing an ANE handoff. The finely alternating tail remains one
GPU-permitted package. This reduces the layer stack from 13 predictions and 12
boundaries to 6 predictions and 5 boundaries while moving 6 of the 10
convolution blocks to ANE-permitted packages.

The experiment is cheap first. Before exporting anything new, benchmark the
existing three convolution-pair packages under GPU and ANE permissions and
test whether a monolithic `.all` model already performs heterogeneous internal
scheduling without public prediction boundaries. Before any multi-bucket work,
only one new Core ML package—the contiguous layers-8-15 tail—is required for
the exact fixed-512 six-piece test.

The Mac decides whether the lower-boundary substrate is economically viable.
The iPad decides whether the device-class placement benefit is large and stable
enough to justify touching the daily-driver iPhone. The iPhone remains last and
is never required for a negative result.

## Problem Statement

- **Observed failure:** The 13-segment plan produced exact paired logits but
  took 57.591 ms versus 36.032 ms for the monolith at bucket 512. Its 21.558 ms
  decomposition tax exceeded the frozen 30% gate.
- **Unresolved question:** Was that result a rejection of heterogeneous
  placement itself, or only a rejection of a partition with 12 public model
  boundaries?
- **Narrow hypothesis:** Three long-enough convolution islands can earn back
  five boundaries while the alternating tail remains on the GPU.
- **Why this split:** It follows the checkpoint topology and prior dispatch
  evidence. It is not an arbitrary layer-count sweep and does not select a
  partition after observing its end-to-end result.

## Relationship to Plan 010

The following facts are frozen and inherited as evidence, not rerun by default:

- Checkpoint:
  `LiquidAI/LFM2.5-350M@b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`.
- Layer order: `C C A C C A C C A C A C A C A C`.
- Buckets: `{128, 256, 512, 1024, 2048}`.
- Precision: fp16 weights and tensors, with the existing GQA mixed-precision
  selector.
- Real prompt, tokenizer, explicit convolution-state, RoPE, and causal-mask
  contracts.
- Stage 0 admission: representative convolution operations admitted to ANE on
  both the M2 Ultra Mac and physical M2 iPad; GQA preferred GPU.
- Authoritative G1a protocol: Release Swift, three warmups, N=20 per candidate,
  alternating AB/BA order, direct fp16 `MLMultiArray` handoffs, package hashes,
  and same-host compute-plan provenance.
- Terminal 13-segment result: 36.032 ms monolithic, 57.591 ms segmented,
  21.558 ms added, 37.43% of segmented total, paired logits `max_abs = 0.0`.

Plan 010 remains a terminal negative result even if this plan succeeds. A
positive result here would support only the narrower claim that coarse,
selective partitioning can work where per-class maximal-run partitioning does
not.

## Frozen Candidate Definitions

Candidate names and compute-unit permissions are frozen before Phase 0 timing.
Embedding and LM head use `.cpuAndGPU` in every timed candidate so that only the
layer-stack scheduling policy changes.

| Candidate | Layer-stack behavior | Purpose |
| --- | --- | --- |
| `M_GPU` | Existing fixed-512 monolith loaded with `.cpuAndGPU`. | No-boundary homogeneous control. |
| `M_ALL` | The same monolith and package hash loaded with `.all`. | Zero-public-boundary scheduler control. |
| `S6_GPU` | Six pieces, all loaded with `.cpuAndGPU`. | Exact five-boundary decomposition-tax control. |
| **`S6_SELECTIVE`** | Conv pairs 0-1, 3-4, and 6-7 use `.cpuAndNeuralEngine`; GQA 2, GQA 5, and tail 8-15 use `.cpuAndGPU`. | Frozen hypothesis. |

No CPU-only mode, all-ANE mode, inversion matrix, or alternative partition is
added during this plan. If the selected split fails, a new plan and new
pre-registration are required before testing another split.

## Goals and Non-Goals

### Goals

- [ ] Determine whether `M_ALL` already obtains useful heterogeneous internal
  scheduling without public model boundaries.
- [ ] Measure the isolated ANE-versus-GPU latency delta of the three existing
  two-convolution packages on real layer-entry tensors.
- [ ] Measure the exact decomposition tax of six packages and five layer-stack
  boundaries at bucket 512.
- [ ] Determine whether `S6_SELECTIVE` beats the best no-boundary monolithic
  policy on physical M2 iPad and, only if justified, A17 Pro iPhone.
- [ ] Publish a reproducible positive or negative report with same-device
  dispatch, package hashes, paired timing, and numerical evidence.

### Non-Goals

- No rescue, reinterpretation, or rerun of plan 010's 13-segment matrix.
- No search over partition points, layer subsets, or compute-unit combinations.
- No model surgery, retraining, quantization, pruning, or stateful Core ML.
- No decode benchmark, KV-cache optimization, energy study, sustained thermal
  study, external-baseline bakeoff, or paper-wide claim expansion.
- No multifunction packaging, linked models, Metal runtime, private ANE API, or
  custom operator. Those change the boundary mechanism and require a separate
  plan.
- No iPhone work until every Mac and iPad continuation gate passes.

## Scope and Constraints

- The plan and final repo-memory pointer live in `kokoro-coreml`.
- Implementation, generated model artifacts, Swift runtime, result JSON, and
  public report live in `/Users/mm/Documents/GitHub/lfm2-surgical-coreml`.
- Generated packages and result payloads remain ignored and uncommitted under
  `outputs/lfm2_surgical/selective/`.
- Preserve the public repo's tracked `docs/huggingface-model-card.md`; it is
  outside this latency experiment unless a later artifact-publishing request
  explicitly brings it into scope.
- Preserve the frozen Stage 1 executable and report. Add a separate selective
  executable instead of changing the meaning of
  `lfm2-surgical-benchmark` or `docs/stage1-report.md`.
- Physical-device results are valid only for the recorded device and OS build.
- The iPhone is Matt's daily driver. Execution pauses for explicit availability
  before Phase 4 even if the preceding gates pass.

## Ground-Truth Contracts

- **One variable at a time.** `M_GPU` and `M_ALL` share one package hash.
  `S6_GPU` and `S6_SELECTIVE` share all six package hashes and differ only in
  load-time compute-unit configuration.
- **No dispatch table, no number.** Every reported timing row has compute-plan
  evidence captured on the same host, with package-tree SHA-256 and requested
  compute units.
- **Real layer-entry tensors.** Pair microbenchmarks use activations produced by
  the frozen real prompt and the preceding real checkpoint layers. Synthetic
  zeros are not valid timing inputs except for the explicit prefill states that
  are zero by contract.
- **Direct Swift boundaries govern.** Python/NumPy timings may diagnose but
  never decide a gate.
- **Cold and warm remain separate.** Package loading and first-use compilation
  occur before the three warmups and never enter steady-state samples.
- **Counterbalance order.** Candidate order alternates by pair. Device runs log
  thermal state and reject samples collected after `.serious` is observed.
- **Correctness before speed.** All structural and hardware-placement
  comparisons must retain the same top token and stay within the frozen `1e-2`
  maximum-absolute logit tolerance. Bit-exact output is recorded when achieved
  but is not required after a newly fused tail changes legal fp16 fusion.
- **The best monolith governs.** A split never claims a win by comparing only
  with `M_GPU` when `M_ALL` is faster.
- **A kill is completion.** Stop at the first failed gate and publish the
  negative result. Do not continue to collect an attractive later datapoint.

## Already Shipped (Do Not Re-Solve)

The public `lfm2-surgical-coreml` repo already contains:

- `scripts/lfm2_surgical/segments.py` with the real checkpoint equations and
  maximal-run segment definitions.
- `scripts/lfm2_surgical/export_segments.py` with fp16 conversion, fixed-512
  monolith export, and synchronized enumerated-shape handling.
- Existing reusable packages for `segment_00_conv_0_1`, `segment_01_gqa_2`,
  `segment_02_conv_3_4`, `segment_03_gqa_5`, and
  `segment_04_conv_6_7`.
- `Sources/LFM2SurgicalRuntime/BenchmarkRuntime.swift` with direct fp16 model
  handoffs, stable tensor construction, artifact hashing, summary statistics,
  and the frozen Stage 1 benchmark.
- `scripts/dump_device_compute_plan.py` and same-host package-hash provenance.
- `scripts/lfm2_surgical/prepare_stage1_input.py` and the real fixed prompt.
- Python tests, Swift Package structure, Stage 0 report, and Stage 1 negative
  report.

Reuse those contracts. Do not create a second checkpoint loader, tensor format,
statistics implementation, or device-dispatch schema.

## Fresh Baseline at Plan Creation

- `kokoro-coreml` is clean before this plan file is created.
- `lfm2-surgical-coreml` is clean at commit `1fe6047`; that revision added the
  tracked `docs/huggingface-model-card.md`, which is outside this plan's scope.
- No layers-8-15 composite package, enumerated monolith, selective Swift
  executable, or selective result report exists.
- The Stage 1 artifacts proved all 797 monolithic and 807 segmented costed
  operations preferred GPU under `.cpuAndGPU` on the M2 Ultra Mac.
- The Stage 1 pair packages are already converted over all five frozen buckets,
  but their runtime latency under `.cpuAndNeuralEngine` has not been measured.
- No ANE runtime timing was collected in plan 010; admission evidence is not a
  latency result.

## Selected Partition

```text
real checkpoint
    C0 C1 A2 C3 C4 A5 C6 C7 A8 C9 A10 C11 A12 C13 A14 C15
    |____| |  |____| |  |____| |____________________________|
     pair   A  pair  A   pair              tail
      ANE  GPU  ANE GPU   ANE               GPU

layer-stack predictions: 6
added layer-stack boundaries versus monolith: 5
conv blocks isolated for ANE permission: 6 of 10
alternating tail blocks kept together: 8
```

The tail is one contiguous package, not a multifunction wrapper around eight
existing packages. It executes layers 8-15 internally in checkpoint order and
emits the same final hidden state plus convolution states 9/11/13/15 and K/V
pairs 8/10/12/14 required by the existing full-prefill contract.

## Implementation Phases

> Execute one phase at a time. Run a phase audit before committing or moving
> on. A failed gate skips every later measurement phase and moves directly to
> Phase 5 closeout.

### Required Skills

- **Whole plan:** `execute-plan` for phase sequencing and checked-in evidence.
- **Every phase:** `phase-audit` before continuation.
- **Core ML judgment:** `ilya-sutskever`, `coreml`, `coreml-profile`, and
  `coreml-validate` for dispatch, placement, and parity decisions.
- **Implementation:** `debug` for root-causing only a failed required contract;
  `documentation` for new Python and Swift APIs.
- **Reports:** `markdown` and `write-notes`; `david-ogilvy` only for final
  public reader-facing prose.
- **Git:** `git-commit` once per completed phase. Use `git-push` or `deploy`
  only when explicitly authorized.
- Do not use `bakeoff`; it is the Kokoro audio harness and does not apply.
- Do not run `create-guide` unless execution discovers a genuinely missing
  external mechanism. The checked-in LFM2 export, split-graph, scheduling, and
  benchmark-hygiene guides already cover this experiment.

### Phase 0: Zero-Boundary and Conv-Pair Economics (Mac, no export)

**Goal:** Establish that ANE placement has enough isolated value to justify
one new package and prove that `.all` does not already solve the problem more
simply.

**Required skills:** `ilya-sutskever`, `coreml-profile`, `coreml-validate`,
`documentation`, `phase-audit`.

**Tasks:**

- [ ] Add a selective benchmark entry point without changing the frozen Stage
  1 executable. It may reuse extracted internal support, but the old Stage 1
  command must still produce exact paired logits and a G1a KILL.
- [ ] Load the existing fixed-512 monolith twice, once as `M_GPU` and once as
  `M_ALL`. Keep embedding and head `.cpuAndGPU` in both candidates.
- [ ] Capture package-hash-bound compute plans for both policies on the Mac.
- [ ] Run Release Swift N=20 after three warmups, alternating `M_GPU/M_ALL` and
  `M_ALL/M_GPU`; record every sample, median, IQR, paired difference, and final
  logits.
- [ ] Generate the real layer-entry hidden tensors for conv pairs 0-1, 3-4,
  and 6-7 outside timed regions by running the frozen prompt through the
  preceding GPU packages.
- [ ] Load each existing conv-pair package under `.cpuAndGPU` and
  `.cpuAndNeuralEngine`. Benchmark each pair with N=20, three warmups, and
  alternating order on its real layer-entry tensor.
- [ ] Record per-pair GPU median, ANE median, paired delta, 95% paired bootstrap
  confidence interval, output `max_abs`, top token where applicable, package
  hash, and same-host dispatch.
- [ ] Write the Phase 0 table and verdict to
  `docs/selective-split-report.md` in the public repo.

**Gates:**

- **G0a — simpler scheduler result:** If `M_ALL` assigns the three target conv
  pairs (or a broader set of conv operations) to ANE, keeps attention on GPU,
  and has no statistically supported regression versus `M_GPU`, stop. Its
  median must be no greater than `M_GPU`, and the paired 95% interval must
  include zero or favor `M_ALL`. The result is positive for heterogeneous
  scheduling but negative for public splitting: Core ML already found the
  zero-boundary solution.
- **G0b — target admission:** Each target conv-pair package must place at least
  80% of costed operations on ANE under `.cpuAndNeuralEngine` at bucket 512.
  Any pair below 80% kills the frozen three-pair mechanism.
- **G0c — aggregate pair value:** Let
  `pair_savings = Σ(median_GPU_i - median_ANE_i)` across the three pairs. Its
  paired-bootstrap 95% lower confidence bound must be greater than zero. If it
  is not, the same six packages cannot become faster merely by changing their
  load-time permissions; stop before exporting the tail.
- **G0d — pair numerics:** GPU and ANE outputs must retain the same top token
  and `max_abs <= 1e-2`. Unexplained failure kills the placement path.

**Verification:** No new Core ML package exists. The report contains the
monolithic scheduler table, all three pair tables, hash-bound dispatch, exact
commands, and one explicit `GO`, `KILL`, or `NO SPLIT NEEDED` verdict.

---

### Phase 1: One Tail Package and Exact Six-Piece Mac Gate

**Goal:** Replace the 12-boundary estimate with the exact five-boundary cost.

**Required skills:** `ilya-sutskever`, `coreml`, `coreml-profile`,
`coreml-validate`, `debug`, `documentation`, `phase-audit`.

**Tasks:**

- [ ] Add one narrow composite-layer spec for contiguous layers 8-15. Do not
  build a general partition-search or graph-planning abstraction.
- [ ] Export `tail_08_15_fixed_512.mlpackage` with the existing equations,
  fixed bucket 512, fp16 tensors, GQA mixed-precision selector, explicit state
  outputs, and the existing checkpoint identity.
- [ ] Extend the new selective Swift runtime with exactly the frozen six-piece
  descriptor list and per-descriptor compute-unit configuration.
- [ ] Keep common embedding, LM head, input tensors, state tensors, attention
  inputs, timing summaries, and artifact hashing shared across candidates.
- [ ] Capture same-Mac compute plans for the new tail and every reused package
  under every compute-unit policy that enters a reported row.
- [ ] Benchmark `M_GPU`, `M_ALL`, `S6_GPU`, and `S6_SELECTIVE` in a balanced
  Latin-square order, Release Swift, three warmups per candidate, N=20 per
  candidate. Package loading and first prediction remain outside timing.
- [ ] Compare paired final logits outside timed regions and write all raw
  samples plus summaries to JSON.
- [ ] Rerun the frozen Stage 1 executable once to prove that support-code reuse
  did not change its exact output or terminal G1a classification.

**Gates:**

- **G1a — exact five-boundary budget:**
  `(median(S6_GPU) - median(M_GPU)) / median(S6_GPU) <= 0.30` at bucket 512.
  This is the same registered definition used by plan 010. With the historical
  36.032 ms monolith it corresponds to `S6_GPU <= 51.474 ms`, but the gate uses
  the newly paired medians, not that historical absolute number.
- **G1b — structural equivalence:** `M_GPU` and `S6_GPU` paired final logits
  must retain the same top token and `max_abs <= 1e-2`. Bit-exact output is
  reported when achieved. This isolates segmentation from placement without
  treating legal fp16 fusion drift as a structural failure.
- **G1c — placement numerics:** `S6_SELECTIVE` must retain the same top token as
  `S6_GPU` and have `max_abs <= 1e-2`.
- **G1d — provenance:** Every timed package hash must match its same-Mac
  compute-plan record. A mismatch invalidates the run; fix provenance and
  rerun before deciding the gate.

Mac placement speed is reported but is not itself an iPad continuation gate.
The M2 Ultra GPU/ANE balance is not the mobile device hypothesis. G1a is the
Mac's decision; G2 is the physical iPad's decision.

**Verification:** `docs/selective-split-report.md` contains the four-candidate
fixed-512 table, exact G1a arithmetic, paired numerics, dispatch summaries,
artifact hashes, and an explicit verdict.

---

### Phase 2: Physical iPad Fixed-512 Decision Gate

**Goal:** Decide on the mobile-OS/M2 proxy whether this split is large enough
to justify a bucket matrix and eventual daily-driver phone use.

**Required skills:** `coreml-profile`, `coreml-validate`, `debug`,
`write-notes`, `phase-audit`.

**Tasks:**

- [ ] Build and install the Release selective benchmark on the physical M2
  iPad using the established `iDesk` device path and foreground policy.
- [ ] Record exact iPad model, OS build, Xcode, app revision, package hashes,
  battery state, thermal state, and screen/foreground conditions.
- [ ] Capture same-iPad, bucket-512 dispatch for every package and policy used
  by `M_GPU`, `M_ALL`, `S6_GPU`, and `S6_SELECTIVE`.
- [ ] Run all four candidates after compilation warmup, N=20 each, using the
  same balanced order and real prompt as Phase 1.
- [ ] Store raw samples, medians, IQRs, paired differences, 95% paired bootstrap
  intervals, logits, and thermal observations in the result JSON.

**Gates:**

- **G2a — device dispatch:** Under `S6_SELECTIVE`, each target conv pair must
  retain at least 80% ANE-preferred costed operations; both GQA singles and the
  tail must remain GPU-preferred. Any contradiction kills the mechanism on the
  proxy device.
- **G2b — device boundary budget:** The same G1a fraction for
  `S6_GPU` versus `M_GPU` must be `<= 0.30` on iPad.
- **G2c — worthwhile proxy win:** `S6_SELECTIVE` must beat the faster of
  `M_GPU` and `M_ALL` by at least 10% in median latency at bucket 512, and the
  paired-bootstrap 95% confidence interval for the improvement must exclude
  zero.
- **G2d — placement attribution:** `S6_SELECTIVE` must also beat `S6_GPU` with
  a paired-bootstrap 95% confidence interval excluding zero. Otherwise a win
  cannot be attributed to the ANE placement.
- **G2e — device numerics:** Same top token and `max_abs <= 1e-2` versus
  `M_GPU`.

**Verification:** The iPad result is self-contained and reproducible from the
public repo. A failed gate means no additional buckets and no iPhone.

---

### Phase 3: Enumerated Tail and Non-Phone Crossover Check

**Goal:** Require a repeatable short-prompt regime, not a single bucket-512
anomaly, before the phone is touched.

**Required skills:** `coreml`, `coreml-profile`, `coreml-validate`,
`documentation`, `write-notes`, `phase-audit`.

**Tasks:**

- [ ] Generalize only the validated tail export from fixed 512 to synchronized
  enumerated inputs over `{128, 256, 512, 1024, 2048}`.
- [ ] Export one enumerated monolithic layer-stack control over the same five
  synchronized shapes. `M_GPU` and `M_ALL` load this identical package under
  different policies; do not compare an enumerated split with the old
  fixed-512 monolith outside the Phase 2 fixed-shape gate.
- [ ] Verify every flexible input retains exactly five enumerated shapes; no
  `RangeDim` is permitted.
- [ ] Repeat package hashing, Mac load/predict smoke tests, and iPad
  bucket-specialized dispatch capture for gate buckets `{128, 256, 512}`.
- [ ] Run `M_GPU`, `M_ALL`, `S6_GPU`, and `S6_SELECTIVE` at all three gate
  buckets on iPad with three warmups, N=20, balanced order, paired confidence
  intervals, and correctness checks.
- [ ] Preserve the fixed-512 Phase 2 artifact and result. Do not overwrite it
  with the enumerated package result.

**Gate G3 — repeatable proxy regime:** At at least two of `{128, 256, 512}`,
`S6_SELECTIVE` must:

1. beat the faster monolith by at least 10% median latency;
2. have a paired-bootstrap 95% improvement interval excluding zero;
3. beat `S6_GPU` with its paired interval excluding zero;
4. satisfy the same dispatch and numerical contracts as G2.

If fewer than two buckets pass, stop. The iPhone is not a discovery device.

**Verification:** The report identifies exactly which buckets pass G3 and
freezes the on-phone command before Phase 4.

---

### Phase 4: A17 Pro iPhone Confirmation (Last)

**Goal:** Confirm or refute the selective split on the target phone only after
the iPad has already demonstrated a material regime.

**Required skills:** `coreml-profile`, `coreml-validate`, `debug`,
`write-notes`, `phase-audit`.

**Precondition:** User confirms the iPhone is available. Passing G3 does not by
itself authorize unattended use of the daily-driver phone.

**Tasks:**

- [ ] Install the Release build and first capture dispatch at bucket 512. Stop
  immediately if target conv pairs do not admit to ANE or the GPU pieces do not
  remain GPU-preferred.
- [ ] Record device/OS/build/package hashes, battery state, screen state,
  foreground state, and thermal state.
- [ ] Run the four frozen candidates over all five buckets, three warmups and
  N=20 per candidate, balanced order, with cold compilation excluded.
- [ ] Reject and rerun a bucket if `.serious` thermal state occurs during its
  samples; do not silently trim individual slow measurements.
- [ ] Capture same-phone compute-plan evidence for every reported package and
  bucket specialization.
- [ ] Compare paired logits outside timing and write raw plus summarized JSON.

**Outcome criteria:**

- **Strong positive:** At at least two buckets, `S6_SELECTIVE` beats the faster
  monolith by at least 15% median latency, beats `S6_GPU`, both paired 95%
  improvement intervals exclude zero, dispatch matches the frozen mechanism,
  and numerics pass.
- **Narrow positive:** At at least two buckets, the improvement over the faster
  monolith is statistically positive and at least 10%, but below the 15%
  strong threshold. Report it as a limited device-specific crossover, not a
  general surgical-inference win.
- **Negative:** Fewer than two buckets meet the narrow criterion, any required
  dispatch contract fails, or numerics fail.

No energy, decode, or thermal-soak claim is attached to any of these outcomes.
A later plan may test those only after a latency-positive result.

**Verification:** Every phone row is bound to same-phone dispatch, exact
package hashes, raw samples, thermal state, and correctness evidence.

---

### Phase 5: Closeout and Publication

**Goal:** Make the result durable without changing plan 010's conclusion.

**Required skills:** `markdown`, `write-notes`, `david-ogilvy`,
`documentation`, `phase-audit`.

**Tasks:**

- [ ] Complete `docs/selective-split-report.md` in the public repo with the
  terminal gate, commands, hashes, raw-result paths, tables, and limitations.
- [ ] Add `README/Notes/lfm2-selective-split-result.md` in `kokoro-coreml` as
  the repo-memory pointer and interpretation. Keep local experimental judgment
  in Notes, not Guides.
- [ ] Update this plan's status, checkboxes, measured values, and terminal
  verdict.
- [ ] Link the new report from the public README without rewriting or diluting
  `docs/stage1-report.md`.
- [ ] Update the paper only if its wording clearly distinguishes the failed
  13-segment partition from this independently registered six-piece result.
- [ ] Run public repo tests, Swift build/tests, the old Stage 1 replay, Markdown
  checks, memory health, and secret scan before calling the tree publishable.
- [ ] Do not publish generated checkpoint-derived packages unless separately
  authorized and the existing LFM license/attribution path is followed.

**Verification:** A stranger can reproduce the terminal gate from the public
repo, and the paper/repo memory state the same outcome without claiming that
plan 010 was reversed.

## Intended Files

Exact names may be adjusted during implementation only when the adjacent code
requires it; responsibilities may not expand.

### `lfm2-surgical-coreml`

- Modify `Package.swift` to add `lfm2-selective-benchmark`.
- Add `Sources/LFM2SelectiveBenchmark/main.swift`.
- Add `Sources/LFM2SurgicalRuntime/SelectiveRuntime.swift`.
- Optionally extract small shared, behavior-preserving Swift helpers from
  `BenchmarkRuntime.swift`; the frozen Stage 1 API and executable remain intact.
- Modify `scripts/lfm2_surgical/segments.py` with one layers-8-15 composite.
- Modify `scripts/lfm2_surgical/export_segments.py` with fixed/enumerated
  selective-tail scopes and one enumerated-monolith scope used only after G2.
- Extend `tests/test_lfm2_surgical_tools.py` for the composite spec, outputs,
  shape contract, and CLI scope.
- Add `docs/selective-split-report.md`.
- Generated only: `outputs/lfm2_surgical/selective/**`.

### `kokoro-coreml`

- This plan.
- Add `README/Notes/lfm2-selective-split-result.md` only during Phase 5.
- Update wiki routing only if required by memory-health checks.

## Executable Memory

These are the intended stable commands after their phases implement the named
entry points. The executor updates them if the final CLI differs.

```bash
cd /Users/mm/Documents/GitHub/lfm2-surgical-coreml

# Phase 0: no new Core ML export.
swift build -c release
.build/release/lfm2-selective-benchmark phase0 \
  --stage1-dir outputs/lfm2_surgical/stage1 \
  --input-json outputs/lfm2_surgical/stage1/g1a_input_512.json \
  --warmups 3 --runs 20 \
  --out outputs/lfm2_surgical/selective/phase0_mac.json

# Phase 1: the only new fixed-shape package.
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --scope selective-tail --fixed-bucket 512 \
  --out-dir outputs/lfm2_surgical/selective

.build/release/lfm2-selective-benchmark fixed-512 \
  --stage1-dir outputs/lfm2_surgical/stage1 \
  --selective-dir outputs/lfm2_surgical/selective \
  --input-json outputs/lfm2_surgical/stage1/g1a_input_512.json \
  --warmups 3 --runs 20 \
  --out outputs/lfm2_surgical/selective/mac_fixed_512.json

# Phase 3: only after the fixed-shape iPad gate passes.
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --scope selective-tail --all-buckets \
  --out-dir outputs/lfm2_surgical/selective/enumerated

.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --scope selective-monolith --all-buckets \
  --out-dir outputs/lfm2_surgical/selective/enumerated

# Regression gates.
.venv/bin/python -m pytest -q
swift build -c release
.build/release/lfm2-surgical-benchmark \
  --stage1-dir outputs/lfm2_surgical/stage1 \
  --input-json outputs/lfm2_surgical/stage1/g1a_input_512.json \
  --warmups 3 --runs 20 \
  --out outputs/lfm2_surgical/selective/stage1_replay.json
```

The device subcommands must emit the same JSON schema as the Mac command and
exit nonzero only after writing evidence when a frozen gate fails.

## Performance and Latency Budget

The historical Stage 1 numbers are planning priors, not substitutes for new
paired measurement:

| Quantity | Historical or derived value | Use |
| --- | ---: | --- |
| `M_GPU` bucket-512 median | 36.032 ms | Prior only. |
| 13-piece GPU median | 57.591 ms | Prior only. |
| 13-piece added cost | 21.558 ms | Prior only. |
| Average added cost per prior boundary | 1.797 ms | Heuristic only; never a gate. |
| Five-boundary linear estimate | 8.983 ms | Build expectation only; never subtracted from timing. |
| Six-piece G1a ceiling at historical monolith | 51.474 ms | Illustration of the frozen 30% formula. |

The experiment never assumes boundary cost is linear. Phase 1 directly
measures `S6_GPU`. The actual savings required to beat the monolith are
`median(S6_GPU) - min(median(M_GPU), median(M_ALL))` on that device and run.

## Test Strategy

### Unit and Export Tests

- Composite spec covers exactly layers 8-15 once and preserves checkpoint
  order.
- Tail flat I/O names and shapes match the existing full-prefill state/KV
  contract.
- Fixed export has no flexible dimensions; enumerated export has exactly five
  synchronized shapes and no range shape.
- Candidate descriptors contain exactly six packages in the frozen order and
  exactly three ANE-permitted entries for `S6_SELECTIVE`.
- Package hashing is deterministic over the package tree and recorded before
  timing.

### Runtime Tests

- Stage 1 replay retains exact logits and terminal G1a classification.
- All candidates use direct fp16 `MLMultiArray` hidden handoffs.
- Loading/compilation and warmups are outside measured regions.
- Balanced order gives N=20 samples per candidate with no missing row.
- Result JSON is written before a gate-triggered nonzero exit.

### Physical-Device Tests

- Same-device dispatch and package hashes accompany every timing table.
- Thermal, battery, screen, and foreground state are recorded.
- No iPhone test runs before G3 and explicit device availability.

## Success Criteria

### Hard Requirements

- [ ] Exactly the frozen six-piece partition; no exploratory alternatives.
- [ ] Real checkpoint and real prompt only.
- [ ] One shared artifact set per candidate pair; compute configuration is the
  only placement variable.
- [ ] Release Swift and direct fp16 boundaries govern every gate.
- [ ] Same-device dispatch, package hash, raw samples, and correctness for every
  reported result.
- [ ] Best monolithic policy is always the baseline.
- [ ] Stop at the first failed continuation gate.
- [ ] iPhone remains untouched until G3 passes and the user confirms
  availability.
- [ ] Positive and negative outcomes receive equally reproducible reports.

### Definition of Done

One of the following is sufficient:

1. `NO SPLIT NEEDED` at G0a, with a reproducible monolithic `.all` scheduler
   result;
2. a gate-triggered negative report at G0, G1, G2, G3, or G4;
3. a narrow or strong positive iPhone report under the frozen Phase 4 criteria.

In every case, this plan is updated to its terminal status, the public report
is complete, the kokoro Notes pointer exists, and no later phase is represented
as run when it was cancelled.

## Rollback and Kill-Switch Strategy

- All implementation is additive or behind a new executable and output root.
- The frozen Stage 1 executable/report remain the regression oracle.
- Generated packages can be deleted and regenerated; no source-of-truth model
  artifact is overwritten.
- A failed gate disables later device commands in the harness unless an
  explicit `--ignore-gate-for-debugging` flag is added. Such debug runs may
  never enter the report or decide an outcome.
- If shared-helper extraction changes Stage 1 output or classification, revert
  that extraction and duplicate the minimal support in `SelectiveRuntime.swift`.
- Do not revert or delete unrelated user files when cleaning the experiment.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Boundary cost is nonlinear. | Measure the exact six-piece all-GPU control; use the 1.797 ms estimate only for planning. |
| `.all` silently changes devices across OS builds. | Capture same-device compute plans and compare the same package hash. |
| ANE pair admission does not imply ANE pair speed. | Phase 0 times each pair under both policies before any new export. |
| Tail export accidentally changes equations or precision. | Reuse checkpoint equations and mixed-precision selector; require the frozen top-token and `1e-2` logit tolerance. |
| Mac result does not transfer to mobile. | Mac gates only structural tax; iPad makes the first placement-value decision. |
| One iPad bucket is a thermal/noise anomaly. | Require two of three short buckets before phone use. |
| Baseline cherry-picking. | Compare with the faster monolith independently at every bucket. |
| Daily-driver disruption. | Phone is last, gated, and requires explicit availability. |
| Scope expands into packaging research. | Multifunction/fused-runtime work is explicitly a separate plan. |

## Open Questions

### Resolved by This Plan

- **Which alternate split?** Exactly the three two-convolution islands plus one
  unsplit alternating tail.
- **How many new packages?** One tail package; reuse five existing packages.
- **Does iPhone testing happen automatically?** No. G3 and explicit availability
  are both required.
- **What baseline matters?** The faster of the same-package `M_GPU` and `M_ALL`
  monoliths at each bucket.
- **What proves the mechanism?** Same-segmentation `S6_SELECTIVE` versus
  `S6_GPU`, plus same-device dispatch.

### Unresolved Until Execution

- Does `.all` choose heterogeneous devices inside the fixed monolith on the
  current macOS/iPadOS/iOS builds?
- Are the three conv pairs faster under `.cpuAndNeuralEngine`, not merely
  admissible?
- Does the layers-8-15 mixed tail remain fully GPU-preferred on mobile OS?
- Is five-boundary cost small enough on both Mac and iPad?

Those are measured questions. None requires further design choice.

## References

### Internal

- [Plan 010 terminal experiment](./010-lfm2-surgical-prefill-plan.md)
- [Stage 1 negative-result note](../Notes/lfm2-stage1-negative-result.md)
- [LFM2 surgical prefill guide](../Guides/apple-silicon/LFM2-surgical-prefill-CoreML-guide.md)
- [Core ML compute-unit scheduling guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [Split-graph and multifunction packaging guide](../Guides/apple-silicon/CoreML-split-graphs-multifunction-packaging-guide.md)
- [Warmed-inference benchmark hygiene guide](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [iPhone device-lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md)
- [Plan workflow skills guide](../Skills/plan-workflow-skills-guide.md)

### Public Experiment Repo

- `/Users/mm/Documents/GitHub/lfm2-surgical-coreml/docs/stage1-report.md`
- `/Users/mm/Documents/GitHub/lfm2-surgical-coreml/Sources/LFM2SurgicalRuntime/BenchmarkRuntime.swift`
- `/Users/mm/Documents/GitHub/lfm2-surgical-coreml/scripts/lfm2_surgical/export_segments.py`
- `/Users/mm/Documents/GitHub/lfm2-surgical-coreml/scripts/lfm2_surgical/segments.py`

## Estimated Effort

- G0 kill or no-split-needed result: less than one day.
- G1 Mac six-piece result: one additional day.
- G2-G3 iPad continuation: one to two additional days.
- G4 phone confirmation and G5 closeout: one to two additional days, subject to
  device availability.

Total maximum: approximately five focused days. Most negative outcomes finish
earlier by design.
