# MoE SSD/DRAM Expert Prefetch Experiment Plan

**Date:** 2026-06-29
**Status:** Complete (Stage 1 killed: planned prefetch depths cannot hide reads)

> This plan is intentionally gate-heavy. The goal is not to build an impressive
> prefetcher. The goal is to cheaply decide whether learned expert prefetching
> across the SSD-to-DRAM boundary can beat demand paging on Apple Silicon UMA
> once cache truth, bandwidth, latency, and joules/token are counted.

## Executive Summary

Build a staged experiment for SSD-tier MoE expert prefetching. Stage 0 measures
the physical storage envelope with synthetic expert blocks; Stage 1 traces MoE
router behavior; Stage 2 replays traces in a deterministic simulator; Stage 3
touches a real runtime only if the cheap gates pass.

## Problem Statement

- **Symptom:** Prior MoE offload work suggests high expert-prediction accuracy,
  but most measured systems hide CPU-to-GPU PCIe transfer rather than true
  NVMe-to-DRAM reads on Apple Silicon UMA.
- **Root Cause:** The repo does not yet have a measured hardware envelope,
  router trace schema, simulator, or evidence contract for SSD-tier MoE
  experiments.
- **Impact:** Building a runtime prefetcher first risks spending weeks on a
  system that is bandwidth-bound even for an oracle predictor.

## Mode Definitions

These are experiment modes, not production runtime modes.

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| `stage0-envelope` | Measures expert-sized direct reads and computes hideability plus oracle bandwidth ceiling. | Kills bandwidth-bound ideas before ML work. |
| `stage1-trace` | Captures router traces and evaluates predictors on trace replay. | Separates predictor signal from storage physics. |
| `stage2-simulate` | Replays traces through a cache/prefetch/energy simulator. | Tests design space before real I/O integration. |
| `stage3-runtime` | Integrates a selected operating point into a real harness. | Runs only after Stage 0-2 prove value. |

## Goals and Non-Goals

### Goals

- [ ] Produce a Stage 0 envelope memo with measured cold sequential/random
      expert-block bandwidth, p50/p95 latency, hideability, oracle bandwidth
      ceiling, `fs_usage` proof, and a go/kill decision.
- [ ] Produce a Stage 1 trace dataset and predictor report across at least
      code, math, and prose prompts.
- [ ] Produce a Stage 2 deterministic simulator with demand-LRU, last-token,
      global-frequency, Markov/n-gram, Expert Activation Matrix, learned
      optional, and oracle policies.
- [ ] Advance to Stage 3 only if learned or EAM-style prediction beats demand
      and the best trivial policy while not regressing joules/token.

### Non-Goals

- Shipping a production MoE runtime in Kokoro.
- Modifying the existing Swift `KokoroPipeline.synthesize(...)` runtime.
- Training or fine-tuning the base MoE model.
- KV-cache prefetching.
- Multi-tenant or batched serving.
- Pre-gated/model-changing methods unless Stages 0-2 prove one-layer lead time
  is insufficient and the user explicitly expands scope.

## Scope and Constraints

- **Scope:** New experimental tooling under `scripts/moe_prefetch/`, generated
  artifacts under `outputs/moe_prefetch/`, and notes/results under
  `README/Notes/`.
- **Constraints:** Stage 0 must use incompressible synthetic expert blocks and
  must prove physical disk reads with `fs_usage -f diskio`.
- **Constraints:** `powermetrics` values are estimated and only valid for
  within-machine comparisons; do not compare joules/token across devices.
- **Guardrails:** Existing Kokoro export, bakeoff, and Swift runtime paths stay
  untouched unless Stage 3 is explicitly authorized.

## Ground Truth Contracts (Do Not Violate)

- **No cache fiction:** A result without `fs_usage -f diskio` evidence is not an
  SSD result.
- **Oracle first:** If the Stage 0 oracle bandwidth ceiling is below the target
  tokens/sec, stop. No predictor can beat the ceiling.
- **Trivial baselines matter:** Learned prediction must beat the best trivial
  prefetcher, not merely demand paging.
- **Energy is part of correctness:** A tokens/sec win that loses joules/token is
  a failed H2 result.
- **Cold and warm are separate:** Never average cold first-run storage/cache
  behavior with warmed steady-state decode.
- **Generated artifacts stay out of git:** Traces, dummy weight files, plots,
  `fs_usage` logs, and `powermetrics` captures live under `outputs/moe_prefetch/`.

## Already Shipped (Do Not Re-Solve)

- **MoE prior-art guide:** `README/Guides/moe-expert-offload-prefetch-prior-art-guide.md`.
- **Apple Silicon measurement guide:** `README/Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md`.
- **Guide triage note:** `README/Notes/moe-ssd-dram-prefetch-guide-triage-2026-06-29.md`.
- **Benchmark hygiene guide:** `README/Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md`.
- **External bakeoff schema patterns:** `scripts/external_bakeoff/schema.py`
  demonstrates result-schema helpers, provenance capture, and generated-output
  layout.

## Fresh Baseline (Current State)

- **Architecture:** No MoE runtime, no MoE trace harness, and no SSD-tier expert
  cache exist in this repo.
- **Metrics:** No measured Apple Silicon expert-block NVMe bandwidth, latency,
  hideability, or joules/token exists for this experiment.
- **Known gaps:** Model choice, quantization, target tokens/sec, expert byte
  size, and target device must be frozen before interpreting Stage 0.

## Build vs Buy

| Component | Decision | Rationale |
| --- | --- | --- |
| Direct-read microbenchmark | Build small C binary plus Python runner. | macOS cache behavior is the experiment; generic tools can hide too much. |
| MoE model execution | Buy via Hugging Face / PyTorch for Stage 1. | Router tracing is easier in Python; no Core ML conversion yet. |
| Predictor | Start with trivial policies plus EAM. | Simpler than a trained MLP and strong enough to falsify learning value. |
| Simulator | Build plain Python. | Deterministic replay is small and must match our Stage 0 parameters. |
| Runtime integration | Defer. | Premature integration is the expensive path. |

## Solution Overview

```text
scripts/moe_prefetch/
  direct_read_bench.c          -> expert-block read benchmark
  run_stage0_envelope.py       -> builds/runs benchmark, captures fs_usage paths
  model_inventory.py           -> records candidate MoE model shape and expert size
  trace_routers.py             -> captures Stage 1 router traces
  predictors.py                -> demand, trivial, EAM, optional learned predictor
  simulate.py                  -> Stage 2 cache/prefetch/energy replay
  summarize.py                 -> writes markdown reports and gate decisions
  schema.py                    -> shared JSON schema helpers

tests/test_moe_prefetch_tools.py

outputs/moe_prefetch/
  stage0/
  stage1/
  stage2/
  stage3/

README/Notes/moe-ssd-dram-prefetch-results.md
```

## Implementation Phases

> Do one phase at a time. Verify before proceeding. A kill decision is a valid
> successful outcome.

### Phase 0: Experiment Inventory and Threshold Freeze

**Goal:** Freeze the minimum facts needed to interpret Stage 0.

**Tasks:**

- [x] Add `scripts/moe_prefetch/schema.py` with dataclasses or typed dicts for
      machine info, model inventory, Stage 0 measurements, and gate decisions.
- [x] Add `scripts/moe_prefetch/model_inventory.py` to record candidate model,
      expert count, active experts per token, quantization assumption, estimated
      expert bytes, target tokens/sec, and target device.
- [x] Write `outputs/moe_prefetch/stage0/thresholds.json` from CLI arguments;
      default provisional thresholds are `speed_win_percent=25`,
      `trivial_margin_percent=10`, and `energy_regression_allowed=false`.
- [x] Add `README/Notes/moe-ssd-dram-prefetch-results.md` with a Stage 0 section
      and links back to this plan.

**Verification:**

```bash
python scripts/moe_prefetch/model_inventory.py \
  --model-id <candidate-model> \
  --quantization-bits 4 \
  --active-experts-per-token <n> \
  --target-tokens-per-second <n> \
  --output outputs/moe_prefetch/stage0/model_inventory.json

python -m pytest tests/test_moe_prefetch_tools.py
```

**Gate:** Do not run Stage 0 until target tokens/sec, expert-byte estimate, and
threshold JSON exist.

---

### Phase 1: Stage 0 Hardware Envelope

**Goal:** Decide whether the SSD boundary is physically worth studying.

**Tasks:**

- [x] Add `scripts/moe_prefetch/direct_read_bench.c` using `open`, `fcntl` with
      `F_NOCACHE`, page-aligned buffers, `pread`, configurable block size,
      offset pattern, queue depth, and JSON output.
- [x] Add `scripts/moe_prefetch/run_stage0_envelope.py` to compile the C
      benchmark, create incompressible synthetic expert files, run sequential
      and random cells, and record command provenance.
- [x] Capture `fs_usage -w -f diskio <pid>` output for each accepted benchmark
      cell and store paths or privileged-tool errors in the result JSON.
- [x] Add optional `powermetrics` capture for sustained read loops using
      `disk,cpu_power,gpu_power,ane_power,thermal` samplers.
- [x] Add `scripts/moe_prefetch/summarize.py stage0` to compute p50/p95
      latency, bandwidth, hideability, oracle bandwidth ceiling, and go/kill.
- [x] Append the Stage 0 memo to
      `README/Notes/moe-ssd-dram-prefetch-results.md`.

**Verification:**

```bash
python scripts/moe_prefetch/run_stage0_envelope.py \
  --thresholds outputs/moe_prefetch/stage0/thresholds.json \
  --output-dir outputs/moe_prefetch/stage0

# If non-interactive sudo cannot run fs_usage, rerun from a human terminal:
sudo -v
python scripts/moe_prefetch/run_stage0_envelope.py \
  --thresholds outputs/moe_prefetch/stage0/thresholds.json \
  --output-dir outputs/moe_prefetch/stage0 \
  --fs-usage-sudo-mode interactive \
  --capture-powermetrics

python scripts/moe_prefetch/summarize.py stage0 \
  --input outputs/moe_prefetch/stage0/results.json \
  --notes README/Notes/moe-ssd-dram-prefetch-results.md

python -m pytest tests/test_moe_prefetch_tools.py
```

**Gate:** Stage 0 now has valid `fs_usage` disk I/O proof and the oracle
bandwidth ceiling clears the provisional target, but hideability is flagged:
the measured one-layer compute budget is much smaller than the p95 expert-block
read latency. Stage 1 may start, but it must treat one-layer-ahead prediction as
insufficient and measure whether multi-layer-ahead prediction produces hideable
recall that beats the trivial baselines.

---

### Phase 2: Stage 1 Router Trace and Predictor Replay

**Goal:** Measure whether expert prediction provides usable, hideable lead time
in the selected model/workloads.

**Tasks:**

- [x] Add `scripts/moe_prefetch/trace_routers.py` using Hugging Face/PyTorch
      forward hooks to capture `(request_id, domain, token_index, layer_index,
      router_input_shape, actual_topk_expert_ids, layer_compute_ns)`.
- [x] Add prompt fixtures for code, math, and long-form prose under
      `scripts/moe_prefetch/prompt_suites/`.
- [x] Add `scripts/moe_prefetch/predictors.py` with demand-LRU, last-token,
      global-frequency, Markov/n-gram, EAM, and oracle policies.
- [x] Add `scripts/moe_prefetch/summarize.py stage1` to report recall@k,
      precision, hideable recall, wasted predictions, and per-domain variance.
- [x] Append Stage 1 results and a go/kill decision to
      `README/Notes/moe-ssd-dram-prefetch-results.md`.

**Verification:**

```bash
python scripts/moe_prefetch/trace_routers.py \
  --model-id <candidate-model> \
  --prompt-suite scripts/moe_prefetch/prompt_suites \
  --output outputs/moe_prefetch/stage1/router_trace.jsonl

python scripts/moe_prefetch/summarize.py stage1 \
  --trace outputs/moe_prefetch/stage1/router_trace.jsonl \
  --stage0 outputs/moe_prefetch/stage0/summary.json \
  --output outputs/moe_prefetch/stage1/predictability.json \
  --notes README/Notes/moe-ssd-dram-prefetch-results.md

python -m pytest tests/test_moe_prefetch_tools.py
```

**Gate:** Kill if EAM or learned prediction does not beat the best trivial
baseline by the frozen margin on hideable recall. High raw recall does not pass
if reads cannot arrive in time. Stage 1 killed the planned path because Stage 0
requires roughly 10 layers of lead time, while the planned prefetch depths
`{1,2,3}` produce zero hideable recall for every policy, including oracle.

---

### Phase 3: Stage 2 Discrete-Event Simulator

**Goal:** Test whether hideable prediction converts to tokens/sec and
joules/token before runtime integration.

**Tasks:**

- [ ] Add `scripts/moe_prefetch/simulate.py` with a fixed-capacity DRAM expert
      cache, SSD latency/bandwidth model from Stage 0, policy plug-ins from
      Stage 1, LRU eviction, prefetch depth, and cache-pollution accounting.
- [ ] Parameterize sweeps over DRAM budget `{25,50,75}%`, prefetch depth
      `{1,2,3}`, and policy `{demand_lru,last_token,frequency,markov,eam,oracle}`.
- [ ] Add an energy model with SSD bytes read, wasted bytes, prefetch active
      time, and optional `powermetrics`-derived coefficients from Stage 0.
- [ ] Add `scripts/moe_prefetch/summarize.py stage2` to write sweep JSON,
      compact markdown tables, and a recommended operating point.
- [ ] Append Stage 2 results and gate decision to
      `README/Notes/moe-ssd-dram-prefetch-results.md`.

**Verification:**

```bash
python scripts/moe_prefetch/simulate.py \
  --trace outputs/moe_prefetch/stage1/router_trace.jsonl \
  --stage0 outputs/moe_prefetch/stage0/results.json \
  --output outputs/moe_prefetch/stage2/sweep.json

python scripts/moe_prefetch/summarize.py stage2 \
  --input outputs/moe_prefetch/stage2/sweep.json \
  --notes README/Notes/moe-ssd-dram-prefetch-results.md

python -m pytest tests/test_moe_prefetch_tools.py
```

**Gate:** Kill if the best non-oracle predictor fails to beat demand-LRU by the
frozen speed threshold, fails to beat the best trivial prefetcher by the frozen
margin, or regresses joules/token.

---

### Phase 4: Stage 3 Runtime Harness Spike

**Goal:** Validate the Stage 2 operating point against real I/O in a contained
runtime harness.

**Tasks:**

- [ ] Choose one runtime harness: `llama.cpp` MoE offload path, MLX harness, or
      PyTorch harness. Record the choice and why in the results note.
- [ ] Store experts as individually addressable files or offsets with explicit
      resident-cache capacity matching the Stage 2 operating point.
- [ ] Implement a prefetch worker, demand-LRU baseline, best trivial baseline,
      selected predictor, and oracle replay where possible.
- [ ] Capture tokens/sec, first-token latency, SSD bytes/token, wasted
      bytes/token, cache hit rate, `fs_usage`, `powermetrics`, and thermal
      status for each policy.
- [ ] Compare real measurements with Stage 2 predictions and record
      simulator-vs-real deltas.

**Verification:**

```bash
python scripts/moe_prefetch/summarize.py stage3 \
  --input outputs/moe_prefetch/stage3/results.json \
  --stage2 outputs/moe_prefetch/stage2/sweep.json \
  --notes README/Notes/moe-ssd-dram-prefetch-results.md
```

**Gate:** Ship the finding only if real tokens/sec and joules/token match the
Stage 2 direction within the recorded tolerance. If real hardware loses where
simulation won, document the confound as the result.

## Executable Memory

- Regression test: `python -m pytest tests/test_moe_prefetch_tools.py`
- Stage 0 proof command:
  `python scripts/moe_prefetch/summarize.py stage0 --input outputs/moe_prefetch/stage0/results.json --notes README/Notes/moe-ssd-dram-prefetch-results.md`
- Stage 2 proof command:
  `python scripts/moe_prefetch/summarize.py stage2 --input outputs/moe_prefetch/stage2/sweep.json --notes README/Notes/moe-ssd-dram-prefetch-results.md`

## Success Criteria

### Hard Requirements (Must Pass)

- [x] Every accepted Stage 0 row has a matching `fs_usage -f diskio` artifact.
- [x] Stage 0 computes and records hideability plus oracle bandwidth ceiling.
- [x] Stage 1 reports hideable recall, not only raw recall.
- [ ] Stage 2 includes demand-LRU, best trivial, selected predictor, and oracle.
- [ ] Stage 2 reports tokens/sec and joules/token.
- [ ] Stage 3 is not started until Stage 0-2 gates pass.

### Definition of Done

- [x] One of these outcomes is recorded in
      `README/Notes/moe-ssd-dram-prefetch-results.md`: Stage 0 kill, Stage 1
      kill, Stage 2 kill, or Stage 3 real-hardware decision.
- [x] Generated artifacts are present under `outputs/moe_prefetch/` and remain
      uncommitted.
- [x] The final decision references exact commands, machine info, model info,
      thresholds, and artifact paths.

## Open Questions

### Resolved

- **Q:** Should we build a learned MLP predictor first?
- **A:** No. Start with trivial policies plus EAM. A learned MLP is only useful
  after the oracle and simple baselines prove there is system value to capture.

- **Q:** Should this plan modify the Kokoro Swift runtime?
- **A:** No. This is an experimental MoE systems lane. Touch production Kokoro
  only after Stage 3 is separately approved.

### Unresolved

- **Q:** Which MoE checkpoint is the first Stage 1 target?
- **Options:** OLMoE-1B-7B for smaller iteration, Qwen3-30B-A3B for more modern
  relevance, or Mixtral-8x7B for literature comparability. Current lean:
  choose the smallest model whose router hooks and expert shapes are easy to
  inspect locally.

- **Q:** What is the target tokens/sec?
- **Options:** freeze a product target, use the in-RAM model baseline, or use a
  literature comparator. Current lean: derive from fully resident in-RAM decode
  on the same machine, then set an explicit acceptable floor.

- **Q:** Is Stage 3 `llama.cpp`, MLX, or PyTorch?
- **Options:** `llama.cpp` for real offload relevance, MLX for Apple-native
  memory behavior, PyTorch for fastest instrumentation. Current lean: PyTorch
  until Stage 2 passes, then choose the runtime with the least custom code.

## References

### Internal

- [MoE expert offload and prefetch prior art](../Guides/moe-expert-offload-prefetch-prior-art-guide.md)
- [Apple Silicon NVMe and energy measurement](../Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
- [MoE SSD/DRAM guide triage note](../Notes/moe-ssd-dram-prefetch-guide-triage-2026-06-29.md)
- [Apple Silicon warmed-inference benchmark hygiene](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Runtime boundary wiki](../Wiki/runtime-boundary.md)
- [External bakeoff wiki](../Wiki/external-bakeoff.md)
- [Plan workflow skills guide](../Skills/plan-workflow-skills-guide.md)

### External

- [Mixtral-Offloading repository](https://github.com/dvmazur/mixtral-offloading)
- [MoE-Infinity / Activation-Aware Expert Offloading](https://arxiv.org/html/2401.14361v3)
- [HOBBIT: A Mixed Precision Expert Offloading System](https://arxiv.org/html/2411.01433v1)
- [Pre-gated MoE, ISCA 2024](https://dl.acm.org/doi/10.1109/ISCA59077.2024.00078)

## Degradation and Rollback

- **If Stage 0 fails:** Stop and write the kill memo. Recommended follow-up is
  compression, smaller experts, fewer active experts, or a different model.
- **If Stage 1 fails:** Stop; expert locality is not usable enough in the
  selected regime.
- **If Stage 2 fails:** Stop; predictor value does not convert to system value.
- **If Stage 3 fails:** Keep the simulator and real traces; document the
  unmodeled confound instead of masking it.
- **Rollback:** Delete `scripts/moe_prefetch/`, `tests/test_moe_prefetch_tools.py`,
  and generated notes for a clean experiment removal. No production runtime
  state is changed by this plan.

## Monitoring and Observability

**Metrics to Track:**

- `expert_read_latency_ns_p50` / `expert_read_latency_ns_p95`
- `expert_read_bandwidth_gbps`
- `hideability_ratio`
- `oracle_bandwidth_ceiling_tokens_per_second`
- `recall_at_k`
- `precision`
- `hideable_recall`
- `prefetch_hit_rate`
- `wasted_byte_fraction`
- `cache_pollution_rate`
- `effective_decode_tokens_per_second`
- `joules_per_token`

**Artifacts to Preserve:**

- `outputs/moe_prefetch/stage0/model_inventory.json`
- `outputs/moe_prefetch/stage0/thresholds.json`
- `outputs/moe_prefetch/stage0/results.json`
- `outputs/moe_prefetch/stage0/fs_usage_*.txt`
- `outputs/moe_prefetch/stage1/router_trace.jsonl`
- `outputs/moe_prefetch/stage1/predictability.json`
- `outputs/moe_prefetch/stage2/sweep.json`
- `outputs/moe_prefetch/stage3/results.json`
