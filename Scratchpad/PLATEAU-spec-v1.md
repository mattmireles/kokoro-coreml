# PLATEAU — Spec v1

**An elastic LLM architecture native to M-series Apple Silicon: product-key memory capacity, a weight-tied looped trunk, and block-parallel decoding, sized from a 16GB floor to Ultra-class machines.**

- **Document version:** v1 (2026-07-29). Revisions are saved as new files (`PLATEAU-spec-v2.md`, …), never overwriting this one.
- **Owner:** Matt Mireles. **Executors:** AI agents + Matt (hardware-in-the-loop steps are marked `[HW: device]`).
- **Status:** Experimental program spec. Architecture constants marked `PROVISIONAL` are set by the staged gates below, not by this document.
- **Name:** "Plateau" = the compute-bound, static-shape regime where surgical placement on Apple Silicon pays off — the opposite side of the "State Is the Cliff" thesis. Every design decision in this document exists to keep the model on the plateau.

---

## 0. How to use this document

1. Read §1–§4 completely before writing any code. They contain the thesis, the hardware model, and the decisions already made (with reasons). Do not relitigate closed decisions; do contest open ones (§11) and kill thresholds (§8) — **in writing, before code exists.**
2. Execute stages in order (§7). Each stage has pre-registered kill criteria. A kill is a valid, reportable outcome. Do not move goalposts after a run; do not set targets relative to a computed ceiling so that they trivially pass (see §9.5 for the failure this rule comes from).
3. Every reported number carries the evidence bundle in §9. Numbers without dispatch evidence and device metadata do not exist.
4. Stage boundaries are human review gates. Stage 3 additionally requires a cross-model adversarial review (§10.4) before spend.

---

## 1. Thesis and objective function

### 1.1 The objective

On Apple Silicon, all compute units (CPU, GPU, ANE) share one LPDDR bus and one unified memory pool. At batch=1 autoregressive decode, the workload is **bandwidth-bound**: model weights stream over DRAM per token regardless of which unit computes. Energy per token is dominated by bytes moved, not FLOPs executed (DRAM access energy ≫ SRAM access ≫ MAC). Therefore:

> **Objective:** maximize intelligence per DRAM byte moved (bandwidth) and per DRAM byte resident (capacity). Watts, tok/s, and battery follow from bytes.

"Maximally utilize ANE+GPU+CPU" is explicitly **not** the objective. On a shared bus, concurrent utilization without arithmetic intensity is contention. Units are used where their structure fits the op (§4.4), not for utilization's sake.

### 1.2 The three levers

There are exactly three ways to raise intelligence-per-byte. Each maps to one architectural mechanism:

| Lever | Mechanism | What it buys |
|---|---|---|
| Reuse each fetched byte across more **tokens** | Block-parallel decoding (semi-autoregressive block diffusion) | Weight bytes/token drop from `W` to `(S·K/B)·W`; every pass is fixed-shape and compute-bound → ANE-eligible |
| Reuse each fetched byte across more **time** | Weight-tied looped trunk (recurrent depth, fixed K per tier in v1) | Serial depth without proportional parameter/streaming cost; amortized further by B |
| **Don't fetch** the byte at all | Product-key memory (PKM) tables replacing dense FFN capacity | KB/token gathered instead of GB/token streamed; capacity bounded by RAM, not bandwidth |

The reductions compose multiplicatively. Illustrative floor-tier oracle (Stage 0 recomputes this properly): trunk ≈ 0.4GB, B=32, S=4 denoise steps, K=4 loops → trunk bytes/token ≈ S·K·0.4GB/32 = 0.2GB → ~2.5ms at ~80GB/s effective → O(100s) tok/s trunk-side ceiling on a 16GB Air-class machine, vs. ~6.7 tok/s for a dense 30B INT4 streamed autoregressively on the same bus. **Caveat:** that comparison is capacity-mismatched and quality-unknown; the honest comparison is quality-matched and is exactly what Stages 2–3 exist to measure.

### 1.3 Why heterogeneity becomes real here (and only here)

The composed architecture gives each unit work it is *structurally* suited for, concurrently, on different sub-computations of the same token, with zero-copy handoff via unified memory:

- **ANE:** trunk block-forward passes (fixed shape, INT8/FP16, compute-bound).
- **CPU (incl. AMX/SME):** PKM key scoring (tiny GEMMs), random row gathers (ANE cannot gather; GPU gathers badly), the loop/tier driver, sampling.
- **GPU (Metal/MLX):** optional global softmax-attention layers; fallback execution for any shell that fails ANE placement; training-time everything.

This is the only version of "use all three units" that is not bus-contention theater. Stage 1 benchmark M2 tests whether even this version survives contact with the bus (§7.2).

---

## 2. Decision log (closed decisions — context for the executor)

These were argued and closed in design sessions. The executor needs the *why*, not just the *what*.

1. **Objective reframed** from "intelligence per watt" to "intelligence per byte" (bandwidth + capacity). On wall-powered Macs, watts are abundant; bandwidth and RAM are the scarce resources. Per-watt remains the *secondary* metric and is load-bearing only on the fanless floor tier.
2. **Deployment target = the whole M-series lattice**, not one SKU. Dev/primary bench machine is Matt's M2 Ultra (64GB); floor is 16GB Air-class. The lattice spans ~8× in bandwidth and ~30×+ in RAM; a point design cannot be optimal across it → the architecture is **elastic** (§4.5).
3. **Ranking by tier** (from roofline analysis, §3.2): on Ultra-class, PKM capacity > block decoding > looping (bandwidth is abundant, RAM and GPU compute are scarce; ANE adds ~2× usable low-precision compute but only for static shapes). On floor-class, block decoding is what rescues the loop cost (base chips are bandwidth-poor with datacenter-like roofline corners). The elastic design serves both rankings from one artifact.
4. **Elasticity knobs and their binding times:** table count T and loop depth K are **runtime** knobs (no recompilation). Block size B and quantization tier are **ship-time** knobs (each (B, quant) pair is a separately compiled Core ML shell). "Fully elastic" honestly means: continuous in knowledge (T) and thinking (K), discrete in throughput (B) and precision. Do not fight the compiler for continuous B.
5. **Capacity elasticity = whole-table dropping in v1**, not row truncation. PKM's product-key structure is combinatorial (√N × √N key grid); truncating rows breaks key combinations. Row-level Matryoshka ordering *within* tables is a v2 refinement — but the **file layout** clusters rows by importance from day one (decision 9).
6. **Floor = the triple (16GB RAM, ~100GB/s, fanless).** 16GB is Apple's own baseline since late 2024. The floor device defines the trunk size everywhere; elasticity above the floor is the game. 8GB machines are out of scope.
7. **v1 posture = "owner"** (wire ~9–10GB on the floor device; apologize to no one). Reason: v1 is an experiment; an ambient-citizen posture makes every measurement conditional on uncontrolled co-tenant load, which is incompatible with pre-registered kill criteria. One variable at a time; the machine is the variable we freeze.
8. **But train the citizen corner now.** Elastic weights bake the supported operating range into the parameters (one-way door). The sandwich training scheme (§5.3) includes a citizen-scale min corner (T=1–2, low K) so a future ambient tier is in-distribution without retraining. Marginal training cost ≈ 0 (the min corner is computed every sandwich step anyway).
9. **Table file format is mmap-able from day one** (one-way door): file layout == memory layout, no load-time swizzling, rows importance-clustered so hot rows share pages (§6). v1 wires the file; v2's citizen posture gets OS-page-cache eviction for free. A format migration after tables are trained and shipped costs a quarter; doing it now costs a weekend.
10. **Tier selection is a runtime probe function**, called once at launch (v1). Never an install-time constant. Ships with its own microbenchmark; correctness never depends on op placement, only the tier choice does. (Game-engine graphics-settings auto-detect, applied to ML.)
11. **v1 uses fixed K per tier. No learned halting.** Adaptive halting × elastic lattice = two hard problems multiplied; keep the first kill criteria clean. Trained with stochastic unroll so multiple K values are in-distribution (§5.3). Learned halting is v2.
12. **Endgame posture = foreground hero**, not resident daemon. Takes the machine while active; quiesces cleanly (releases wired memory, checkpoints state) when backgrounded. Consequence: **mid-stream tier migration is never built** — tier changes happen at session/generation boundaries only. This deletes difficulty items (pressure-response machinery, shell-swapping mid-generation, paging-under-gather-load) from the roadmap entirely, not just from v1.
13. **Elasticity's ceiling is set by dollars, not DRAM.** A 512GB M3 Ultra does not get a 300GB table; training budget caps trained table capacity (realistically 10–40GB of rows for this program). Big machines spend surplus RAM on concurrency and context. State this in any paper before a reviewer does.
14. **Fallback if elastic training fails** (E1 kill, §7.3): train N per-tier students by distillation from the same curated data. The dataset is the durable capital asset; adapters/checkpoints are disposable. Boring, honest, already the house methodology.

---

## 3. Hardware model of the M-series lattice

### 3.1 The lattice (figures are approximate; several are estimates — Stage 1 measures what matters empirically)

| Tier | Example chips | DRAM BW (GB/s) | RAM | ANE (TOPS)† | SLC (est.)‡ | Thermals |
|---|---|---|---|---|---|---|
| Floor | M2/M3/M4 (Air, iPad Pro) | ~100–120 | 16–32GB | ~16–38 | ~8MB | **Fanless** |
| Mid | M2/M3/M4 Pro | ~150–273 | 24–64GB | ~16–38 | ~24MB | Active |
| High | M2/M3/M4 Max | ~300–546 | 36–128GB | ~16–38 | ~48MB | Active |
| Ultra | M2/M3 Ultra | ~800+ | 64–512GB | ~32–76 (2× fused clusters) | ~96MB | Active, sustained |

† ANE TOPS figures across generations are **not on the same precision basis** (M2's 15.8 is FP16-basis; M4's 38 is INT8-basis). Never compare marketing TOPS across generations; measure achieved throughput (Stage 1, M1).
‡ SLC sizes are die-analysis estimates, not Apple-documented. What matters is *effective residency*, measured by M5, not the nominal size.

Reference devices on hand: **M2 Ultra Mac Studio 64GB (800GB/s)** = top endpoint + primary bench; **iPad Pro 11" M2 (~100GB/s, fanless)** = floor proxy + sustained-load instrument; iPhone 15 Pro Max / iPhone 12 Pro = out of scope for v1 (A-series is a future program), usable for generality spot-checks only.

### 3.2 Roofline structure (why tier changes ranking)

- **M2 Ultra:** ~27 TFLOPS FP32-class GPU vs 800GB/s → corner ≈ 35 FLOPs/byte; with ANE (~32 TOPS INT8/FP16) ≈ 70 ops/byte. **Bandwidth-rich, compute-poor** — the inverse of a datacenter GPU (H100 ≈ 295). Dense 30B INT4 AR decode oracle ≈ 53 tok/s: decode bandwidth is not Ultra's pain; parallel/batched work hits the compute wall fast. ANE's value on Ultra = roughly doubling usable low-precision compute, reachable only via fixed-shape block work.
- **Floor chips:** ANE ~16 TOPS vs ~100GB/s → corner ≈ 160 ops/byte — **datacenter-like**. Base chips need block-parallel decoding *more* than the Ultra does, and SLC-residency arguments for the loop mostly don't apply (8MB SLC < trunk). At the floor, **B amortization rescues K cost**; on Ultra, SLC residency additionally helps.

### 3.3 ANE constraints (design religion)

- Static shapes only; compiled via Core ML; no data-dependent control flow; no gathers; INT8/FP16 (per-channel weight quant supported); prefers conv-style 4D tensor layouts (see Apple's "Deploying Transformers on the Apple Neural Engine").
- Op placement is decided by the Core ML runtime and **varies by OS build even on identical silicon** → per-device, per-build dispatch evidence is mandatory for every number (§9.1).
- The recurrence loop and denoise loop are driven from CPU; each iteration is one fixed-shape ANE dispatch (surgical decomposition applied temporally).
- Wired-memory cap: raise via `sudo sysctl iogpu.wired_limit_mb=<MiB>` (session-only; automate via LaunchDaemon for benches). MLX training auto-adopts the runtime's recommended wired budget; still verify under Memory Pressure.

---

## 4. Architecture v1 (constants PROVISIONAL unless marked FIXED)

### 4.1 Trunk (reasoning core)

- Weight-tied looped transformer core: `L_core` = 8–12 physical layers, `d_model` = 1536–2048, params **600–800M**, INT4 weights (+ INT8 embeddings) → **~300–400MB resident**. PROVISIONAL; final size set by E2 and the floor memory budget (§4.6). FIXED: the trunk is sized once, at the floor, for the entire lattice.
- Looping: trunk applied **K ∈ {2, 4, 8, 16}** times per denoise step (K fixed per tier at runtime; stochastic-unroll trained). Latent recurrence (hidden-state carry between loops), Huginn-style, with input injection each loop.
- Tokenizer/vocab: reuse an existing ~32k BPE (small embedding table matters at the floor). PROVISIONAL.

### 4.2 Knowledge: product-key memory (PKM) tables

- Per table: **N = 1M value rows**, product keys = 2 × 1024 sub-keys (1024² addressable), `d_v` = 1024, values FP16 → **~2KB/row, ~2GB/table** (INT8 value tier = ~1GB/table as a quant shell). Query: multi-head top-k, **k = 32 total** → **~64KB gathered per table per forward pass**.
- Placement: memory layers interleaved at fixed loop positions (e.g., after loops 1 and K/2). PROVISIONAL; E3 ablates placement.
- Capacity axis: **T ∈ {1, 2, 4, 8, 16, 20} tables** (2–40GB). Whole-table dropping only (v1). Tables are trained jointly under sandwich sampling of T.
- Per-token traffic: T=4 → ~256KB/table-reads per forward pass; with S=4 denoise steps ≈ ~1MB/token. Three-plus orders of magnitude under dense-FFN streaming. (This is the corrected version of the killed MoE/SSD experiment: same bandwidth-ceiling arithmetic, expert granularity taken to the KB limit instead of 5.64GB/token.)

### 4.3 Sequence mixing and decoding

- **Decoder: semi-autoregressive block diffusion** (BD3-LM-style). Autoregressive across blocks; iterative refinement (masked denoising) within a block. Denoise steps **S ∈ {4, 8}** per tier. PROVISIONAL; E4 informs S.
- Within-block: bidirectional attention over the fixed block (static shape → ANE).
- Cross-block context: **fixed-size recurrent block-summary state** (GRU/Mamba-style update, static shape → ANE). No growing KV cache in the base design. FIXED in spirit (no unbounded state on the ANE path); mechanism PROVISIONAL.
- Optional module (Stage-2 ablation, off by default): 2–4 global softmax-attention layers over compressed block summaries, executed on GPU, for long-range recall. Only added if the fixed-state design measurably fails recall evals.
- Block-size axis: **B ∈ {16, 32, 64}** — one compiled shell per (B, quant) pair.

### 4.4 Unit mapping (v1)

| Computation | Unit | Why |
|---|---|---|
| Trunk block-forward (per loop, per denoise step) | ANE | Fixed shape, compute-bound, best ops/W and frees GPU |
| PKM key scoring | CPU/AMX | Two tiny GEMMs over sub-keys; low-latency small matmul |
| PKM row gathers | CPU | Random access; ANE can't, GPU shouldn't |
| Cross-block state update, sampling, drivers, tier selection | CPU | Control flow |
| Optional global attention; any shell that fails ANE placement | GPU (Metal/MLX) | Dynamic shapes; fallback |

### 4.5 Elasticity system

- **Compiled shell lattice:** {B: 16/32/64} × {quant: W4A16-ANE primary, W8-table tier, FP16-GPU fallback}. Each shell precompiled and shipped; CPU driver selects at launch.
- **Runtime knobs:** T (tables enabled), K (loop depth), S (denoise steps). Changeable between generations; never mid-stream (decision 12).
- **Tier conditioning:** trunk receives a learned tier embedding (configuration token) encoding (T, K, B); **per-tier RMSNorm gains** (the transformer analogue of switchable BatchNorm — cheap, and historically where cross-config interference hides).
- **Monotonicity requirement (pre-registered property):** no configuration may outperform a strictly larger configuration by more than noise (defined: >0.5% absolute on the eval suite across 3 seeds). Enforced during training via nested self-distillation (max config teaches sub-configs), verified in E1.
- **Launch-time tier probe:** ship a ~10-second microbench (ANE dispatch latency, achieved gather bandwidth, thermal headroom class); select operating point empirically per device+OS build. Log the probe result with every session for fleet telemetry later.

### 4.6 Operating points (v1 targets)

| Tier | Device class | Wired budget | Config (T / K / B / S) | Notes |
|---|---|---|---|---|
| **Citizen corner** | (trained, not operated in v1) | ~3–5GB | 1–2 / 2–4 / 16–32 / 4 | Exists so v2 ambient tier is in-distribution |
| **Floor-owner** | 16GB Air-class | **~9–10GB** (trunk ~0.4 + shells/scratch ~1 + tables 6–8) | 3–4 / 4 / 32 / 4 | Leave macOS ~6GB; you cannot wire 14 of 16 regardless of posture |
| Mid | 36–64GB Pro/Max | ~20–40GB | 8 / 8 / 32–64 / 4–8 | |
| **Ultra** | 64GB+ Studio | ~40–50GB | 16–20 / 8–16 / 64 / 8 | Surplus RAM/BW → concurrency + context, not more tables (decision 13) |

**Thermal non-exemption:** owner posture owns RAM, not thermals. The floor device is fanless; all floor-tier numbers are steady-state numbers from the hour-long soak protocol (§9.3). This is the same instrument and the same debt as the Aperture hour-4 measurement.

---

## 5. Training design (Stage 2 toys → Stage 3)

### 5.1 Objective

Masked-denoising block-diffusion objective with semi-AR block factorization (BD3-LM family), + auxiliary losses: PKM load-balancing / dead-row revival, nested self-distillation (max-config logits → sub-configs), monotonicity-aware weighting.

### 5.2 PKM training hygiene

Known failure modes to instrument from the first toy run: dead rows (track per-row hit rates; revive or reinit), key-space collapse (track sub-key usage entropy), value-norm blowup (norm clamps). Report these curves in every training run — they are the PKM equivalent of router-entropy checks.

### 5.3 Elastic (sandwich) training — FIXED scheme, PROVISIONAL constants

Each step: compute loss at (a) max corner (T_max, K_max, B_max), (b) min corner = **citizen corner** (T=1, K=2), (c) 1–2 random middle configurations. Stochastic unroll makes all K ∈ lattice in-distribution. Tier embedding + per-tier norms active from step 0. Distill (a) → (b,c).

### 5.4 Compute envelopes

- Toys (E1–E3): 60–240M params × 2–5B tokens ≈ 10¹⁸–10¹⁹ FLOPs ≈ single-digit H100-hours per run (~$10–20). Runable locally on the Ultra in days if preferred; rent for iteration speed.
- Stage 3: 1–2B params × ~40B tokens ≈ 3.6–7×10²⁰ FLOPs ≈ 300–700 H100-hours at realistic MFU, **×2 margin** for gather/recurrence inefficiency → **$2–8K**. Data plan: reuse existing curated corpora + the house gold-data pipeline; the dataset is the asset either way (decision 14).

---

## 6. Table file format (one-way door — implement in Stage 1 window)

- Single file per table. Header (versioned, self-describing) → key blocks (sub-key matrices, contiguous) → value rows.
- **Layout == memory layout.** `mmap` + `mlock`/wire in v1; no deserialization, no swizzling, alignment = 16KB (macOS page size).
- **Rows clustered by learned importance** (importance = training-time hit-rate percentile; re-clustered at export). 8 rows/page at 2KB — clustering makes hot pages dense so v2's unwired posture gets page-cache locality for free.
- Include per-table metadata: importance histogram, row-hit stats, quant params, training-run provenance hash.
- Deliverable: format spec doc + reference reader (C/Swift) + property tests (round-trip, alignment, endian).

---

## 7. Experimental program (stages, budgets, kill criteria)

### Stage 0 — Analytical model. $0, 1 day.

Build the bytes/energy notebook: `E/token ≈ Σ bytes_moved · e_source + FLOPs · e_unit`; tok/s oracles = BW / bytes-per-token. Compute oracles for all candidate configs at both lattice endpoints (Ultra 800GB/s, floor ~100GB/s), including the B/K/S/T interactions in §1.2 and §3.2. Deliverable: notebook + a one-page table of which configs are already dead on paper.
**Rule (from the MoE Stage-0 failure):** targets are set from product requirements *before* computing the oracle — never as "oracle minus ε."

### Stage 1 — Physics on random weights. $0 marginal. ~10 days agent time. `[HW: M2 Ultra + iPad Pro M2]`

Random weights are **valid here** — joules and bytes don't care about training. (They are *never* valid for quality; §9.4.) All benchmarks follow §9 discipline. Run every benchmark on both devices unless noted; the iPad is the floor proxy and all its numbers are post-soak steady-state.

| ID | Benchmark | Method (summary) | Kill criterion (pre-registered) |
|---|---|---|---|
| **M1** | ANE topology probe `[Ultra]` | Sweep fixed-shape block-forwards; measure achieved TOPS for one dispatch; then two concurrent model instances | Single dispatch caps at ~one fused cluster **and** dual instances don't scale → halve all Ultra ANE-compute claims; Idea-1's Ultra justification weakens accordingly |
| **M2** | Bus contention | Concurrent GPU weight-streaming + CPU random gathers + ANE fixed-shape dispatch; compare aggregate vs isolated sums | Aggregate degradation > **30%** → 3-unit concurrency is theater; collapse mapping to 2 units and re-run Stage 0 |
| **M3** | Gather bench | k=32 × 2KB random gathers from a wired 16–32GB table (200-line harness, no model); measure effective random BW, SLC pollution (co-run M5 variant), latency distribution | Gather serialization tax > **20%** of a trunk-step budget, or gather energy ≥ 1/10 of equivalent streamed bytes on the floor proxy → Idea-3 mapping fails at that tier |
| **M4** | Block-forward vs AR decode | Same small model: fixed-shape B∈{16,32,64} forward on ANE (Core ML) vs AR decode on GPU (MLX). Ultra: primary metric tok/s-per-otherwise-idle-unit; floor: primary metric J/token (powermetrics rails) | ANE block path < **3×** better on the tier-primary metric at iso-model-bytes → block decoding insufficient on this silicon; program pivots to PKM-only |
| **M5** | Loop amortization | ~300M random trunk, K=1→32; marginal energy & latency per extra iteration | Ultra: marginal iteration ≥ **30%** of first-iteration cost → SLC/ANE residency story is fiction (informational at floor — no residency expected). **This harness doubles as the monolithic-Core-ML-export control experiment owed to the Surgical Inference reviewers; run and report both.** |

### Stage 2 — Quality proxies. ~$100–300 total. 1–2 weeks.

| ID | Experiment | Design | Kill criterion |
|---|---|---|---|
| **E1** | Elasticity tax | Toy composed model (~60M trunk + 2 small tables), sandwich-trained incl. citizen corner, vs fixed-max-config control at matched tokens. Ablations: tier-conditioned norms on/off; table-drop vs row-truncate; monotonicity check across corners (3 seeds) | Tax > **3%** avg or > **5%** at max corner → elastic training dies; fallback = per-tier distilled students from shared gold data (decision 14). Monotonicity violation > 0.5% → fix or kill |
| **E2** | Loop vs dense | 60M trunk × K=4 vs 240M dense, matched tokens (~2–5B) | Looped model recovers < **50%** of the quality gap to the 4× dense → recurrence doesn't substitute for depth even at toy scale; demote loop axis to K∈{1,2} |
| **E3** | PKM vs dense | Param-matched and FLOP-matched dense baselines vs trunk+tables; ablate table placement | PKM model closes < **50%** of gap to param-matched dense while using < 10% of streamed bytes → PKM thesis fails at toy scale; program halts for redesign |
| **E4** | Block-diffusion reality check | **No pretraining:** port a denoising step of open-weight LLaDA-8B or Dream-7B to Metal/ANE; measure quality-adjusted tok/s and J/token vs an AR 8B at matched benchmarks `[HW: Ultra]` | < **2×** tok/s at ≤2% benchmark delta → S-step creep has eaten the win on this silicon; revisit S/B lattice or demote diffusion axis |

### Gate review (human + cross-model, §10.4)

Inputs: all Stage 0–2 reports vs kill table. Output: Stage-3 architecture constants finalized (freeze §4's PROVISIONALs) or program pivot/kill memo. **Scale-transfer caveat is mandatory in the memo:** toy exponents and an 8B port do not guarantee the 1–2B composed regime; the pre-registration says so rather than the reviewers.

### Stage 3 — Composed pretrain. $2–8K. 2–4 weeks. Gated.

1–2B composed model, ~40B tokens, full sandwich lattice, per §5. Deliverables: the **quality × bytes/token × tier frontier** (the paper's central figure), monotonicity surface, elasticity tax at scale, and shipped inference artifact (shells + tables + probe) running on both reference devices. Publishable in either direction — a clean negative on any axis is a result.

---

## 8. Kill-criteria summary (all thresholds contestable **before code**, frozen after)

M1 cluster-scaling ✗ → halve ANE claims · M2 >30% contention → 2-unit design · M3 >20% tax / energy fail → PKM mapping fails per tier · M4 <3× → pivot to PKM-only · M5 ≥30% marginal → drop residency claims · E1 >3%/5% tax or monotonicity ✗ → per-tier distillation fallback · E2 <50% gap recovery → demote K · E3 <50% gap closure → halt/redesign · E4 <2× → demote diffusion axis. **Global rule:** distinguish *zero data* (untested) from *negative data* (tested, failed) in every report; untested assumptions are never cited as evidence.

---

## 9. Measurement discipline (non-negotiable)

1. **Dispatch evidence per number.** Every benchmark result attaches Core ML placement proof: Xcode performance report or programmatic `MLComputePlan` dump showing per-op device assignment, archived alongside the result. Core ML silently re-places ops across OS builds; a number without placement proof does not exist. Record: device model, chip, RAM, macOS build, power source, `iogpu.wired_limit_mb`, thermal state.
2. **Power rails.** `sudo powermetrics` with CPU/GPU/ANE samplers at ≤200ms intervals (verify sampler names per OS build via `powermetrics -h`); report per-rail J/token medians + p95 over ≥5 runs; subtract measured idle baseline.
3. **Thermal steady state.** Fanless floor proxy: 60-minute soak at target load before measurement windows; report sustained, never burst. Log thermal-pressure notifications. (Same protocol closes the open Aperture hour-4 question — run it once, spend it twice.)
4. **Random-weights rule.** Random/`tiny-random-*` weights are valid **only** for physics (bytes, joules, latency). Any quality metric requires the harness to assert a trained checkpoint ID; for routed/keyed components, assert key-usage entropy above a sanity floor before computing recall/precision. (This rule exists because a predictability table was once generated on an untrained router; never again.)
5. **No goalpost anchoring.** Success thresholds are written in this spec before harness code exists and are never set relative to a just-computed ceiling.
6. **One variable at a time.** Configuration sweeps change one axis per sweep; interaction studies are explicitly labeled as such.

---

## 10. Execution logistics

### 10.1 Repo layout

```
plateau/
  spec/            # this file + revisions + decision-change memos
  stage0/          # analytical notebook + oracle tables
  harness/         # powermetrics wrapper, MLComputePlan dumper, soak protocol, device-metadata collector
  bench/           # m1_ane_topology/ m2_contention/ m3_gather/ m4_block_vs_ar/ m5_loop/
  toys/            # e1_elasticity/ e2_loop/ e3_pkm/ e4_diffusion_port/
  formats/         # table file format spec + reference reader + property tests
  runtime/         # shell driver, tier probe, loop driver (CPU)
  reports/         # one markdown per result, template below
```

### 10.2 Report template (per result)

`reports/<stage>_<id>_<device>_<date>.md`: claim → method → device/build metadata block → dispatch-evidence artifact links → raw data location → medians/p95 table → verdict vs pre-registered kill → anomalies → *zero-data vs negative-data* ledger for anything not run.

### 10.3 Division of labor

- **Agents:** all code, Stage 0, toy training orchestration (rented H100s), analysis, report drafting, format implementation.
- **`[HW: …]` steps require Matt's physical devices:** Stage-1 runs on the Ultra and iPad (agent prepares one-command harnesses; Matt executes and uploads raw logs), E4 Ultra measurements, launch-probe validation.
- **Human gates:** kill-threshold contest window (before any code), each stage boundary, Stage-3 spend approval.

### 10.4 Cross-model review (house BFT rule)

Before the Stage-3 gate closes: this spec + all Stage 0–2 reports go to ≥2 heterogeneous frontier models (different providers) with an adversarial brief: *find the un-pre-registered assumption, the anchored threshold, the capacity-mismatched comparison, the scale-transfer overclaim.* Their findings and dispositions are appended to the gate memo.

### 10.5 Timeline (elapsed, assuming parallel agent execution)

Stage 0: days 1–2 → Stage 1: days 2–14 → Stage 2: days 7–21 (overlaps; toys don't need Matt's hardware) → Gate: ~day 22–24 → Stage 3: weeks 4–8.

---

## 11. Open questions (contest these; do not silently resolve them)

1. Exact trunk geometry (L_core, d_model) and vocab — set at gate from E2/E3 + floor budget.
2. PKM placement within the loop; multi-head memory configuration (h×k split of k=32).
3. Cross-block state mechanism (GRU-style vs SSM update) and whether the optional GPU global-attention module is ever needed (recall evals decide).
4. S (denoise steps) per tier after E4; whether S should join the runtime-knob set (it can — it's loop-count, not shape).
5. INT8-value table tier: quality cost unknown; measure in E3.
6. Whether Ultra-tier surplus goes to multi-agent concurrency (multiple simultaneous generations sharing tables — tables are read-only at inference, so this is free structurally) — design sketch only in v1.
7. Data mixture for Stage 3 (reuse house gold pipeline vs public corpora blend).

## 12. Known risks (accepted, monitored)

Scale transfer of toy results (mitigated: pre-registered caveat, cheap toys first) · gradient conflict across the coupled elastic manifold worse than OFA/Flextron precedent suggests (mitigated: E1 is the cheapest experiment in the program) · diffusion step-creep under quality pressure (E4 measures) · PKM training pathologies (§5.2 instrumentation) · Core ML placement drift across OS updates (probe + dispatch evidence make it detectable, not preventable) · gather behavior on LPDDR at 2KB granularity (M3 exists precisely for this).

## 13. References (agent: verify links before citing in reports)

Product-key memories — Lample et al. 2019, arXiv:1907.05242 · Memory Layers at Scale — Berges et al. 2024, arXiv:2412.09764 · Recurrent depth / latent reasoning ("Huginn") — Geiping et al. 2025, arXiv:2502.05171 · Universal Transformers — Dehghani et al. 2018, arXiv:1807.03819 · Adaptive Computation Time — Graves 2016, arXiv:1603.08983 · Block Diffusion (BD3-LM) — Arriola et al. 2025, arXiv:2503.09573 · LLaDA — Nie et al. 2025, arXiv:2502.09992 · Dream 7B — HKU NLP 2025 (open weights) · Mercury — Inception Labs (commercial evidence for diffusion-LM throughput) · Once-for-All — Cai et al. 2019, arXiv:1908.09791 · Slimmable Networks — Yu et al. 2018, arXiv:1812.08928 · MatFormer — arXiv:2310.07707 · Flextron — arXiv:2406.10260 · Gemma 3n (shipped MatFormer-style elasticity) — Google model card · Mamba-2 — Dao & Gu 2024, arXiv:2405.21060 · Apple: "Deploying Transformers on the Apple Neural Engine" (ML research note) · MLX / mlx-lm — github.com/ml-explore.

---

*End of spec v1. Change control: revisions append a decision-change memo to §2 and save as a new file.*
