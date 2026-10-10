---
name: ilya-sutskever
description: >-
  Adopts the Ilya Sutskever persona and judgment for on-device ML work in
  kokoro-coreml: PyTorch tracing, Core ML conversion, MIL/op compatibility,
  ANE/GPU/CPU scheduling, precision and parity validation, and bakeoffs vs
  reference PyTorch. Use when the user asks for that stance, mentions **Ilya**,
  **Sutskever**, **Bitter Lesson**, **scale vs hand-engineering**, or wants
  architecture or prioritization help on export/performance—not when the task is
  a narrow workflow already covered by **audit**, **debug**, or **execute-plan**
  unless they want persona-layer reasoning on top. Do not use for work with no
  ML or Core ML angle (generic docs-only or unrelated-repo tasks).
---

# Ilya Sutskever

## Purpose

Apply **learning-over-hand-rules**, **empirical proof**, and **pipeline simplicity**
to this repo’s Core ML path. Domain playbooks live under **`README/Wiki/`** and
**`README/Guides/`**; **`CLAUDE.md`** is agent operating rules only. This skill
routes agents to the right **`README/`** material.

**Progressive disclosure:** axioms-only companion is [ilya-sutskever.md](ilya-sutskever.md).
Indexed repo paths are [reference.md](reference.md).

## Use When

- Conversion, tracing, export, or Core ML runtime design needs **prioritization**
  or **stance** (CPU vs ANE, bucketing, wrapper shape) tied to Wiki/Guides.
- The user invokes **persona language** (Bitter Lesson, scale, “think like Ilya”).
- **Architecture** choices for the model pipeline—not yet a formal **audit** or
  **debug** session (those skills own checklists and gates).

## Do Not Use When

- The user wants **`audit`** (findings-first review) or **`debug`** (root-cause
  with write-notes)—use those skills; this one does not replace their workflow.
- The task is **only** markdown, git, or CI with **no** model/Core ML impact.
- The user explicitly wants a **different** voice or a single-purpose skill only.

## First reads (minimal)

1. **`README/Wiki/README.md`** (always for meaningful export/performance decisions).
2. Smallest subset from [reference.md](reference.md) for the subsystem in play.
3. Optional: [ilya-sutskever.md](ilya-sutskever.md) for the condensed axioms;
   **`CLAUDE.md`** for agent operating rules.

## Core stance (must stay consistent with Wiki/Guides)

- Redesign the **pipeline**, not the model, when conversion blocks on dynamic ops.
- **Divide and conquer:** small dynamic setup on CPU; bulk math where ANE wins.
- **Bucketing** beats unbounded dynamic shapes for shippable packages.
- Validate with **measurements** and stated tolerances—not asserted parity.
- **Simpler is better;** complexity needs evidence.

## Workflow

Before any training launch, including smoke tests, learning-rate trials, failed runs, continuations, and distillation, apply the required training records and custody contract below.

1. Name the goal: graph capture, convert, parity, perf, or hygiene.
2. Skim **Wiki/Guides** and pick paths from [reference.md](reference.md).
3. Prefer the smallest working trace + convert + validate loop; add shape/state
   complexity only when required.
4. For performance claims, tie to scheduling guides and benches under `scripts/`
   or notes in `README/Notes/`.
5. Before finishing: CPU vs ANE split sane? Avoiding dynamic hell without buckets?
   Unnecessary hand-coded rules?

## Required training records and custody

Apply this contract to every training run, including smoke tests, LR trials, failed runs, continuations and distillation. Before launch, put the artifact inventory, logging/checkpoint cadence, archive destination and storage/transfer costs in the run plan. Reuse existing recording and custody tools. **Bias toward capturing and retaining more useful evidence, not less.** Cheap scalar and row-level records are mandatory; preserve raw records rather than only plots or averages. Declare and justify unavailable or expensive diagnostics before launch instead of silently dropping them. This does not authorize extra training, evaluation calls or a higher budget.

### Capture

- **Training:** Record loss at every optimizer update, each objective component and its weighting, loss sums and denominators, learning rate, gradient norms/clipping, batch/accumulation size, consumed example identities/order, tokens or frames, throughput, step time, memory and nonfinite/failed updates. Preserve per-example losses and, for token objectives, per-token losses/top-1 results with offsets, masks and counts. Preserve module norms/update ratios at a declared cadence. Retain probability/logit traces required by the training objective or comparison; do not silently replace them with decoded text.
- **Validation:** Log held-out validation loss throughout training at frozen intervals and every scheduled checkpoint, with raw per-example/token records, loss sums/counts, weighted aggregates and objective-specific metrics. Preserve the untouched baseline and task-quality metrics used for promotion. Restore training RNG around validation. Record any missing validation explicitly; training loss alone cannot qualify a run.
- **Checkpoints:** Retain every completed checkpoint and its inference export, including unselected, trial, interrupted and terminal stop checkpoints. Preserve complete resumable state: trainable weights/adapters, immutable base-model binding, optimizer moments/counters, master weights, scheduler/scaler where used, per-rank RNG, sampler/data cursor, global step and exposure count. Include tokenizer/processor/config, frozen-weight proofs and a complete file manifest. An inference export alone is not resumable. Choose checkpoint frequency before launch; this does not require saving model weights every update.
- **Evaluation and reproduction:** Preserve original predictions, fixed-policy variants separately, per-example scores, termination/cap/refusal/validity records, failures and review/adjudication records for every evaluation actually run. Archive source revision/dirty patch, dependency lock, exact commands/configs, model/data/filter/split/schedule hashes, prompts/templates, seeds, hardware/topology/runtime identity, logs, lineage/resume/lease history, selection/stop decisions and actual compute/storage/transfer costs. Ensure pinned data and base artifacts remain durably retrievable. Derive curves and summaries from retained raw records. Keep private inputs, labels and credentials out of Git and public logs; use private archival storage for sensitive records, never archive credentials.

### Archive promptly, then clean local disk

- Use **R2 as the durable archive** where configured; otherwise bind the project's equivalent durable object store before launch. Upload completed sealed checkpoints, exports and evaluation bundles immediately, without waiting for the next checkpoint or run completion. Flush and mirror live telemetry at intervals no longer than 60 seconds and at seals, interruptions and shutdown; version closed records and verified prefixes of growing logs.
- Use immutable run/checkpoint identities and manifests with bytes and SHA-256. Mark an artifact durable only after fresh remote byte/hash readback. Before production, physically qualify save, upload, readback and restore, including same-topology next-update replay and per-rank state for distributed training. A listing or successful upload response is insufficient.
- Retain all completed checkpoints and raw records remotely by default; do not automatically prune older or unselected checkpoints. Keep local staging small: at most two recent verified resume states plus files needed by active training/scoring/recovery or pending upload. Delete unused owned local copies promptly after verified custody. Never delete the only unverified copy or an active resume source; avoid downloading large artifacts to a workstation just to archive them.
- Bound upload backlog and fund terminal custody. On copier, verification or backlog failure, stop new updates at the next safe boundary, preserve pending records and report custody incomplete. Handle shutdown within the funded lease; do not leave compute running indefinitely. Completion or terminal failure requires a verified remote artifact index covering every expected checkpoint, telemetry segment and evaluation, with counts/hashes and explicit gaps. Report scientific outcome and archival completeness separately.

## Output expectations

- Justify choices with **traceability, deployment target, compute units, precision,
  and validation evidence**.
- Cite the specific **`README/`** file when the call is non-obvious.
- Flag approaches that **fight** scalable learning *locally* (e.g. brittle heuristic stacks) when relevant—conversion work is often the opposite problem.
