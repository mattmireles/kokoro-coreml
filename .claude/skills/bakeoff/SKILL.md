---
name: bakeoff
description: >-
  Run the controlled bakeoff benchmark on this machine. Prepares inputs,
  builds the Swift pipeline, runs the counterbalanced harness for all
  available configs (A, D, E, F), records results, and updates
  performance-notes.md. Use when the user says "run the bakeoff",
  "benchmark this machine", or invokes $bakeoff.
---

# Bakeoff

## Purpose

Run the full bakeoff benchmark suite on the current machine and record
publication-grade results. This is the one-command path from "I have a
machine" to "results are in performance-notes.md."

## Use When

- The user wants to benchmark the current machine.
- A new machine is available and needs bakeoff data.
- The user says "run the bakeoff", "benchmark", or invokes `$bakeoff`.

## Do Not Use When

- The user only wants to run a single quick test (use the harness directly).
- The user wants to create or modify the bakeoff plan (use `create-plan`).
- The user wants to audit existing results (use `audit`).

## Prerequisites

Everything is handled by the setup script. If this is a fresh machine
(or you're not sure), run it first:

```bash
bash scripts/setup_bakeoff.sh
```

This takes ~10 minutes and handles: Python deps from
`requirements-bakeoff.txt`, model downloads,
all model exports (Duration [32-512], F0Ntrain, DecoderPre,
GeneratorFromHar for buckets [3,7,10,15,30]s), Swift binary build,
and benchmark input preparation. Use `--skip-download` if models are already
local. After setup, use `uv run --no-sync` for the publication command so `uv`
does not auto-sync the project lockfile over the explicitly installed bakeoff
environment.

If setup has already been run, verify quickly:

```bash
ls swift/.build/release/kokoro-bench && \
ls outputs/bakeoff/input_manifest.json && \
ls outputs/swift_bench_inputs/hnsf_weights.json && \
echo "Ready"
```

Always rebuild the Swift benchmark binary before a publication run. The
checked-out Swift sources may be newer than `swift/.build/release/kokoro-bench`;
an old binary can lack `--batch` support and will make Config F fail with the
single-shot usage message.

```bash
(cd swift && swift build -c release --product kokoro-bench)
swift/.build/release/kokoro-bench --help
```

The help output must include:

```text
[--input-key KEY | --batch]
```

## Procedure

### 1. Run setup (if needed)

```bash
bash scripts/setup_bakeoff.sh          # full setup (~10 min)
bash scripts/setup_bakeoff.sh --skip-download  # skip HF download
```

### 2. Identify the machine and quiet-host gate

Read `README/Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md`
first. Irvine M1 numbers are publishable only with warmed per-bucket timing and a
quiet-host pass.

For remote Irvine hosts, run the quiet gate before benchmarking:

```bash
uv run python scripts/external_bakeoff/check_remote_host_quiet.py \
  --machine-id irvine_m1
```

Record:
- Chip (e.g., M1 Mini, M2 Ultra, M2 MacBook Air)
- RAM
- macOS version
- Any relevant config (plugged in vs battery, thermal state)
- Quiet-host status JSON (when remote)

Choose a `--machine-id` slug: `m1_mini`, `m2_ultra`, `m2_air`, etc.

### 3. Rebuild and sanity-check Config F

Rebuild the Swift benchmark binary even if setup was already run:

```bash
(cd swift && swift build -c release --product kokoro-bench)
```

Then run a focused Config F batch warmup. This catches stale binaries, missing
inputs, and batch protocol issues before the full benchmark spends time loading
all Python baselines:

```bash
uv run --no-sync python - <<'PY'
from scripts.bakeoff_harness import SwiftPipelineContext

ctx = SwiftPipelineContext()
ctx.warmup("")
ctx.close()
print("Config F batch warmup completed")
PY
```

If this stalls after an ANE compiler diagnostic, check
`scripts/bakeoff_harness.py`: the batch reader must accept stdout lines that
end with `DONE` or `ERROR`, not only lines equal to those exact strings. Core ML
can emit diagnostics such as `E5RT encountered an STL exception...` onto stdout
immediately before `DONE`, yielding one combined line.

### 4. Run the bakeoff

Config D (MPS) requires `PYTORCH_ENABLE_MPS_FALLBACK=1`. Set it for
all runs to keep the command uniform.

```bash
BAKEOFF_SKIP_SMOKE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs a,d,e,f \
  --iterations 5 \
  --order-seed 0 \
  --machine-id <machine_id>
```

Expected runtime: 10–20 minutes depending on machine speed.

On 16 GB machines, if the combined A/D/E/F process swaps heavily, split the
publication run into two commands with the same setup and order seed:

```bash
BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs a,f \
  --iterations 5 \
  --order-seed 0 \
  --machine-id <machine_id>_af

BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs d,e \
  --iterations 5 \
  --order-seed 0 \
  --machine-id <machine_id>_de
```

### 5. Verify results

For a single combined run, set `RESULT_FILES` to one file. For a split 16 GB
run, set it to both split files:

```bash
export RESULT_FILES="outputs/bakeoff/results_<machine_id>.json"
uv run --no-sync python - <<'PY'
import json, statistics
from collections import defaultdict
import os
from pathlib import Path

paths = [Path(p) for p in os.environ['RESULT_FILES'].split()]
statuses = defaultdict(int)
groups = defaultdict(list)

for path in paths:
    data = json.loads(path.read_text())
    for r in data['results']:
        statuses[(r.get('config'), r.get('status'))] += 1
        if r.get('status') == 'ok':
            groups[(r['config'], r['input_key'])].append(r['wall_time_s'] * 1000)

print('statuses:')
for key, count in sorted(statuses.items()):
    print(f'  {key}: {count}')

for ik in ['3s', '7s', '15s', '30s']:
    row = {
        c: statistics.median(groups.get((c, ik), [0]))
        for c in ['a', 'd', 'e', 'f']
    }
    print(
        f'{ik:4s}  A={row["a"]:.0f}ms  D={row["d"]:.0f}ms  '
        f'E={row["e"]:.0f}ms  F={row["f"]:.0f}ms'
    )
PY
```

For a split run, keep the same verifier and change only the first line:

```bash
export RESULT_FILES="outputs/bakeoff/results_<machine_id>_af.json outputs/bakeoff/results_<machine_id>_de.json"
```

All configs should show `status: ok` for all inputs. Config D may show
`config_unavailable` if MPS fallback wasn't set — that's acceptable.

### 6. Update performance-notes.md

Add a new section to `README/Notes/performance-notes.md` following the
existing pattern (see "Bakeoff v2: Controlled benchmark on M2 MacBook Air"
or "Bakeoff v3: Swift pipeline" for the template). Include:

- Machine identification and provenance
- End-to-end wall time table (warm median, ms)
- RTF table
- Speedup: F vs A, F vs E
- Cross-machine comparison table (if prior machine data exists)
- Interpretation (2-4 bullet points)

### 7. Commit and push

Use `git-commit` to stage the notes changes. Do not force-add
`outputs/bakeoff/*.json`; `outputs/` is intentionally ignored, so cite result
paths in notes instead of committing generated benchmark artifacts. Then
`git-push` to sync with origin.

## Configs Reference

| Config | What it measures | Runtime |
| --- | --- | --- |
| A | Shipping Python HAR-post hybrid (PyTorch prefix + CoreML decoder) | ~2 min |
| D | PyTorch end-to-end on MPS (GPU with CPU fallback) | ~2 min |
| E | PyTorch end-to-end on CPU | ~3 min |
| F | Swift + CoreML pipeline (5 models + Swift hn-nsf DSP) | ~1 min |

Configs B/C (decoder-only) are omitted by default — they measure ANE
participation, not pipeline speed. Add `b,c` to `--configs` if needed.

## Canonical Docs

- Bakeoff plan: `README/Plans/kokoro-bakeoff-v2.md`
- Swift pipeline plan: `README/Plans/swift-prefix-rewrite-v1.md`
- Harness: `scripts/bakeoff_harness.py`
- Swift CLI: `swift/Sources/KokoroBenchmark/main.swift`
- Performance notes: `README/Notes/performance-notes.md`
