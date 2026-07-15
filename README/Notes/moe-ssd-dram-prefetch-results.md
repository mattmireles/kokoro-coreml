# MoE SSD/DRAM Prefetch Results

**First spotted:** 2026-06-29
**Status:** Complete (Stage 1 kill)

## Summary

This note records the staged results for
[MoE SSD/DRAM expert prefetch experiment plan](../Plans/moe-ssd-dram-prefetch-v1.md).
Each stage must end with a written go/kill decision before the next stage
starts. A kill decision is a valid result.

## Related Guides

- [MoE expert offload and prefetch prior art](../Guides/moe-expert-offload-prefetch-prior-art-guide.md)
- [Apple Silicon NVMe and energy measurement](../Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
- [MoE SSD/DRAM guide triage](moe-ssd-dram-prefetch-guide-triage-2026-06-29.md)

## Stage 0: Hardware Envelope

**Status:** Complete.

### Frozen Assumptions

- Model inventory: `outputs/moe_prefetch/stage0/model_inventory.json`
- Gate thresholds: `outputs/moe_prefetch/stage0/thresholds.json`
- Model: `mistralai/Mixtral-8x7B-v0.1`
- Expert bytes: `88080384`
- Active expert bytes/token: `5637144576`
- Target decode rate: `1.0` token/sec

### Measurements

| Pattern | Bandwidth GB/s | p50 latency ns | p95 latency ns | fs_usage proof |
| --- | ---: | ---: | ---: | --- |
| sequential | 5.929 | 14167000 | 17164000 | True |
| random | 5.962 | 14241000 | 16671000 | True |

Oracle bandwidth ceiling: `1.057558` tokens/sec.
One-layer compute p50: `1.773188` ms (`outputs/moe_prefetch/stage0/compute.json`).
Hideability ratio (max read p95 / compute p50): `9.679741`.

### Privileged Capture Status

- `fs_usage`: present for every accepted read cell.
- `powermetrics`: captured at `outputs/moe_prefetch/stage0/powermetrics_stage0.plist`

### Decision

FLAG: bandwidth passes but one-layer lead time is insufficient.

## Stage 1: Router Trace and Predictor Replay

**Status:** Complete.

### Trace

- Trace: `outputs/moe_prefetch/stage1/router_trace.jsonl`
- Rows: `180`
- Model IDs: `hf-internal-testing/tiny-random-MixtralForCausalLM`
- Domains: `code, math, prose`
- Required prefetch depth from Stage 0 hideability: `10` layers
- Evaluated prefetch depths: `1, 2, 3`

### Predictor Replay

| Depth | Policy | Recall | Precision | Hideable recall | Wasted prediction fraction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | demand_lru | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | last_token | 0.517 | 0.534 | 0.000 | 0.466 |
| 1 | frequency | 0.558 | 0.565 | 0.000 | 0.435 |
| 1 | markov | 0.467 | 0.519 | 0.000 | 0.481 |
| 1 | eam | 0.264 | 0.528 | 0.000 | 0.472 |
| 1 | oracle | 1.000 | 1.000 | 0.000 | 0.000 |
| 2 | demand_lru | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | last_token | 0.517 | 0.534 | 0.000 | 0.466 |
| 2 | frequency | 0.558 | 0.565 | 0.000 | 0.435 |
| 2 | markov | 0.467 | 0.519 | 0.000 | 0.481 |
| 2 | eam | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | oracle | 1.000 | 1.000 | 0.000 | 0.000 |
| 3 | demand_lru | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | last_token | 0.517 | 0.534 | 0.000 | 0.466 |
| 3 | frequency | 0.558 | 0.565 | 0.000 | 0.435 |
| 3 | markov | 0.467 | 0.519 | 0.000 | 0.481 |
| 3 | eam | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | oracle | 1.000 | 1.000 | 0.000 | 0.000 |

### Decision

KILL: planned prefetch depths cannot hide Stage 0 p95 expert reads.

## Stage 2: Offline Simulator

**Status:** Not run. Stage 1 killed the planned experiment path because
prefetch depths `{1,2,3}` cannot hide the Stage 0 p95 expert read latency.

## Stage 3: Runtime Harness

**Status:** Not run. Stage 2 did not start.

## Required Executable Memory

Regression test:

```bash
python3 -m pytest tests/test_moe_prefetch_tools.py
uv run --no-sync python scripts/moe_prefetch/summarize.py stage1 \
  --trace outputs/moe_prefetch/stage1/router_trace.jsonl \
  --stage0 outputs/moe_prefetch/stage0/summary.json \
  --output outputs/moe_prefetch/stage1/predictability.json \
  --notes README/Notes/moe-ssd-dram-prefetch-results.md
```
