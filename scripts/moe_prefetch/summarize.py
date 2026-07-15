#!/usr/bin/env python3
"""Summarize MoE prefetch experiment stages and update notes."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.moe_prefetch.predictors import evaluate_policies, load_trace
from scripts.moe_prefetch.schema import load_json, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    stage0 = sub.add_parser("stage0")
    stage0.add_argument("--input", type=Path, required=True)
    stage0.add_argument("--notes", type=Path, required=True)
    stage0.add_argument("--output", type=Path, default=None)
    stage0.add_argument("--compute", type=Path, default=None)
    stage0.add_argument("--one-layer-compute-ms", type=float, default=None)
    stage1 = sub.add_parser("stage1")
    stage1.add_argument("--trace", type=Path, required=True)
    stage1.add_argument("--stage0", type=Path, required=True)
    stage1.add_argument("--output", type=Path, required=True)
    stage1.add_argument("--notes", type=Path, required=True)
    stage1.add_argument("--prefetch-depths", default="1,2,3")
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _has_fs_usage_diskio_proof(path_text: str) -> bool:
    """Return whether an `fs_usage -f diskio` artifact contains disk I/O rows.

    Called by `_cell_summary` for Stage 0 acceptance. A filesystem path alone is
    not proof: failed privileged captures can leave an empty file behind, and
    the MoE SSD/DRAM plan explicitly forbids treating timing rows as SSD data
    without `fs_usage` evidence.
    """
    if not path_text:
        return False
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return False
    diskio_markers = ("RdData", "WrData", "RdMeta", "WrMeta")
    return any(
        any(marker in line for marker in diskio_markers)
        for line in path.read_text(errors="replace").splitlines()
    )


def _cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
    measurement = cell.get("measurement") or {}
    latencies = [float(v) for v in measurement.get("latencies_ns", [])]
    total_bytes = float(measurement.get("total_bytes_read", 0))
    wall_ns = float(measurement.get("wall_time_ns", 0))
    bandwidth_gbps = (total_bytes / wall_ns) if wall_ns > 0 else 0.0
    # bytes/ns is numerically equal to GB/s when using decimal GB.
    fs_usage_path = cell.get("fs_usage_path", "")
    fs_usage_error = cell.get("fs_usage_error", "")
    has_fs_usage = not fs_usage_error and _has_fs_usage_diskio_proof(fs_usage_path)
    return {
        "pattern": cell.get("pattern"),
        "returncode": cell.get("returncode"),
        "successful_reads": measurement.get("successful_reads", 0),
        "failed_reads": measurement.get("failed_reads", 0),
        "bandwidth_gbps": bandwidth_gbps,
        "latency_p50_ns": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ns": _percentile(latencies, 0.95),
        "fs_usage_path": fs_usage_path,
        "fs_usage_error": fs_usage_error,
        "has_fs_usage": has_fs_usage,
    }


def _replace_stage0_section(note_text: str, stage0_markdown: str) -> str:
    start = note_text.index("## Stage 0: Hardware Envelope")
    end = note_text.index("## Stage 1: Router Trace and Predictor Replay")
    return note_text[:start] + stage0_markdown.rstrip() + "\n\n" + note_text[end:]


def _replace_stage1_section(note_text: str, stage1_markdown: str) -> str:
    start = note_text.index("## Stage 1: Router Trace and Predictor Replay")
    end = note_text.index("## Stage 2: Offline Simulator")
    return note_text[:start] + stage1_markdown.rstrip() + "\n\n" + note_text[end:]


def summarize_stage0(args: argparse.Namespace) -> int:
    payload = load_json(args.input)
    inventory = payload["inventory"]
    thresholds = payload["thresholds"]
    compute_payload = load_json(args.compute) if args.compute else None
    if args.compute and args.one_layer_compute_ms is not None:
        raise SystemExit("provide --compute or --one-layer-compute-ms, not both")
    cells = [_cell_summary(cell) for cell in payload.get("cells", [])]
    usable = [cell for cell in cells if cell["returncode"] == 0 and cell["failed_reads"] == 0]
    random_cells = [cell for cell in usable if cell["pattern"] == "random"]
    ceiling_cell = random_cells[0] if random_cells else (usable[0] if usable else None)
    active_bytes = float(inventory["active_expert_bytes_per_token"])
    ceiling_tps = 0.0
    if ceiling_cell and ceiling_cell["bandwidth_gbps"] > 0:
        ceiling_tps = (ceiling_cell["bandwidth_gbps"] * 1_000_000_000.0) / active_bytes

    target_tps = float(inventory["target_tokens_per_second"])
    fs_usage_missing = any(not cell["has_fs_usage"] for cell in usable)
    bandwidth_kill = ceiling_tps < target_tps
    one_layer_compute_ms = args.one_layer_compute_ms
    if compute_payload:
        one_layer_compute_ms = float(compute_payload["latency_p50_ms"])
    hideability = None
    max_latency_ns = max((cell["latency_p95_ns"] for cell in usable), default=0.0)
    if fs_usage_missing:
        decision = "KILL: missing fs_usage disk I/O proof for accepted measurements."
    elif bandwidth_kill:
        decision = "KILL: oracle bandwidth ceiling is below target tokens/sec."
    elif one_layer_compute_ms is None:
        decision = "HOLD: bandwidth passes, but one-layer compute time is required for hideability."
    else:
        compute_ns = one_layer_compute_ms * 1_000_000.0
        hideability = max_latency_ns / compute_ns if compute_ns > 0 else 0.0
        decision = (
            "GO: bandwidth passes and hideability <= 1."
            if hideability <= 1.0
            else "FLAG: bandwidth passes but one-layer lead time is insufficient."
        )

    summary = {
        "inventory": inventory,
        "thresholds": thresholds,
        "config": payload.get("config", {}),
        "cells": cells,
        "oracle_bandwidth_ceiling_tokens_per_second": ceiling_tps,
        "target_tokens_per_second": target_tps,
        "one_layer_compute_ms": one_layer_compute_ms,
        "hideability_fetch_latency_p95_ns": max_latency_ns if one_layer_compute_ms else None,
        "hideability_ratio": hideability,
        "compute": {
            "path": str(args.compute) if args.compute else "",
            "benchmark": compute_payload.get("benchmark", "") if compute_payload else "",
            "latency_p50_ms": compute_payload.get("latency_p50_ms") if compute_payload else None,
            "latency_p95_ms": compute_payload.get("latency_p95_ms") if compute_payload else None,
            "config": compute_payload.get("config", {}) if compute_payload else {},
            "model_shape": compute_payload.get("model_shape", {}) if compute_payload else {},
        },
        "decision": decision,
    }
    output = args.output or args.input.parent / "summary.json"
    write_json(output, summary)

    rows = "\n".join(
        "| {pattern} | {bandwidth_gbps:.3f} | {latency_p50_ns:.0f} | {latency_p95_ns:.0f} | {has_fs_usage} |".format(
            **cell
        )
        for cell in cells
    )
    config = payload.get("config", {})
    powermetrics_status = (
        f"captured at `{config['powermetrics_path']}`"
        if config.get("powermetrics_path")
        else config.get("powermetrics_error", "not captured or no error recorded")
    )
    fs_usage_status = (
        "missing for at least one accepted read cell. This run is not\n"
        "  valid SSD proof."
        if fs_usage_missing
        else "present for every accepted read cell."
    )
    if one_layer_compute_ms is None:
        compute_section = "One-layer compute: not measured."
    else:
        compute_section = (
            f"One-layer compute p50: `{one_layer_compute_ms:.6f}` ms"
            + (
                f" (`{args.compute}`)." if args.compute else "."
            )
            + f"\nHideability ratio (max read p95 / compute p50): `{hideability:.6f}`."
        )
    valid_rerun = ""
    if fs_usage_missing:
        output_dir = args.input.parent
        valid_rerun = f"""
### Valid Rerun Path

Run Stage 0 from a terminal that can accept a sudo password prompt:

```bash
sudo -v
python scripts/moe_prefetch/run_stage0_envelope.py \\
  --thresholds outputs/moe_prefetch/stage0/thresholds.json \\
  --output-dir {output_dir} \\
  --fs-usage-sudo-mode interactive \\
  --capture-powermetrics
python scripts/moe_prefetch/summarize.py stage0 \\
  --input {args.input} \\
  --notes {args.notes}
```

Do not proceed to Stage 1 unless the summary reports `fs_usage proof` as true
for every accepted read cell.
"""
    stage0_markdown = f"""## Stage 0: Hardware Envelope

**Status:** Complete.

### Frozen Assumptions

- Model inventory: `outputs/moe_prefetch/stage0/model_inventory.json`
- Gate thresholds: `outputs/moe_prefetch/stage0/thresholds.json`
- Model: `{inventory["model_id"]}`
- Expert bytes: `{inventory["expert_bytes"]}`
- Active expert bytes/token: `{inventory["active_expert_bytes_per_token"]}`
- Target decode rate: `{target_tps}` token/sec

### Measurements

| Pattern | Bandwidth GB/s | p50 latency ns | p95 latency ns | fs_usage proof |
| --- | ---: | ---: | ---: | --- |
{rows}

Oracle bandwidth ceiling: `{ceiling_tps:.6f}` tokens/sec.
{compute_section}

### Privileged Capture Status

- `fs_usage`: {fs_usage_status}
- `powermetrics`: {powermetrics_status}

### Decision

{decision}
{valid_rerun}
"""
    args.notes.write_text(_replace_stage0_section(args.notes.read_text(), stage0_markdown))
    print(f"wrote {output}")
    print(f"updated {args.notes}")
    print(decision)
    return 0


def _parse_depths(depths_text: str) -> list[int]:
    depths = [int(part.strip()) for part in depths_text.split(",") if part.strip()]
    if not depths or any(depth <= 0 for depth in depths):
        raise SystemExit("--prefetch-depths must contain positive integers")
    return depths


def summarize_stage1(args: argparse.Namespace) -> int:
    rows = load_trace(args.trace)
    if not rows:
        raise SystemExit("stage1 trace is empty")
    stage0 = load_json(args.stage0)
    thresholds = stage0.get("thresholds", {})
    margin = 1.0 + (float(thresholds.get("trivial_margin_percent", 10.0)) / 100.0)
    hideability_ratio = float(stage0.get("hideability_ratio") or 0.0)
    required_depth = max(1, int(hideability_ratio + 0.999999))
    depths = _parse_depths(args.prefetch_depths)
    evaluations = {
        str(depth): evaluate_policies(rows, prefetch_depth=depth, required_depth=required_depth)
        for depth in depths
    }

    best_depth = max(
        depths,
        key=lambda depth: evaluations[str(depth)]["eam"]["hideable_recall"],
    )
    best = evaluations[str(best_depth)]
    trivial_policies = ("last_token", "frequency", "markov")
    best_trivial = max(best[policy]["hideable_recall"] for policy in trivial_policies)
    eam_hideable = best["eam"]["hideable_recall"]
    if max(depths) < required_depth:
        decision = (
            "KILL: planned prefetch depths cannot hide Stage 0 p95 expert reads."
        )
    elif eam_hideable >= best_trivial * margin and eam_hideable > 0.0:
        decision = "GO: EAM hideable recall clears the best trivial baseline."
    else:
        decision = "KILL: EAM hideable recall does not clear the best trivial baseline."

    domains = sorted({str(row["domain"]) for row in rows})
    model_ids = sorted({str(row["model_id"]) for row in rows})
    output = {
        "trace": str(args.trace),
        "stage0": str(args.stage0),
        "row_count": len(rows),
        "domains": domains,
        "model_ids": model_ids,
        "required_prefetch_depth": required_depth,
        "evaluated_prefetch_depths": depths,
        "evaluations": evaluations,
        "best_depth": best_depth,
        "best_trivial_hideable_recall": best_trivial,
        "eam_hideable_recall": eam_hideable,
        "decision": decision,
    }
    write_json(args.output, output)

    rows_md = "\n".join(
        "| {depth} | {policy} | {recall:.3f} | {precision:.3f} | {hideable_recall:.3f} | {wasted_prediction_fraction:.3f} |".format(
            depth=depth,
            policy=policy,
            **metrics,
        )
        for depth in depths
        for policy, metrics in evaluations[str(depth)].items()
    )
    stage1_markdown = f"""## Stage 1: Router Trace and Predictor Replay

**Status:** Complete.

### Trace

- Trace: `{args.trace}`
- Rows: `{len(rows)}`
- Model IDs: `{", ".join(model_ids)}`
- Domains: `{", ".join(domains)}`
- Required prefetch depth from Stage 0 hideability: `{required_depth}` layers
- Evaluated prefetch depths: `{", ".join(str(depth) for depth in depths)}`

### Predictor Replay

| Depth | Policy | Recall | Precision | Hideable recall | Wasted prediction fraction |
| ---: | --- | ---: | ---: | ---: | ---: |
{rows_md}

### Decision

{decision}
"""
    args.notes.write_text(_replace_stage1_section(args.notes.read_text(), stage1_markdown))
    print(f"wrote {args.output}")
    print(f"updated {args.notes}")
    print(decision)
    return 0


def main() -> int:
    args = _parse_args()
    if args.stage == "stage0":
        return summarize_stage0(args)
    if args.stage == "stage1":
        return summarize_stage1(args)
    raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
