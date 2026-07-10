#!/usr/bin/env python3
"""Estimate whether saved optimization probes can close frontier losses.

This script is intentionally conservative: a probe only applies to a frontier
cell when both the runtime bucket and hardware platform can be inferred and
matched exactly. The result is an estimate, not promotion proof, because most
probe reports time a sub-stack rather than the full end-to-end Config F path.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from summarize_optimization_candidates import collect_rows


DEFAULT_FRONTIER_JSON = Path("outputs/external_bakeoff/competitive_frontier.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/frontier_gap_candidates.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/frontier_gap_candidates.json")
BUCKET_RE = re.compile(r"(?<!\d)(3s|7s|10s|15s|30s)(?!\d)")


def infer_bucket(row: dict[str, Any]) -> str | None:
    """Infer runtime bucket from candidate label or report path."""

    haystack = " ".join(str(row.get(key) or "") for key in ("label", "path"))
    match = BUCKET_RE.search(haystack)
    return None if not match else match.group(1)


def infer_machine(row: dict[str, Any]) -> str | None:
    """Infer hardware platform from report path/label tokens."""

    haystack = " ".join(str(row.get(key) or "") for key in ("label", "path")).lower()
    normalized = haystack.replace("_", "-")
    if "iphone-12-pro" in normalized or "iphone12" in normalized:
        return "iphone-12-pro"
    if "irvine" in normalized or "irvine-m1" in normalized:
        return "irvine-m1"
    if "m2-air" in normalized or "m2air" in normalized:
        return "m2-air"
    if "m2-studio" in normalized or "m2studio" in normalized:
        return "m2-studio"
    return None


def candidate_delta_ms(row: dict[str, Any]) -> float | None:
    """Return measured probe savings in milliseconds when available."""

    baseline = row.get("baseline_ms")
    candidate = row.get("candidate_ms")
    if baseline is None or candidate is None:
        return None
    return float(baseline) - float(candidate)


def _load_frontier(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} is missing summary")
    return summary


def annotate_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    bucket = infer_bucket(row)
    machine = infer_machine(row)
    delta = candidate_delta_ms(row)
    if bucket is None or machine is None or delta is None:
        return None
    annotated = dict(row)
    annotated.update(
        {
            "inferred_bucket": bucket,
            "inferred_machine_id": machine,
            "delta_ms": delta,
        }
    )
    return annotated


def summarize_gap_candidates(
    frontier_summary: dict[str, Any], candidate_rows: list[dict[str, Any]], top_per_cell: int
) -> dict[str, Any]:
    """Return loss cells with exact-machine candidate closure estimates."""

    candidates = [item for row in candidate_rows if (item := annotate_candidate(row))]
    cells: list[dict[str, Any]] = []
    for cell in frontier_summary.get("config_f_losses") or []:
        machine = str(cell["machine_id"])
        bucket = str(cell["input_key"])
        config_ms = float(cell["config_f_warm_median_ms"])
        best_ms = float(cell["best_warm_median_ms"])
        required_ms = config_ms - best_ms
        matches = [
            row
            for row in candidates
            if row["inferred_machine_id"] == machine and row["inferred_bucket"] == bucket
        ]
        matches.sort(key=lambda row: float(row["delta_ms"]), reverse=True)
        estimates: list[dict[str, Any]] = []
        for row in matches[:top_per_cell]:
            estimated_config_ms = config_ms - float(row["delta_ms"])
            estimates.append(
                {
                    "label": row["label"],
                    "family": row["family"],
                    "path": row["path"],
                    "passes": bool(row["passes"]),
                    "quality_status": "strict-pass" if row["passes"] else "quality-fail",
                    "baseline_ms": row["baseline_ms"],
                    "candidate_ms": row["candidate_ms"],
                    "delta_ms": row["delta_ms"],
                    "speedup_pct": row["speedup_pct"],
                    "corr": row.get("corr"),
                    "snr_db": row.get("snr_db"),
                    "estimated_config_f_ms": estimated_config_ms,
                    "estimated_margin_ms": best_ms - estimated_config_ms,
                    "would_close_gap": estimated_config_ms < best_ms,
                }
            )
        cells.append(
            {
                "machine_id": machine,
                "input_key": bucket,
                "best_impl_label": cell["best_impl_label"],
                "best_warm_median_ms": best_ms,
                "config_f_warm_median_ms": config_ms,
                "required_reduction_ms": required_ms,
                "required_reduction_pct": 100.0 * required_ms / config_ms,
                "candidate_count": len(matches),
                "strict_pass_closers": len(
                    [row for row in estimates if row["passes"] and row["would_close_gap"]]
                ),
                "quality_fail_closers": len(
                    [row for row in estimates if not row["passes"] and row["would_close_gap"]]
                ),
                "top_candidates": estimates,
            }
        )
    return {
        "loss_cells": cells,
        "loss_cell_count": len(cells),
        "candidate_rows_with_exact_machine_bucket": len(candidates),
        "strict_pass_closers": sum(cell["strict_pass_closers"] for cell in cells),
        "quality_fail_closers": sum(cell["quality_fail_closers"] for cell in cells),
    }


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the gap decomposition as Markdown."""

    lines = [
        "# Frontier Gap Candidates",
        "",
        "Warmed inference only. Candidate savings are exact-machine, exact-bucket",
        "sub-stack substitution estimates; they are not full-path promotion proof.",
        "",
        f"Loss cells analyzed: `{summary['loss_cell_count']}`.",
        f"Strict-pass candidates that would close a loss: `{summary['strict_pass_closers']}`.",
        f"Quality-fail candidates that would close a loss: `{summary['quality_fail_closers']}`.",
        "",
        "| Machine | Bucket | Fastest impl | Required reduction | Exact-machine candidates | Best strict-pass estimate | Best quality-fail estimate |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for cell in summary["loss_cells"]:
        strict = next((row for row in cell["top_candidates"] if row["passes"]), None)
        fail = next((row for row in cell["top_candidates"] if not row["passes"]), None)
        strict_text = "none"
        if strict:
            strict_text = (
                f"`{strict['label']}` -> {_fmt_ms(strict['estimated_config_f_ms'])} "
                f"({'closes' if strict['would_close_gap'] else 'short'})"
            )
        fail_text = "none"
        if fail:
            fail_text = (
                f"`{fail['label']}` -> {_fmt_ms(fail['estimated_config_f_ms'])} "
                f"({'closes' if fail['would_close_gap'] else 'short'})"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    cell["machine_id"],
                    cell["input_key"],
                    str(cell["best_impl_label"]),
                    f"{_fmt_ms(cell['required_reduction_ms'])} / {_fmt_pct(cell['required_reduction_pct'])}",
                    str(cell["candidate_count"]),
                    strict_text,
                    fail_text,
                ]
            )
            + " |"
        )

    for cell in summary["loss_cells"]:
        if not cell["top_candidates"]:
            continue
        lines.extend(
            [
                "",
                f"## {cell['machine_id']} {cell['input_key']}",
                "",
                "| Quality | Candidate | Delta | Estimated Config F | Margin vs fastest | Corr | SNR dB | Report |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in cell["top_candidates"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["quality_status"],
                        f"`{row['label']}`",
                        _fmt_ms(row["delta_ms"]),
                        _fmt_ms(row["estimated_config_f_ms"]),
                        _fmt_ms(row["estimated_margin_ms"]),
                        "n/a" if row["corr"] is None else f"{float(row['corr']):.6f}",
                        "n/a" if row["snr_db"] is None else f"{float(row['snr_db']):.2f}",
                        f"`{row['path']}`",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER_JSON)
    parser.add_argument("--report-glob", action="append", default=["outputs/**/report*.json"])
    parser.add_argument("--top-per-cell", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    frontier = _load_frontier(args.frontier_json)
    rows = collect_rows(args.report_glob)
    summary = summarize_gap_candidates(frontier, rows, args.top_per_cell)
    markdown = render_markdown(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "loss_cell_count": summary["loss_cell_count"],
                "strict_pass_closers": summary["strict_pass_closers"],
                "quality_fail_closers": summary["quality_fail_closers"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
