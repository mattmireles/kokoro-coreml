#!/usr/bin/env python3
"""Summarize the frontier impact of the HAR-post upsample rewrite candidate.

This is a projection helper, not a replacement for real remote timing. It uses
measured local package speedups to estimate how much of each lower-end Mac
source/body gap the candidate could close, then keeps the candidate out of the
paper frontier until a quiet-device run proves it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_PACKAGE_REPORT = Path("outputs/export_rewrite_smoke/report_all_buckets_cpu_gpu.json")
DEFAULT_LOCAL_BASELINE = DEFAULT_OUTPUT_DIR / "results_config_f_reference_m2-studio-local_vector_noise_batch.json"
DEFAULT_LOCAL_CANDIDATE = DEFAULT_OUTPUT_DIR / "results_config_f_reference_m2-studio-local_rewrite_ups_as_conv.json"
DEFAULT_STAGE_GAPS = DEFAULT_OUTPUT_DIR / "stage_gap_decomposition.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "rewrite_candidate_impact.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "rewrite_candidate_impact.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _median_ms(record: dict[str, Any]) -> float:
    """Return the warm median latency in milliseconds for one result record."""

    values = record.get("warm_wall_times_s")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{record.get('input_key', '<unknown>')} missing warm_wall_times_s")
    return float(statistics.median(float(value) for value in values) * 1000.0)


def _package_speedups(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Map runtime bucket to measured package-level rewrite speedup."""

    rows: dict[str, dict[str, float]] = {}
    for row in report.get("rows") or []:
        bucket = str(row["bucket"])
        med = row["warm_predict_median_ms"]
        rows[bucket] = {
            "baseline_ms": float(med["baseline"]),
            "candidate_ms": float(med["candidate"]),
            "speedup_pct": float(row["speedup_vs_baseline_pct"]),
        }
    return rows


def _local_end_to_end_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare local end-to-end baseline and rewrite candidate medians."""

    base_records = {str(row["input_key"]): row for row in baseline.get("records") or [] if row.get("status") == "ok"}
    cand_records = {str(row["input_key"]): row for row in candidate.get("records") or [] if row.get("status") == "ok"}
    rows: list[dict[str, Any]] = []
    for bucket in sorted(cand_records, key=lambda item: int(item.rstrip("s"))):
        if bucket not in base_records:
            continue
        baseline_ms = _median_ms(base_records[bucket])
        candidate_ms = _median_ms(cand_records[bucket])
        rows.append(
            {
                "bucket": bucket,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "delta_ms": candidate_ms - baseline_ms,
                "speedup_pct": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
            }
        )
    return rows


def _projection_rows(stage_gaps: dict[str, Any], package_speedups: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Project package speedups onto lower-end Mac stage decompositions."""

    rows: list[dict[str, Any]] = []
    for row in stage_gaps.get("rows") or []:
        bucket = str(row["input_key"])
        speed = package_speedups.get(bucket)
        if speed is None:
            continue
        config = row["config"]
        laishere = row["laishere"]
        config_total_ms = float(config["total_s"]) * 1000.0
        config_generator_ms = float(config["generator_s"]) * 1000.0
        laishere_total_ms = float(laishere["total_s"]) * 1000.0
        projected_save_ms = config_generator_ms * speed["speedup_pct"] / 100.0
        projected_total_ms = config_total_ms - projected_save_ms
        projected_gap_ms = projected_total_ms - laishere_total_ms
        frontier_best_ms = float(row["frontier_best_ms"])
        rows.append(
            {
                "machine_id": str(row["machine_id"]),
                "bucket": bucket,
                "package_speedup_pct": speed["speedup_pct"],
                "config_total_ms": config_total_ms,
                "config_generator_ms": config_generator_ms,
                "laishere_total_ms": laishere_total_ms,
                "projected_save_ms": projected_save_ms,
                "projected_total_ms": projected_total_ms,
                "projected_gap_ms": projected_gap_ms,
                "closes_profile_gap": projected_total_ms <= laishere_total_ms,
                "frontier_best_ms": frontier_best_ms,
                "projected_frontier_gap_ms": projected_total_ms - frontier_best_ms,
                "closes_paper_frontier": projected_total_ms <= frontier_best_ms,
            }
        )
    return rows


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Build the rewrite candidate impact summary payload."""

    package_speedups = _package_speedups(_load_json(args.package_report))
    local_rows = _local_end_to_end_rows(_load_json(args.local_baseline), _load_json(args.local_candidate))
    projection_rows = _projection_rows(_load_json(args.stage_gaps), package_speedups)
    irvine_rows = [row for row in projection_rows if row["machine_id"] == "irvine-m1"]
    return {
        "package_report": str(args.package_report),
        "local_baseline": str(args.local_baseline),
        "local_candidate": str(args.local_candidate),
        "stage_gaps": str(args.stage_gaps),
        "package_speedups": package_speedups,
        "local_end_to_end_rows": local_rows,
        "projection_rows": projection_rows,
        "summary": {
            "local_end_to_end_positive_buckets": sum(1 for row in local_rows if row["speedup_pct"] > 0),
            "projection_closes_profile_gap_count": sum(1 for row in projection_rows if row["closes_profile_gap"]),
            "projection_closes_paper_frontier_count": sum(1 for row in projection_rows if row["closes_paper_frontier"]),
            "irvine_projection_closes_profile_gap_count": sum(1 for row in irvine_rows if row["closes_profile_gap"]),
            "irvine_projection_closes_paper_frontier_count": sum(1 for row in irvine_rows if row["closes_paper_frontier"]),
        },
    }


def _fmt_ms(value: float) -> str:
    """Format milliseconds for Markdown."""

    return f"{value:.1f} ms"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a Markdown summary of candidate impact."""

    lines = [
        "# Rewrite Candidate Impact",
        "",
        "The HAR-post upsample rewrite is a measured local win, but this report keeps",
        "it separate from the paper-facing frontier until quiet lower-end-device",
        "timing proves the projection.",
        "",
        "## Local End-to-End Proof",
        "",
        "| Bucket | Baseline | Rewrite | Delta | Speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["local_end_to_end_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["baseline_ms"]),
                    _fmt_ms(row["candidate_ms"]),
                    _fmt_ms(row["delta_ms"]),
                    f"{row['speedup_pct']:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Lower-End Projection",
            "",
            "Projection uses measured local package-level generator speedup applied only",
            "to each machine's current Config F generator stage. It does not assume",
            "non-generator stages improve.",
            "",
            "| Machine | Bucket | Generator | Package speedup | Projected save | Projected Config F | laishere | Gap after rewrite | Closes profile gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["projection_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['machine_id']}`",
                    f"`{row['bucket']}`",
                    _fmt_ms(row["config_generator_ms"]),
                    f"{row['package_speedup_pct']:.2f}%",
                    _fmt_ms(row["projected_save_ms"]),
                    _fmt_ms(row["projected_total_ms"]),
                    _fmt_ms(row["laishere_total_ms"]),
                    _fmt_ms(row["projected_gap_ms"]),
                    "`yes`" if row["closes_profile_gap"] else "`no`",
                ]
            )
            + " |"
        )
    s = summary["summary"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Local end-to-end positive buckets: `{s['local_end_to_end_positive_buckets']}`.",
            f"- Projected lower-end profile gaps closed: `{s['projection_closes_profile_gap_count']}`.",
            f"- Projected Irvine profile gaps closed: `{s['irvine_projection_closes_profile_gap_count']}`.",
            "- Decision: keep the rewrite candidate, but it is not sufficient alone to",
            "  prove absolute fastest on Irvine M1. The next strict win still needs",
            "  either quiet Irvine timing plus another source/body improvement, or a",
            "  stronger single-package graph change.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-report", type=Path, default=DEFAULT_PACKAGE_REPORT)
    parser.add_argument("--local-baseline", type=Path, default=DEFAULT_LOCAL_BASELINE)
    parser.add_argument("--local-candidate", type=Path, default=DEFAULT_LOCAL_CANDIDATE)
    parser.add_argument("--stage-gaps", type=Path, default=DEFAULT_STAGE_GAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary) + "\n")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "irvine_projection_closes_profile_gap_count": summary["summary"][
                    "irvine_projection_closes_profile_gap_count"
                ],
                "local_end_to_end_positive_buckets": summary["summary"]["local_end_to_end_positive_buckets"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
