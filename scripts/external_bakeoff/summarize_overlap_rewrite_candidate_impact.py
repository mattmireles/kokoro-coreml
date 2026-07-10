#!/usr/bin/env python3
"""Summarize the combined HnSF-overlap plus HAR-post rewrite candidate.

This is a measurement ledger and projection helper, not a frontier updater. It
keeps the strict runtime overlap, the HAR-post graph rewrite, and their combined
local wall-time behavior in one place so quiet lower-end-device runs have a
single source of truth.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_SERIAL = DEFAULT_OUTPUT_DIR / "results_config_f_reference_m2-studio-local_serial_hnsf_all_buckets.json"
DEFAULT_OVERLAP = DEFAULT_OUTPUT_DIR / "results_config_f_reference_m2-studio-local_overlap_baseline_all_buckets.json"
DEFAULT_COMBINED = DEFAULT_OUTPUT_DIR / "results_config_f_reference_m2-studio-local_overlap_rewrite_overlay_all_buckets.json"
DEFAULT_PACKAGE_REPORT = Path("outputs/export_rewrite_smoke/report_all_buckets_cpu_gpu.json")
DEFAULT_STAGE_GAPS = DEFAULT_OUTPUT_DIR / "stage_gap_decomposition.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "overlap_rewrite_candidate_impact.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "overlap_rewrite_candidate_impact.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _median(values: list[float]) -> float:
    """Return the median from a non-empty list."""

    if not values:
        raise ValueError("expected at least one value")
    return float(statistics.median(values))


def _warm_result_rows(payload: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    """Map bucket to warm wall/stage medians from a result-file payload."""

    rows: dict[str, dict[str, float | str]] = {}
    for record in payload.get("records") or []:
        if record.get("status") != "ok":
            continue
        bucket = str(record["input_key"])
        raw_warm = record.get("provenance", {}).get("raw_warm_results") or []
        if not raw_warm:
            raise ValueError(f"{bucket} missing provenance.raw_warm_results")

        def stage_ms(key: str) -> float:
            return 1000.0 * _median([float(row.get(key, 0.0) or 0.0) for row in raw_warm])

        rows[bucket] = {
            "wall_ms": 1000.0 * _median([float(value) for value in record["warm_wall_times_s"]]),
            "generator_ms": stage_ms("t_coreml_predict_s"),
            "decoder_pre_ms": stage_ms("t_decoder_pre_coreml_s"),
            "hnsf_ms": stage_ms("t_hnsf_swift_s"),
            "decoder_pre_hnsf_overlap_ms": stage_ms("t_decoder_pre_hnsf_overlap_s"),
            "output_sha256": str(record.get("output_sha256") or ""),
        }
    return rows


def _package_speedups(report: dict[str, Any]) -> dict[str, float]:
    """Map bucket to measured HAR-post package speedup percentage."""

    speedups: dict[str, float] = {}
    for row in report.get("rows") or []:
        speedups[str(row["bucket"])] = float(row["speedup_vs_baseline_pct"])
    return speedups


def _compare_rows(
    before: dict[str, dict[str, float | str]],
    after: dict[str, dict[str, float | str]],
) -> list[dict[str, Any]]:
    """Compare two bucket-indexed local benchmark summaries."""

    rows: list[dict[str, Any]] = []
    for bucket in sorted(set(before) & set(after), key=lambda item: int(item.rstrip("s"))):
        before_ms = float(before[bucket]["wall_ms"])
        after_ms = float(after[bucket]["wall_ms"])
        save_ms = before_ms - after_ms
        rows.append(
            {
                "bucket": bucket,
                "before_ms": before_ms,
                "after_ms": after_ms,
                "save_ms": save_ms,
                "speedup_pct": 100.0 * save_ms / before_ms if before_ms else 0.0,
                "before_generator_ms": float(before[bucket]["generator_ms"]),
                "after_generator_ms": float(after[bucket]["generator_ms"]),
                "after_overlap_ms": float(after[bucket]["decoder_pre_hnsf_overlap_ms"]),
                "hash_identical": before[bucket]["output_sha256"] == after[bucket]["output_sha256"],
            }
        )
    return rows


def _projection_rows(
    stage_gaps: dict[str, Any],
    package_speedups: dict[str, float],
) -> list[dict[str, Any]]:
    """Project overlap plus rewrite savings onto lower-end stage decompositions."""

    rows: list[dict[str, Any]] = []
    for row in stage_gaps.get("rows") or []:
        bucket = str(row["input_key"])
        rewrite_speedup_pct = package_speedups.get(bucket)
        if rewrite_speedup_pct is None:
            continue
        config = row["config"]
        current_overlap_ms = 1000.0 * float(config.get("decoder_pre_hnsf_overlap_s") or 0.0)
        possible_overlap_ms = 1000.0 * min(float(config["decoder_pre_s"]), float(config["hnsf_s"]))
        projected_overlap_save_ms = max(0.0, possible_overlap_ms - current_overlap_ms)
        generator_ms = 1000.0 * float(config["generator_s"])
        projected_rewrite_save_ms = generator_ms * rewrite_speedup_pct / 100.0
        current_total_ms = 1000.0 * float(config["total_s"])
        projected_total_ms = current_total_ms - projected_overlap_save_ms - projected_rewrite_save_ms
        laishere_ms = 1000.0 * float(row["laishere"]["total_s"])
        frontier_best_ms = float(row["frontier_best_ms"])
        rows.append(
            {
                "machine_id": str(row["machine_id"]),
                "bucket": bucket,
                "current_total_ms": current_total_ms,
                "generator_ms": generator_ms,
                "rewrite_speedup_pct": rewrite_speedup_pct,
                "projected_rewrite_save_ms": projected_rewrite_save_ms,
                "projected_overlap_save_ms": projected_overlap_save_ms,
                "projected_total_ms": projected_total_ms,
                "laishere_ms": laishere_ms,
                "gap_vs_laishere_ms": projected_total_ms - laishere_ms,
                "frontier_best_ms": frontier_best_ms,
                "gap_vs_frontier_ms": projected_total_ms - frontier_best_ms,
                "closes_profile_gap": projected_total_ms <= laishere_ms,
                "closes_frontier_gap": projected_total_ms <= frontier_best_ms,
            }
        )
    rows.sort(key=lambda item: (item["machine_id"], int(item["bucket"].rstrip("s"))))
    return rows


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Build the combined-candidate summary payload."""

    serial = _warm_result_rows(_load_json(args.serial_results))
    overlap = _warm_result_rows(_load_json(args.overlap_results))
    combined = _warm_result_rows(_load_json(args.combined_results))
    package_speedups = _package_speedups(_load_json(args.package_report))
    projection_rows = _projection_rows(_load_json(args.stage_gaps), package_speedups)
    irvine_rows = [row for row in projection_rows if row["machine_id"] == "irvine-m1"]
    return {
        "serial_results": str(args.serial_results),
        "overlap_results": str(args.overlap_results),
        "combined_results": str(args.combined_results),
        "package_report": str(args.package_report),
        "stage_gaps": str(args.stage_gaps),
        "local_overlap_rows": _compare_rows(serial, overlap),
        "local_combined_rows": _compare_rows(serial, combined),
        "projection_rows": projection_rows,
        "summary": {
            "local_overlap_positive_buckets": sum(
                1 for row in _compare_rows(serial, overlap) if row["save_ms"] > 0
            ),
            "local_combined_positive_buckets": sum(
                1 for row in _compare_rows(serial, combined) if row["save_ms"] > 0
            ),
            "projected_profile_gaps_closed": sum(1 for row in projection_rows if row["closes_profile_gap"]),
            "projected_frontier_gaps_closed": sum(1 for row in projection_rows if row["closes_frontier_gap"]),
            "irvine_profile_gaps_closed": sum(1 for row in irvine_rows if row["closes_profile_gap"]),
            "irvine_frontier_gaps_closed": sum(1 for row in irvine_rows if row["closes_frontier_gap"]),
        },
    }


def _fmt_ms(value: float) -> str:
    """Format milliseconds for Markdown."""

    return f"{value:.1f} ms"


def _fmt_pct(value: float) -> str:
    """Format a percent for Markdown."""

    return f"{value:.2f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the combined-candidate summary as Markdown."""

    lines = [
        "# Overlap + Rewrite Candidate Impact",
        "",
        "This is a warmed-inference measurement ledger, not a frontier update. The",
        "runtime overlap is strict and hash-identical locally. The HAR-post rewrite",
        "changes the generator package and therefore changes waveform hashes; keep",
        "using the package-level parity/correlation reports for that quality gate.",
        "",
        "## Local Overlap Only",
        "",
        "| Bucket | Serial shipped | Overlap shipped | Save | Speedup | Measured overlap | Hash |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["local_overlap_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["before_ms"]),
                    _fmt_ms(row["after_ms"]),
                    _fmt_ms(row["save_ms"]),
                    _fmt_pct(row["speedup_pct"]),
                    _fmt_ms(row["after_overlap_ms"]),
                    "`identical`" if row["hash_identical"] else "`changed`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Local Overlap + Rewrite Overlay",
            "",
            "| Bucket | Serial shipped | Overlap + rewrite | Save | Speedup | Generator before/after | Hash |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["local_combined_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["before_ms"]),
                    _fmt_ms(row["after_ms"]),
                    _fmt_ms(row["save_ms"]),
                    _fmt_pct(row["speedup_pct"]),
                    f"{_fmt_ms(row['before_generator_ms'])} -> {_fmt_ms(row['after_generator_ms'])}",
                    "`identical`" if row["hash_identical"] else "`changed`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Lower-End Projection",
            "",
            "Projection applies only two independent strict/measured effects to existing",
            "stage medians: HAR-post package speedup on the generator stage and",
            "`min(decoder_pre, hnsf)` for the new overlap. It assumes no other stage",
            "changes and remains non-publishable until quiet-device timing confirms it.",
            "",
            "| Machine | Bucket | Current Config F | Overlap save | Rewrite save | Projected Config F | laishere | Gap | Frontier gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["projection_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['machine_id']}`",
                    f"`{row['bucket']}`",
                    _fmt_ms(row["current_total_ms"]),
                    _fmt_ms(row["projected_overlap_save_ms"]),
                    _fmt_ms(row["projected_rewrite_save_ms"]),
                    _fmt_ms(row["projected_total_ms"]),
                    _fmt_ms(row["laishere_ms"]),
                    _fmt_ms(row["gap_vs_laishere_ms"]),
                    _fmt_ms(row["gap_vs_frontier_ms"]),
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
            f"- Local overlap positive buckets: `{s['local_overlap_positive_buckets']}`.",
            f"- Local combined positive buckets: `{s['local_combined_positive_buckets']}`.",
            f"- Projected lower-end profile gaps closed: `{s['projected_profile_gaps_closed']}`.",
            f"- Projected Irvine profile gaps closed: `{s['irvine_profile_gaps_closed']}`.",
            "- Decision: keep both candidates. The overlap is low-risk and strict; the",
            "  rewrite is a generator-stage win but still needs quiet lower-end proof.",
            "  The combined projection is useful for target sizing, not for paper rows.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-results", type=Path, default=DEFAULT_SERIAL)
    parser.add_argument("--overlap-results", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--combined-results", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--package-report", type=Path, default=DEFAULT_PACKAGE_REPORT)
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
                "output": str(args.output),
                "local_combined_positive_buckets": summary["summary"]["local_combined_positive_buckets"],
                "irvine_profile_gaps_closed": summary["summary"]["irvine_profile_gaps_closed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
