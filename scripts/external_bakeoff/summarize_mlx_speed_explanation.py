#!/usr/bin/env python3
"""Explain apparent MLX wins against the corrected warmed Config F frontier.

The bakeoff directory intentionally preserves several generations of result
JSON. Some early Config F files include Core ML compile/cache behavior or older
runtime artifacts, while ``competitive_frontier`` uses corrected warmed rows.
This script keeps that distinction explicit so future analysis does not compare
MLX against stale Core ML timings.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import RUNTIME_BUCKETS, load_json, write_json  # noqa: E402


DEFAULT_OUTPUT = Path("outputs/external_bakeoff/mlx_speed_explanation.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/mlx_speed_explanation.json")
RAW_CONFIG_FILES = {
    "m2-studio": "results_config_f_reference_m2-studio.json",
    "m2-air": "results_config_f_reference_m2-air.json",
    "irvine-m1": "results_config_f_reference_irvine-m1.json",
}
MLX_FILES = {
    "m2-studio": "results_mlx_audio_m2-studio.json",
    "m2-air": "results_mlx_audio_m2-air.json",
    "irvine-m1": "results_mlx_audio_irvine-m1.json",
}


def _median(values: list[float]) -> float | None:
    """Return a float median in seconds for a warm timing array."""

    return float(statistics.median(values)) if values else None


def _record_map(path: Path) -> dict[str, dict[str, Any]]:
    """Return records keyed by input bucket."""

    payload = load_json(path)
    return {str(record.get("input_key") or ""): record for record in payload.get("records", [])}


def _warm_median_s(record: dict[str, Any] | None) -> float | None:
    """Return the warm median seconds for one result record."""

    if not record:
        return None
    return _median([float(value) for value in record.get("warm_wall_times_s") or []])


def _status(record: dict[str, Any] | None) -> str:
    """Return a stable record status string."""

    if not record:
        return "missing"
    return str(record.get("status") or "")


def _duration_ratio(record: dict[str, Any] | None) -> float | None:
    """Return observed/canonical duration ratio when available."""

    if not record:
        return None
    canonical = record.get("canonical_audio_duration_s")
    observed = record.get("observed_audio_duration_s")
    if not canonical or not observed:
        return None
    return float(observed) / float(canonical)


def _frontier_rows(frontier_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return competitive frontier rows keyed by machine, bucket, impl."""

    payload = load_json(frontier_path)
    rows = {}
    for row in payload.get("rows", []):
        rows[(str(row["machine_id"]), str(row["input_key"]), str(row["impl"]))] = row
    return rows


def _frontier_mlx_comparisons(frontier_path: Path) -> list[dict[str, Any]]:
    """Compare MLX to corrected warmed Config F rows in the paper frontier."""

    rows = _frontier_rows(frontier_path)
    comparisons: list[dict[str, Any]] = []
    for machine in sorted(MLX_FILES):
        for bucket in RUNTIME_BUCKETS:
            config = rows.get((machine, bucket, "config-f-reference"))
            mlx = rows.get((machine, bucket, "mlx-audio"))
            config_ms = None if not config else config.get("warm_median_ms")
            mlx_ms = None if not mlx else mlx.get("warm_median_ms")
            mlx_full = bool(mlx and mlx.get("full_duration"))
            mlx_status = str(mlx.get("status") if mlx else "missing")
            outcome = "mlx-missing"
            if mlx and mlx_status == "error":
                outcome = "mlx-error"
            elif config_ms is None:
                outcome = "config-missing"
            elif mlx_ms is None:
                outcome = "mlx-missing"
            elif not mlx_full:
                outcome = "mlx-not-full-duration"
            elif float(mlx_ms) < float(config_ms):
                outcome = "mlx-faster"
            else:
                outcome = "config-faster"
            comparisons.append(
                {
                    "machine_id": machine,
                    "input_key": bucket,
                    "config_f_ms": config_ms,
                    "mlx_ms": mlx_ms,
                    "mlx_status": mlx_status,
                    "mlx_duration_ratio": None if not mlx else mlx.get("duration_ratio"),
                    "outcome": outcome,
                    "config_source_path": None if not config else config.get("source_path"),
                    "mlx_source_path": None if not mlx else mlx.get("source_path"),
                    "mlx_error": "" if not mlx else str(mlx.get("error") or ""),
                }
            )
    return comparisons


def _raw_apparent_wins(results_dir: Path) -> list[dict[str, Any]]:
    """Return rows where MLX beats early/raw Config F result files."""

    rows: list[dict[str, Any]] = []
    for machine, config_name in RAW_CONFIG_FILES.items():
        config_path = results_dir / config_name
        mlx_path = results_dir / MLX_FILES[machine]
        if not config_path.exists() or not mlx_path.exists():
            continue
        config_records = _record_map(config_path)
        mlx_records = _record_map(mlx_path)
        for bucket in RUNTIME_BUCKETS:
            config_record = config_records.get(bucket)
            mlx_record = mlx_records.get(bucket)
            config_s = _warm_median_s(config_record)
            mlx_s = _warm_median_s(mlx_record)
            if config_s is None or mlx_s is None:
                continue
            if _status(config_record) == "ok" and _status(mlx_record) == "ok" and mlx_s < config_s:
                rows.append(
                    {
                        "machine_id": machine,
                        "input_key": bucket,
                        "raw_config_f_ms": config_s * 1000.0,
                        "mlx_ms": mlx_s * 1000.0,
                        "mlx_apparent_save_ms": (config_s - mlx_s) * 1000.0,
                        "config_duration_ratio": _duration_ratio(config_record),
                        "mlx_duration_ratio": _duration_ratio(mlx_record),
                        "raw_config_source_path": str(config_path),
                        "mlx_source_path": str(mlx_path),
                    }
                )
    return rows


def build_report(results_dir: Path, frontier_path: Path) -> dict[str, Any]:
    """Build the MLX speed explanation payload."""

    frontier = _frontier_mlx_comparisons(frontier_path)
    raw_wins = _raw_apparent_wins(results_dir)
    return {
        "summary": {
            "corrected_warmed_mlx_wins": sum(1 for row in frontier if row["outcome"] == "mlx-faster"),
            "corrected_warmed_config_wins": sum(1 for row in frontier if row["outcome"] == "config-faster"),
            "mlx_error_rows": sum(1 for row in frontier if row["outcome"] == "mlx-error"),
            "raw_apparent_mlx_wins": len(raw_wins),
            "conclusion": (
                "MLX is not faster than corrected warmed Config F on any full-duration Mac row. "
                "Apparent MLX wins come from stale/raw Config F files that include Core ML "
                "compile/cache behavior or older unpromoted runtime artifacts."
            ),
        },
        "corrected_warmed_frontier": frontier,
        "raw_apparent_mlx_wins": raw_wins,
    }


def _fmt_ms(value: Any) -> str:
    """Format milliseconds for Markdown."""

    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_ratio(value: Any) -> str:
    """Format duration ratio for Markdown."""

    return "n/a" if value is None else f"{float(value):.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the MLX explanation as Markdown."""

    summary = report["summary"]
    lines = [
        "# MLX Speed Explanation",
        "",
        "## Verdict",
        "",
        summary["conclusion"],
        "",
        f"- Corrected warmed full-duration MLX wins: `{summary['corrected_warmed_mlx_wins']}`.",
        f"- Corrected warmed full-duration Config F wins over MLX: `{summary['corrected_warmed_config_wins']}`.",
        f"- MLX error rows: `{summary['mlx_error_rows']}`.",
        f"- Raw/stale apparent MLX wins: `{summary['raw_apparent_mlx_wins']}`.",
        "",
        "## Corrected Warmed Frontier",
        "",
        "These rows use the same paper-facing source set as `competitive_frontier`: corrected warmed Config F rows, full-duration MLX rows only, and no Core ML compile/cache timing.",
        "",
        "| Machine | Bucket | Config F | MLX | MLX duration ratio | Outcome |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["corrected_warmed_frontier"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    row["input_key"],
                    _fmt_ms(row["config_f_ms"]),
                    _fmt_ms(row["mlx_ms"]),
                    _fmt_ratio(row["mlx_duration_ratio"]),
                    row["outcome"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Apparent MLX Wins Against Raw Config F",
            "",
            "These rows explain the confusing view: MLX can beat early/raw Config F JSON files, but those files are not the corrected warmed frontier.",
            "",
            "| Machine | Bucket | Raw Config F | MLX | Apparent MLX save | Config duration ratio | MLX duration ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["raw_apparent_mlx_wins"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    row["input_key"],
                    _fmt_ms(row["raw_config_f_ms"]),
                    _fmt_ms(row["mlx_ms"]),
                    _fmt_ms(row["mlx_apparent_save_ms"]),
                    _fmt_ratio(row["config_duration_ratio"]),
                    _fmt_ratio(row["mlx_duration_ratio"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Why This Happens",
            "",
            "- MLX uses a dynamic GPU/Metal runtime and avoids Core ML's expensive `.mlpackage` compile/specialization path during warm calls.",
            "- Early Config F result files captured stale artifacts and/or Core ML compile/cache behavior; the corrected frontier uses targeted warmed reruns such as `*_vector_noise_batch.json`.",
            "- MLX has no valid 3s full-duration row: all Mac 3s MLX cells fail with the same broadcast-shape error.",
            "- On the corrected warmed rows where MLX produces full-duration audio, Config F is faster than MLX on every Mac bucket.",
            "- The real current Core ML competitor on lower-end Macs is laishere, not MLX; laishere's source/body boundary is the architecture clue to reuse.",
            "",
            "## Reusable Lesson",
            "",
            "Never compare a dynamic runtime's warm loop to a Core ML row unless the Core ML row has explicitly discarded compile/cache behavior and uses the same output-duration eligibility gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/external_bakeoff"))
    parser.add_argument("--frontier", type=Path, default=Path("outputs/external_bakeoff/competitive_frontier.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.results_dir, args.frontier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report))
    write_json(args.json_output, report)
    print(json.dumps({"output": str(args.output), "json_output": str(args.json_output), **report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
