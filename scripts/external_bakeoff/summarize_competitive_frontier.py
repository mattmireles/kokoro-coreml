#!/usr/bin/env python3
"""Summarize the current warmed-inference competitive frontier.

The external bakeoff directory contains historical and targeted probe result
files. This script defaults to the paper-facing corrected result files only,
then identifies the fastest full-duration implementation per machine/bucket and
the exact Config F win/loss state.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import RUNTIME_BUCKETS, load_json  # noqa: E402


DEFAULT_RESULT_FILES = (
    "results_config_f_reference_m2-studio-local_vector_noise_batch.json",
    "results_config_f_reference_m2-air_vector_noise_batch.json",
    "results_config_f_reference_irvine-m1_vector_noise_batch.json",
    "results_config_f_reference_ios_iphone-12-pro.json",
    "results_mlx_audio_m2-studio.json",
    "results_mlx_audio_m2-air.json",
    "results_mlx_audio_irvine-m1.json",
    "results_soniqo_speech_swift_kokoro_m2-studio.json",
    "results_soniqo_speech_swift_kokoro_m2-air.json",
    "results_soniqo_speech_swift_kokoro_irvine-m1.json",
    "results_soniqo_speech_swift_kokoro_ios_iphone-12-pro.json",
    "results_laishere_kokoro_coreml_m2-studio.json",
    "results_laishere_kokoro_coreml_m2-air.json",
    "results_laishere_kokoro_coreml_irvine-m1.json",
)
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/competitive_frontier.md")
CONFIG_F_IMPLS = {"config-f-reference", "config-f-reference-ios"}


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _impl_label(impl: str) -> str:
    labels = {
        "config-f-reference": "Config F",
        "config-f-reference-ios": "Config F iOS",
        "mlx-audio": "MLX",
        "soniqo-speech-swift-kokoro": "Soniqo",
        "soniqo-speech-swift-kokoro-ios": "Soniqo iOS",
        "laishere-kokoro-coreml": "laishere",
    }
    return labels.get(impl, impl)


def _machine_id(raw: str) -> str:
    """Normalize targeted rerun machine IDs back to hardware platform IDs."""

    suffixes = (
        "-vector-noise-batch",
        "-local-vector-noise-batch",
    )
    machine = raw
    for suffix in suffixes:
        if machine.endswith(suffix):
            machine = machine[: -len(suffix)]
    if machine == "m2-studio-local":
        return "m2-studio"
    return machine


def default_result_paths(results_dir: Path) -> list[Path]:
    """Return existing paper-facing result files in stable order."""

    return [path for name in DEFAULT_RESULT_FILES if (path := results_dir / name).exists()]


def load_records(paths: list[Path], min_duration_ratio: float) -> list[dict[str, Any]]:
    """Load flattened result rows with warm medians and duration eligibility."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        for record in payload.get("records", []):
            warm = [float(value) for value in record.get("warm_wall_times_s") or []]
            median_s = _median(warm)
            canonical = record.get("canonical_audio_duration_s")
            observed = record.get("observed_audio_duration_s")
            status = str(record.get("status") or "")
            duration_ratio = None
            if canonical and observed:
                duration_ratio = float(observed) / float(canonical)
            full_duration = bool(
                status == "ok"
                and median_s is not None
                and duration_ratio is not None
                and duration_ratio >= min_duration_ratio
            )
            machine_id = _machine_id(str(record.get("machine_id") or payload.get("machine_id") or ""))
            rows.append(
                {
                    "machine_id": machine_id,
                    "raw_machine_id": str(record.get("machine_id") or payload.get("machine_id") or ""),
                    "impl": str(record.get("impl") or payload.get("impl") or ""),
                    "impl_label": _impl_label(str(record.get("impl") or payload.get("impl") or "")),
                    "input_key": str(record.get("input_key") or ""),
                    "status": status,
                    "warm_median_s": median_s,
                    "warm_median_ms": None if median_s is None else median_s * 1000.0,
                    "canonical_audio_duration_s": canonical,
                    "observed_audio_duration_s": observed,
                    "duration_ratio": duration_ratio,
                    "full_duration": full_duration,
                    "source_path": str(path),
                    "error": record.get("error") or "",
                }
            )
    return rows


def summarize_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-machine/bucket fastest rows and Config F outcome."""

    machines = sorted({row["machine_id"] for row in rows if row["machine_id"]})
    cells: list[dict[str, Any]] = []
    for machine in machines:
        for key in RUNTIME_BUCKETS:
            candidates = [
                row
                for row in rows
                if row["machine_id"] == machine
                and row["input_key"] == key
                and row["full_duration"]
                and row["warm_median_s"] is not None
            ]
            candidates.sort(key=lambda row: float(row["warm_median_s"]))
            config = next((row for row in candidates if row["impl"] in CONFIG_F_IMPLS), None)
            best = candidates[0] if candidates else None
            next_best = candidates[1] if len(candidates) > 1 else None
            if not best:
                outcome = "no-full-duration-result"
                gap_pct = None
            elif not config:
                outcome = "config-f-missing"
                gap_pct = None
            elif best["impl"] in CONFIG_F_IMPLS:
                outcome = "config-f-wins"
                gap_pct = None
                if next_best:
                    gap_pct = 100.0 * (float(next_best["warm_median_s"]) / float(config["warm_median_s"]) - 1.0)
            else:
                outcome = "config-f-loses"
                gap_pct = 100.0 * (float(config["warm_median_s"]) / float(best["warm_median_s"]) - 1.0)
            cells.append(
                {
                    "machine_id": machine,
                    "input_key": key,
                    "best_impl": None if not best else best["impl"],
                    "best_impl_label": None if not best else best["impl_label"],
                    "best_warm_median_ms": None if not best else best["warm_median_ms"],
                    "config_f_warm_median_ms": None if not config else config["warm_median_ms"],
                    "outcome": outcome,
                    "gap_pct": gap_pct,
                    "eligible_impls": [row["impl_label"] for row in candidates],
                }
            )
    losing = [cell for cell in cells if cell["outcome"] == "config-f-loses"]
    missing = [cell for cell in cells if cell["outcome"] == "config-f-missing"]
    no_full = [cell for cell in cells if cell["outcome"] == "no-full-duration-result"]
    mac_cells = [cell for cell in cells if cell["machine_id"] != "iphone-12-pro"]
    return {
        "runtime_buckets": list(RUNTIME_BUCKETS),
        "cells": cells,
        "absolute_fastest_verified": not losing and not missing and not no_full,
        "mac_full_duration_cells": len(mac_cells),
        "mac_config_f_losses": len([cell for cell in mac_cells if cell["outcome"] == "config-f-loses"]),
        "config_f_losses": losing,
        "config_f_missing": missing,
        "no_full_duration_result": no_full,
    }


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_gap(cell: dict[str, Any]) -> str:
    gap = cell.get("gap_pct")
    if gap is None:
        return "n/a"
    if cell["outcome"] == "config-f-wins":
        return f"Config F ahead by {gap:.1f}%"
    if cell["outcome"] == "config-f-loses":
        return f"Config F needs {gap:.1f}%"
    return "n/a"


def render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], min_duration_ratio: float) -> str:
    """Render a paper-facing frontier table."""

    lines = [
        "# Competitive Frontier",
        "",
        "Warmed inference only. Rows are eligible for fastest-ranking only when",
        f"`observed_audio_duration_s / canonical_audio_duration_s >= {min_duration_ratio:.2f}`.",
        "This excludes short public-artifact outputs from full-bucket wins.",
        "",
        f"Absolute fastest verified: `{str(summary['absolute_fastest_verified']).lower()}`.",
        f"Mac full-duration Config F losses: `{summary['mac_config_f_losses']}`.",
        "",
        "| Machine | Bucket | Fastest full-duration impl | Fastest median | Config F median | Outcome | Gap |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for cell in summary["cells"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    cell["machine_id"],
                    cell["input_key"],
                    str(cell["best_impl_label"] or "none"),
                    _fmt_ms(cell["best_warm_median_ms"]),
                    _fmt_ms(cell["config_f_warm_median_ms"]),
                    cell["outcome"],
                    _fmt_gap(cell),
                ]
            )
            + " |"
        )

    excluded = [
        row
        for row in rows
        if row["status"] == "ok"
        and row["warm_median_s"] is not None
        and not row["full_duration"]
    ]
    if excluded:
        lines.extend(
            [
                "",
                "## Excluded Short Outputs",
                "",
                "| Machine | Bucket | Impl | Median | Observed / canonical |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in sorted(excluded, key=lambda item: (item["machine_id"], item["input_key"], item["impl_label"])):
            ratio = row["duration_ratio"]
            lines.append(
                f"| {row['machine_id']} | {row['input_key']} | {row['impl_label']} | "
                f"{_fmt_ms(row['warm_median_ms'])} | {ratio:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/external_bakeoff"))
    parser.add_argument("--min-duration-ratio", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    files = args.results or default_result_paths(args.results_dir)
    rows = load_records(files, args.min_duration_ratio)
    summary = summarize_frontier(rows)
    markdown = render_markdown(summary, rows, args.min_duration_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "absolute_fastest_verified": summary["absolute_fastest_verified"],
                "config_f_losses": len(summary["config_f_losses"]),
                "config_f_missing": len(summary["config_f_missing"]),
                "no_full_duration_result": len(summary["no_full_duration_result"]),
                "output": str(args.output),
                "records": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
