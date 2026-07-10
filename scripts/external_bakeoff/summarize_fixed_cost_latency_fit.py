#!/usr/bin/env python3
"""Fit warmed latency to a fixed-cost plus duration model.

This is a diagnostic companion to ``competitive_frontier``. It uses only
paper-facing warmed frontier rows and fits:

    latency_ms = fixed_ms + slope_ms_per_audio_s * canonical_audio_duration_s

The fit is intentionally simple. Its job is to reveal whether an implementation
looks like it pays a higher per-call offset, a higher duration-scaled cost, or
both. It is not a replacement for per-bucket benchmark rows.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import load_json, write_json  # noqa: E402


DEFAULT_FRONTIER = Path("outputs/external_bakeoff/competitive_frontier.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/fixed_cost_latency_fit.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/fixed_cost_latency_fit.json")


def _fit(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return ordinary least-squares fit details for one point set."""

    if len(points) < 3:
        return None
    xs = [float(point["canonical_audio_duration_s"]) for point in points]
    ys = [float(point["warm_median_ms"]) for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    fixed = y_mean - slope * x_mean
    residuals = [y - (fixed + slope * x) for x, y in zip(xs, ys)]
    rss = sum(value**2 for value in residuals)
    tss = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 if tss == 0 else 1.0 - rss / tss
    return {
        "fixed_ms": fixed,
        "slope_ms_per_audio_s": slope,
        "r2": r2,
        "median_abs_error_ms": statistics.median(abs(value) for value in residuals),
        "max_abs_error_ms": max(abs(value) for value in residuals),
        "points": points,
        "fit_warning": _fit_warning(fixed, r2),
    }


def _fit_warning(fixed_ms: float, r2: float) -> str:
    """Return a short warning when the linear fit should be treated carefully."""

    warnings: list[str] = []
    if r2 < 0.95:
        warnings.append("low-r2")
    if fixed_ms < 0:
        warnings.append("negative-fixed-term")
    return ",".join(warnings) if warnings else ""


def _eligible_rows(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    """Return warmed, full-duration, successful frontier rows."""

    rows = []
    for row in frontier.get("rows", []):
        if str(row.get("status") or "") != "ok":
            continue
        if not row.get("full_duration"):
            continue
        if row.get("warm_median_ms") is None:
            continue
        if row.get("canonical_audio_duration_s") is None:
            continue
        rows.append(row)
    return rows


def _fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit every machine/implementation cell with enough warmed rows."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["machine_id"]), str(row["impl"])), []).append(row)

    fits: list[dict[str, Any]] = []
    for (machine_id, impl), points in sorted(grouped.items()):
        points = sorted(points, key=lambda item: float(item["canonical_audio_duration_s"]))
        fit = _fit(points)
        if not fit:
            continue
        label = str(points[0].get("impl_label") or impl)
        fits.append(
            {
                "machine_id": machine_id,
                "impl": impl,
                "impl_label": label,
                "n_points": len(points),
                "buckets": [str(point["input_key"]) for point in points],
                **fit,
            }
        )
    return fits


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return warmed frontier rows keyed by machine, implementation, and bucket."""

    return {
        (str(row["machine_id"]), str(row["impl"]), str(row["input_key"])): row
        for row in rows
    }


def _fit_index(fits: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Return fit rows keyed by machine and implementation."""

    return {(str(row["machine_id"]), str(row["impl"])): row for row in fits}


def _crossover_audio_seconds(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """Return fitted audio-duration crossover where a and b have equal latency."""

    denominator = float(a["slope_ms_per_audio_s"]) - float(b["slope_ms_per_audio_s"])
    if abs(denominator) < 1e-9:
        return None
    value = (float(b["fixed_ms"]) - float(a["fixed_ms"])) / denominator
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _comparison_rows(rows: list[dict[str, Any]], fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build machine-level Config F comparisons against laishere and MLX."""

    row_index = _row_index(rows)
    fit_index = _fit_index(fits)
    machines = sorted({str(row["machine_id"]) for row in rows})
    comparisons: list[dict[str, Any]] = []
    for machine_id in machines:
        config_fit = fit_index.get((machine_id, "config-f-reference"))
        if not config_fit:
            continue
        for competitor in ("laishere-kokoro-coreml", "mlx-audio"):
            competitor_fit = fit_index.get((machine_id, competitor))
            if not competitor_fit:
                continue
            deltas: list[dict[str, Any]] = []
            for bucket in config_fit["buckets"]:
                config = row_index.get((machine_id, "config-f-reference", bucket))
                other = row_index.get((machine_id, competitor, bucket))
                if not config or not other:
                    continue
                deltas.append(
                    {
                        "bucket": bucket,
                        "config_f_ms": float(config["warm_median_ms"]),
                        "competitor_ms": float(other["warm_median_ms"]),
                        "config_minus_competitor_ms": (
                            float(config["warm_median_ms"]) - float(other["warm_median_ms"])
                        ),
                    }
                )
            comparisons.append(
                {
                    "machine_id": machine_id,
                    "competitor_impl": competitor,
                    "competitor_label": competitor_fit["impl_label"],
                    "config_fixed_minus_competitor_fixed_ms": (
                        float(config_fit["fixed_ms"]) - float(competitor_fit["fixed_ms"])
                    ),
                    "config_slope_minus_competitor_slope_ms_per_audio_s": (
                        float(config_fit["slope_ms_per_audio_s"])
                        - float(competitor_fit["slope_ms_per_audio_s"])
                    ),
                    "fit_crossover_audio_s": _crossover_audio_seconds(config_fit, competitor_fit),
                    "fit_reliable": not config_fit["fit_warning"] and not competitor_fit["fit_warning"],
                    "config_fit_warning": config_fit["fit_warning"],
                    "competitor_fit_warning": competitor_fit["fit_warning"],
                    "bucket_deltas": deltas,
                }
            )
    return comparisons


def build_report(frontier_path: Path) -> dict[str, Any]:
    """Build the fixed-cost latency-fit report payload."""

    frontier = load_json(frontier_path)
    rows = _eligible_rows(frontier)
    fits = _fit_rows(rows)
    comparisons = _comparison_rows(rows, fits)
    config_vs_mlx = [
        item for item in comparisons if item["competitor_impl"] == "mlx-audio"
    ]
    config_vs_laishere = [
        item for item in comparisons if item["competitor_impl"] == "laishere-kokoro-coreml"
    ]
    return {
        "source_frontier": str(frontier_path),
        "summary": {
            "eligible_rows": len(rows),
            "fit_rows": len(fits),
            "config_vs_mlx_machines": len(config_vs_mlx),
            "config_vs_laishere_machines": len(config_vs_laishere),
            "config_f_beats_mlx_full_duration_buckets": sum(
                1
                for item in config_vs_mlx
                for delta in item["bucket_deltas"]
                if delta["config_minus_competitor_ms"] < 0
            ),
            "config_f_loses_laishere_full_duration_buckets": sum(
                1
                for item in config_vs_laishere
                for delta in item["bucket_deltas"]
                if delta["config_minus_competitor_ms"] > 0
            ),
        },
        "fits": fits,
        "comparisons": comparisons,
    }


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_slope(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f} ms/s"


def _fmt_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _fmt_audio_s(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}s"


def _delta_summary(deltas: list[dict[str, Any]]) -> str:
    """Return compact per-bucket delta text. Positive means Config F is slower."""

    if not deltas:
        return "n/a"
    parts = []
    for row in deltas:
        delta = float(row["config_minus_competitor_ms"])
        sign = "+" if delta >= 0 else ""
        parts.append(f"{row['bucket']} {sign}{delta:.1f} ms")
    return ", ".join(parts)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the fixed-cost latency-fit report as Markdown."""

    summary = report["summary"]
    lines = [
        "# Fixed-Cost Latency Fit",
        "",
        "Warmed inference only. This fits each eligible frontier implementation to",
        "`latency_ms = fixed_ms + slope_ms_per_audio_s * canonical_audio_duration_s`.",
        "The fit is diagnostic: per-bucket warmed rows remain the source of truth.",
        "",
        "## Summary",
        "",
        f"- Eligible warmed full-duration rows: `{summary['eligible_rows']}`.",
        f"- Machine/implementation fits: `{summary['fit_rows']}`.",
        f"- Config F full-duration bucket wins over MLX in fitted comparisons: `{summary['config_f_beats_mlx_full_duration_buckets']}`.",
        f"- Config F full-duration bucket losses to laishere in fitted comparisons: `{summary['config_f_loses_laishere_full_duration_buckets']}`.",
        "",
        "## Fit Rows",
        "",
        "| Machine | Impl | Buckets | Fixed term | Slope | R^2 | Median abs error | Warning |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["fits"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    row["impl_label"],
                    ", ".join(row["buckets"]),
                    _fmt_ms(row["fixed_ms"]),
                    _fmt_slope(row["slope_ms_per_audio_s"]),
                    _fmt_float(row["r2"]),
                    _fmt_ms(row["median_abs_error_ms"]),
                    row["fit_warning"] or "",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Config F Comparisons",
            "",
            "Delta is `Config F - competitor`; positive means Config F is slower.",
            "",
            "| Machine | Competitor | Fixed delta | Slope delta | Fit crossover | Reliable fit | Bucket deltas |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["comparisons"]:
        reliable = "yes" if row["fit_reliable"] else "no"
        warnings = ", ".join(
            value
            for value in [row["config_fit_warning"], row["competitor_fit_warning"]]
            if value
        )
        if warnings:
            reliable = f"{reliable} ({warnings})"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    row["competitor_label"],
                    _fmt_ms(row["config_fixed_minus_competitor_fixed_ms"]),
                    _fmt_slope(row["config_slope_minus_competitor_slope_ms_per_audio_s"]),
                    _fmt_audio_s(row["fit_crossover_audio_s"]),
                    reliable,
                    _delta_summary(row["bucket_deltas"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- MLX is not the current warmed Mac blocker: the corrected frontier has Config F faster on every full-duration MLX bucket, and MLX has no valid 3s full-duration Mac row.",
            "- laishere remains the lower-end strict blocker. Use this report to size fixed-cost versus duration-scaled gaps, then use `strict_win_budget_after_overlap_rewrite.md` for the concrete post-candidate save required.",
            "- Negative fixed terms or low `R^2` mean the implementation is not well described by one global line across buckets; treat those rows as heuristic, not proof.",
            "- The next strict optimization should remove a Core ML call boundary or materially reduce generator-stage duration. Another 1-3% host tweak is below the remaining Irvine 3s/7s budget.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.frontier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report))
    write_json(args.json_output, report)
    print(json.dumps({"output": str(args.output), "json_output": str(args.json_output), **report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
