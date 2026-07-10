#!/usr/bin/env python3
"""Summarize saved optimization probe reports by speed and quality status."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_GLOB = "outputs/**/report*.json"
DEFAULT_OUTPUT = Path("outputs/optimization_candidate_frontier.md")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _metric_block(report: dict[str, Any]) -> dict[str, Any]:
    benchmark = report.get("benchmark") or {}
    metrics = benchmark.get("metrics") or {}
    for key in (
        "candidate_vs_baseline_trimmed",
        "candidate_vs_fused_trimmed",
        "style_vs_fused_trimmed",
        "candidate_vs_baseline_full",
        "candidate_vs_fused_full",
        "style_vs_fused_full",
    ):
        value = metrics.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _warm_medians(benchmark: dict[str, Any]) -> tuple[float | None, float | None]:
    med = benchmark.get("warm_predict_median_ms") or {}
    if not isinstance(med, dict):
        return None, None
    baseline = (
        med.get("baseline")
        or med.get("baseline_total")
        or med.get("fused")
        or med.get("fused_total")
    )
    candidate = (
        med.get("candidate")
        or med.get("candidate_total")
        or med.get("style_specialized")
        or med.get("candidate_stack")
    )
    if baseline is None or candidate is None:
        return None, None
    return float(baseline), float(candidate)


def _speedup(report: dict[str, Any], benchmark: dict[str, Any]) -> float | None:
    for key in ("speedup_vs_baseline_pct", "speedup_vs_fused_pct"):
        value = report.get(key)
        if value is not None:
            return float(value)
        value = benchmark.get(key)
        if value is not None:
            return float(value)
    baseline, candidate = _warm_medians(benchmark)
    if baseline and candidate is not None:
        return 100.0 * (baseline - candidate) / baseline
    return None


def _label(report: dict[str, Any], path: Path) -> str:
    report_path = str(report.get("report") or "")
    if report_path:
        parent = Path(report_path).parent.name
        if parent:
            return parent
    return path.parent.name


def summarize_report(path: Path) -> dict[str, Any] | None:
    """Return one flattened probe summary row."""

    report = _load_json(path)
    if not report:
        return None
    benchmark = report.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        return None
    baseline_ms, candidate_ms = _warm_medians(benchmark)
    speedup_pct = _speedup(report, benchmark)
    metrics = _metric_block(report)
    passes = report.get("passes")
    if passes is None:
        return None

    parts = path.parts
    family = parts[1] if len(parts) > 2 and parts[0] == "outputs" else path.parent.name
    return {
        "family": family,
        "label": _label(report, path),
        "path": str(path),
        "passes": bool(passes),
        "speedup_pct": speedup_pct,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "corr": metrics.get("correlation"),
        "snr_db": metrics.get("snr_db"),
        "max_abs_error": metrics.get("max_abs_error"),
    }


def collect_rows(report_globs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in report_globs:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match)
            if path in seen:
                continue
            seen.add(path)
            row = summarize_report(path)
            if row and row["speedup_pct"] is not None:
                rows.append(row)
    return sorted(rows, key=lambda row: float(row["speedup_pct"]), reverse=True)


def classify(row: dict[str, Any], material_speedup_pct: float) -> str:
    """Classify a candidate by quality and whether its speedup can move a frontier."""

    speedup = float(row["speedup_pct"])
    if row["passes"] and speedup >= material_speedup_pct:
        return "quality-safe material speedup"
    if row["passes"] and speedup > 0:
        return "quality-safe noise-sized speedup"
    if row["passes"]:
        return "quality-safe slower"
    if speedup > 0:
        return "speed-positive quality fail"
    return "slower quality fail"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(rows: list[dict[str, Any]], material_speedup_pct: float, top: int) -> str:
    """Render candidate frontier as Markdown."""

    quality_safe_speedups = [
        row for row in rows if row["passes"] and float(row["speedup_pct"]) > 0.0
    ]
    material = [
        row
        for row in quality_safe_speedups
        if float(row["speedup_pct"]) >= material_speedup_pct
    ]
    speed_positive_quality_fail = [
        row for row in rows if not row["passes"] and float(row["speedup_pct"]) > 0.0
    ]

    lines = [
        "# Optimization Candidate Frontier",
        "",
        f"Material speed threshold: `{material_speedup_pct:.1f}%`.",
        f"Quality-safe speed-positive candidates: `{len(quality_safe_speedups)}`.",
        f"Quality-safe material candidates: `{len(material)}`.",
        f"Speed-positive quality-fail candidates: `{len(speed_positive_quality_fail)}`.",
        "",
        "| Class | Family | Candidate | Speedup | Baseline ms | Candidate ms | Corr | SNR dB | Report |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    emitted = 0
    for row in rows:
        if emitted >= top:
            break
        speedup = float(row["speedup_pct"])
        if speedup <= 0 and emitted >= top // 2:
            continue
        emitted += 1
        lines.append(
            "| "
            + " | ".join(
                [
                    classify(row, material_speedup_pct),
                    row["family"],
                    f"`{row['label']}`",
                    _fmt(row["speedup_pct"], 2) + "%",
                    _fmt(row["baseline_ms"], 1),
                    _fmt(row["candidate_ms"], 1),
                    _fmt(row["corr"], 6),
                    _fmt(row["snr_db"], 2),
                    f"`{row['path']}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-glob", action="append", default=[DEFAULT_REPORT_GLOB])
    parser.add_argument("--material-speedup-pct", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    rows = collect_rows(args.report_glob)
    markdown = render_markdown(rows, args.material_speedup_pct, args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    material = [
        row
        for row in rows
        if row["passes"]
        and row["speedup_pct"] is not None
        and float(row["speedup_pct"]) >= args.material_speedup_pct
    ]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "quality_safe_material_candidates": len(material),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
