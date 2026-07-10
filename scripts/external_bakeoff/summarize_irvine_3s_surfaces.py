#!/usr/bin/env python3
"""Summarize saved Irvine 3s graph-surface probes."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_GLOB = "outputs/**/report*.json"
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/irvine_3s_surfaces.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/irvine_3s_surfaces.json")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}%"


def _metric_block(report: dict[str, Any]) -> dict[str, Any]:
    benchmark = report.get("benchmark") or {}
    metrics = report.get("metrics") or benchmark.get("metrics") or {}
    for key in (
        "candidate_vs_baseline_trimmed",
        "candidate_vs_fused_trimmed",
        "split_vs_fused_trimmed",
        "cos_vs_fused_trimmed",
        "candidate_vs_dump_trimmed",
        "split_vs_dump_trimmed",
        "cos_vs_dump_trimmed",
    ):
        value = metrics.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _warm_medians(report: dict[str, Any]) -> tuple[float | None, float | None]:
    benchmark = report.get("benchmark") or {}
    med = report.get("warm_predict_median_ms") or benchmark.get("warm_predict_median_ms") or {}
    if not isinstance(med, dict):
        return None, None
    baseline = (
        med.get("baseline_total")
        or med.get("baseline_generator")
        or med.get("baseline")
        or med.get("fused")
        or med.get("fused_total")
    )
    candidate = (
        med.get("candidate_total")
        or med.get("candidate_stack")
        or med.get("candidate")
        or med.get("split_total")
        or med.get("candidate_har_source_fused")
        or med.get("cos")
        or med.get("style_specialized")
    )
    if baseline is None or candidate is None:
        return None, None
    return float(baseline), float(candidate)


def _speedup(report: dict[str, Any], baseline: float | None, candidate: float | None) -> float | None:
    for key in ("speedup_vs_baseline_pct", "speedup_vs_fused_pct", "speedup_vs_generator_pct"):
        if report.get(key) is not None:
            return float(report[key])
        benchmark = report.get("benchmark") or {}
        if benchmark.get(key) is not None:
            return float(benchmark[key])
    if baseline and candidate is not None:
        return 100.0 * (baseline - candidate) / baseline
    return None


def _label(report: dict[str, Any], path: Path) -> str:
    for key in ("label", "report", "report_path"):
        value = report.get(key)
        if value:
            text = str(value)
            if key == "label":
                return text
            parent = Path(text).parent.name
            if parent:
                return parent
    return path.parent.name


def _is_irvine(path: Path, report: dict[str, Any]) -> bool:
    haystack = " ".join(str(value) for value in [path, report.get("report"), report.get("report_path")] if value)
    return "irvine" in haystack.lower()


def _is_3s(path: Path, report: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(value)
        for value in [path, report.get("label"), report.get("report"), report.get("report_path"), report.get("tensor_dump")]
        if value
    )
    return "3s" in haystack


def summarize_report(path: Path) -> dict[str, Any] | None:
    report = _load_json(path)
    if not report or not _is_3s(path, report) or not _is_irvine(path, report):
        return None
    passes = report.get("passes")
    if passes is None:
        metrics = _metric_block(report)
        corr = metrics.get("correlation")
        snr = metrics.get("snr_db")
        passes = bool(corr is not None and float(corr) >= 0.99998 and snr is not None and float(snr) >= 45.0)
    baseline, candidate = _warm_medians(report)
    speedup = _speedup(report, baseline, candidate)
    if speedup is None:
        return None
    metrics = _metric_block(report)
    family = path.parts[1] if len(path.parts) > 2 and path.parts[0] == "outputs" else path.parent.name
    return {
        "family": family,
        "label": _label(report, path),
        "path": str(path),
        "passes": bool(passes),
        "quality_status": "strict-pass" if passes else "quality-fail",
        "baseline_ms": baseline,
        "candidate_ms": candidate,
        "delta_ms": None if baseline is None or candidate is None else baseline - candidate,
        "speedup_pct": speedup,
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
            if row:
                rows.append(row)
    return sorted(rows, key=lambda row: float(row["speedup_pct"]), reverse=True)


def summarize_surfaces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "machine_id": "irvine-m1",
        "input_key": "3s",
        "rows": rows,
        "row_count": len(rows),
        "strict_pass_positive_count": sum(
            1 for row in rows if row["passes"] and float(row["speedup_pct"]) > 0.0
        ),
        "quality_fail_positive_count": sum(
            1 for row in rows if not row["passes"] and float(row["speedup_pct"]) > 0.0
        ),
        "best_strict_pass": next(
            (row for row in rows if row["passes"] and float(row["speedup_pct"]) > 0.0),
            None,
        ),
        "best_quality_fail": next(
            (row for row in rows if not row["passes"] and float(row["speedup_pct"]) > 0.0),
            None,
        ),
    }


def render_markdown(summary: dict[str, Any], top: int) -> str:
    best_strict = summary["best_strict_pass"]
    best_fail = summary["best_quality_fail"]
    lines = [
        "# Irvine 3s Graph Surfaces",
        "",
        "Saved Irvine 3s reports only. This scanner understands split/fused",
        "benchmark keys that the generic optimization scanner does not classify.",
        "",
        f"Rows classified: `{summary['row_count']}`.",
        f"Strict-pass positive rows: `{summary['strict_pass_positive_count']}`.",
        f"Quality-fail positive rows: `{summary['quality_fail_positive_count']}`.",
        f"Best strict-pass row: `{None if best_strict is None else best_strict['label']}`.",
        f"Best quality-fail row: `{None if best_fail is None else best_fail['label']}`.",
        "",
        "| Quality | Family | Candidate | Speedup | Delta | Baseline | Candidate | Corr | SNR dB | Report |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"][:top]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["quality_status"],
                    row["family"],
                    f"`{row['label']}`",
                    _fmt_pct(row["speedup_pct"]),
                    _fmt_ms(row["delta_ms"]),
                    _fmt_ms(row["baseline_ms"]),
                    _fmt_ms(row["candidate_ms"]),
                    "n/a" if row.get("corr") is None else f"{float(row['corr']):.6f}",
                    "n/a" if row.get("snr_db") is None else f"{float(row['snr_db']):.2f}",
                    f"`{row['path']}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-glob", action="append", default=[DEFAULT_REPORT_GLOB])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    rows = collect_rows(args.report_glob)
    summary = summarize_surfaces(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary, args.top))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": summary["row_count"],
                "strict_pass_positive_count": summary["strict_pass_positive_count"],
                "quality_fail_positive_count": summary["quality_fail_positive_count"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
