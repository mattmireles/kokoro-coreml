#!/usr/bin/env python3
"""Summarize and rank F0-source speed candidates from saved probe reports."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_GLOB = "outputs/f0_noise_exact_shape/**/report*.json"
DEFAULT_DECISIONS_GLOB = "outputs/f0_source_listening/**/f0_source_listening_decisions.csv"
DEFAULT_OUTPUT = Path("outputs/f0_source_listening/f0_source_candidate_summary.md")


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load one JSON object, returning None for malformed files."""

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _machine_from_path(path: Path) -> str:
    """Infer the machine from the report path when the report lacks a field."""

    text = path.as_posix().lower()
    if "irvine" in text:
        return "irvine-m1"
    if "m2-air" in text or "m2_air" in text:
        return "m2-air"
    return "m2-studio"


def _label(report: dict[str, Any], path: Path) -> str:
    """Return a stable human-readable candidate label."""

    report_path = str(report.get("report") or "")
    if report_path:
        parent = Path(report_path).parent.name
        if parent:
            return parent
    noise_package = str(report.get("noise_package") or "")
    if noise_package:
        parent = Path(noise_package).parent.name
        if parent:
            return parent
    return path.parent.name


def _decision_key(label: str, source_report: str) -> tuple[str, str]:
    """Return a normalized key for joining decision rows."""

    return (label, source_report)


def load_decisions(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    """Load F0-source human decision rows keyed by label and source report."""

    decisions: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = str(row.get("label") or "")
                source_report = str(row.get("source_report") or "")
                if label:
                    decisions[_decision_key(label, source_report)] = row
    return decisions


def summarize_report(
    path: Path,
    decisions: dict[tuple[str, str], dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Return one flattened summary row for a saved F0-source probe report."""

    report = _load_json(path)
    if not report:
        return None
    benchmark = report.get("benchmark") or {}
    med = benchmark.get("warm_predict_median_ms") or {}
    metrics = (benchmark.get("metrics") or {}).get("candidate_vs_baseline_trimmed") or {}
    baseline = med.get("baseline_total")
    candidate = med.get("candidate_total")
    if baseline is None or candidate is None:
        return None

    export = report.get("export") or {}
    label = _label(report, path)
    source_report = str(path)
    decision = {}
    if decisions:
        decision = decisions.get(_decision_key(label, source_report), {})
        if not decision:
            # Generated listening packs store repo-relative report paths. Remote
            # reports may be summarized before a pack exists, so this fallback
            # keeps the row useful without inventing a decision.
            decision = decisions.get(_decision_key(label, str(report.get("report") or "")), {})

    speedup = report.get("speedup_vs_baseline_pct")
    if speedup is None and float(baseline) > 0:
        speedup = 100.0 * (float(baseline) - float(candidate)) / float(baseline)
    tensor_dump = str(report.get("tensor_dump") or "")
    bucket = Path(tensor_dump).name if tensor_dump else ""
    row = {
        "label": label,
        "machine": _machine_from_path(path),
        "bucket": bucket,
        "report": source_report,
        "baseline_ms": float(baseline),
        "candidate_ms": float(candidate),
        "speedup_pct": float(speedup or 0.0),
        "corr": metrics.get("correlation"),
        "snr_db": metrics.get("snr_db"),
        "max_abs_error": metrics.get("max_abs_error"),
        "passes_strict_gate": bool(report.get("passes")),
        "natural_asr": bool(export.get("natural_asr")),
        "deployment_target": str(export.get("deployment_target") or ""),
        "native_instance_norm": bool(export.get("native_instance_norm")),
        "palettize_body": bool(export.get("palettize_body")),
        "source_mode": str(export.get("source_mode") or "current"),
        "phase_mode": str(export.get("phase_mode") or "atan2"),
        "human_decision": str(decision.get("human_decision") or ""),
        "decision_notes": str(decision.get("notes") or ""),
    }
    return row


def _fmt_float(value: Any, digits: int = 2) -> str:
    """Format optional numeric values for Markdown."""

    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """Render ranked candidate rows as Markdown."""

    lines = [
        "# F0 Source Candidate Ranking",
        "",
        "Sorted by warm median speedup versus the strict-equivalent baseline stack.",
        "Strict waveform failures are not production approvals; rows with blank",
        "`Human` still need no-ASR listening decisions.",
        "",
        "| Rank | Machine | Bucket | Candidate | Speedup | Baseline ms | Candidate ms | Corr | SNR dB | Strict | Human | Notes |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        strict = "pass" if row["passes_strict_gate"] else "fail"
        human = row["human_decision"] or "blank"
        notes = row["decision_notes"].replace("|", "/")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["machine"],
                    row["bucket"],
                    f"`{row['label']}`",
                    _fmt_float(row["speedup_pct"], 1) + "%",
                    _fmt_float(row["baseline_ms"], 1),
                    _fmt_float(row["candidate_ms"], 1),
                    _fmt_float(row["corr"], 6),
                    _fmt_float(row["snr_db"], 2),
                    strict,
                    human,
                    notes,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def collect_rows(report_globs: list[str], decisions: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    """Collect and rank summary rows from report globs."""

    paths: list[Path] = []
    for pattern in report_globs:
        paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
    rows = []
    seen = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        row = summarize_report(path, decisions)
        if row:
            rows.append(row)
    return sorted(rows, key=lambda row: row["speedup_pct"], reverse=True)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-glob", action="append", default=[DEFAULT_REPORT_GLOB])
    parser.add_argument("--decisions", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=0, help="Limit emitted rows; 0 keeps all rows.")
    args = parser.parse_args()

    decision_paths = args.decisions or [
        Path(match) for match in glob.glob(DEFAULT_DECISIONS_GLOB, recursive=True)
    ]
    decisions = load_decisions(decision_paths)
    rows = collect_rows(args.report_glob, decisions)
    if args.top > 0:
        rows = rows[: args.top]
    markdown = render_markdown(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
