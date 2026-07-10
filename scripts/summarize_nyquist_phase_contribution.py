#!/usr/bin/env python3
"""Summarize Nyquist-phase contribution reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORTS = [
    Path("outputs/nyquist_phase_contribution/report_3s.json"),
    Path("outputs/nyquist_phase_contribution/report_3s_padded.json"),
    Path("outputs/nyquist_phase_contribution/report_7s.json"),
    Path("outputs/nyquist_phase_contribution/report_7s_padded.json"),
    Path("outputs/nyquist_phase_contribution/report_10s.json"),
    Path("outputs/nyquist_phase_contribution/report_10s_padded.json"),
    Path("outputs/nyquist_phase_contribution/report_15s.json"),
    Path("outputs/nyquist_phase_contribution/report_15s_padded.json"),
    Path("outputs/nyquist_phase_contribution/report_30s.json"),
    Path("outputs/nyquist_phase_contribution/report_30s_padded.json"),
]
DEFAULT_OUTPUT = Path("outputs/nyquist_phase_contribution/summary.md")
DEFAULT_JSON_OUTPUT = Path("outputs/nyquist_phase_contribution/summary.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _metric(report: dict[str, Any], variant: str, key: str) -> float | None:
    value = ((report.get("waveform_metrics_vs_dump") or {}).get(variant) or {}).get(key)
    return None if value is None else float(value)


def _feature_metric(report: dict[str, Any], name: str, key: str) -> float | int | None:
    value = ((report.get("feature_metrics") or {}).get(name) or {}).get(key)
    if value is None:
        return None
    return int(value) if isinstance(value, int) else float(value)


def _fmt(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _fmt_corr(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def summarize_reports(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        report = _load_json(path)
        metadata = report.get("manifest_metadata") or {}
        row = {
            "path": str(path),
            "input_key": metadata.get("input_key"),
            "bucket_seconds": metadata.get("bucket_seconds"),
            "pad_har_to": report.get("pad_har_to"),
            "geometry": "padded" if report.get("pad_har_to") is not None else "natural",
            "dumped_har_corr": _metric(report, "dumped_har", "correlation"),
            "dumped_har_snr_db": _metric(report, "dumped_har", "snr_db"),
            "recomputed_manual_corr": _metric(report, "recomputed_manual", "correlation"),
            "recomputed_manual_snr_db": _metric(report, "recomputed_manual", "snr_db"),
            "recomputed_manual_dumped_nyquist_corr": _metric(
                report, "recomputed_manual_dumped_nyquist", "correlation"
            ),
            "recomputed_manual_dumped_nyquist_snr_db": _metric(
                report, "recomputed_manual_dumped_nyquist", "snr_db"
            ),
            "zero_nyquist_corr": _metric(report, "dumped_har_zero_nyquist", "correlation"),
            "zero_nyquist_snr_db": _metric(report, "dumped_har_zero_nyquist", "snr_db"),
            "affine_nyquist_corr": _metric(report, "recomputed_manual_affine_nyquist", "correlation"),
            "affine_nyquist_snr_db": _metric(report, "recomputed_manual_affine_nyquist", "snr_db"),
            "negated_nyquist_corr": _metric(report, "recomputed_manual_negated_nyquist", "correlation"),
            "negated_nyquist_snr_db": _metric(report, "recomputed_manual_negated_nyquist", "snr_db"),
            "swift_basis_nyquist_corr": _metric(
                report, "recomputed_manual_swift_basis_nyquist", "correlation"
            ),
            "swift_basis_nyquist_snr_db": _metric(
                report, "recomputed_manual_swift_basis_nyquist", "snr_db"
            ),
            "swift_basis_atan2_nyquist_corr": _metric(
                report, "recomputed_manual_swift_basis_atan2_nyquist", "correlation"
            ),
            "swift_basis_atan2_nyquist_snr_db": _metric(
                report, "recomputed_manual_swift_basis_atan2_nyquist", "snr_db"
            ),
            "nyquist_wrapped_max_abs_error": _feature_metric(
                report, "recomputed_nyquist_phase_vs_dumped", "wrapped_max_abs_error"
            ),
            "nyquist_two_pi_branch_errors": _feature_metric(
                report, "recomputed_nyquist_phase_vs_dumped", "two_pi_branch_errors"
            ),
            "swift_basis_nyquist_two_pi_branch_errors": _feature_metric(
                report, "swift_basis_nyquist_phase_vs_dumped", "two_pi_branch_errors"
            ),
            "swift_basis_atan2_nyquist_two_pi_branch_errors": _feature_metric(
                report, "swift_basis_atan2_nyquist_phase_vs_dumped", "two_pi_branch_errors"
            ),
        }
        row["dumped_nyquist_delta_snr_db"] = None
        if row["recomputed_manual_dumped_nyquist_snr_db"] is not None and row["recomputed_manual_snr_db"] is not None:
            row["dumped_nyquist_delta_snr_db"] = (
                row["recomputed_manual_dumped_nyquist_snr_db"] - row["recomputed_manual_snr_db"]
            )
        row["passes_strict_waveform_gate"] = (
            row["recomputed_manual_dumped_nyquist_corr"] is not None
            and row["recomputed_manual_dumped_nyquist_corr"] >= 0.99998
            and row["recomputed_manual_dumped_nyquist_snr_db"] is not None
            and row["recomputed_manual_dumped_nyquist_snr_db"] >= 45.0
        )
        rows.append(row)
    rows.sort(key=lambda row: (int(row.get("bucket_seconds") or 0), row["geometry"]))
    strict_rows = [row for row in rows if row["passes_strict_waveform_gate"]]
    return {
        "rows": rows,
        "row_count": len(rows),
        "strict_waveform_gate_pass_count": len(strict_rows),
        "strict_waveform_gate_pass_rows": [
            {"input_key": row["input_key"], "geometry": row["geometry"], "path": row["path"]} for row in strict_rows
        ],
        "interpretation": (
            "Using the raw trimmed waveform reference, dumped Nyquist phase plus "
            "padded shipping HAR geometry repairs the source-boundary path across "
            "3s/7s/10s/15s/30s. Natural HAR geometry still fails strict waveform "
            "parity. Branch-only Swift-basis Nyquist repair fails, but exact "
            "Swift Float real/imag dot products followed by atan2 matches the "
            "dumped-Nyquist oracle. Prior fused-source timing still shows padded "
            "geometry removes the direct speed edge, so exact Nyquist repair is "
            "a strict contract unlock rather than a standalone production win."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Nyquist Phase Contribution Summary",
        "",
        "PyTorch-only sensitivity probe for the compact `har_source -> waveform`",
        "path. The table compares natural versus padded HAR geometry and whether",
        "splicing the dumped Swift Nyquist phase is sufficient.",
        "",
        f"Rows: `{summary['row_count']}`.",
        f"Strict waveform gate pass rows: `{summary['strict_waveform_gate_pass_count']}`.",
        "",
        "| Bucket | Geometry | Dumped HAR SNR | Recomputed SNR | + dumped Nyquist SNR | Swift-branch Nyquist SNR | Swift-atan2 Nyquist SNR | Affine Nyquist SNR | Negated Nyquist SNR | Zero-Nyquist SNR | Nyquist wrapped max | 2pi errors | Swift-branch 2pi errors | Swift-atan2 2pi errors | Report |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['input_key']}`",
                    row["geometry"],
                    _fmt(row["dumped_har_snr_db"]),
                    _fmt(row["recomputed_manual_snr_db"]),
                    _fmt(row["recomputed_manual_dumped_nyquist_snr_db"]),
                    _fmt(row["swift_basis_nyquist_snr_db"]),
                    _fmt(row["swift_basis_atan2_nyquist_snr_db"]),
                    _fmt(row["affine_nyquist_snr_db"]),
                    _fmt(row["negated_nyquist_snr_db"]),
                    _fmt(row["zero_nyquist_snr_db"]),
                    _fmt(row["nyquist_wrapped_max_abs_error"], 4),
                    "n/a" if row["nyquist_two_pi_branch_errors"] is None else str(row["nyquist_two_pi_branch_errors"]),
                    "n/a"
                    if row["swift_basis_nyquist_two_pi_branch_errors"] is None
                    else str(row["swift_basis_nyquist_two_pi_branch_errors"]),
                    "n/a"
                    if row["swift_basis_atan2_nyquist_two_pi_branch_errors"] is None
                    else str(row["swift_basis_atan2_nyquist_two_pi_branch_errors"]),
                    f"`{row['path']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()
    paths = args.report if args.report else DEFAULT_REPORTS
    summary = summarize_reports(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "row_count": summary["row_count"],
                "strict_waveform_gate_pass_count": summary["strict_waveform_gate_pass_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
