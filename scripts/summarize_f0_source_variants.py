#!/usr/bin/env python3
"""Summarize F0 source formulation probes across runtime buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("outputs/f0_source_variants/report_3s_7s_10s_15s_30s.json")
DEFAULT_OUTPUT = Path("outputs/f0_source_variants/summary_3s_7s_10s_15s_30s.md")
DEFAULT_JSON_OUTPUT = Path("outputs/f0_source_variants/summary_3s_7s_10s_15s_30s.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _metric(report: dict[str, Any], section: str, variant: str, key: str) -> float | None:
    value = ((report.get(section) or {}).get(variant) or {}).get(key)
    return None if value is None else float(value)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_corr(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def summarize_source_variants(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for report in payload.get("reports") or []:
        row = {
            "input_key": report.get("input_key"),
            "bucket_seconds": report.get("bucket_seconds"),
            "tensor_dump": report.get("tensor_dump"),
            "swift_like_source_corr": _metric(
                report, "metrics_vs_dump_har_source", "swift_like_seeded", "correlation"
            ),
            "swift_like_source_snr_db": _metric(
                report, "metrics_vs_dump_har_source", "swift_like_seeded", "snr_db"
            ),
            "best_simplified_source_corr": None,
            "best_simplified_source_snr_db": None,
            "best_simplified_source_variant": None,
            "dump_source_recomputed_har_corr": _metric(
                report, "metrics_vs_dump_har_padded", "dump_source_recomputed_stft", "correlation"
            ),
            "dump_source_recomputed_har_snr_db": _metric(
                report, "metrics_vs_dump_har_padded", "dump_source_recomputed_stft", "snr_db"
            ),
            "dump_source_recomputed_har_max_abs_error": _metric(
                report, "metrics_vs_dump_har_padded", "dump_source_recomputed_stft", "max_abs_error"
            ),
            "swift_like_recomputed_har_corr": _metric(
                report, "metrics_vs_dump_har_padded", "swift_like_seeded", "correlation"
            ),
            "swift_like_recomputed_har_snr_db": _metric(
                report, "metrics_vs_dump_har_padded", "swift_like_seeded", "snr_db"
            ),
        }
        simplified = []
        for name, metrics in (report.get("metrics_vs_dump_har_source") or {}).items():
            if name in {"swift_like_seeded", "original_pytorch_seeded"}:
                continue
            simplified.append(
                (
                    float(metrics.get("snr_db") or float("-inf")),
                    name,
                    float(metrics.get("correlation") or 0.0),
                )
            )
        if simplified:
            best_snr, best_name, best_corr = max(simplified, key=lambda item: item[0])
            row["best_simplified_source_variant"] = best_name
            row["best_simplified_source_snr_db"] = best_snr
            row["best_simplified_source_corr"] = best_corr
        rows.append(row)
    rows.sort(key=lambda row: str(row.get("input_key")))
    min_swift_source_snr = min(
        (row["swift_like_source_snr_db"] for row in rows if row["swift_like_source_snr_db"] is not None),
        default=None,
    )
    max_recomputed_har_snr = max(
        (
            row["dump_source_recomputed_har_snr_db"]
            for row in rows
            if row["dump_source_recomputed_har_snr_db"] is not None
        ),
        default=None,
    )
    return {
        "rows": rows,
        "row_count": len(rows),
        "min_swift_like_source_snr_db": min_swift_source_snr,
        "max_dump_source_recomputed_har_snr_db": max_recomputed_har_snr,
        "source_equation_is_solved": min_swift_source_snr is not None and min_swift_source_snr >= 100.0,
        "recomputed_stft_har_is_solved": max_recomputed_har_snr is not None and max_recomputed_har_snr >= 35.0,
        "interpretation": (
            "Swift-like seeded source generation matches dumped har_source across buckets, "
            "but recomputing HAR/STFT from even the dumped source stays near 8 dB SNR with "
            "2*pi phase-wrap max errors. The remaining quality blocker is the HAR/STFT "
            "representation contract, not the sine source equation."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# F0 Source Variant Summary",
        "",
        "Cheap PyTorch-only source formulation probe across runtime buckets.",
        "This summary exists to prevent re-exporting Core ML packages for source",
        "equations that already fail before conversion.",
        "",
        f"Rows: `{summary['row_count']}`.",
        f"Source equation solved: `{str(summary['source_equation_is_solved']).lower()}`.",
        f"Recomputed HAR/STFT solved: `{str(summary['recomputed_stft_har_is_solved']).lower()}`.",
        "",
        "| Bucket | Swift-like source corr | Swift-like source SNR | Best simplified source | Simplified corr | Simplified SNR | Dump source -> HAR corr | Dump source -> HAR SNR | HAR max abs |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['input_key']}`",
                    _fmt_corr(row["swift_like_source_corr"]),
                    _fmt(row["swift_like_source_snr_db"]),
                    f"`{row['best_simplified_source_variant']}`",
                    _fmt_corr(row["best_simplified_source_corr"]),
                    _fmt(row["best_simplified_source_snr_db"]),
                    _fmt_corr(row["dump_source_recomputed_har_corr"]),
                    _fmt(row["dump_source_recomputed_har_snr_db"]),
                    _fmt(row["dump_source_recomputed_har_max_abs_error"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()
    summary = summarize_source_variants(_load_json(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_equation_is_solved": summary["source_equation_is_solved"],
                "recomputed_stft_har_is_solved": summary["recomputed_stft_har_is_solved"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
