#!/usr/bin/env python3
"""Summarize whether a strict ``har_source -> Core ML`` boundary pays off.

The strict padded/Nyquist source-boundary packages preserve waveform quality,
but their Core ML generator graph is larger than the shipped generator-only
graph. This report combines those package medians with direct Swift HnSF timing
so the decision uses the full warmed inference boundary:

    net_delta = candidate_har_source_fused - baseline_generator - swift_stft

Positive ``net_delta`` means the candidate is still slower after crediting it
for the Swift STFT work it removes. Swift source generation is reported but not
included in the delta because both paths still need it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_HNSF_TIMING = Path("outputs/external_bakeoff/hnsf_source_stft_timing_local.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/hnsf_source_boundary_net.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/hnsf_source_boundary_net.json")
DEFAULT_FUSED_REPORTS = {
    "3s": Path("outputs/har_source_fused/3s_atan_manual_fp32_nyquist_padded/report_har_source_fused.json"),
    "7s": Path("outputs/har_source_fused/7s_atan_manual_fp32_nyquist_padded/report_har_source_fused.json"),
    "10s": Path("outputs/har_source_fused/10s_atan_manual_fp32_nyquist_padded/report_har_source_fused.json"),
    "30s": Path("outputs/har_source_fused/30s_atan_manual_fp32_nyquist_padded/report_har_source_fused.json"),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _metric(metrics: dict[str, Any], key: str, field: str) -> float | None:
    value = metrics.get(key, {}).get(field)
    return None if value is None else float(value)


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} ms"


def _fmt_db(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} dB"


def _fmt_corr(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.9f}"


def build_report(hnsf_timing: Path, fused_reports: dict[str, Path]) -> dict[str, Any]:
    """Build a net source-boundary report from HnSF and fused-package timings."""

    hnsf_payload = _load_json(hnsf_timing)
    hnsf_by_bucket = {f"{row['bucket_s']}s": row for row in hnsf_payload.get("buckets", [])}
    rows: list[dict[str, Any]] = []

    for bucket, report_path in sorted(fused_reports.items(), key=lambda item: int(item[0].rstrip("s"))):
        hnsf = hnsf_by_bucket.get(bucket)
        if hnsf is None:
            rows.append({"bucket": bucket, "status": "missing_hnsf_timing", "path": str(report_path)})
            continue
        if not report_path.exists():
            rows.append({"bucket": bucket, "status": "missing_fused_report", "path": str(report_path)})
            continue

        fused = _load_json(report_path)
        medians = fused.get("warm_predict_median_ms") or {}
        baseline_ms = float(medians["baseline_generator"])
        candidate_ms = float(medians["candidate_har_source_fused"])
        generator_delta_ms = candidate_ms - baseline_ms
        removable_stft_ms = float(hnsf["stft_median_ms"])
        net_delta_ms = generator_delta_ms - removable_stft_ms
        metrics = fused.get("metrics") or {}

        rows.append(
            {
                "bucket": bucket,
                "status": "ok",
                "path": str(report_path),
                "compute_units": fused.get("compute_units"),
                "baseline_generator_ms": baseline_ms,
                "candidate_har_source_fused_ms": candidate_ms,
                "candidate_minus_generator_ms": generator_delta_ms,
                "removable_swift_stft_ms": removable_stft_ms,
                "net_delta_after_stft_credit_ms": net_delta_ms,
                "source_still_required_ms": float(hnsf["source_median_ms"]),
                "build_har_ms": float(hnsf["build_har_median_ms"]),
                "candidate_vs_baseline_correlation": _metric(
                    metrics, "candidate_vs_baseline_trimmed", "correlation"
                ),
                "candidate_vs_baseline_snr_db": _metric(metrics, "candidate_vs_baseline_trimmed", "snr_db"),
                "candidate_vs_dump_correlation": _metric(metrics, "candidate_vs_dump_trimmed", "correlation"),
                "candidate_vs_dump_snr_db": _metric(metrics, "candidate_vs_dump_trimmed", "snr_db"),
            }
        )

    ok_rows = [row for row in rows if row["status"] == "ok"]
    winning_rows = [row for row in ok_rows if row["net_delta_after_stft_credit_ms"] < 0]
    return {
        "hnsf_timing": str(hnsf_timing),
        "timing_boundary": hnsf_payload.get("timing_boundary"),
        "rows": rows,
        "summary": {
            "measured_buckets": len(ok_rows),
            "net_winning_buckets_after_stft_credit": len(winning_rows),
            "missing_buckets": len(rows) - len(ok_rows),
            "best_net_delta_ms": None
            if not ok_rows
            else min(float(row["net_delta_after_stft_credit_ms"]) for row in ok_rows),
            "worst_net_delta_ms": None
            if not ok_rows
            else max(float(row["net_delta_after_stft_credit_ms"]) for row in ok_rows),
            "decision": "reject strict har_source boundary unless a future change removes a Core ML call boundary or reduces body cost",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the net source-boundary report as Markdown."""

    summary = report["summary"]
    lines = [
        "# HnSF Source Boundary Net Timing",
        "",
        "This report credits strict `har_source -> Core ML` candidates for the Swift STFT work they remove.",
        "Swift source generation still remains required and therefore cancels out of the net delta.",
        "",
        "## Summary",
        "",
        f"- Measured buckets: `{summary['measured_buckets']}`.",
        f"- Net winning buckets after STFT credit: `{summary['net_winning_buckets_after_stft_credit']}`.",
        f"- Best net delta: `{_fmt_ms(summary['best_net_delta_ms'])}`.",
        f"- Worst net delta: `{_fmt_ms(summary['worst_net_delta_ms'])}`.",
        f"- Decision: {summary['decision']}.",
        "",
        "## Rows",
        "",
        "| Bucket | Baseline generator | Candidate fused | Generator delta | STFT credit | Net after STFT credit | Source still required | Replacement quality |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        if row["status"] != "ok":
            lines.append(
                "| {bucket} | n/a | n/a | n/a | n/a | n/a | n/a | `{status}`: `{path}` |".format(
                    bucket=row["bucket"],
                    status=row["status"],
                    path=row["path"],
                )
            )
            continue
        quality = (
            f"vs current generator corr `{_fmt_corr(row['candidate_vs_baseline_correlation'])}`, "
            f"SNR `{_fmt_db(row['candidate_vs_baseline_snr_db'])}`"
        )
        lines.append(
            " | ".join(
                [
                    f"| {row['bucket']}",
                    _fmt_ms(row["baseline_generator_ms"]),
                    _fmt_ms(row["candidate_har_source_fused_ms"]),
                    _fmt_ms(row["candidate_minus_generator_ms"]),
                    _fmt_ms(row["removable_swift_stft_ms"]),
                    _fmt_ms(row["net_delta_after_stft_credit_ms"]),
                    _fmt_ms(row["source_still_required_ms"]),
                    quality + " |",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The 3s strict fused candidate is nearly a wash after STFT credit, but it still does not win.",
            "- Longer buckets get worse because padded/Nyquist geometry adds more Core ML generator work than Swift STFT removes.",
            "- The next useful source-boundary work must remove a prediction boundary, produce `x_source` without an extra noise call, or change the phase representation consumed by the first noise convolutions.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_report_mapping(values: list[str]) -> dict[str, Path]:
    reports = dict(DEFAULT_FUSED_REPORTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--fused-report must be BUCKET=PATH, got {value!r}")
        bucket, path = value.split("=", 1)
        reports[bucket] = Path(path)
    return reports


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hnsf-timing", type=Path, default=DEFAULT_HNSF_TIMING)
    parser.add_argument(
        "--fused-report",
        action="append",
        default=[],
        help="Override/add strict fused report as BUCKET=PATH.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.hnsf_timing, _parse_report_mapping(args.fused_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report))
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
