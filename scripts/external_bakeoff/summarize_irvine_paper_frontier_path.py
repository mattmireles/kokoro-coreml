#!/usr/bin/env python3
"""Summarize the Irvine M1 path to the paper-facing frontier.

This joins the listening-gated source/body candidates with the measured
HAR-post rewrite projection. The purpose is to avoid a false conclusion from
profile-only wins: absolute-fastest claims must beat the stricter
``competitive_frontier`` paper rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_LOWER_END_GATE_JSON = DEFAULT_OUTPUT_DIR / "lower_end_mac_win_gate.json"
DEFAULT_REWRITE_IMPACT_JSON = DEFAULT_OUTPUT_DIR / "rewrite_candidate_impact.json"
DEFAULT_DIRECT_SOURCE_REWRITE_REPORTS = [
    (
        Path("outputs/f0_noise_exact_shape/3s_natural_asr_cos_rsqrt/report_f0_noise_exact_3s_local.json"),
        Path(
            "outputs/f0_noise_exact_shape/3s_natural_asr_cos_rsqrt_ups_as_conv_natural_asr_cos_rsqrt/"
            "report_cos_resblock_ups_as_conv_local.json"
        ),
    ),
    (
        Path("outputs/f0_noise_exact_shape/7s_natural_asr_cos_rsqrt/report_f0_noise_exact_7s_local.json"),
        Path(
            "outputs/f0_noise_exact_shape/7s_natural_asr_cos_rsqrt_ups_as_conv_natural_asr_cos_rsqrt/"
            "report_cos_resblock_ups_as_conv_local.json"
        ),
    ),
    (
        Path("outputs/f0_noise_exact_shape/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/report_cos_resblock.json"),
        Path(
            "outputs/f0_noise_exact_shape/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt_ups_as_conv_natural_asr_cos_rsqrt/"
            "report_cos_resblock_ups_as_conv_local.json"
        ),
    ),
    (
        Path("outputs/f0_noise_exact_shape/15s_padded_cos_resblock_cos_rsqrt/report_cos_resblock.json"),
        Path(
            "outputs/f0_noise_exact_shape/15s_padded_cos_resblock_cos_rsqrt_ups_as_conv_cos_rsqrt/"
            "report_cos_resblock_ups_as_conv_local.json"
        ),
    )
]
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "irvine_paper_frontier_path.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "irvine_paper_frontier_path.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _rewrite_save_by_bucket(rewrite_impact: dict[str, Any]) -> dict[str, float]:
    saves: dict[str, float] = {}
    for row in rewrite_impact.get("projection_rows") or []:
        if row.get("machine_id") == "irvine-m1":
            saves[str(row["bucket"])] = float(row["projected_save_ms"])
    return saves


def _report_label(report: dict[str, Any], path: Path) -> str:
    report_path = str(report.get("report") or "")
    return Path(report_path).parent.name if report_path else path.parent.name


def _report_bucket(report: dict[str, Any]) -> str:
    tensor_dump = str(report.get("tensor_dump") or "")
    return Path(tensor_dump).name if tensor_dump else ""


def _candidate_ms(report: dict[str, Any]) -> float:
    return float(report["benchmark"]["warm_predict_median_ms"]["candidate_total"])


def _direct_source_rewrite_by_bucket(pairs: list[tuple[Path, Path]]) -> dict[str, dict[str, Any]]:
    direct: dict[str, dict[str, Any]] = {}
    for original_path, rewritten_path in pairs:
        if not original_path.exists() or not rewritten_path.exists():
            continue
        original = _load_json(original_path)
        rewritten = _load_json(rewritten_path)
        bucket = _report_bucket(original)
        if not bucket:
            continue
        original_ms = _candidate_ms(original)
        rewritten_ms = _candidate_ms(rewritten)
        direct[bucket] = {
            "original_label": _report_label(original, original_path),
            "rewritten_label": _report_label(rewritten, rewritten_path),
            "original_report": str(original_path),
            "rewritten_report": str(rewritten_path),
            "original_candidate_ms": original_ms,
            "rewritten_candidate_ms": rewritten_ms,
            "local_direct_rewrite_save_ms": original_ms - rewritten_ms,
            "rewritten_upsample_layers": (rewritten.get("export") or {}).get("rewritten_upsample_layers"),
        }
    return direct


def build_summary(
    lower_end_gate: dict[str, Any],
    rewrite_impact: dict[str, Any],
    direct_source_rewrite_reports: list[tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    """Build the Irvine paper-frontier combined-path summary."""

    rewrite_saves = _rewrite_save_by_bucket(rewrite_impact)
    direct_rewrite = _direct_source_rewrite_by_bucket(direct_source_rewrite_reports or [])
    rows: list[dict[str, Any]] = []
    for row in lower_end_gate.get("irvine_rows") or []:
        bucket = str(row["input_key"])
        source_projected_ms = float(row["projected_full_ms"])
        rewrite_save_ms = rewrite_saves.get(bucket, 0.0)
        combined_projected_ms = source_projected_ms - rewrite_save_ms
        paper_ms = float(row["paper_competitor_ms"])
        combined_margin_ms = paper_ms - combined_projected_ms
        direct = direct_rewrite.get(bucket)
        direct_save_ms = None if direct is None else float(direct["local_direct_rewrite_save_ms"])
        direct_projected_ms = None if direct_save_ms is None else source_projected_ms - direct_save_ms
        direct_margin_ms = None if direct_projected_ms is None else paper_ms - direct_projected_ms
        rows.append(
            {
                "bucket": bucket,
                "candidate": row["candidate"],
                "human_decision": row.get("human_decision") or "",
                "paper_competitor": row.get("paper_competitor"),
                "paper_competitor_ms": paper_ms,
                "source_projected_ms": source_projected_ms,
                "source_paper_margin_ms": float(row["paper_margin_ms"]),
                "rewrite_save_ms": rewrite_save_ms,
                "combined_projected_ms": combined_projected_ms,
                "combined_paper_margin_ms": combined_margin_ms,
                "would_beat_paper_with_rewrite": combined_margin_ms > 0,
                "additional_save_required_ms": max(0.0, -combined_margin_ms),
                "direct_source_rewrite": direct,
                "direct_source_rewrite_projected_ms": direct_projected_ms,
                "direct_source_rewrite_paper_margin_ms": direct_margin_ms,
                "would_beat_paper_with_direct_source_rewrite": bool(
                    direct_margin_ms is not None and direct_margin_ms > 0
                ),
                "direct_source_rewrite_additional_save_required_ms": (
                    None if direct_margin_ms is None else max(0.0, -direct_margin_ms)
                ),
                "corr": row.get("corr"),
                "snr_db": row.get("snr_db"),
            }
        )
    rows.sort(key=lambda item: (item["additional_save_required_ms"], item["bucket"]))
    return {
        "rows": rows,
        "paper_rows_closed_by_independent_projection": sum(
            1 for row in rows if row["would_beat_paper_with_rewrite"]
        ),
        "paper_rows_remaining_after_independent_projection": sum(
            1 for row in rows if not row["would_beat_paper_with_rewrite"]
        ),
        "paper_rows_closed_by_direct_source_rewrite": sum(
            1 for row in rows if row["would_beat_paper_with_direct_source_rewrite"]
        ),
        "paper_rows_with_direct_source_rewrite_measurement": sum(
            1 for row in rows if row["direct_source_rewrite"] is not None
        ),
        "decision": (
            "The independent source/body plus production-rewrite projection is "
            "optimistic and must not be treated as a direct combined measurement. "
            "Direct local source/body+upsample-rewrite probes on all accepted buckets are "
            "speed-positive, but much smaller than the production-rewrite projection; "
            "Irvine paper rows still require another implementation win."
        ),
    }


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_num(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the combined Irvine paper-frontier path as Markdown."""

    lines = [
        "# Irvine Paper Frontier Path",
        "",
        "Warmed inference only. This report combines the best saved Irvine",
        "source/body candidate per bucket with the measured HAR-post rewrite",
        "projection. It uses the stricter paper-facing `competitive_frontier`",
        "rows, not only newer stage-profile rows.",
        "",
        f"Paper rows closed by independent source+rewrite projection: `{summary['paper_rows_closed_by_independent_projection']}`.",
        f"Paper rows still open after independent projection: `{summary['paper_rows_remaining_after_independent_projection']}`.",
        f"Paper rows closed by direct measured source/body rewrite: `{summary['paper_rows_closed_by_direct_source_rewrite']}`.",
        f"Rows with direct source/body rewrite measurement: `{summary['paper_rows_with_direct_source_rewrite_measurement']}`.",
        "",
        "| Bucket | Candidate | Human | Paper frontier | Source projected | Independent rewrite save | Independent projected | Independent margin | Direct source rewrite | Direct margin | Quality |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        margin = row["combined_paper_margin_ms"]
        margin_text = _fmt_ms(margin)
        if margin > 0:
            margin_text = f"+{margin_text}"
        direct = row.get("direct_source_rewrite") or {}
        direct_margin = row.get("direct_source_rewrite_paper_margin_ms")
        direct_margin_text = "n/a" if direct_margin is None else _fmt_ms(direct_margin)
        if direct_margin is not None and direct_margin > 0:
            direct_margin_text = f"+{direct_margin_text}"
        direct_text = (
            "n/a"
            if not direct
            else f"{_fmt_ms(direct['local_direct_rewrite_save_ms'])} local save"
        )
        quality = f"corr {_fmt_num(row.get('corr'), 3)}, SNR {_fmt_num(row.get('snr_db'), 2)} dB"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    f"`{row['candidate']}`",
                    row.get("human_decision") or "blank",
                    f"{row['paper_competitor']}: {_fmt_ms(row['paper_competitor_ms'])}",
                    _fmt_ms(row["source_projected_ms"]),
                    _fmt_ms(row["rewrite_save_ms"]),
                    _fmt_ms(row["combined_projected_ms"]),
                    margin_text,
                    direct_text,
                    direct_margin_text,
                    quality,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
            "## Immediate Work",
            "",
            "- Do not publish the independent source+rewrite projection as a win.",
            "- Do not promote direct source/body+upsample-rewrite for Irvine; local direct saves are too small on all accepted buckets.",
            "- Find another source/body implementation win of about `10 ms` for `10s`, `20 ms` for `15s`, and much larger wins for `3s`/`7s`, or prove a better direct stack on quiet Irvine.",
            "- Treat Irvine `3s` and `7s` as unsolved implementation work; current saved source candidates remain far short of the paper frontier.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-end-gate-json", type=Path, default=DEFAULT_LOWER_END_GATE_JSON)
    parser.add_argument("--rewrite-impact-json", type=Path, default=DEFAULT_REWRITE_IMPACT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(
        _load_json(args.lower_end_gate_json),
        _load_json(args.rewrite_impact_json),
        DEFAULT_DIRECT_SOURCE_REWRITE_REPORTS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_rows_closed_by_independent_projection": summary[
                    "paper_rows_closed_by_independent_projection"
                ],
                "paper_rows_remaining_after_independent_projection": summary[
                    "paper_rows_remaining_after_independent_projection"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
