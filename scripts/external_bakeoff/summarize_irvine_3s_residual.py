#!/usr/bin/env python3
"""Summarize the remaining Irvine 3s gap after saved speed branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_NEXT_TARGETS_JSON = Path("outputs/external_bakeoff/irvine_next_targets.json")
DEFAULT_GAP_CANDIDATES_JSON = Path("outputs/external_bakeoff/frontier_gap_candidates.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/irvine_3s_residual.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/irvine_3s_residual.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _irvine_3s_target(next_targets: dict[str, Any]) -> dict[str, Any]:
    for row in next_targets.get("rows") or []:
        if row.get("machine_id") == "irvine-m1" and row.get("input_key") == "3s":
            return row
    raise ValueError("missing irvine-m1 3s row in next targets")


def _irvine_3s_candidates(gap_candidates: dict[str, Any]) -> list[dict[str, Any]]:
    for cell in gap_candidates.get("loss_cells") or []:
        if cell.get("machine_id") == "irvine-m1" and cell.get("input_key") == "3s":
            return list(cell.get("top_candidates") or [])
    return []


def _best_positive_strict(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    strict = [
        item
        for item in candidates
        if item.get("quality_status") == "strict-pass" and float(item.get("delta_ms") or 0.0) > 0.0
    ]
    if not strict:
        return None
    return max(strict, key=lambda item: float(item.get("delta_ms") or 0.0))


def _quality_fail_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        item
        for item in candidates
        if item.get("quality_status") == "quality-fail" and float(item.get("delta_ms") or 0.0) > 0.0
    ]
    return sorted(rows, key=lambda item: float(item.get("delta_ms") or 0.0), reverse=True)


def summarize_residual(next_targets: dict[str, Any], gap_candidates: dict[str, Any]) -> dict[str, Any]:
    """Return the numeric residual budget for the remaining Irvine 3s loss."""

    target = _irvine_3s_target(next_targets)
    candidates = _irvine_3s_candidates(gap_candidates)
    best_quality = target.get("best_quality_fail_signal") or {}
    best_strict = _best_positive_strict(candidates)
    config_ms = float(target["config_f_ms"])
    laishere_ms = float(target["laishere_ms"])
    profile_gap_ms = config_ms - laishere_ms
    quality_delta_ms = float(best_quality.get("delta_ms") or 0.0)
    strict_delta_ms = 0.0 if best_strict is None else float(best_strict.get("delta_ms") or 0.0)
    upstream_gap_ms = max(0.0, float(target.get("upstream_runtime_gap_ms") or 0.0))
    source_body_gap_ms = max(0.0, float(target.get("source_body_gap_ms") or 0.0))
    after_quality_ms = profile_gap_ms - quality_delta_ms
    after_quality_and_strict_ms = after_quality_ms - strict_delta_ms
    after_quality_and_upstream_ms = after_quality_ms - upstream_gap_ms
    after_all_known_positive_ms = after_quality_ms - strict_delta_ms - upstream_gap_ms
    return {
        "machine_id": "irvine-m1",
        "input_key": "3s",
        "config_f_ms": config_ms,
        "laishere_profile_ms": laishere_ms,
        "profile_gap_ms": profile_gap_ms,
        "source_body_gap_ms": source_body_gap_ms,
        "upstream_runtime_gap_ms": upstream_gap_ms,
        "best_quality_fail_signal": best_quality,
        "best_positive_strict_signal": None
        if best_strict is None
        else {
            "label": best_strict.get("label"),
            "family": best_strict.get("family"),
            "delta_ms": strict_delta_ms,
            "corr": best_strict.get("corr"),
            "snr_db": best_strict.get("snr_db"),
        },
        "residual_after_best_quality_fail_ms": after_quality_ms,
        "residual_after_quality_plus_best_strict_ms": after_quality_and_strict_ms,
        "residual_after_quality_plus_upstream_gap_ms": after_quality_and_upstream_ms,
        "residual_after_all_known_positive_estimates_ms": after_all_known_positive_ms,
        "known_positive_estimates_close_3s": after_all_known_positive_ms <= 0.0,
        "quality_fail_candidates": [
            {
                "label": row.get("label"),
                "delta_ms": row.get("delta_ms"),
                "corr": row.get("corr"),
                "snr_db": row.get("snr_db"),
                "path": row.get("path"),
            }
            for row in _quality_fail_rows(candidates)
        ],
        "interpretation": (
            "Saved 3s signals do not close warmed laishere. Even after the best "
            "quality-fail F0/source branch, the estimated residual is material; "
            "removing the matched upstream gap still leaves a positive residual."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    strict = summary.get("best_positive_strict_signal")
    quality = summary.get("best_quality_fail_signal") or {}
    lines = [
        "# Irvine 3s Residual",
        "",
        "Warmed profile denominator. Additive rows are estimates from saved",
        "sub-stack probes, not full-path promotion proof.",
        "",
        f"Config F: `{_fmt_ms(summary['config_f_ms'])}`.",
        f"laishere warmed profile: `{_fmt_ms(summary['laishere_profile_ms'])}`.",
        f"Profile gap: `{_fmt_ms(summary['profile_gap_ms'])}`.",
        f"Matched source/body gap: `{_fmt_ms(summary['source_body_gap_ms'])}`.",
        f"Matched upstream/runtime gap: `{_fmt_ms(summary['upstream_runtime_gap_ms'])}`.",
        "",
        "## Residual Budget",
        "",
        "| Scenario | Remaining gap vs warmed laishere |",
        "| --- | ---: |",
        f"| Best quality-fail signal `{quality.get('label')}` | {_fmt_ms(summary['residual_after_best_quality_fail_ms'])} |",
        f"| Best quality-fail + best strict signal `{None if strict is None else strict.get('label')}` | {_fmt_ms(summary['residual_after_quality_plus_best_strict_ms'])} |",
        f"| Best quality-fail + eliminate matched upstream/runtime gap | {_fmt_ms(summary['residual_after_quality_plus_upstream_gap_ms'])} |",
        f"| Best quality-fail + best strict + eliminate upstream/runtime gap | {_fmt_ms(summary['residual_after_all_known_positive_estimates_ms'])} |",
        "",
        "## Positive Quality-Fail Signals",
        "",
        "| Candidate | Delta | Corr | SNR dB | Report |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["quality_fail_candidates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    _fmt_ms(row["delta_ms"]),
                    "n/a" if row.get("corr") is None else f"{float(row['corr']):.6f}",
                    "n/a" if row.get("snr_db") is None else f"{float(row['snr_db']):.2f}",
                    f"`{row['path']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next-targets-json", type=Path, default=DEFAULT_NEXT_TARGETS_JSON)
    parser.add_argument("--gap-candidates-json", type=Path, default=DEFAULT_GAP_CANDIDATES_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()
    summary = summarize_residual(_load_json(args.next_targets_json), _load_json(args.gap_candidates_json))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "residual_after_best_quality_fail_ms": summary["residual_after_best_quality_fail_ms"],
                "residual_after_all_known_positive_estimates_ms": summary[
                    "residual_after_all_known_positive_estimates_ms"
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
