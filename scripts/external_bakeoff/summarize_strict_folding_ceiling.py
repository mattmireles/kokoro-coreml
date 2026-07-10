#!/usr/bin/env python3
"""Summarize the speed ceiling for strict HAR/STFT folding work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_HNSF_BOUNDARY = Path("outputs/external_bakeoff/hnsf_source_boundary_net.json")
DEFAULT_STRICT_BUDGET = Path("outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.json")
DEFAULT_FOLDING_SURFACE = Path("outputs/external_bakeoff/pre_noise_folding_surface.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/strict_folding_ceiling.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/strict_folding_ceiling.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def build_summary(
    hnsf_boundary: dict[str, Any],
    strict_budget: dict[str, Any],
    folding_surface: dict[str, Any],
) -> dict[str, Any]:
    """Build a joined strict-folding ceiling summary."""

    boundary_by_bucket = {str(row.get("bucket")): row for row in hnsf_boundary.get("rows") or []}
    surface_by_bucket = {str(row.get("input_key")): row for row in folding_surface.get("rows") or []}
    rows = []
    profile_closable = 0
    paper_closable = 0
    for budget in strict_budget.get("rows") or []:
        bucket = str(budget.get("bucket"))
        boundary = boundary_by_bucket.get(bucket, {})
        surface = surface_by_bucket.get(bucket, {})
        removable_stft_ms = float(boundary.get("removable_swift_stft_ms") or 0.0)
        profile_needed = float(budget.get("additional_profile_save_required_ms") or 0.0)
        paper_needed = float(budget.get("additional_paper_save_required_ms") or 0.0)
        profile_coverage = removable_stft_ms / profile_needed * 100.0 if profile_needed else None
        paper_coverage = removable_stft_ms / paper_needed * 100.0 if paper_needed else None
        profile_closes = profile_needed > 0.0 and removable_stft_ms >= profile_needed
        paper_closes = paper_needed > 0.0 and removable_stft_ms >= paper_needed
        profile_closable += int(profile_closes)
        paper_closable += int(paper_closes)
        rows.append(
            {
                "bucket": bucket,
                "removable_swift_stft_ms": removable_stft_ms,
                "strict_profile_save_needed_ms": profile_needed,
                "strict_paper_save_needed_ms": paper_needed,
                "profile_gap_coverage_pct": profile_coverage,
                "paper_gap_coverage_pct": paper_coverage,
                "profile_gap_closes_with_stft_only": profile_closes,
                "paper_gap_closes_with_stft_only": paper_closes,
                "folding_surface_decision": folding_surface.get("decision"),
                "touched_har_frame_pct": surface.get("union_touched_har_frame_pct"),
                "pre_noise_to_har_value_ratio": surface.get("pre_noise_to_har_value_ratio"),
            }
        )
    return {
        "inputs": {
            "hnsf_boundary": hnsf_boundary.get("hnsf_timing"),
            "strict_budget_candidate": strict_budget.get("candidate_label"),
            "folding_surface_decision": folding_surface.get("decision"),
        },
        "rows": rows,
        "summary": {
            "profile_rows_closed_by_stft_only": profile_closable,
            "paper_rows_closed_by_stft_only": paper_closable,
            "decision": (
                "do_not_build_materialized_pre_noise_boundary; only pursue fused runtime/kernel "
                "if it also reduces generator body scheduling or synchronization cost"
            ),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render markdown for the joined ceiling summary."""

    lines = [
        "# Strict Folding Ceiling",
        "",
        "This report joins the removable Swift STFT timing, the post-overlap+rewrite",
        "strict win budget, and the pre-noise folding surface.",
        "",
        f"- Decision: `{summary['summary']['decision']}`.",
        f"- Folding surface: `{summary['inputs']['folding_surface_decision']}`.",
        f"- Profile rows closed by STFT-only removal: `{summary['summary']['profile_rows_closed_by_stft_only']}`.",
        f"- Paper rows closed by STFT-only removal: `{summary['summary']['paper_rows_closed_by_stft_only']}`.",
        "",
        "| Bucket | Removable STFT | Profile save needed | Profile coverage | Paper save needed | Paper coverage | HAR frames touched | Pre-noise/HAR values |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        profile_coverage = row["profile_gap_coverage_pct"]
        paper_coverage = row["paper_gap_coverage_pct"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["removable_swift_stft_ms"]),
                    _fmt_ms(row["strict_profile_save_needed_ms"]),
                    "n/a" if profile_coverage is None else f"{profile_coverage:.2f}%",
                    _fmt_ms(row["strict_paper_save_needed_ms"]),
                    "n/a" if paper_coverage is None else f"{paper_coverage:.2f}%",
                    "n/a" if row["touched_har_frame_pct"] is None else f"{float(row['touched_har_frame_pct']):.2f}%",
                    "n/a" if row["pre_noise_to_har_value_ratio"] is None else f"{float(row['pre_noise_to_har_value_ratio']):.2f}x",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Removing Swift STFT alone is much smaller than the remaining lower-end Mac",
            "strict gaps. Since the pre-noise surface touches all HAR frames and",
            "pre-noise tensors are larger than HAR, a materialized pre-noise boundary",
            "is the wrong shape. A strict fused implementation must remove",
            "synchronization or improve generator-body scheduling; otherwise this",
            "branch cannot close the M1 rows.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hnsf-boundary-json", type=Path, default=DEFAULT_HNSF_BOUNDARY)
    parser.add_argument("--strict-budget-json", type=Path, default=DEFAULT_STRICT_BUDGET)
    parser.add_argument("--folding-surface-json", type=Path, default=DEFAULT_FOLDING_SURFACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(
        _load_json(args.hnsf_boundary_json),
        _load_json(args.strict_budget_json),
        _load_json(args.folding_surface_json),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "decision": summary["summary"]["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
