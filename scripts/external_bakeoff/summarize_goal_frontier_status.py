#!/usr/bin/env python3
"""Summarize the current fastest-implementation goal status.

This report is intentionally small and downstream of the authoritative
frontier artifacts. It answers the operational question for the active goal:
what proves we are not absolute fastest yet, what can move the frontier next,
and what external state is blocking device evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_FRONTIER_JSON = DEFAULT_OUTPUT_DIR / "competitive_frontier.json"
DEFAULT_FRESHNESS_JSON = DEFAULT_OUTPUT_DIR / "frontier_freshness.json"
DEFAULT_NEXT_TARGETS_JSON = DEFAULT_OUTPUT_DIR / "irvine_next_targets.json"
DEFAULT_LISTENING_JSON = DEFAULT_OUTPUT_DIR / "irvine_listening_targets.json"
DEFAULT_IOS_INSTALL_JSON = DEFAULT_OUTPUT_DIR / "config_f_ios_manual_install_latest.json"
DEFAULT_IOS_RUN_JSON = DEFAULT_OUTPUT_DIR / "config_f_ios_run_latest.json"
DEFAULT_DECISIONS = (
    Path("outputs")
    / "f0_source_listening"
    / "irvine_exact_speed_branch"
    / "f0_source_listening_decisions.csv"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "goal_frontier_status.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "goal_frontier_status.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any]:
    """Load a JSON object if it exists, otherwise return an empty object."""

    if not path.exists():
        return {}
    return _load_json(path)


def _merge_ios_status(install: dict[str, Any], run_status: dict[str, Any]) -> dict[str, Any]:
    """Merge install evidence with the latest Config F launch/run evidence."""

    if not run_status:
        return install
    merged = dict(install)
    if run_status.get("app_bundle_id") and not merged.get("bundle_id"):
        merged["bundle_id"] = run_status["app_bundle_id"]
    for key in (
        "checked_at_local",
        "launch_blocker",
        "launch_error",
        "launch_ok",
        "lock_state",
        "ok",
        "runner",
    ):
        if key in run_status:
            merged[key] = run_status[key]
    merged["latest_run_status"] = run_status
    return merged


def _decision_counts(path: Path) -> dict[str, int]:
    """Return counts for human listening decisions in ``path``."""

    if not path.exists():
        return {"missing_csv": 1}
    counts: dict[str, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            decision = (row.get("human_decision") or "").strip().lower() or "blank"
            counts[decision] = counts.get(decision, 0) + 1
    return dict(sorted(counts.items()))


def summarize_status(
    frontier: dict[str, Any],
    freshness: dict[str, Any],
    next_targets: dict[str, Any],
    listening: dict[str, Any],
    ios_install: dict[str, Any],
    decision_counts: dict[str, int],
    *,
    irvine_load_note: str,
) -> dict[str, Any]:
    """Build the current fastest-goal status summary."""

    frontier_summary = frontier.get("summary") or {}
    real_losses = [
        row for row in freshness.get("loss_rows") or [] if not row.get("frontier_loss_looks_stale_or_tie")
    ]
    blank_decisions = int(decision_counts.get("blank", 0))
    blockers: list[str] = []
    if not frontier_summary.get("absolute_fastest_verified"):
        blockers.append("absolute_fastest_verified is false")
    if next_targets.get("strict_pass_closers", 0) == 0:
        blockers.append("no saved strict-pass candidate closes a real Irvine loss")
    if blank_decisions:
        blockers.append(f"{blank_decisions} Irvine no-ASR listening decisions are blank")
    if ios_install.get("launch_blocker"):
        blockers.append(f"iPhone Config F launch blocker: {ios_install.get('launch_blocker')}")
    if irvine_load_note:
        blockers.append(irvine_load_note)

    next_actions = []
    if blank_decisions:
        next_actions.append("Collect no-ASR human decisions for Irvine F0/source speed candidates.")
    next_actions.extend(
        [
            "Retry iPhone Config F runner only after the physical device is unlocked.",
            "Run publishable Irvine timings only after mediaanalysisd/Spotlight load is idle.",
            "Create a new strict single-package or source/body formulation; existing strict probes do not close losses.",
        ]
    )

    return {
        "absolute_fastest_verified": bool(frontier_summary.get("absolute_fastest_verified")),
        "config_f_loss_cells": int(frontier_summary.get("config_f_loss_count") or len(frontier_summary.get("config_f_losses") or [])),
        "config_f_missing_cells": int(frontier_summary.get("config_f_missing_count") or len(frontier_summary.get("config_f_missing") or [])),
        "no_full_duration_result_cells": int(
            frontier_summary.get("no_full_duration_result_count")
            or len(frontier_summary.get("no_full_duration_result") or [])
        ),
        "real_irvine_loss_count": len(real_losses),
        "stale_or_tie_loss_count": int(freshness.get("stale_or_tie_loss_count") or 0),
        "strict_pass_closers": int(next_targets.get("strict_pass_closers") or 0),
        "quality_fail_closers": int(next_targets.get("quality_fail_closers") or 0),
        "quality_fail_warmed_profile_closers": int(next_targets.get("quality_fail_warmed_profile_closers") or 0),
        "listening_targets": {
            "rows": int(listening.get("row_count") or 0),
            "mapped": int(listening.get("mapped_count") or 0),
            "exact_timing_report_artifacts": int(listening.get("exact_timing_report_listening_artifact_count") or 0),
            "decision_counts": decision_counts,
        },
        "iphone": {
            "install_ok": bool(ios_install.get("install_ok")),
            "launch_ok": bool(ios_install.get("launch_ok")),
            "launch_blocker": ios_install.get("launch_blocker"),
            "bundle_id": ios_install.get("bundle_id"),
            "latest_run_checked_at_local": ios_install.get("checked_at_local"),
            "device": ios_install.get("device") or {},
        },
        "blockers": blockers,
        "next_actions": next_actions,
        "real_irvine_losses": [
            {
                "bucket": row.get("input_key"),
                "config_f_ms": row.get("profile_config_f_ms"),
                "laishere_ms": row.get("profile_laishere_ms"),
                "gap_ms": row.get("profile_gap_ms"),
                "gap_pct": row.get("profile_gap_pct"),
            }
            for row in real_losses
        ],
    }


def _fmt_ms(value: Any) -> str:
    """Format milliseconds for Markdown."""

    return "n/a" if value is None else f"{float(value):.1f} ms"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a concise Markdown status report."""

    listening = summary["listening_targets"]
    iphone = summary["iphone"]
    lines = [
        "# Fastest Goal Frontier Status",
        "",
        f"Absolute fastest verified: `{str(summary['absolute_fastest_verified']).lower()}`.",
        f"Config F loss cells: `{summary['config_f_loss_cells']}`.",
        f"Real Irvine loss cells: `{summary['real_irvine_loss_count']}`.",
        f"Stale/tie loss cells: `{summary['stale_or_tie_loss_count']}`.",
        f"Strict-pass closers: `{summary['strict_pass_closers']}`.",
        f"Quality-fail warmed-profile closers: `{summary['quality_fail_warmed_profile_closers']}`.",
        "",
        "## Real Irvine Losses",
        "",
        "| Bucket | Config F | laishere | Gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["real_irvine_losses"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["config_f_ms"]),
                    _fmt_ms(row["laishere_ms"]),
                    f"{_fmt_ms(row['gap_ms'])} / {float(row['gap_pct']):.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## No-ASR Listening",
            "",
            f"Targets: `{listening['rows']}`; mapped artifacts: `{listening['mapped']}`; "
            f"exact timing artifacts: `{listening['exact_timing_report_artifacts']}`.",
            f"Decision counts: `{json.dumps(listening['decision_counts'], sort_keys=True)}`.",
            "",
            "## iPhone",
            "",
            f"Install OK: `{str(iphone['install_ok']).lower()}`.",
            f"Launch OK: `{str(iphone['launch_ok']).lower()}`.",
            f"Launch blocker: `{iphone['launch_blocker'] or '-'}`.",
            f"Latest run check: `{iphone.get('latest_run_checked_at_local') or '-'}`.",
            f"Bundle: `{iphone['bundle_id'] or '-'}`.",
            "",
            "## Blockers",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["blockers"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in summary["next_actions"])
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER_JSON)
    parser.add_argument("--freshness-json", type=Path, default=DEFAULT_FRESHNESS_JSON)
    parser.add_argument("--next-targets-json", type=Path, default=DEFAULT_NEXT_TARGETS_JSON)
    parser.add_argument("--listening-json", type=Path, default=DEFAULT_LISTENING_JSON)
    parser.add_argument("--ios-install-json", type=Path, default=DEFAULT_IOS_INSTALL_JSON)
    parser.add_argument("--ios-run-json", type=Path, default=DEFAULT_IOS_RUN_JSON)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--irvine-load-note", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    ios_status = _merge_ios_status(_load_json(args.ios_install_json), _load_optional_json(args.ios_run_json))
    summary = summarize_status(
        _load_json(args.frontier_json),
        _load_json(args.freshness_json),
        _load_json(args.next_targets_json),
        _load_json(args.listening_json),
        ios_status,
        _decision_counts(args.decisions),
        irvine_load_note=args.irvine_load_note,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "absolute_fastest_verified": summary["absolute_fastest_verified"],
                "blocker_count": len(summary["blockers"]),
                "output": str(args.output),
                "real_irvine_loss_count": summary["real_irvine_loss_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
