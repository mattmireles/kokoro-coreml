#!/usr/bin/env python3
"""Summarize the lower-end Mac gates needed to beat every competitor.

The report is deliberately narrower than the global frontier. It focuses on
the machines where Config F is not already plainly ahead: Irvine M1 and M2 Air.
All projections use warmed medians only.
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
DEFAULT_IRVINE_TARGETS_JSON = DEFAULT_OUTPUT_DIR / "irvine_next_targets.json"
DEFAULT_IRVINE_DECISIONS = (
    Path("outputs")
    / "f0_source_listening"
    / "irvine_exact_speed_branch"
    / "f0_source_listening_decisions.csv"
)
DEFAULT_M2_DECISIONS = (
    Path("outputs")
    / "f0_source_listening"
    / "m2_air_3s_source_body"
    / "f0_source_listening_decisions.csv"
)
DEFAULT_M2_REPORTS = [
    Path("outputs/f0_noise_exact_shape/3s_natural_asr_cos_rsqrt/report_f0_noise_exact_3s_m2_air.json"),
    Path(
        "outputs/f0_noise_exact_shape/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/"
        "report_native_in_ios17_nopal_m2_air.json"
    ),
]
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "lower_end_mac_win_gate.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "lower_end_mac_win_gate.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_decisions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    decisions: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("label") or "").strip()
            if label:
                decisions[label] = (row.get("human_decision") or "").strip().lower()
    return decisions


def _frontier_cell(frontier: dict[str, Any], machine_id: str, input_key: str) -> dict[str, Any]:
    for cell in frontier.get("summary", {}).get("cells") or []:
        if cell.get("machine_id") == machine_id and cell.get("input_key") == input_key:
            return cell
    raise ValueError(f"missing frontier cell for {machine_id} {input_key}")


def _freshness_row(freshness: dict[str, Any], machine_id: str, input_key: str) -> dict[str, Any] | None:
    for row in freshness.get("loss_rows") or []:
        if row.get("machine_id") == machine_id and row.get("input_key") == input_key:
            return row
    return None


def _fmt_label_from_report(path: Path, report: dict[str, Any]) -> str:
    report_path = str(report.get("report") or path)
    return Path(report_path).parent.name


def _metric(report: dict[str, Any], name: str) -> float | None:
    metrics = report.get("benchmark", {}).get("metrics", {}).get("candidate_vs_baseline_trimmed", {})
    value = metrics.get(name)
    return None if value is None else float(value)


def _m2_candidate_rows(
    frontier: dict[str, Any],
    freshness: dict[str, Any],
    reports: list[Path],
    decisions: dict[str, str],
) -> list[dict[str, Any]]:
    cell = _frontier_cell(frontier, "m2-air", "3s")
    fresh = _freshness_row(freshness, "m2-air", "3s") or {}
    rows: list[dict[str, Any]] = []
    for path in reports:
        report = _load_json(path)
        med = report.get("benchmark", {}).get("warm_predict_median_ms", {})
        baseline_stack_ms = float(med["baseline_total"])
        candidate_stack_ms = float(med["candidate_total"])
        config_ms = float(cell["config_f_warm_median_ms"])
        paper_best_ms = float(cell["best_warm_median_ms"])
        profile_best_ms = fresh.get("profile_laishere_ms")
        if profile_best_ms is not None:
            profile_best_ms = float(profile_best_ms)
        projected_full_ms = config_ms - baseline_stack_ms + candidate_stack_ms
        label = _fmt_label_from_report(path, report)
        rows.append(
            {
                "machine_id": "m2-air",
                "input_key": "3s",
                "candidate": label,
                "gate": "human-listening",
                "human_decision": decisions.get(label, ""),
                "current_config_f_ms": config_ms,
                "paper_competitor_ms": paper_best_ms,
                "paper_competitor": cell.get("best_impl_label"),
                "profile_competitor_ms": profile_best_ms,
                "profile_competitor": "laishere" if profile_best_ms is not None else None,
                "baseline_stack_ms": baseline_stack_ms,
                "candidate_stack_ms": candidate_stack_ms,
                "projected_full_ms": projected_full_ms,
                "paper_margin_ms": paper_best_ms - projected_full_ms,
                "profile_margin_ms": None if profile_best_ms is None else profile_best_ms - projected_full_ms,
                "would_win_paper_if_accepted": projected_full_ms < paper_best_ms,
                "would_win_profile_if_accepted": (
                    False if profile_best_ms is None else projected_full_ms < profile_best_ms
                ),
                "corr": _metric(report, "correlation"),
                "snr_db": _metric(report, "snr_db"),
                "frontier_loss_looks_stale_or_tie": bool(fresh.get("frontier_loss_looks_stale_or_tie")),
                "source_report": str(path),
            }
        )
    return rows


def _irvine_candidate_rows(
    frontier: dict[str, Any], irvine_targets: dict[str, Any], decisions: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in irvine_targets.get("rows") or []:
        signal = row.get("best_quality_fail_signal") or {}
        profile = row.get("quality_fail_vs_warmed_profile") or {}
        label = signal.get("label")
        if not label:
            continue
        cell = _frontier_cell(frontier, "irvine-m1", str(row.get("input_key")))
        paper_best_ms = float(cell["best_warm_median_ms"])
        projected_full_ms = profile.get("projected_config_f_ms")
        profile_margin_ms = profile.get("profile_margin_ms")
        paper_margin_ms = None if projected_full_ms is None else paper_best_ms - float(projected_full_ms)
        rows.append(
            {
                "machine_id": "irvine-m1",
                "input_key": row.get("input_key"),
                "candidate": label,
                "gate": (
                    "human-listening"
                    if paper_margin_ms is not None and paper_margin_ms > 0
                    else "paper-frontier-gap"
                    if profile.get("would_beat_warmed_laishere_profile")
                    else "implementation-gap"
                ),
                "human_decision": decisions.get(str(label), ""),
                "current_config_f_ms": row.get("config_f_ms"),
                "paper_competitor_ms": paper_best_ms,
                "paper_competitor": cell.get("best_impl_label"),
                "profile_competitor_ms": row.get("laishere_ms"),
                "profile_competitor": "laishere",
                "projected_full_ms": projected_full_ms,
                "paper_margin_ms": paper_margin_ms,
                "profile_margin_ms": profile_margin_ms,
                "would_win_paper_if_accepted": bool(paper_margin_ms is not None and paper_margin_ms > 0),
                "would_win_profile_if_accepted": bool(profile.get("would_beat_warmed_laishere_profile")),
                "corr": signal.get("corr"),
                "snr_db": signal.get("snr_db"),
                "source_body_gap_ms": row.get("source_body_gap_ms"),
                "upstream_runtime_gap_ms": row.get("upstream_runtime_gap_ms"),
            }
        )
    return rows


def build_summary(
    frontier: dict[str, Any],
    freshness: dict[str, Any],
    irvine_targets: dict[str, Any],
    *,
    irvine_decisions: dict[str, str],
    m2_decisions: dict[str, str],
    m2_reports: list[Path],
) -> dict[str, Any]:
    """Build the lower-end Mac win-gate summary."""

    m2_rows = _m2_candidate_rows(frontier, freshness, m2_reports, m2_decisions)
    irvine_rows = _irvine_candidate_rows(frontier, irvine_targets, irvine_decisions)
    all_rows = m2_rows + irvine_rows
    pending_paper_wins = [
        row
        for row in all_rows
        if row["would_win_paper_if_accepted"] and row.get("human_decision") not in {"pass", "caveat"}
    ]
    accepted_paper_wins = [
        row
        for row in all_rows
        if row["would_win_paper_if_accepted"] and row.get("human_decision") in {"pass", "caveat"}
    ]
    pending_profile_wins = [
        row
        for row in all_rows
        if row["would_win_profile_if_accepted"] and row.get("human_decision") not in {"pass", "caveat"}
    ]
    accepted_profile_wins = [
        row
        for row in all_rows
        if row["would_win_profile_if_accepted"] and row.get("human_decision") in {"pass", "caveat"}
    ]
    paper_blocked_rows = [row for row in all_rows if not row["would_win_paper_if_accepted"]]
    return {
        "scope": ["m2-air", "irvine-m1"],
        "warmed_only": True,
        "m2_air_candidate_count": len(m2_rows),
        "irvine_candidate_count": len(irvine_rows),
        "pending_paper_listening_win_count": len(pending_paper_wins),
        "accepted_paper_win_count": len(accepted_paper_wins),
        "pending_profile_listening_win_count": len(pending_profile_wins),
        "accepted_profile_win_count": len(accepted_profile_wins),
        "paper_blocked_row_count": len(paper_blocked_rows),
        "m2_air_rows": m2_rows,
        "irvine_rows": irvine_rows,
        "pending_paper_listening_wins": pending_paper_wins,
        "pending_profile_listening_wins": pending_profile_wins,
        "accepted_profile_wins": accepted_profile_wins,
        "paper_blocked_rows": paper_blocked_rows,
        "decision": (
            "M2 Air 3s has paper-frontier wins gated only by human listening. "
            "Irvine source/body candidates can beat several newer warmed profile "
            "rows after no-ASR listening acceptance, but none beats the stricter "
            "paper-facing frontier yet. Irvine "
            "still needs a combined implementation win, not just listening acceptance."
        ),
    }


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_num(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _render_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Machine | Bucket | Candidate | Gate | Human | Current Config F | Projected | Paper margin | Profile margin | Quality |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        quality = f"corr {_fmt_num(row.get('corr'), 3)}, SNR {_fmt_num(row.get('snr_db'), 2)} dB"
        paper_margin = row.get("paper_margin_ms")
        paper_margin_text = _fmt_ms(paper_margin)
        if paper_margin is not None and float(paper_margin) > 0:
            paper_margin_text = f"+{paper_margin_text}"
        profile_margin = row.get("profile_margin_ms")
        profile_margin_text = _fmt_ms(profile_margin)
        if profile_margin is not None and float(profile_margin) > 0:
            profile_margin_text = f"+{profile_margin_text}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["machine_id"]),
                    f"`{row['input_key']}`",
                    f"`{row['candidate']}`",
                    str(row["gate"]),
                    row.get("human_decision") or "blank",
                    _fmt_ms(row.get("current_config_f_ms")),
                    _fmt_ms(row.get("projected_full_ms")),
                    f"{paper_margin_text} vs {row.get('paper_competitor')}: {_fmt_ms(row.get('paper_competitor_ms'))}",
                    f"{profile_margin_text} vs {row.get('profile_competitor') or '-'}: {_fmt_ms(row.get('profile_competitor_ms'))}",
                    quality,
                ]
            )
            + " |"
        )
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    immediate_work = []
    if summary["pending_paper_listening_win_count"]:
        immediate_work.append("- Get no-ASR human listening decisions for paper-frontier listening wins.")
    if summary["pending_profile_listening_win_count"]:
        immediate_work.append("- Get no-ASR human listening decisions for remaining profile-only listening wins.")
    if summary["accepted_profile_win_count"]:
        immediate_work.append(
            "- Treat accepted Irvine profile-only source/body wins as quality-approved speed signals, not paper-frontier wins."
        )
    immediate_work.extend(
        [
            "- Keep Irvine `3s`, `7s`, `10s`, and `15s` as implementation targets for the stricter paper-facing frontier.",
            "- After acceptance, rerun warmed publishable lower-end rows on quiet hosts and refresh `competitive_frontier`.",
        ]
    )
    lines = [
        "# Lower-End Mac Win Gate",
        "",
        "Warmed inference only. This report tracks the remaining `m2-air` and",
        "`irvine-m1` gates needed before claiming we beat the external Apple",
        "Silicon implementations on lower-end Macs.",
        "",
        f"Pending paper-frontier listening wins: `{summary['pending_paper_listening_win_count']}`.",
        f"Accepted paper-frontier wins: `{summary['accepted_paper_win_count']}`.",
        f"Pending profile-only listening wins: `{summary['pending_profile_listening_win_count']}`.",
        f"Accepted profile-only wins: `{summary['accepted_profile_win_count']}`.",
        f"Paper-frontier blocked rows: `{summary['paper_blocked_row_count']}`.",
        "",
        "## Candidate Gates",
        "",
    ]
    lines.extend(_render_rows(summary["m2_air_rows"] + summary["irvine_rows"]))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
            "## Immediate Work",
            "",
        ]
    )
    lines.extend(immediate_work)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER_JSON)
    parser.add_argument("--freshness-json", type=Path, default=DEFAULT_FRESHNESS_JSON)
    parser.add_argument("--irvine-targets-json", type=Path, default=DEFAULT_IRVINE_TARGETS_JSON)
    parser.add_argument("--irvine-decisions", type=Path, default=DEFAULT_IRVINE_DECISIONS)
    parser.add_argument("--m2-decisions", type=Path, default=DEFAULT_M2_DECISIONS)
    parser.add_argument("--m2-report", action="append", type=Path, dest="m2_reports")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(
        _load_json(args.frontier_json),
        _load_json(args.freshness_json),
        _load_json(args.irvine_targets_json),
        irvine_decisions=_load_decisions(args.irvine_decisions),
        m2_decisions=_load_decisions(args.m2_decisions),
        m2_reports=args.m2_reports or DEFAULT_M2_REPORTS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pending_paper_listening_win_count": summary["pending_paper_listening_win_count"],
                "paper_blocked_row_count": summary["paper_blocked_row_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
