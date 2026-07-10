#!/usr/bin/env python3
"""Summarize the remaining real Irvine M1 frontier targets.

This report is intentionally downstream of the warmed-inference frontier,
freshness, stage-gap, and saved-candidate reports. It does not promote new
timing data; it turns existing machine-readable evidence into the next concrete
optimization targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_FRESHNESS_JSON = Path("outputs/external_bakeoff/frontier_freshness.json")
DEFAULT_STAGE_GAPS_JSON = Path("outputs/external_bakeoff/stage_gap_decomposition.json")
DEFAULT_GAP_CANDIDATES_JSON = Path("outputs/external_bakeoff/frontier_gap_candidates.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/irvine_next_targets.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/irvine_next_targets.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _ms_from_seconds(value: Any) -> float | None:
    return None if value is None else float(value) * 1000.0


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _candidate_text(candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "none"
    delta = float(candidate.get("delta_ms") or 0.0)
    margin = abs(float(candidate.get("estimated_margin_ms") or 0.0))
    if delta >= 0.0:
        change = f"{_fmt_ms(delta)} saved"
    else:
        change = f"{_fmt_ms(abs(delta))} slower"
    return f"`{candidate['label']}` ({change}; still {_fmt_ms(margin)} short)"


def _quality_profile_counterfactual(
    config_ms: Any,
    laishere_ms: Any,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if config_ms is None or laishere_ms is None or candidate is None:
        return None
    delta_ms = float(candidate.get("delta_ms") or 0.0)
    projected_ms = float(config_ms) - delta_ms
    margin_ms = float(laishere_ms) - projected_ms
    return {
        "projected_config_f_ms": projected_ms,
        "profile_margin_ms": margin_ms,
        "would_beat_warmed_laishere_profile": margin_ms > 0.0,
    }


def _stage_index(stage_gaps: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = stage_gaps.get("rows") or []
    return {(str(row.get("machine_id")), str(row.get("input_key"))): row for row in rows}


def _candidate_index(gap_candidates: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cells = gap_candidates.get("loss_cells") or []
    return {(str(cell.get("machine_id")), str(cell.get("input_key"))): cell for cell in cells}


def _positive_quality_fail_signal(cell: dict[str, Any] | None) -> dict[str, Any] | None:
    if not cell:
        return None
    for candidate in cell.get("top_candidates") or []:
        if candidate.get("quality_status") == "quality-fail" and float(candidate.get("delta_ms") or 0.0) > 0.0:
            return candidate
    return None


def _best_strict_candidate(cell: dict[str, Any] | None) -> dict[str, Any] | None:
    if not cell:
        return None
    strict = [item for item in cell.get("top_candidates") or [] if item.get("quality_status") == "strict-pass"]
    if not strict:
        return None
    return max(strict, key=lambda item: float(item.get("delta_ms") or 0.0))


def _target_class(total_gap_ms: float | None, source_gap_ms: float | None, other_gap_ms: float | None) -> str:
    source = float(source_gap_ms or 0.0)
    other = float(other_gap_ms or 0.0)
    total = abs(float(total_gap_ms or 0.0))
    material_other = other > 0.0 and total > 0.0 and other >= total * 0.25
    if source > 0.0 and material_other:
        return "source/body primary; upstream/runtime material"
    if source > max(0.0, other):
        return "source/body dominates"
    if other > 0.0:
        return "upstream/runtime dominates"
    return "no positive matched-stage gap"


def summarize_targets(
    freshness: dict[str, Any],
    stage_gaps: dict[str, Any],
    gap_candidates: dict[str, Any],
) -> dict[str, Any]:
    """Return the remaining real Irvine loss cells and next actions."""

    stages = _stage_index(stage_gaps)
    candidates = _candidate_index(gap_candidates)
    rows: list[dict[str, Any]] = []
    for loss in freshness.get("loss_rows") or []:
        if loss.get("machine_id") != "irvine-m1":
            continue
        if loss.get("profile_outcome") != "profile-config-f-loses":
            continue
        key = (str(loss["machine_id"]), str(loss["input_key"]))
        stage = stages.get(key, {})
        cell = candidates.get(key)
        strict = _best_strict_candidate(cell)
        quality_fail = _positive_quality_fail_signal(cell)
        source_gap_ms = _ms_from_seconds(stage.get("config_generator_minus_laishere_nvt_s"))
        other_gap_ms = _ms_from_seconds(stage.get("config_nongenerator_minus_laishere_other_prepare_s"))
        total_gap_ms = loss.get("profile_gap_ms")
        rows.append(
            {
                "machine_id": key[0],
                "input_key": key[1],
                "config_f_ms": loss.get("profile_config_f_ms"),
                "laishere_ms": loss.get("profile_laishere_ms"),
                "gap_ms": total_gap_ms,
                "gap_pct": loss.get("profile_gap_pct"),
                "source_body_gap_ms": source_gap_ms,
                "upstream_runtime_gap_ms": other_gap_ms,
                "target_class": _target_class(total_gap_ms, source_gap_ms, other_gap_ms),
                "strict_pass_closers": 0 if cell is None else int(cell.get("strict_pass_closers") or 0),
                "quality_fail_closers": 0 if cell is None else int(cell.get("quality_fail_closers") or 0),
                "best_strict_candidate": None
                if strict is None
                else {
                    "label": strict.get("label"),
                    "family": strict.get("family"),
                    "delta_ms": strict.get("delta_ms"),
                    "estimated_margin_ms": strict.get("estimated_margin_ms"),
                },
                "best_quality_fail_signal": None
                if quality_fail is None
                else {
                    "label": quality_fail.get("label"),
                    "family": quality_fail.get("family"),
                    "delta_ms": quality_fail.get("delta_ms"),
                    "estimated_margin_ms": quality_fail.get("estimated_margin_ms"),
                    "corr": quality_fail.get("corr"),
                    "snr_db": quality_fail.get("snr_db"),
                },
                "quality_fail_vs_warmed_profile": _quality_profile_counterfactual(
                    loss.get("profile_config_f_ms"),
                    loss.get("profile_laishere_ms"),
                    quality_fail,
                ),
            }
        )

    rows.sort(key=lambda row: ["3s", "7s", "10s", "15s", "30s"].index(row["input_key"]))
    return {
        "machine_id": "irvine-m1",
        "real_loss_count": len(rows),
        "rows": rows,
        "strict_pass_closers": sum(row["strict_pass_closers"] for row in rows),
        "quality_fail_closers": sum(row["quality_fail_closers"] for row in rows),
        "quality_fail_warmed_profile_closers": sum(
            1
            for row in rows
            if (row.get("quality_fail_vs_warmed_profile") or {}).get(
                "would_beat_warmed_laishere_profile"
            )
        ),
        "next_actions": [
            "Do not promote fresh Irvine timing until background indexing/media analysis is idle.",
            "Strict path: find a single-package or narrower exact-HAR source/body graph surface; saved strict-pass probes do not close any current Irvine loss.",
            "Non-strict path: F0/source simplification has speed signal but remains quality-fail and needs human listening acceptance or reformulation.",
            "iPhone path: once the device is unlocked, launch the installed Config F runner, wait for compile/warm inference, then pull and ingest the app Documents result.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Irvine Next Targets",
        "",
        "Warmed inference only. This narrows the current frontier to real Irvine M1",
        "losses after filtering stale/tie paper-facing rows.",
        "",
        f"Real Irvine loss rows: `{summary['real_loss_count']}`.",
        f"Saved strict-pass candidates that close these losses: `{summary['strict_pass_closers']}`.",
        f"Saved quality-fail candidates that close these losses: `{summary['quality_fail_closers']}`.",
        f"Quality-fail candidates that would beat warmed laishere profile: `{summary['quality_fail_warmed_profile_closers']}`.",
        "",
        "| Bucket | Config F | laishere | Gap | Source/body gap | Upstream/runtime gap | Target class | Best saved strict pass | Best quality-fail speed signal | Quality-fail vs warmed profile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        strict_text = _candidate_text(row["best_strict_candidate"])
        quality_text = _candidate_text(row["best_quality_fail_signal"])
        profile = row.get("quality_fail_vs_warmed_profile") or {}
        profile_margin = profile.get("profile_margin_ms")
        if profile:
            outcome = "win" if profile["would_beat_warmed_laishere_profile"] else "loss"
            profile_text = (
                f"{outcome}: projected {_fmt_ms(profile['projected_config_f_ms'])}, "
                f"margin {_fmt_ms(profile_margin)}"
            )
        else:
            profile_text = "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["input_key"],
                    _fmt_ms(row["config_f_ms"]),
                    _fmt_ms(row["laishere_ms"]),
                    f"{_fmt_ms(row['gap_ms'])} / {float(row['gap_pct']):.2f}%",
                    _fmt_ms(row["source_body_gap_ms"]),
                    _fmt_ms(row["upstream_runtime_gap_ms"]),
                    row["target_class"],
                    strict_text,
                    quality_text,
                    profile_text,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Next Actions", ""])
    for action in summary["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freshness-json", type=Path, default=DEFAULT_FRESHNESS_JSON)
    parser.add_argument("--stage-gaps-json", type=Path, default=DEFAULT_STAGE_GAPS_JSON)
    parser.add_argument("--gap-candidates-json", type=Path, default=DEFAULT_GAP_CANDIDATES_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = summarize_targets(
        _load_json(args.freshness_json),
        _load_json(args.stage_gaps_json),
        _load_json(args.gap_candidates_json),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "real_loss_count": summary["real_loss_count"],
                "strict_pass_closers": summary["strict_pass_closers"],
                "quality_fail_closers": summary["quality_fail_closers"],
                "quality_fail_warmed_profile_closers": summary["quality_fail_warmed_profile_closers"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
