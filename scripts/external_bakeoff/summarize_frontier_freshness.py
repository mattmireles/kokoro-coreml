#!/usr/bin/env python3
"""Compare paper-facing frontier losses against newer stage-profile evidence.

The competitive frontier intentionally uses only corrected paper-facing result
files. Some stage-profile reruns use the same warmed chain boundary plus extra
stage timing, but are not direct replacements for the paper table. This report
flags where a frontier loss is likely stale or measurement-scale rather than a
real remaining graph/runtime deficit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import DEFAULT_OUTPUT_DIR, RUNTIME_BUCKETS, load_json  # noqa: E402


DEFAULT_FRONTIER_JSON = DEFAULT_OUTPUT_DIR / "competitive_frontier.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "frontier_freshness.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "frontier_freshness.json"
DEFAULT_CONFIG_FILES = {
    "m2-air": DEFAULT_OUTPUT_DIR / "results_config_f_reference_m2-air_vector_noise_batch.json",
    "irvine-m1": DEFAULT_OUTPUT_DIR / "results_config_f_reference_irvine-m1_vector_noise_batch.json",
}
DEFAULT_STAGE_PROFILE_FILES = {
    "m2-air": DEFAULT_OUTPUT_DIR / "placement" / "results_laishere_stage_profile_m2-air.json",
    "irvine-m1": DEFAULT_OUTPUT_DIR / "placement" / "results_laishere_stage_profile_irvine-m1.json",
}


def _record_by_bucket(payload: dict[str, Any], bucket: str) -> dict[str, Any] | None:
    for record in payload.get("records") or []:
        if str(record.get("input_key")) == bucket:
            return record
    return None


def _median_ms_from_config(record: dict[str, Any]) -> float | None:
    warm = record.get("warm_wall_times_s") or []
    if not warm:
        return None
    values = sorted(float(value) for value in warm)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle] * 1000.0
    return ((values[middle - 1] + values[middle]) / 2.0) * 1000.0


def _stage_profile_total_ms(record: dict[str, Any]) -> float | None:
    medians = record.get("warm_median_s") or {}
    value = medians.get("total_s")
    return None if value is None else float(value) * 1000.0


def _profile_outcome(gap_ms: float, tie_threshold_pct: float, config_ms: float) -> str:
    threshold_ms = abs(config_ms) * tie_threshold_pct / 100.0
    if abs(gap_ms) <= threshold_ms:
        return "profile-tie"
    return "profile-config-f-loses" if gap_ms > 0 else "profile-config-f-wins"


def summarize_freshness(
    frontier_payload: dict[str, Any],
    config_payloads: dict[str, dict[str, Any]],
    stage_profile_payloads: dict[str, dict[str, Any]],
    tie_threshold_pct: float,
) -> dict[str, Any]:
    """Return loss rows annotated with newer stage-profile evidence."""

    rows: list[dict[str, Any]] = []
    losses = frontier_payload.get("summary", {}).get("config_f_losses") or []
    for cell in losses:
        machine = str(cell.get("machine_id") or "")
        bucket = str(cell.get("input_key") or "")
        if bucket not in RUNTIME_BUCKETS:
            continue
        config_record = _record_by_bucket(config_payloads.get(machine, {}), bucket)
        stage_record = _record_by_bucket(stage_profile_payloads.get(machine, {}), bucket)
        config_ms = None if config_record is None else _median_ms_from_config(config_record)
        profile_ms = None if stage_record is None else _stage_profile_total_ms(stage_record)
        profile_gap_ms = None
        profile_gap_pct = None
        profile_outcome = "profile-missing"
        stale_or_tie = False
        if config_ms is not None and profile_ms is not None:
            profile_gap_ms = config_ms - profile_ms
            profile_gap_pct = 100.0 * profile_gap_ms / profile_ms if profile_ms else None
            profile_outcome = _profile_outcome(profile_gap_ms, tie_threshold_pct, config_ms)
            stale_or_tie = profile_outcome in {"profile-config-f-wins", "profile-tie"}
        rows.append(
            {
                "machine_id": machine,
                "input_key": bucket,
                "frontier_best_impl": cell.get("best_impl_label"),
                "frontier_best_ms": cell.get("best_warm_median_ms"),
                "frontier_config_f_ms": cell.get("config_f_warm_median_ms"),
                "frontier_required_reduction_ms": (
                    None
                    if cell.get("config_f_warm_median_ms") is None or cell.get("best_warm_median_ms") is None
                    else float(cell["config_f_warm_median_ms"]) - float(cell["best_warm_median_ms"])
                ),
                "profile_config_f_ms": config_ms,
                "profile_laishere_ms": profile_ms,
                "profile_gap_ms": profile_gap_ms,
                "profile_gap_pct": profile_gap_pct,
                "profile_outcome": profile_outcome,
                "frontier_loss_looks_stale_or_tie": stale_or_tie,
            }
        )
    return {
        "tie_threshold_pct": tie_threshold_pct,
        "loss_rows": rows,
        "frontier_loss_count": len(rows),
        "stale_or_tie_loss_count": sum(1 for row in rows if row["frontier_loss_looks_stale_or_tie"]),
        "real_profile_loss_count": sum(1 for row in rows if row["profile_outcome"] == "profile-config-f-loses"),
        "missing_profile_count": sum(1 for row in rows if row["profile_outcome"] == "profile-missing"),
    }


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render frontier freshness as Markdown."""

    lines = [
        "# Frontier Freshness",
        "",
        "Compares current paper-facing frontier losses with newer laishere",
        "stage-profile reruns. Stage-profile rows are diagnostic evidence, not",
        "automatic replacements for paper table rows.",
        "",
        f"Frontier loss rows: `{summary['frontier_loss_count']}`.",
        f"Loss rows that look stale or measurement-scale by stage profile: `{summary['stale_or_tie_loss_count']}`.",
        f"Loss rows still real by stage profile: `{summary['real_profile_loss_count']}`.",
        f"Tie threshold: `{summary['tie_threshold_pct']:.2f}%` of Config F profile median.",
        "",
        "| Machine | Bucket | Frontier required reduction | Profile Config F | Profile laishere | Profile gap | Profile outcome | Interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["loss_rows"]:
        if row["profile_outcome"] == "profile-config-f-wins":
            interp = "frontier loss likely stale"
        elif row["profile_outcome"] == "profile-tie":
            interp = "measurement-scale tie"
        elif row["profile_outcome"] == "profile-config-f-loses":
            interp = "real remaining loss"
        else:
            interp = "missing diagnostic rerun"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    row["input_key"],
                    _fmt_ms(row["frontier_required_reduction_ms"]),
                    _fmt_ms(row["profile_config_f_ms"]),
                    _fmt_ms(row["profile_laishere_ms"]),
                    f"{_fmt_ms(row['profile_gap_ms'])} / {_fmt_pct(row['profile_gap_pct'])}",
                    row["profile_outcome"],
                    interp,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _load_existing(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for machine, path in paths.items():
        if path.exists():
            payloads[machine] = load_json(path)
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER_JSON)
    parser.add_argument("--tie-threshold-pct", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = summarize_freshness(
        load_json(args.frontier_json),
        _load_existing(DEFAULT_CONFIG_FILES),
        _load_existing(DEFAULT_STAGE_PROFILE_FILES),
        args.tie_threshold_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frontier_loss_count": summary["frontier_loss_count"],
                "stale_or_tie_loss_count": summary["stale_or_tie_loss_count"],
                "real_profile_loss_count": summary["real_profile_loss_count"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
