#!/usr/bin/env python3
"""Decompose Config F frontier losses against laishere stage profiles.

The competitive frontier tells us where Config F loses. This script explains
those losses in stage terms using warmed inference only:

- Config F stage medians from ``raw_warm_results`` in the same-window result.
- laishere stage medians from ``profile_laishere_stages.py`` outputs.

The boundaries are not identical. Config F's ``t_coreml_predict_s`` is the
current HAR-post generator predict timing, while laishere's
``noise+vocoder+tail`` includes its F0-noise source, broader vocoder body, and
fp32 tail. The comparison is still useful because it identifies whether a
frontier loss is dominated by source/vocoder work or by upstream/runtime work.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_FRONTIER_JSON = Path("outputs/external_bakeoff/competitive_frontier.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/stage_gap_decomposition.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/stage_gap_decomposition.json")
DEFAULT_RESULTS_DIR = Path("outputs/external_bakeoff")
DEFAULT_PLACEMENT_DIR = Path("outputs/external_bakeoff/placement")

CONFIG_STAGE_KEYS = (
    "t_duration_coreml_s",
    "t_f0ntrain_coreml_s",
    "t_decoder_pre_coreml_s",
    "t_decoder_pre_hnsf_overlap_s",
    "t_coreml_predict_s",
    "t_hnsf_swift_s",
    "t_matrix_ops_s",
    "t_padding_s",
    "t_trim_s",
    "t_alignment_s",
)
LAISHERE_UPSTREAM_KEYS = ("albert_s", "post_albert_s", "alignment_s", "prosody_s")
LAISHERE_SOURCE_KEYS = ("noise_s", "vocoder_s", "tail_s")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(statistics.median(clean)) if clean else None


def _machine_id(raw: str) -> str:
    machine = raw
    for suffix in ("-vector-noise-batch", "-local-vector-noise-batch"):
        if machine.endswith(suffix):
            machine = machine[: -len(suffix)]
    if machine == "m2-studio-local":
        return "m2-studio"
    return machine


def _result_path_for_machine(results_dir: Path, machine_id: str) -> Path:
    names = {
        "m2-studio": "results_config_f_reference_m2-studio-local_vector_noise_batch.json",
        "m2-air": "results_config_f_reference_m2-air_vector_noise_batch.json",
        "irvine-m1": "results_config_f_reference_irvine-m1_vector_noise_batch.json",
    }
    if machine_id not in names:
        raise KeyError(f"no Config F result path known for {machine_id}")
    return results_dir / names[machine_id]


def _laishere_profile_path_for_machine(placement_dir: Path, machine_id: str) -> Path:
    return placement_dir / f"results_laishere_stage_profile_{machine_id}.json"


def _record_by_bucket(payload: dict[str, Any], bucket: str) -> dict[str, Any] | None:
    for record in payload.get("records") or []:
        if str(record.get("input_key")) == bucket and str(record.get("status") or "ok") == "ok":
            return record
    return None


def summarize_config_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return warmed Config F stage medians in seconds for one record."""

    raw_warm = (record.get("provenance") or {}).get("raw_warm_results") or []
    if not raw_warm:
        raw = (record.get("provenance") or {}).get("raw_last_result") or {}
        raw_warm = [raw] if raw else []

    medians = {key: _median([run.get(key) for run in raw_warm]) for key in CONFIG_STAGE_KEYS}
    wall_s = _median([run.get("wall_time_s") for run in raw_warm])
    if wall_s is None:
        wall_values = [float(value) for value in record.get("warm_wall_times_s") or []]
        wall_s = _median(wall_values)
    overlap_s = float(medians["t_decoder_pre_hnsf_overlap_s"] or 0.0)
    known_sum = sum(
        float(value)
        for key, value in medians.items()
        if value is not None and key != "t_decoder_pre_hnsf_overlap_s"
    ) - overlap_s
    host_other_s = None if wall_s is None else wall_s - known_sum
    non_generator_s = None
    generator_s = medians["t_coreml_predict_s"]
    if wall_s is not None and generator_s is not None:
        non_generator_s = wall_s - generator_s
    return {
        "total_s": wall_s,
        "duration_s": medians["t_duration_coreml_s"],
        "f0n_s": medians["t_f0ntrain_coreml_s"],
        "decoder_pre_s": medians["t_decoder_pre_coreml_s"],
        "decoder_pre_hnsf_overlap_s": overlap_s,
        "generator_s": generator_s,
        "hnsf_s": medians["t_hnsf_swift_s"],
        "matrix_ops_s": medians["t_matrix_ops_s"],
        "padding_s": medians["t_padding_s"],
        "trim_s": medians["t_trim_s"],
        "alignment_s": medians["t_alignment_s"],
        "known_sum_s": known_sum,
        "host_other_s": host_other_s,
        "non_generator_s": non_generator_s,
    }


def summarize_laishere_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return warmed laishere stage medians in seconds for one record."""

    warm = record.get("warm_median_s") or {}
    if not isinstance(warm, dict):
        raise ValueError("laishere record is missing warm_median_s object")
    upstream_s = sum(float(warm.get(key) or 0.0) for key in LAISHERE_UPSTREAM_KEYS)
    source_s = sum(float(warm.get(key) or 0.0) for key in LAISHERE_SOURCE_KEYS)
    prepare_s = float(record.get("prepare_wall_time_s") or 0.0)
    total_s = float(warm.get("total_s") or 0.0)
    return {
        "total_s": total_s,
        "prepare_s": prepare_s,
        "total_plus_prepare_s": total_s + prepare_s,
        "upstream_s": upstream_s,
        "noise_s": float(warm.get("noise_s") or 0.0),
        "vocoder_s": float(warm.get("vocoder_s") or 0.0),
        "tail_s": float(warm.get("tail_s") or 0.0),
        "noise_vocoder_tail_s": source_s,
        "other_plus_prepare_s": upstream_s + prepare_s,
        "python_overhead_s": float(warm.get("python_overhead_s") or 0.0),
    }


def summarize_stage_gaps(
    frontier: dict[str, Any],
    results_dir: Path,
    placement_dir: Path,
) -> dict[str, Any]:
    """Return per-loss stage decomposition rows."""

    rows: list[dict[str, Any]] = []
    for cell in frontier.get("summary", {}).get("config_f_losses") or []:
        machine = _machine_id(str(cell["machine_id"]))
        bucket = str(cell["input_key"])
        try:
            config_payload = _load_json(_result_path_for_machine(results_dir, machine))
            laishere_payload = _load_json(_laishere_profile_path_for_machine(placement_dir, machine))
        except (KeyError, FileNotFoundError):
            continue
        config_record = _record_by_bucket(config_payload, bucket)
        laishere_record = _record_by_bucket(laishere_payload, bucket)
        if not config_record or not laishere_record:
            continue
        config = summarize_config_record(config_record)
        laishere = summarize_laishere_record(laishere_record)
        config_total = float(config["total_s"] or 0.0)
        laishere_total = float(laishere["total_s"])
        generator_gap_s = None
        if config["generator_s"] is not None:
            generator_gap_s = float(config["generator_s"]) - float(laishere["noise_vocoder_tail_s"])
        upstream_gap_s = None
        if config["non_generator_s"] is not None:
            upstream_gap_s = float(config["non_generator_s"]) - float(laishere["other_plus_prepare_s"])
        rows.append(
            {
                "machine_id": machine,
                "input_key": bucket,
                "frontier_config_f_ms": cell.get("config_f_warm_median_ms"),
                "frontier_best_ms": cell.get("best_warm_median_ms"),
                "frontier_best_impl_label": cell.get("best_impl_label"),
                "config": config,
                "laishere": laishere,
                "total_gap_s": config_total - laishere_total,
                "total_gap_vs_frontier_s": (
                    float(cell["config_f_warm_median_ms"]) - float(cell["best_warm_median_ms"])
                )
                / 1000.0,
                "config_generator_minus_laishere_nvt_s": generator_gap_s,
                "config_nongenerator_minus_laishere_other_prepare_s": upstream_gap_s,
            }
        )
    return {
        "rows": rows,
        "row_count": len(rows),
    }


def _ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 1000.0:.1f} ms"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render stage-gap decomposition as Markdown."""

    lines = [
        "# Stage Gap Decomposition",
        "",
        "Warmed inference only. Boundaries are not identical: Config F",
        "`generator` is the current HAR-post predict timing, while laishere",
        "`noise+vocoder+tail` is its F0-noise source, broader vocoder body,",
        "and fp32 tail.",
        "",
        f"Loss rows analyzed: `{summary['row_count']}`.",
        "",
        "| Machine | Bucket | Config F total | laishere total | Total gap | Config generator | laishere noise+vocoder+tail | Source/body gap | Config non-generator | laishere other+prep | Other gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        config = row["config"]
        laishere = row["laishere"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    row["input_key"],
                    _ms(config["total_s"]),
                    _ms(laishere["total_s"]),
                    _ms(row["total_gap_s"]),
                    _ms(config["generator_s"]),
                    _ms(laishere["noise_vocoder_tail_s"]),
                    _ms(row["config_generator_minus_laishere_nvt_s"]),
                    _ms(config["non_generator_s"]),
                    _ms(laishere["other_plus_prepare_s"]),
                    _ms(row["config_nongenerator_minus_laishere_other_prepare_s"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Config F Stage Details",
            "",
            "| Machine | Bucket | Duration | F0Ntrain | DecoderPre | HnSF | Overlap | Generator | Matrix/pad/trim | Host other |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        config = row["config"]
        aux = sum(
            float(config[key] or 0.0)
            for key in ("matrix_ops_s", "padding_s", "trim_s", "alignment_s")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    row["input_key"],
                    _ms(config["duration_s"]),
                    _ms(config["f0n_s"]),
                    _ms(config["decoder_pre_s"]),
                    _ms(config["hnsf_s"]),
                    _ms(config["decoder_pre_hnsf_overlap_s"]),
                    _ms(config["generator_s"]),
                    _ms(aux),
                    _ms(config["host_other_s"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER_JSON)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--placement-dir", type=Path, default=DEFAULT_PLACEMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    frontier = _load_json(args.frontier_json)
    summary = summarize_stage_gaps(frontier, args.results_dir, args.placement_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "row_count": summary["row_count"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
