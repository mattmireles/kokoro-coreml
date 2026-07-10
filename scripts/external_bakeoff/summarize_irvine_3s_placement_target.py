#!/usr/bin/env python3
"""Summarize the remaining Irvine 3s Core ML placement target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OURS_CPU_NE = Path("outputs/external_bakeoff/compute_plan/ours_har_post_3s_cpu_ne.json")
DEFAULT_OURS_CPU_GPU = Path("outputs/graph_surface/irvine_m1/compute_plan_generator_3s_cpu_gpu.json")
DEFAULT_LAISHERE_CPU_NE = Path("outputs/external_bakeoff/compute_plan/laishere_vocoder_cpu_ne.json")
DEFAULT_EXACT_BODY_CPU_NE = Path(
    "outputs/decoder_vocoder_split/3s_har_cos_rsqrt_native_in_broadcast_ios17/irvine/"
    "compute_plan_body_cpu_ne_irvine.json"
)
DEFAULT_SURFACES_JSON = Path("outputs/external_bakeoff/irvine_3s_surfaces.json")
DEFAULT_RESIDUAL_JSON = Path("outputs/external_bakeoff/irvine_3s_residual.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/irvine_3s_placement_target.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/irvine_3s_placement_target.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:.1f}%"


def _plan_summary(label: str, path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    counts = dict(plan.get("preferredDeviceCounts") or {})
    weights = dict(plan.get("costWeightByPreferredDevice") or {})
    return {
        "label": label,
        "path": str(path),
        "model": plan.get("model"),
        "compiled_model": plan.get("compiledModel"),
        "compute_units": plan.get("computeUnits"),
        "operation_count": plan.get("operationCount"),
        "preferred_device_counts": counts,
        "cost_weight_by_preferred_device": weights,
        "neural_engine_preferred_ops": int(counts.get("neuralEngine") or 0),
        "neural_engine_cost_weight": float(weights.get("neuralEngine") or 0.0),
        "gpu_cost_weight": float(weights.get("gpu") or 0.0),
        "cpu_cost_weight": float(weights.get("cpu") or 0.0),
    }


def _surface_row(surfaces: dict[str, Any], label: str, *, worst_speedup: bool = False) -> dict[str, Any] | None:
    rows = [row for row in surfaces.get("rows") or [] if row.get("label") == label]
    if not rows:
        return None
    if worst_speedup:
        return min(rows, key=lambda row: float(row.get("speedup_pct") or 0.0))
    return rows[0]


def summarize_placement_target(
    ours_cpu_ne: dict[str, Any],
    ours_cpu_gpu: dict[str, Any],
    laishere_cpu_ne: dict[str, Any],
    surfaces: dict[str, Any],
    residual: dict[str, Any],
    exact_body_cpu_ne: dict[str, Any] | None = None,
    *,
    ours_cpu_ne_path: Path = DEFAULT_OURS_CPU_NE,
    ours_cpu_gpu_path: Path = DEFAULT_OURS_CPU_GPU,
    laishere_cpu_ne_path: Path = DEFAULT_LAISHERE_CPU_NE,
    exact_body_cpu_ne_path: Path = DEFAULT_EXACT_BODY_CPU_NE,
) -> dict[str, Any]:
    """Return the repeatable target for the remaining Irvine 3s loss."""

    ours_ne = _plan_summary("ours_har_post_3s_cpu_ne", ours_cpu_ne_path, ours_cpu_ne)
    ours_gpu = _plan_summary("ours_har_post_3s_cpu_gpu", ours_cpu_gpu_path, ours_cpu_gpu)
    laishere_ne = _plan_summary("laishere_vocoder_cpu_ne", laishere_cpu_ne_path, laishere_cpu_ne)
    exact_body_ne = None
    if exact_body_cpu_ne is not None:
        exact_body_ne = _plan_summary("exact_decoder_vocoder_body_3s_cpu_ne", exact_body_cpu_ne_path, exact_body_cpu_ne)
    partial_ne_strict = _surface_row(surfaces, "3s_broadcast_adain_native_in_ios17", worst_speedup=True)
    best_strict = surfaces.get("best_strict_pass")
    residual_after_known = residual.get("residual_after_all_known_positive_estimates_ms")
    plans = [ours_ne, ours_gpu, laishere_ne]
    if exact_body_ne is not None:
        plans.append(exact_body_ne)
    target = {
        "machine_id": "irvine-m1",
        "input_key": "3s",
        "plans": plans,
        "ours_cpu_ne_neural_engine_preferred_ops": ours_ne["neural_engine_preferred_ops"],
        "laishere_cpu_ne_neural_engine_preferred_ops": laishere_ne["neural_engine_preferred_ops"],
        "laishere_cpu_ne_neural_engine_cost_weight": laishere_ne["neural_engine_cost_weight"],
        "exact_body_cpu_ne_neural_engine_preferred_ops": None
        if exact_body_ne is None
        else exact_body_ne["neural_engine_preferred_ops"],
        "exact_body_cpu_ne_neural_engine_cost_weight": None
        if exact_body_ne is None
        else exact_body_ne["neural_engine_cost_weight"],
        "strict_partial_ne_counterexample": None
        if partial_ne_strict is None
        else {
            "label": partial_ne_strict.get("label"),
            "baseline_ms": partial_ne_strict.get("baseline_ms"),
            "candidate_ms": partial_ne_strict.get("candidate_ms"),
            "speedup_pct": partial_ne_strict.get("speedup_pct"),
            "passes": partial_ne_strict.get("passes"),
            "path": partial_ne_strict.get("path"),
        },
        "best_strict_positive_surface": None
        if best_strict is None
        else {
            "label": best_strict.get("label"),
            "delta_ms": best_strict.get("delta_ms"),
            "speedup_pct": best_strict.get("speedup_pct"),
            "path": best_strict.get("path"),
        },
        "residual_after_all_known_positive_estimates_ms": residual_after_known,
        "compute_unit_flag_flip_is_sufficient": False,
        "target": (
            "Build a laishere-like mixed CPU/Neural Engine body plan that is "
            "runtime-positive for the strict Swift HAR/source contract. Existing "
            "strict graph surfaces show that merely obtaining partial Neural "
            "Engine placement is not enough; the body boundary and operator "
            "surface must avoid the observed synchronization penalty."
        ),
        "deep_research_request": (
            "Design an M1 MLProgram source/STFT/vocoder body that preserves "
            "current Swift HAR/source semantics, keeps strict waveform parity, "
            "and shifts the expensive conv/add/mul/instance_norm body work into "
            "a laishere-like mixed CPU/Neural Engine plan without the existing "
            "split-boundary synchronization penalty or 3s warmed regression."
        ),
    }
    return target


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Irvine 3s Placement Target",
        "",
        "Warmed Irvine M1 3s only. This report turns saved Core ML compute-plan",
        "JSONs into the next implementation target for the remaining laishere",
        "loss.",
        "",
        "## Compute Plans",
        "",
        "| Plan | Compute units | Ops | Preferred counts | CPU weight | GPU weight | NE weight | Source |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for plan in summary["plans"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{plan['label']}`",
                    f"`{plan['compute_units']}`",
                    str(plan["operation_count"]),
                    f"`{json.dumps(plan['preferred_device_counts'], sort_keys=True)}`",
                    _fmt_pct(plan["cpu_cost_weight"]),
                    _fmt_pct(plan["gpu_cost_weight"]),
                    _fmt_pct(plan["neural_engine_cost_weight"]),
                    f"`{plan['path']}`",
                ]
            )
            + " |"
        )
    counter = summary.get("strict_partial_ne_counterexample")
    best_strict = summary.get("best_strict_positive_surface")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Our strict CPU+NE plan has `{summary['ours_cpu_ne_neural_engine_preferred_ops']}` "
            "Neural Engine-preferred ops.",
            f"laishere's CPU+NE vocoder plan has `{summary['laishere_cpu_ne_neural_engine_preferred_ops']}` "
            "Neural Engine-preferred ops and "
            f"`{_fmt_pct(summary['laishere_cpu_ne_neural_engine_cost_weight'])}` estimated cost on Neural Engine.",
            "A compute-unit flag flip is not sufficient.",
        ]
    )
    if counter is not None:
        lines.extend(
            [
                "",
                "## Partial-NE Counterexample",
                "",
                f"`{counter['label']}` is a strict-pass surface but measured "
                f"`{_fmt_ms(counter['candidate_ms'])}` versus "
                f"`{_fmt_ms(counter['baseline_ms'])}` "
                f"(`{float(counter['speedup_pct']):.1f}%` speedup).",
                "That rejects the hypothesis that any partial Neural Engine placement wins.",
            ]
        )
    if summary.get("exact_body_cpu_ne_neural_engine_preferred_ops") is not None:
        lines.extend(
            [
                "",
                "## Exact Body Placement Trap",
                "",
                "The existing strict decoder+vocoder body split already gets "
                f"`{summary['exact_body_cpu_ne_neural_engine_preferred_ops']}` "
                "Neural Engine-preferred ops and "
                f"`{_fmt_pct(summary['exact_body_cpu_ne_neural_engine_cost_weight'])}` "
                "estimated cost on Neural Engine.",
                "That still was not a warmed runtime win, so placement alone is not the target.",
            ]
        )
    if best_strict is not None:
        lines.extend(
            [
                "",
                "## Best Existing Strict Positive Surface",
                "",
                f"`{best_strict['label']}` saves only `{_fmt_ms(best_strict['delta_ms'])}` "
                f"(`{float(best_strict['speedup_pct']):.1f}%`).",
            ]
        )
    lines.extend(
        [
            "",
            "## Target",
            "",
            summary["target"],
            "",
            "## Deep Research Request",
            "",
            summary["deep_research_request"],
            "",
            "## Residual",
            "",
            "Known saved positive estimates leave "
            f"`{_fmt_ms(summary['residual_after_all_known_positive_estimates_ms'])}` "
            "against warmed laishere.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-cpu-ne", type=Path, default=DEFAULT_OURS_CPU_NE)
    parser.add_argument("--ours-cpu-gpu", type=Path, default=DEFAULT_OURS_CPU_GPU)
    parser.add_argument("--laishere-cpu-ne", type=Path, default=DEFAULT_LAISHERE_CPU_NE)
    parser.add_argument("--exact-body-cpu-ne", type=Path, default=DEFAULT_EXACT_BODY_CPU_NE)
    parser.add_argument("--surfaces-json", type=Path, default=DEFAULT_SURFACES_JSON)
    parser.add_argument("--residual-json", type=Path, default=DEFAULT_RESIDUAL_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()
    summary = summarize_placement_target(
        _load_json(args.ours_cpu_ne),
        _load_json(args.ours_cpu_gpu),
        _load_json(args.laishere_cpu_ne),
        _load_json(args.surfaces_json),
        _load_json(args.residual_json),
        _load_json(args.exact_body_cpu_ne) if args.exact_body_cpu_ne.exists() else None,
        ours_cpu_ne_path=args.ours_cpu_ne,
        ours_cpu_gpu_path=args.ours_cpu_gpu,
        laishere_cpu_ne_path=args.laishere_cpu_ne,
        exact_body_cpu_ne_path=args.exact_body_cpu_ne,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "absolute_fastest_verified": False,
                "output": str(args.output),
                "ours_ne_ops": summary["ours_cpu_ne_neural_engine_preferred_ops"],
                "laishere_ne_ops": summary["laishere_cpu_ne_neural_engine_preferred_ops"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
