#!/usr/bin/env python3
"""Summarize the remaining source-contract frontier for Irvine M1.

This report joins three existing evidence streams:

- F0/source formulation probes, which show whether the source equation itself
  can match Swift.
- Irvine next-target analysis, which shows whether quality-fail source/body
  branches would close warmed laishere rows.
- Generator noise/body split reports, which isolate the body-only counterfactual
  when ``x_source_*`` tensors are treated as already available.

It does not promote a candidate. It records why the next useful work is a
cheaper strict source/HAR contract rather than more package-boundary splitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_VARIANTS_JSON = Path("outputs/f0_source_variants/summary_3s_7s_10s_15s_30s.json")
DEFAULT_IRVINE_TARGETS_JSON = Path("outputs/external_bakeoff/irvine_next_targets.json")
DEFAULT_M2_BODY_REPORT = Path(
    "outputs/generator_noise_split/3s_native_in_broadcast_ios17/report_native_in_ios17_cpu_gpu.json"
)
DEFAULT_IRVINE_BODY_REPORT = Path(
    "outputs/generator_noise_split/3s_native_in_broadcast_ios17/irvine/report_native_in_ios17_cpu_gpu_irvine.json"
)
DEFAULT_STRICT_BUDGET = Path("outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.json")
DEFAULT_XSOURCE_FEASIBILITY = Path(
    "outputs/xsource_distillation_feasibility/3s_pre_noise_conv_har_conv_geometry_ridge.json"
)
DEFAULT_PRE_NOISE_FOLDING_SURFACE = Path("outputs/external_bakeoff/pre_noise_folding_surface.json")
DEFAULT_STRICT_FOLDING_CEILING = Path("outputs/external_bakeoff/strict_folding_ceiling.json")
DEFAULT_IRVINE_LISTENING_TARGETS = Path("outputs/external_bakeoff/irvine_listening_targets.json")
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/source_contract_frontier.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/source_contract_frontier.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load an optional JSON object if it exists."""

    if not path.exists():
        return None
    return _load_json(path)


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_db(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f} dB"


def _fmt_mib(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) / (1024.0 * 1024.0):.2f} MiB"


def _body_counterfactual(report: dict[str, Any], machine_id: str) -> dict[str, Any]:
    med = report["benchmark"]["warm_predict_median_ms"]
    fused_ms = float(med["fused"])
    body_ms = float(med["split_body"])
    source_ms = float(med["split_noise"])
    total_ms = float(med["split_total"])
    return {
        "machine_id": machine_id,
        "fused_ms": fused_ms,
        "body_only_ms": body_ms,
        "source_noise_ms": source_ms,
        "split_total_ms": total_ms,
        "body_only_saves_ms": fused_ms - body_ms,
        "full_split_delta_ms": fused_ms - total_ms,
        "passes": bool(report.get("passes")),
    }


def _build_implementation_queue(
    profile_budget_rows: list[dict[str, Any]],
    paper_budget_rows: list[dict[str, Any]],
    quality_profile_closers: list[dict[str, Any]],
    quality_listening: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the next source-contract experiments in priority order."""

    profile_buckets = [str(row.get("bucket")) for row in profile_budget_rows]
    paper_buckets = [str(row.get("bucket")) for row in paper_budget_rows]
    accepted_buckets = set()
    pending_buckets = set()
    if quality_listening is not None:
        accepted_buckets = set(quality_listening.get("accepted_buckets") or [])
        pending_buckets = set(quality_listening.get("pending_buckets") or [])
    listening_buckets = [
        str(row.get("input_key"))
        for row in quality_profile_closers
        if str(row.get("input_key")) not in accepted_buckets
    ]
    max_profile_save = max(
        (float(row.get("additional_profile_save_required_ms") or 0.0) for row in profile_budget_rows),
        default=0.0,
    )
    max_paper_save = max(
        (float(row.get("additional_paper_save_required_ms") or 0.0) for row in paper_budget_rows),
        default=0.0,
    )
    return [
        {
            "priority": 1,
            "track": "strict",
            "experiment": "algebraically fold STFT/HAR into first noise convolutions",
            "why": (
                "The expensive materialized HAR representation is the blocker; the first noise "
                "convs are linear over HAR before nonlinear residual work, so a fixed-bucket "
                "filterbank/folding attempt could avoid raw phase tensor materialization. This "
                "must preserve the first strided noise-conv geometry; cheap source-window adapters failed."
            ),
            "target_buckets": profile_buckets,
            "required_profile_save_ms": max_profile_save,
            "required_paper_save_ms": max_paper_save,
            "promotion_gate": (
                "same x_source_* or early activation parity first, then waveform parity and "
                "warmed lower-end timing"
            ),
        },
        {
            "priority": 2,
            "track": "strict",
            "experiment": "distill a different strict boundary, not compact direct x_source tensors",
            "why": (
                "Compact direct x_source adapters fail on x_source_0 even with strict HAR windows "
                "and temporal Conv1d capacity; only continue distillation if the target moves to "
                "early/later activations or includes the residual/AdaIN behavior that x_source_0 needs."
            ),
            "target_buckets": profile_buckets,
            "required_profile_save_ms": max_profile_save,
            "required_paper_save_ms": max_paper_save,
            "promotion_gate": (
                "activation or waveform parity first, then warmed end-to-end win on quiet Irvine M1; "
                "no new hot Core ML boundary unless it removes more cost than it adds"
            ),
        },
        {
            "priority": 3,
            "track": "quality-changing",
            "experiment": "no-ASR human listening review for saved source/body speed branches",
            "why": (
                "Saved source/body branches would beat warmed laishere profile on several buckets, "
                "but they are not strict waveform-parity candidates."
            ),
            "target_buckets": listening_buckets,
            "required_profile_save_ms": 0.0,
            "required_paper_save_ms": 0.0,
            "promotion_gate": (
                "filled no-ASR listening decisions plus waveform-health review; pending buckets: "
                + (", ".join(sorted(pending_buckets)) if pending_buckets else "none")
                + "; keep separate from "
                "strict paper claims unless the methodology explicitly accepts listening-equivalent quality"
            ),
        },
        {
            "priority": 4,
            "track": "stop",
            "experiment": (
                "do not repeat exact HAR-post splits, sine-source variants, compact direct x_source "
                "or pre-noise adapters, direct source/body upsample-rewrite-only packages, "
                "or no-side-input phase+rewrite packages"
            ),
            "why": (
                "Exact split source/noise production is slower end-to-end, and Swift-like source "
                "generation already matches dumped har_source; compact direct x_source adapters fail "
                "on x_source_0; cheap pre-noise adapters fail before the first strided noise conv; "
                "direct source/body upsample rewrite saves only a few local milliseconds and closes "
                "zero Irvine paper rows; no-side-input phase+rewrite is either fp16 quality-failing or fp32 slower. The "
                "remaining issue is HAR/STFT representation or boundary choice."
            ),
            "target_buckets": [],
            "required_profile_save_ms": 0.0,
            "required_paper_save_ms": 0.0,
            "promotion_gate": "n/a",
        },
    ]


def summarize_source_contract_frontier(
    source_variants: dict[str, Any],
    irvine_targets: dict[str, Any],
    m2_body_report: dict[str, Any],
    irvine_body_report: dict[str, Any],
    strict_budget: dict[str, Any] | None = None,
    xsource_feasibility: dict[str, Any] | None = None,
    pre_noise_folding_surface: dict[str, Any] | None = None,
    strict_folding_ceiling: dict[str, Any] | None = None,
    irvine_listening_targets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the source-contract frontier summary."""

    rows = source_variants.get("rows") or []
    min_swift_source_snr = source_variants.get("min_swift_like_source_snr_db")
    max_recomputed_har_snr = source_variants.get("max_dump_source_recomputed_har_snr_db")
    quality_profile_closers = [
        row
        for row in (irvine_targets.get("rows") or [])
        if (row.get("quality_fail_vs_warmed_profile") or {}).get(
            "would_beat_warmed_laishere_profile"
        )
    ]
    source_body_rows = [
        row for row in (irvine_targets.get("rows") or []) if "source/body" in str(row.get("target_class"))
    ]
    budget_rows = [] if strict_budget is None else list(strict_budget.get("rows") or [])
    profile_budget_rows = [
        row for row in budget_rows if float(row.get("additional_profile_save_required_ms") or 0.0) > 0.0
    ]
    paper_budget_rows = [
        row for row in budget_rows if float(row.get("additional_paper_save_required_ms") or 0.0) > 0.0
    ]
    quality_listening = _summarize_quality_listening(irvine_listening_targets)
    return {
        "source_equation_is_solved": bool(source_variants.get("source_equation_is_solved")),
        "recomputed_stft_har_is_solved": bool(source_variants.get("recomputed_stft_har_is_solved")),
        "source_variant_rows": len(rows),
        "min_swift_like_source_snr_db": min_swift_source_snr,
        "max_dump_source_recomputed_har_snr_db": max_recomputed_har_snr,
        "irvine_real_loss_count": int(irvine_targets.get("real_loss_count") or 0),
        "irvine_source_body_loss_count": len(source_body_rows),
        "strict_pass_closers": int(irvine_targets.get("strict_pass_closers") or 0),
        "quality_fail_warmed_profile_closers": len(quality_profile_closers),
        "quality_fail_warmed_profile_buckets": [row.get("input_key") for row in quality_profile_closers],
        "post_overlap_rewrite_budget": {
            "candidate_label": None if strict_budget is None else strict_budget.get("candidate_label"),
            "profile_rows_remaining": len(profile_budget_rows),
            "paper_rows_remaining": len(paper_budget_rows),
            "profile_rows": [
                {
                    "bucket": row.get("bucket"),
                    "additional_save_required_ms": float(row.get("additional_profile_save_required_ms") or 0.0),
                    "additional_generator_speedup_required_pct": float(
                        row.get("additional_profile_generator_speedup_required_pct") or 0.0
                    ),
                }
                for row in profile_budget_rows
            ],
            "paper_rows": [
                {
                    "bucket": row.get("bucket"),
                    "additional_save_required_ms": float(row.get("additional_paper_save_required_ms") or 0.0),
                    "additional_generator_speedup_required_pct": float(
                        row.get("additional_paper_generator_speedup_required_pct") or 0.0
                    ),
                }
                for row in paper_budget_rows
            ],
        },
        "implementation_queue": _build_implementation_queue(
            profile_budget_rows,
            paper_budget_rows,
            quality_profile_closers,
            quality_listening,
        ),
        "quality_listening_status": quality_listening,
        "xsource_distillation_feasibility": _summarize_xsource_feasibility(xsource_feasibility),
        "pre_noise_folding_surface": _summarize_pre_noise_folding_surface(pre_noise_folding_surface),
        "strict_folding_ceiling": _summarize_strict_folding_ceiling(strict_folding_ceiling),
        "body_counterfactuals": [
            _body_counterfactual(m2_body_report, "m2-studio"),
            _body_counterfactual(irvine_body_report, "irvine-m1"),
        ],
        "decision": (
            "The Swift-like source equation is solved, but recomputed HAR/STFT is not. "
            "The body package is fast if x_source tensors are free, and quality-fail "
            "F0/source branches would close several warmed Irvine profile rows only "
            "after no-ASR listening acceptance. The next useful work is a cheaper "
            "strict source/HAR contract or listening-accepted source replacement, "
            "not another exact HAR-post split."
        ),
    }


def _summarize_quality_listening(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Summarize no-ASR human decision status for quality-changing speed branches."""

    if report is None:
        return None
    rows = report.get("rows") or []
    decision_counts: dict[str, int] = {}
    gate_counts: dict[str, int] = {}
    accepted_buckets: list[str] = []
    pending_buckets: list[str] = []
    failing_buckets: list[str] = []
    accepted_decisions = {"pass", "caveat"}
    for row in rows:
        bucket = str(row.get("bucket") or "")
        decision = str(row.get("human_decision") or "").strip().lower()
        gate = str(row.get("waveform_gate_decision") or "").strip().lower()
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        if decision in accepted_decisions:
            accepted_buckets.append(bucket)
        elif decision == "fail":
            failing_buckets.append(bucket)
        else:
            pending_buckets.append(bucket)
    return {
        "rows": len(rows),
        "mapped_count": int(report.get("mapped_count") or 0),
        "exact_timing_report_listening_artifact_count": int(
            report.get("exact_timing_report_listening_artifact_count") or 0
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "gate_counts": dict(sorted(gate_counts.items())),
        "accepted_count": len(accepted_buckets),
        "pending_count": len(pending_buckets),
        "failed_count": len(failing_buckets),
        "accepted_buckets": accepted_buckets,
        "pending_buckets": pending_buckets,
        "failed_buckets": failing_buckets,
        "asr_used": False,
    }


def _summarize_xsource_feasibility(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Summarize the optional direct x_source distillation smoke report."""

    if report is None:
        return None
    rows = []
    for row in report.get("rows") or []:
        validation = row.get("validation_metrics") or {}
        rows.append(
            {
                "name": row.get("name"),
                "feature_count": (row.get("feature_shape") or [None, None])[1],
                "validation_snr_db": validation.get("snr_db"),
                "validation_correlation": validation.get("correlation"),
                "validation_max_abs_error": validation.get("max_abs_error"),
            }
        )
    return {
        "tensor_dump": report.get("tensor_dump"),
        "decision": report.get("decision"),
        "target_mode": report.get("target_mode"),
        "model": report.get("model"),
        "feature_set": report.get("feature_set"),
        "radius": report.get("radius"),
        "holdout_stride": report.get("holdout_stride"),
        "hidden": report.get("hidden"),
        "steps": report.get("steps"),
        "conv_kernel": report.get("conv_kernel"),
        "conv_depth": report.get("conv_depth"),
        "rows": rows,
    }


def _summarize_pre_noise_folding_surface(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Summarize optional pre-noise folding-surface evidence."""

    if report is None:
        return None
    rows = []
    for row in report.get("rows") or []:
        rows.append(
            {
                "input_key": row.get("input_key"),
                "union_touched_har_frame_pct": row.get("union_touched_har_frame_pct"),
                "full_har_bytes_fp16": row.get("full_har_bytes_fp16"),
                "total_pre_noise_bytes_fp16": row.get("total_pre_noise_bytes_fp16"),
                "pre_noise_to_har_value_ratio": row.get("pre_noise_to_har_value_ratio"),
            }
        )
    return {"decision": report.get("decision"), "rows": rows}


def _summarize_strict_folding_ceiling(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Summarize optional strict folding speed-ceiling evidence."""

    if report is None:
        return None
    rows = []
    for row in report.get("rows") or []:
        rows.append(
            {
                "bucket": row.get("bucket"),
                "removable_swift_stft_ms": row.get("removable_swift_stft_ms"),
                "strict_profile_save_needed_ms": row.get("strict_profile_save_needed_ms"),
                "profile_gap_coverage_pct": row.get("profile_gap_coverage_pct"),
                "strict_paper_save_needed_ms": row.get("strict_paper_save_needed_ms"),
                "paper_gap_coverage_pct": row.get("paper_gap_coverage_pct"),
            }
        )
    return {
        "decision": (report.get("summary") or {}).get("decision"),
        "profile_rows_closed_by_stft_only": (report.get("summary") or {}).get(
            "profile_rows_closed_by_stft_only"
        ),
        "paper_rows_closed_by_stft_only": (report.get("summary") or {}).get(
            "paper_rows_closed_by_stft_only"
        ),
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Source Contract Frontier",
        "",
        "Warmed-inference evidence for the remaining Irvine M1 source/body gap.",
        "",
        "## Summary",
        "",
        f"- Source equation solved: `{str(summary['source_equation_is_solved']).lower()}`.",
        f"- Recomputed HAR/STFT solved: `{str(summary['recomputed_stft_har_is_solved']).lower()}`.",
        f"- Source-variant buckets scanned: `{summary['source_variant_rows']}`.",
        f"- Swift-like source minimum SNR: `{_fmt_db(summary['min_swift_like_source_snr_db'])}`.",
        f"- Dumped source recomputed-HAR maximum SNR: `{_fmt_db(summary['max_dump_source_recomputed_har_snr_db'])}`.",
        f"- Irvine real loss rows: `{summary['irvine_real_loss_count']}`.",
        f"- Irvine source/body loss rows: `{summary['irvine_source_body_loss_count']}`.",
        f"- Saved strict candidates closing Irvine rows: `{summary['strict_pass_closers']}`.",
        f"- Quality-fail source candidates that beat warmed laishere profile: `{summary['quality_fail_warmed_profile_closers']}`.",
        "",
        "## Post Overlap + Rewrite Budget",
        "",
        "Additional strict speed still required after the current runtime overlap",
        "and HAR-post upsample rewrite projection:",
        "",
        "| Target | Bucket | Extra save needed | Extra generator speedup needed |",
        "| --- | --- | ---: | ---: |",
    ]
    budget = summary["post_overlap_rewrite_budget"]
    for row in budget["profile_rows"]:
        lines.append(
            "| warmed profile | "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["additional_save_required_ms"]),
                    f"{row['additional_generator_speedup_required_pct']:.2f}%",
                ]
            )
            + " |"
        )
    for row in budget["paper_rows"]:
        lines.append(
            "| paper frontier | "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["additional_save_required_ms"]),
                    f"{row['additional_generator_speedup_required_pct']:.2f}%",
                ]
            )
            + " |"
        )
    if not budget["profile_rows"] and not budget["paper_rows"]:
        lines.append("| n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Implementation Queue",
            "",
            "| Priority | Track | Experiment | Target buckets | Promotion gate |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for row in summary["implementation_queue"]:
        buckets = ", ".join(f"`{bucket}`" for bucket in row["target_buckets"]) or "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["priority"]),
                    row["track"],
                    row["experiment"],
                    buckets,
                    row["promotion_gate"],
                ]
            )
            + " |"
        )
    quality_listening = summary.get("quality_listening_status")
    if quality_listening is not None:
        accepted = ", ".join(f"`{bucket}`" for bucket in quality_listening["accepted_buckets"]) or "none"
        pending = ", ".join(f"`{bucket}`" for bucket in quality_listening["pending_buckets"]) or "none"
        failed = ", ".join(f"`{bucket}`" for bucket in quality_listening["failed_buckets"]) or "none"
        lines.extend(
            [
                "",
                "## Quality-Branch Listening Status",
                "",
                "- No ASR/Whisper gate is used for this branch.",
                f"- Listening rows: `{quality_listening['rows']}`.",
                f"- Rows with listening artifacts: `{quality_listening['mapped_count']}`.",
                f"- Accepted human decisions: `{quality_listening['accepted_count']}`.",
                f"- Pending human decisions: `{quality_listening['pending_count']}`.",
                f"- Failed human decisions: `{quality_listening['failed_count']}`.",
                "",
                "| Status | Buckets |",
                "| --- | --- |",
                f"| accepted | {accepted} |",
                f"| pending | {pending} |",
                f"| failed | {failed} |",
            ]
        )
    feasibility = summary.get("xsource_distillation_feasibility")
    if feasibility is not None:
        lines.extend(
            [
                "",
                "## Source-Side Feasibility Smoke",
                "",
                f"- Tensor dump: `{feasibility['tensor_dump']}`.",
                f"- Decision: `{feasibility['decision']}`.",
                f"- Target mode: `{feasibility.get('target_mode')}`.",
                f"- Model: `{feasibility.get('model')}`.",
                f"- Feature set: `{feasibility.get('feature_set')}`.",
                f"- Radius: `{feasibility['radius']}`.",
                f"- Holdout stride: `{feasibility['holdout_stride']}`.",
                f"- Hidden: `{feasibility.get('hidden')}`.",
                f"- Steps: `{feasibility.get('steps')}`.",
                f"- Conv kernel: `{feasibility.get('conv_kernel')}`.",
                f"- Conv depth: `{feasibility.get('conv_depth')}`.",
                "",
                "| Target | Features | Validation SNR | Validation corr | Validation max abs |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in feasibility["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['name']}`",
                        str(row["feature_count"]),
                        _fmt_db(row["validation_snr_db"]),
                        f"{float(row['validation_correlation']):.6f}",
                        f"{float(row['validation_max_abs_error']):.6f}",
                    ]
                )
                + " |"
            )
    folding = summary.get("pre_noise_folding_surface")
    if folding is not None:
        lines.extend(
            [
                "",
                "## Pre-Noise Folding Surface",
                "",
                f"- Decision: `{folding['decision']}`.",
                "",
                "| Bucket | HAR fp16 | Touched HAR frames | Pre-noise fp16 | Pre-noise/HAR values |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in folding["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['input_key']}`",
                        _fmt_mib(row["full_har_bytes_fp16"]),
                        f"{float(row['union_touched_har_frame_pct']):.2f}%",
                        _fmt_mib(row["total_pre_noise_bytes_fp16"]),
                        f"{float(row['pre_noise_to_har_value_ratio']):.2f}x",
                    ]
                )
                + " |"
            )
    ceiling = summary.get("strict_folding_ceiling")
    if ceiling is not None:
        lines.extend(
            [
                "",
                "## Strict Folding Ceiling",
                "",
                f"- Decision: `{ceiling['decision']}`.",
                f"- Profile rows closed by STFT-only removal: `{ceiling['profile_rows_closed_by_stft_only']}`.",
                f"- Paper rows closed by STFT-only removal: `{ceiling['paper_rows_closed_by_stft_only']}`.",
                "",
                "| Bucket | Removable STFT | Profile coverage | Paper coverage |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in ceiling["rows"]:
            profile_cov = row["profile_gap_coverage_pct"]
            paper_cov = row["paper_gap_coverage_pct"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['bucket']}`",
                        _fmt_ms(row["removable_swift_stft_ms"]),
                        "n/a" if profile_cov is None else f"{float(profile_cov):.2f}%",
                        "n/a" if paper_cov is None else f"{float(paper_cov):.2f}%",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Body Counterfactual",
            "",
            "| Machine | Fused | Body only | Source/noise | Full split | Body-only save | Full split delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["body_counterfactuals"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    _fmt_ms(row["fused_ms"]),
                    _fmt_ms(row["body_only_ms"]),
                    _fmt_ms(row["source_noise_ms"]),
                    _fmt_ms(row["split_total_ms"]),
                    _fmt_ms(row["body_only_saves_ms"]),
                    _fmt_ms(row["full_split_delta_ms"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Quality-Fail Closers",
            "",
            "Quality-fail buckets that would beat the warmed laishere profile if accepted:",
            (
                "`" + "`, `".join(str(v) for v in summary["quality_fail_warmed_profile_buckets"]) + "`."
                if summary["quality_fail_warmed_profile_buckets"]
                else "None."
            ),
            "",
            "## Decision",
            "",
            summary["decision"],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-variants-json", type=Path, default=DEFAULT_SOURCE_VARIANTS_JSON)
    parser.add_argument("--irvine-targets-json", type=Path, default=DEFAULT_IRVINE_TARGETS_JSON)
    parser.add_argument("--m2-body-report", type=Path, default=DEFAULT_M2_BODY_REPORT)
    parser.add_argument("--irvine-body-report", type=Path, default=DEFAULT_IRVINE_BODY_REPORT)
    parser.add_argument("--strict-budget-json", type=Path, default=DEFAULT_STRICT_BUDGET)
    parser.add_argument("--xsource-feasibility-json", type=Path, default=DEFAULT_XSOURCE_FEASIBILITY)
    parser.add_argument("--pre-noise-folding-surface-json", type=Path, default=DEFAULT_PRE_NOISE_FOLDING_SURFACE)
    parser.add_argument("--strict-folding-ceiling-json", type=Path, default=DEFAULT_STRICT_FOLDING_CEILING)
    parser.add_argument("--irvine-listening-targets-json", type=Path, default=DEFAULT_IRVINE_LISTENING_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = summarize_source_contract_frontier(
        _load_json(args.source_variants_json),
        _load_json(args.irvine_targets_json),
        _load_json(args.m2_body_report),
        _load_json(args.irvine_body_report),
        _load_json(args.strict_budget_json),
        _load_optional_json(args.xsource_feasibility_json),
        _load_optional_json(args.pre_noise_folding_surface_json),
        _load_optional_json(args.strict_folding_ceiling_json),
        _load_optional_json(args.irvine_listening_targets_json),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_equation_is_solved": summary["source_equation_is_solved"],
                "recomputed_stft_har_is_solved": summary["recomputed_stft_har_is_solved"],
                "quality_fail_warmed_profile_closers": summary[
                    "quality_fail_warmed_profile_closers"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
