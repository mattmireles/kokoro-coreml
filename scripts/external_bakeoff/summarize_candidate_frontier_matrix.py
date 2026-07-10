#!/usr/bin/env python3
"""Summarize known Config F speed candidates and next useful gates.

This ledger is intentionally evidence-driven. It records both successes and
rejections so future optimization passes do not repeat measured dead ends while
the remaining lower-end Mac and iPhone gates are waiting on external state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_frontier_matrix.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_frontier_matrix.json"
DEFAULT_STRICT_BUDGET = DEFAULT_OUTPUT_DIR / "strict_win_budget_after_rewrite.json"
DEFAULT_OVERLAP_REWRITE_BUDGET = DEFAULT_OUTPUT_DIR / "strict_win_budget_after_overlap_rewrite.json"
DEFAULT_IOS_INSTALL = DEFAULT_OUTPUT_DIR / "config_f_ios_manual_install_latest.json"


@dataclass(frozen=True)
class Candidate:
    """One measured optimization family."""

    family: str
    scope: str
    best_signal: str
    quality: str
    strict: bool
    production_ready: bool
    decision: str
    evidence: str
    next_gate: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        family="DecoderPre/HnSF runtime overlap",
        scope="Swift runtime scheduling inside current Config F boundary",
        best_signal="local M2 Studio hash-identical all-bucket save: +4.80% 3s, +5.36% 7s, +6.92% 10s, +7.24% 15s, +14.25% 30s versus same-binary serial shipped path",
        quality="strict; overlap-only WAV hashes identical on all five buckets",
        strict=True,
        production_ready=True,
        decision="keep enabled by default; fallback env KOKORO_DISABLE_DECODER_HNSF_OVERLAP=1",
        evidence="outputs/external_bakeoff/lower_end_mac_win_attempts.md",
        next_gate="quiet lower-end A/B; if a host regresses, use fallback and inspect decoder-pre CPU contention",
    ),
    Candidate(
        family="HAR-post upsample ConvT rewrite",
        scope="single-package GeneratorFromHar",
        best_signal="M2 Studio package +4.28% 3s, +3.15% 7s, +3.17% 10s, +2.60% 15s, +2.20% 30s; local E2E +1.22-2.58%",
        quality="strict-like spotchecks: corr >=0.999994895, SNR >=46.51 dB, max abs <=0.008453",
        strict=True,
        production_ready=True,
        decision="keep; promote to quiet Irvine timing before replacing checked-in packages",
        evidence="outputs/external_bakeoff/rewrite_candidate_impact.md",
        next_gate="quiet Irvine M1 warmed run; quiet M2 Air rerun for fresh lower-end rows",
    ),
    Candidate(
        family="Generator outputBackings",
        scope="Swift generator-isolation harness using MLPredictionOptions.outputBackings",
        best_signal="local M2 Studio CPU+GPU generator-only: 3s -0.077 ms, 7s +0.414 ms versus plain prediction(from:)",
        quality="strict; 3s waveform_full and trimmed waveform dumps were bit-identical, max_abs 0.0",
        strict=True,
        production_ready=False,
        decision="reject as current production win; keep --generator-output-backing as a device-check harness flag",
        evidence="README/Notes/kokoro-restarted-guide-triage-2026-06-06.md; outputs/generator_output_backing/",
        next_gate="only revisit if a lower-end Mac or iPhone shows a material >1 ms warmed median gain",
    ),
    Candidate(
        family="Reusable Swift input MLMultiArrays",
        scope="Swift benchmark runtime input-buffer reuse for DecoderPre and Generator features",
        best_signal="local M2 Studio staged A/B is hash-identical but slower: 3s 49.412 -> 50.011 ms (-1.21%), 7s 93.877 -> 97.355 ms (-3.71%)",
        quality="strict; role-keyed reusable arrays produce hash-identical 3s/7s WAVs and exact 3s tensor-dump parity after fixing same-shape aliasing and channel-major HAR padding",
        strict=True,
        production_ready=False,
        decision="reject as production win; keep --reuse-input-arrays as a diagnostic flag only",
        evidence="outputs/external_bakeoff/results_config_f_reference_m2-studio-local_reuse_arrays_baseline_v3_short.json; outputs/external_bakeoff/results_config_f_reference_m2-studio-local_reuse_arrays_candidate_v3_short.json",
        next_gate="only revisit if a lower-end Mac shows a contradictory material win; do not enable by default",
    ),
    Candidate(
        family="DecoderPre + Generator merged package",
        scope="single-package DecoderPre plus GeneratorFromHar while preserving Swift HAR/STFT boundary",
        best_signal="local M2 Studio 3s merged CPU+GPU is slower than two predictions: 31.929 -> 33.589 ms (-5.20%); .all is also slower (-4.70%); CPU+NE hard-rejects at 1568.424 ms with ANE compiler failure",
        quality="strict-like versus current two-prediction baseline: SNR 47.90 dB, corr 0.999991, max abs 0.002686",
        strict=True,
        production_ready=False,
        decision="reject; removing this boundary makes the MLProgram larger and less schedulable than the cheap separate DecoderPre call plus GPU generator",
        evidence="README/Notes/performance-notes.md; outputs/decoder_pre_generator_merge/3s/report.json; outputs/decoder_pre_generator_merge/3s/report_all.json; outputs/decoder_pre_generator_merge/3s/report_cpune.json",
        next_gate="do not promote; only revisit if a graph rewrite removes the handoff without absorbing DecoderPre into the generator scheduling surface",
    ),
    Candidate(
        family="Full visible surface rewrite",
        scope="single-package GeneratorFromHar",
        best_signal="local E2E only wins 3s (49.532 ms vs production rewrite 49.669 ms) and noise-ties 10s",
        quality="strict-like spotchecks: corr >=0.999993675, SNR >=48.01 dB, max abs <=0.007263",
        strict=True,
        production_ready=False,
        decision="reject as production replacement; simpler rewrite wins more buckets",
        evidence="outputs/external_bakeoff/results_config_f_reference_m2-studio-local_full_surface_ups_as_conv.json",
        next_gate="none unless a new operator rewrite changes runtime behavior beyond surface matching",
    ),
    Candidate(
        family="Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite",
        scope="single-package GeneratorFromHar",
        best_signal="local 3s roughly +0.08-0.26%; .all/CPU+NE remains harmful",
        quality="strict",
        strict=True,
        production_ready=False,
        decision="reject as too small; graph-surface parity alone is not enough",
        evidence="README/Notes/performance-notes.md",
        next_gate="do not repeat without a new layout, fusion, or partitioning mechanism",
    ),
    Candidate(
        family="HAR input trim",
        scope="single-package GeneratorFromHar with shorter strict HAR axis",
        best_signal="Irvine 3s +0.43%; M2 Studio strict point is slower (-1.15%)",
        quality="strict at har_time=28561; shorter har_time=27601 fails max-abs gate",
        strict=True,
        production_ready=False,
        decision="reject; less than one millisecond on M1 cannot close laishere gap",
        evidence="README/Notes/performance-notes.md",
        next_gate="do not repeat tail trim unless a new source/HAR representation removes much more padding",
    ),
    Candidate(
        family="Style-specialized fused generator",
        scope="single-package GeneratorFromHar with fixed af_heart projections",
        best_signal="Irvine 3s -3.0 ms; M2 Air 3s -2.2 ms; local native-IN variant only +0.07 ms",
        quality="strict on CPU+GPU; CPU+NE fails quality and is far slower",
        strict=True,
        production_ready=False,
        decision="reject; freezing style is not a speed path",
        evidence="README/Notes/performance-notes.md",
        next_gate="none unless combined with a material new operator rewrite",
    ),
    Candidate(
        family="Style-specialized generator plus upsample rewrite",
        scope="single-package fixed-voice GeneratorFromHar with native-IN and zero-insert upsample rewrite",
        best_signal="local 3s +4.54% vs shipped fused, only +0.17% versus production upsample rewrite at N=30; CPU+NE still CPU-preferred after ANE compile failure",
        quality="strict: corr 0.999993, SNR 49.09 dB, max abs 0.002197",
        strict=True,
        production_ready=False,
        decision="reject; noise-sized over the simpler rewrite and does not fix partitioning",
        evidence="outputs/generator_style_specialization/3s_style_native_in_ups_as_conv_ios17/report_cpu_gpu_vs_rewrite_n30.json",
        next_gate="do not promote unless multi-bucket local evidence beats production rewrite by a material margin",
    ),
    Candidate(
        family="LUT-palettized full surface plus upsample rewrite",
        scope="single-package GeneratorFromHar with native-IN, broadcast AdaIN, fp16 inputs, pal8 weights, and zero-insert upsample rewrite",
        best_signal="local 3s -2.78% versus production upsample rewrite; CPU+NE still CPU-preferred after ANE compile failure",
        quality="strict by fused-output gate but weak margin: corr 0.999880, SNR 36.57 dB, max abs 0.009857",
        strict=True,
        production_ready=False,
        decision="reject; reproduces laishere-like LUT surface but is slower and does not fix placement",
        evidence="outputs/generator_cos_snake/3s_native_broadcast_fp16_pal8_ups_as_conv_vs_rewrite_plain_broadcast_adain_native_in_pal8_fp16_inputs_ups_as_conv_ios17/report_cpu_gpu_vs_rewrite.json",
        next_gate="do not repeat palettized final-waveform packages unless compression is moved behind a separate strict tail or changes placement",
    ),
    Candidate(
        family="CT8/CT9/iOS17 toolchain-only rebuild",
        scope="single-package GeneratorFromHar rebuild with newer conversion target",
        best_signal="initial local 3s CT9 +2.14%, but 10s -0.16% and 15s -0.27%; later same-process rows tied",
        quality="strict",
        strict=True,
        production_ready=False,
        decision="reject; toolchain metadata alone is below material threshold",
        evidence="README/Notes/performance-notes.md",
        next_gate="only revisit if paired with a real graph rewrite and same-process baseline",
    ),
    Candidate(
        family="Exact decoder+vocoder split",
        scope="multi-package exact Swift HAR contract",
        best_signal="Irvine 3s CPU+GPU -24.8 ms; CPU+NE -138.3 ms",
        quality="strict versus fused path",
        strict=True,
        production_ready=False,
        decision="reject; split boundary/sync overhead exceeds body savings",
        evidence="README/Kokoro-M1-vocoder-boundary-research-brief.md",
        next_gate="do not repeat broad exact split; only try if a Core ML call boundary is removed",
    ),
    Candidate(
        family="Exact generator noise/body split",
        scope="multi-package exact HAR-post generator split",
        best_signal="body-only is faster if x_source tensors are free (M2 17.6 vs 26.4 ms; Irvine 105.9 vs 168.3 ms), but full strict split loses once noise/source is included; decoderPre overlap cannot hide noise because noise waits for HAR and HnSF already exceeds decoderPre on measured 3s hosts; padded/Nyquist 3s source-noise split is quality-good but slower (34.4 vs 30.4 ms, -13.0%)",
        quality="strict on CPU+GPU; CPU+NE quality fails",
        strict=True,
        production_ready=False,
        decision="reject; x_source body package is promising only with a cheaper strict source contract",
        evidence="outputs/external_bakeoff/lower_end_mac_win_attempts.md",
        next_gate="only revisit if x_source is produced without a separate Core ML noise call or with a new source representation",
    ),
    Candidate(
        family="HAR-source fused strict path",
        scope="source/STFT/HAR fused path",
        best_signal="natural har_source is speed-positive but quality-failing; strict padded/Nyquist fused path has replacement-quality versus the current generator but no net win after Swift STFT credit (+0.051 ms 3s, +1.326 ms 7s, +2.231 ms 10s, +14.977 ms 30s); exact Swift Float Nyquist atan2 matches the dumped-Nyquist oracle but keeps the same direct speed envelope",
        quality="strict with padded geometry and dumped Nyquist phase",
        strict=True,
        production_ready=False,
        decision="reject; preserving strict source contract loses the speed edge",
        evidence="README/Notes/har-stft-phase-contract.md; scripts/external_bakeoff/summarize_hnsf_source_boundary.py",
        next_gate="new representation only: phase reparameterization, weight folding, or a no-extra-boundary Nyquist side input",
    ),
    Candidate(
        family="Swift exact Nyquist phase repair",
        scope="PyTorch-only source/HAR sensitivity probe",
        best_signal="exact Swift Float real/imag dot products followed by atan2 match the dumped-Nyquist oracle on padded buckets: 50.06/49.14/49.87/49.21/48.42 dB SNR; branch-only Swift basis remains only 25.59/25.84/26.55/25.80/25.41 dB",
        quality="strict formula repair; scalar/affine/branch-only variants fail the 45 dB padded waveform gate",
        strict=True,
        production_ready=False,
        decision="keep as source-contract unlock; reject as standalone speed candidate because padded strict source path still has no net warmed win",
        evidence="outputs/nyquist_phase_contribution/summary.md",
        next_gate="put the exact formula inside a single no-extra-boundary graph, fold it into first noise-conv weights, or use it to design a strict source representation with less padding",
    ),
    Candidate(
        family="Exact Nyquist padding trim",
        scope="PyTorch-only source/HAR padding sweep",
        best_signal="first strict exact-Nyquist padding points are 3s har_time=28561 and 7s har_time=66601, saving only 240/28801 frames (0.83%) and 600/67201 frames (0.89%) versus full padded HAR",
        quality="strict at first pass points: 3s SNR 46.52 dB max abs 0.005349; 7s SNR 46.57 dB max abs 0.006771",
        strict=True,
        production_ready=False,
        decision="reject as standalone win; the strict padding context is still almost the full shipping HAR axis",
        evidence="README/Notes/har-stft-phase-contract.md",
        next_gate="only revisit if a graph rewrite makes HAR length reductions superlinear or combines trim with a removed Core ML call boundary",
    ),
    Candidate(
        family="Embedded DC-branch + exact Nyquist fused Core ML graph",
        scope="single-package har_source -> waveform graph with explicit DC branch and exact Swift Nyquist atan2 inside MLProgram",
        best_signal="corrected graph is strict but slower: local 3s 27.53 ms vs 26.98 ms (-2.05%); local 7s 58.36 ms vs 56.40 ms (-3.47%)",
        quality="strict after explicit phase repairs: 3s SNR 47.75 dB, 7s SNR 49.15 dB; Core ML HAR reaches 147.21 dB vs PyTorch after DC branch repair",
        strict=True,
        production_ready=False,
        decision="keep as correctness repair for no-side-input source graphs; reject as current speed win",
        evidence="README/Notes/har-stft-phase-contract.md",
        next_gate="only revisit if this source graph is fused with another removed boundary or if MIL/profile evidence shows the added STFT ops can be scheduled cheaper",
    ),
    Candidate(
        family="RangeDim/flexible input generator",
        scope="single-package GeneratorFromHar with bounded dynamic time axes",
        best_signal="local 3s 343-1561 ms candidate latency versus 31-50 ms fused baseline",
        quality="fails parity: corr about 0.999135, SNR about 27.76 dB, max abs about 0.028-0.031",
        strict=False,
        production_ready=False,
        decision="reject; dynamic broadcast/shape propagation is both slower and not strict",
        evidence="README/Notes/performance-notes.md",
        next_gate="do not use RangeDim for the fused generator hot path; keep fixed buckets",
    ),
    Candidate(
        family="Per-stage prefix compute-unit overrides",
        scope="Swift runtime policy for duration/F0Ntrain/decoder-pre",
        best_signal="local 3s duration+F0Ntrain CPU+ANE N=5 looked +1.858 ms, but N=20 shrank to +0.213 ms and 7s/10s regressed by 11.241/18.366 ms",
        quality="not strict: 3s CPU+ANE prefix WAV vs staged baseline corr 0.691758, SNR 2.38 dB",
        strict=False,
        production_ready=False,
        decision="reject as production candidate; keep per-stage override harness for diagnostics",
        evidence="README/Notes/stage-compute-policy-ablation.md",
        next_gate="none unless a future export makes duration/F0Ntrain CPU+ANE numerically stable and materially faster",
    ),
    Candidate(
        family="Linear weight quantization",
        scope="single-package final-waveform GeneratorFromHar compression",
        best_signal="int8 CPU-only +4.27% but CPU+GPU crashes; int4 iOS18 is slower",
        quality="not strict: int8 CPU-only SNR 27.62 dB, int4 SNR 11.28 dB",
        strict=False,
        production_ready=False,
        decision="reject for final-waveform generator; compression is not the missing speed path",
        evidence="README/Notes/performance-notes.md",
        next_gate="only revisit on discarded-output intermediate stages with separate final-quality tail",
    ),
    Candidate(
        family="Fast F0/source simplification",
        scope="laishere-like source/body branch",
        best_signal="Irvine 3s +10.9 to +18.7 ms depending branch",
        quality="quality-fail: corr 0.813995-0.931840, SNR 5.08-9.19 dB",
        strict=False,
        production_ready=False,
        decision="not paper-strict; only useful with source recovery or no-ASR listening acceptance",
        evidence="outputs/f0_source_listening/cos_resblock_speed_branch/README.md",
        next_gate="human listening decisions or source/STFT representation repair",
    ),
)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _default_strict_budget() -> Path:
    """Prefer the overlap+rewrite budget when the generated artifact exists."""

    if DEFAULT_OVERLAP_REWRITE_BUDGET.exists():
        return DEFAULT_OVERLAP_REWRITE_BUDGET
    return DEFAULT_STRICT_BUDGET


def _status_order(candidate: Candidate) -> tuple[int, str]:
    if candidate.production_ready:
        return (0, candidate.family)
    if candidate.strict:
        return (1, candidate.family)
    return (2, candidate.family)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build the matrix payload."""

    candidates = sorted(CANDIDATES, key=_status_order)
    budget = _load_optional_json(args.strict_budget)
    ios_install = _load_optional_json(args.ios_install)
    profile_rows_remaining = (
        budget.get("summary", {}).get("profile_rows_remaining")
        if isinstance(budget.get("summary"), dict)
        else None
    )
    ios_launch_blocker = ios_install.get("launch_blocker") or "unknown"
    return {
        "strict_budget": str(args.strict_budget),
        "ios_install": str(args.ios_install),
        "summary": {
            "candidate_count": len(candidates),
            "production_ready_strict_candidates": sum(1 for item in candidates if item.production_ready and item.strict),
            "strict_rejected_or_too_small": sum(1 for item in candidates if item.strict and not item.production_ready),
            "non_strict_candidates": sum(1 for item in candidates if not item.strict),
            "profile_rows_remaining_after_rewrite": profile_rows_remaining,
            "iphone_launch_blocker": ios_launch_blocker,
        },
        "candidates": [asdict(item) for item in candidates],
        "next_actions": [
            "Run scripts/external_bakeoff/check_remote_host_quiet.py before any lower-end Mac promotion run.",
            "Retest the HAR-post upsample rewrite on Irvine M1 and M2 Air only when outputs/external_bakeoff/remote_host_quiet_latest.md reports quiet=yes.",
            "Use scripts/external_bakeoff/run_rewrite_promotion_when_quiet.py for lower-end rewrite promotion runs; it wraps the quiet gate and passes --generator-models-dir outputs/export_rewrite_smoke.",
            "Do not use cold compile/cache timings; every frontier update must use warmed medians.",
            "Use README/Notes/fixed-cost-latency-fit.md to separate fixed-boundary overhead from duration-scaled generator cost before promoting a new optimization family.",
            "Use README/Notes/har-stft-phase-contract.md and scripts/external_bakeoff/summarize_hnsf_source_boundary.py before revisiting any har_source boundary; exact Swift Nyquist atan2 solves parity, but tail padding trim saves <1% of HAR frames and Swift STFT credit alone does not make the strict padded/Nyquist path win.",
            "Use README/Kokoro-M1-HAR-STFT-contract-deep-research-prompt.md for the next external research pass; the source equation is solved, the HAR/STFT contract is not.",
            "For a new strict candidate, require a single-package graph or a removed Core ML call boundary before lower-end promotion.",
            "Run the installed Config F iPhone runner only after the physical iPhone is unlocked; current launch blocker is device_locked.",
            "Keep fast F0/source branches separate from strict paper claims unless no-ASR human listening accepts the exact WAVs.",
        ],
    }


def _yes_no(value: bool) -> str:
    return "`yes`" if value else "`no`"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the matrix as Markdown."""

    summary = payload["summary"]
    lines = [
        "# Candidate Frontier Matrix",
        "",
        "This matrix records the current measured optimization frontier for Config F.",
        "It uses warmed-inference evidence only and separates strict production",
        "candidates from non-strict or quality-changing branches.",
        "",
        "## Summary",
        "",
        f"- Candidates recorded: `{summary['candidate_count']}`.",
        f"- Production-ready strict candidates: `{summary['production_ready_strict_candidates']}`.",
        f"- Strict rejected or too-small candidates: `{summary['strict_rejected_or_too_small']}`.",
        f"- Non-strict or quality-changing candidates: `{summary['non_strict_candidates']}`.",
        f"- Irvine profile rows remaining after current projection: `{summary['profile_rows_remaining_after_rewrite']}`.",
        f"- iPhone Config F launch blocker: `{summary['iphone_launch_blocker']}`.",
        "",
        "## Matrix",
        "",
        "| Family | Scope | Best signal | Strict | Production-ready | Decision | Next gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["candidates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    row["scope"],
                    row["best_signal"],
                    _yes_no(bool(row["strict"])),
                    _yes_no(bool(row["production_ready"])),
                    row["decision"],
                    row["next_gate"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Evidence Links", ""])
    for row in payload["candidates"]:
        lines.append(f"- {row['family']}: `{row['evidence']}`.")
    lines.extend(["", "## Next Actions", ""])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-budget", type=Path, default=_default_strict_budget())
    parser.add_argument("--ios-install", type=Path, default=DEFAULT_IOS_INSTALL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "candidate_count": payload["summary"]["candidate_count"],
                "iphone_launch_blocker": payload["summary"]["iphone_launch_blocker"],
                "output": str(args.output),
                "profile_rows_remaining_after_rewrite": payload["summary"][
                    "profile_rows_remaining_after_rewrite"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
