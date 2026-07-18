#!/usr/bin/env python3
"""Create a listening pack for F0-source exact-shape candidates.

The F0-noise exact-shape probes found a real speed opportunity on M1, but the
waveform does not meet strict tensor parity against the current Swift HAR path.
This script turns those probe outputs into a reviewable audio artifact without
re-exporting packages and without ASR:

- load the saved probe report and its Core ML packages;
- render the checked-in baseline path and F0-source candidate on the same tensor
  dump;
- write peak-normalized mono WAVs for listening;
- run the repo's objective audio-health gate and optional plots;
- emit a compact markdown review that records waveform metrics and provenance.

Generated files stay under ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from audio_quality_probe import (  # noqa: E402
    compute_metrics,
    derive_thresholds,
    sample_record,
    write_quality_report,
)
from external_bakeoff.schema import write_wav_mono16  # noqa: E402
from probe_f0_noise_exact_shape import (  # noqa: E402
    _baseline_predict,
    _candidate_predict,
    _metrics,
    _np_dtype,
    _select_inputs,
)
from probe_generator_exact_geometry import _compute_units  # noqa: E402


SAMPLE_RATE = 24_000


def _load_model(package: Path, compute_units: str) -> Any:
    """Load one Core ML package with the requested compute units."""

    import coremltools as ct

    return ct.models.MLModel(str(package), compute_units=_compute_units(ct, compute_units))


def _relative(path: Path) -> str:
    """Return a stable repo-relative path string when possible."""

    try:
        return str(path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        return str(path)


def _duration_label(report: dict[str, Any], report_path: Path) -> str:
    """Return a readable label for one F0-source probe report."""

    export = report.get("export") or {}
    parent = Path(str(report.get("noise_package") or report_path.parent)).parent.name
    if parent and parent != ".":
        return parent
    label = Path(str(report.get("tensor_dump", "f0_source"))).name
    if export.get("natural_asr"):
        label += "_natural"
    return label


def _render_pair(
    report_path: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Render baseline and candidate WAVs for one saved probe report."""

    report = json.loads(report_path.read_text())
    label = _duration_label(report, report_path)
    out_dir = out_root / label
    wav_dir = out_dir / "wav"
    plots_dir = out_dir / "plots" if args.plots else None
    wav_dir.mkdir(parents=True, exist_ok=True)

    tensor_dump = Path(str(report["tensor_dump"]))
    manifest, tensors = load_tensor_dump(tensor_dump)
    natural_asr = bool((report.get("export") or {}).get("natural_asr"))
    inputs = _select_inputs(tensors, natural_asr)

    decoder_pre = _load_model(Path(str(report["decoder_pre_package"])), args.decoder_pre_compute_units)
    fused = _load_model(Path(str(report["fused_package"])), args.fused_compute_units)
    noise = _load_model(Path(str(report["noise_package"])), args.noise_compute_units)
    body = _load_model(Path(str(report["body_package"])), args.body_compute_units)
    tail = _load_model(Path(str(report["tail_package"])), args.tail_compute_units)

    baseline_wave, baseline_times = _baseline_predict(decoder_pre, fused, inputs)
    body_input_dtype = _np_dtype(str((report.get("export") or {}).get("body_input_dtype", "fp32")))
    candidate_wave, candidate_times = _candidate_predict(noise, body, tail, inputs, body_input_dtype)
    trim_len = min(
        int(tensors["waveform"].size),
        int(baseline_wave.size),
        int(candidate_wave.size),
    )
    dump_trim = tensors["waveform"].reshape(-1)[:trim_len]
    baseline_trim = baseline_wave.reshape(-1)[:trim_len]
    candidate_trim = candidate_wave.reshape(-1)[:trim_len]

    baseline_wav = wav_dir / f"{label}_baseline.wav"
    candidate_wav = wav_dir / f"{label}_candidate.wav"
    dump_wav = wav_dir / f"{label}_swift_dump.wav"
    write_wav_mono16(baseline_wav, baseline_trim, SAMPLE_RATE)
    write_wav_mono16(candidate_wav, candidate_trim, SAMPLE_RATE)
    write_wav_mono16(dump_wav, dump_trim, SAMPLE_RATE)

    reference_metrics = [compute_metrics(baseline_wav), compute_metrics(dump_wav)]
    thresholds = derive_thresholds(reference_metrics)
    records = [
        sample_record(dump_wav, "reference", thresholds, plots_dir),
        sample_record(baseline_wav, "reference", thresholds, plots_dir),
        sample_record(candidate_wav, "candidate", thresholds, plots_dir),
    ]
    quality_report, quality_summary = write_quality_report(out_dir / "quality", thresholds, records)

    metrics = {
        "baseline_vs_dump_trimmed": _metrics(dump_trim, baseline_trim),
        "candidate_vs_dump_trimmed": _metrics(dump_trim, candidate_trim),
        "candidate_vs_baseline_trimmed": _metrics(baseline_trim, candidate_trim),
    }
    provenance = {
        "label": label,
        "asr_used": False,
        "asr_note": "No ASR/Whisper gate used; this pack is waveform health plus human listening only.",
        "source_report": _relative(report_path),
        "tensor_dump": _relative(tensor_dump),
        "manifest_metadata": manifest.get("metadata", {}),
        "natural_asr": natural_asr,
        "body_input_dtype": str((report.get("export") or {}).get("body_input_dtype", "fp32")),
        "packages": {
            "decoder_pre": report["decoder_pre_package"],
            "fused": report["fused_package"],
            "noise": report["noise_package"],
            "body": report["body_package"],
            "tail": report["tail_package"],
        },
        "compute_units": {
            "decoder_pre": args.decoder_pre_compute_units,
            "fused": args.fused_compute_units,
            "noise": args.noise_compute_units,
            "body": args.body_compute_units,
            "tail": args.tail_compute_units,
        },
        "single_predict_ms": {
            "baseline": baseline_times,
            "candidate": candidate_times,
        },
        "wavs": {
            "swift_dump": _relative(dump_wav),
            "baseline": _relative(baseline_wav),
            "candidate": _relative(candidate_wav),
        },
        "quality_report": _relative(quality_report),
        "quality_summary": _relative(quality_summary),
        "metrics": metrics,
        "waveform_gate_decision": records[-1]["decision"],
        "waveform_gate_reasons": records[-1]["reject_reasons"],
    }
    provenance_path = out_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    review_path = out_dir / "listening_review.md"
    _write_review(review_path, provenance)
    provenance["review"] = _relative(review_path)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def _metric_cell(metrics: dict[str, Any]) -> str:
    """Format waveform parity metrics for markdown."""

    corr = metrics.get("correlation")
    snr = metrics.get("snr_db")
    max_abs = metrics.get("max_abs_error")
    corr_s = "n/a" if corr is None else f"{corr:.6f}"
    snr_s = "n/a" if snr is None else f"{snr:.2f} dB"
    max_s = "n/a" if max_abs is None else f"{max_abs:.5f}"
    return f"corr {corr_s}, SNR {snr_s}, max {max_s}"


def _write_review(path: Path, provenance: dict[str, Any]) -> None:
    """Write a compact human listening review for one candidate."""

    metrics = provenance["metrics"]
    wavs = provenance["wavs"]
    lines = [
        f"# F0 Source Listening Review: {provenance['label']}",
        "",
        "This review intentionally does not use ASR or Whisper. It is a waveform-health and human-listening artifact.",
        "",
        "## Files",
        "",
        f"- Swift dump reference: `{wavs['swift_dump']}`",
        f"- Baseline Core ML path: `{wavs['baseline']}`",
        f"- F0-source candidate: `{wavs['candidate']}`",
        f"- Objective quality report: `{provenance['quality_report']}`",
        "",
        "## Waveform Metrics",
        "",
        "| Comparison | Metrics |",
        "| --- | --- |",
        f"| Baseline vs Swift dump | {_metric_cell(metrics['baseline_vs_dump_trimmed'])} |",
        f"| Candidate vs Swift dump | {_metric_cell(metrics['candidate_vs_dump_trimmed'])} |",
        f"| Candidate vs baseline | {_metric_cell(metrics['candidate_vs_baseline_trimmed'])} |",
        "",
        "## Machine Gate",
        "",
        f"- Decision: `{provenance['waveform_gate_decision']}`",
        f"- Reject reasons: `{'; '.join(provenance['waveform_gate_reasons']) or '-'}`",
        "",
        "## Listening Decision",
        "",
        "- [ ] Candidate sounds acceptable versus the baseline.",
        "- [ ] Candidate has unacceptable artifacts.",
        "- [ ] Unsure; needs more samples.",
        "",
        "Notes:",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create listening packs for all requested reports."""

    records = [_render_pair(path, args.out_dir, args) for path in args.report]
    index = {
        "asr_used": False,
        "asr_note": "No ASR/Whisper gate used; packs require human listening after waveform-health checks.",
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    review_path = args.out_dir / "README.md"
    lines = [
        "# F0 Source Listening Packs",
        "",
        "No ASR/Whisper gate is used here. These artifacts support human listening after objective waveform-health checks.",
        "",
        "| Label | Waveform gate | Candidate WAV | Review |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| `{record['label']}` | `{record['waveform_gate_decision']}` | "
            f"`{record['wavs']['candidate']}` | `{record['review']}` |"
        )
    review_path.write_text("\n".join(lines) + "\n")
    index["index"] = _relative(index_path)
    index["readme"] = _relative(review_path)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        nargs="+",
        type=Path,
        required=True,
        help="Saved probe_f0_noise_exact_shape report JSON(s).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/f0_source_listening"))
    parser.add_argument("--decoder-pre-compute-units", default="cpuAndNeuralEngine")
    parser.add_argument("--fused-compute-units", default="cpuAndGPU")
    parser.add_argument("--noise-compute-units", default="all")
    parser.add_argument("--body-compute-units", default="cpuAndGPU")
    parser.add_argument("--tail-compute-units", default="all")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    result = run(args)
    print(json.dumps({"index": result["index"], "readme": result["readme"]}, indent=2))


if __name__ == "__main__":
    main()
