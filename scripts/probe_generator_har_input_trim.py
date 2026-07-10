#!/usr/bin/env python3
"""Probe trimming ``GeneratorFromHar`` HAR input while keeping bucket ``x_pre``.

The shipping decoder-har packages trace ``GeneratorFromHar`` with doubled
internal audio geometry. For the 3s bucket, that means ``x_pre`` has 240 frames
but ``har`` has 28,801 frames, even though the Swift harmonic source dump has
14,401 real frames followed by zero padding. This probe tests a narrower
runtime change than exact output geometry:

- keep the bucketed decoder output shape unchanged;
- export a temporary generator package with a shorter static ``har`` axis;
- crop the existing Swift ``har_padded`` tensor to that axis;
- compare warm Core ML predict latency and waveform parity against the checked
  in generator package.

Generated packages and reports stay under ``outputs/``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _duration_label_from_dump, _precision_arg, _remove_existing_package  # noqa: E402


def _package_version(package: str) -> str | None:
    """Return installed package version for report provenance."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _toolchain_report() -> dict[str, str | None]:
    """Return conversion/runtime package versions."""

    return {
        "coremltools": _package_version("coremltools"),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
    }


def _trim_or_pad_last_dim(array: np.ndarray, length: int) -> np.ndarray:
    """Return ``array`` with its last dimension cropped or zero-padded."""

    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    arr = np.asarray(array, dtype=np.float32)
    current = int(arr.shape[-1])
    if current == length:
        return np.ascontiguousarray(arr)
    if current > length:
        return np.ascontiguousarray(arr[..., :length])
    out_shape = list(arr.shape)
    out_shape[-1] = length
    out = np.zeros(out_shape, dtype=np.float32)
    out[..., :current] = arr
    return out


def _inputs(tensors: dict[str, np.ndarray], har_time: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return baseline and candidate Core ML prediction inputs."""

    baseline = {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }
    candidate = {
        "x_pre": baseline["x_pre"],
        "ref_s": baseline["ref_s"],
        "har": _trim_or_pad_last_dim(tensors["har_padded"], har_time),
    }
    return baseline, candidate


def _export_package(
    package: Path,
    tensors: dict[str, np.ndarray],
    har_time: int,
    precision: str,
) -> dict[str, Any]:
    """Export a temporary generator package with the requested HAR axis."""

    import coremltools as ct
    import torch

    from export_synth.wrappers import GeneratorFromHar, remove_dropout

    kmodel = _load_kmodel()
    gen_from_har = GeneratorFromHar(kmodel.decoder.generator).eval()
    removed_dropouts = remove_dropout(gen_from_har)

    x_pre_shape = tuple(int(v) for v in tensors["x_pre_padded"].shape)
    ref_s_shape = tuple(int(v) for v in tensors["ref_s"].shape)
    har_shape = (
        int(tensors["har_padded"].shape[0]),
        int(tensors["har_padded"].shape[1]),
        int(har_time),
    )

    x_pre = torch.zeros(x_pre_shape, dtype=torch.float32)
    ref_s = torch.zeros(ref_s_shape, dtype=torch.float32)
    har = torch.zeros(har_shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(
            gen_from_har,
            (x_pre, ref_s, har),
            strict=False,
            check_trace=False,
        )
        traced_out = traced(x_pre, ref_s, har)

    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="har", shape=har_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    package.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_package(package)
    model.save(str(package))

    return {
        "toolchain": _toolchain_report(),
        "package": str(package),
        "precision": precision,
        "x_pre_shape": list(x_pre_shape),
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
        "traced_output_shape": [int(v) for v in traced_out.shape],
        "removed_dropouts": removed_dropouts,
    }


def _predict(model: Any, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = model.predict(feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    waveform = out.get("waveform", next(iter(out.values())))
    return np.asarray(waveform, dtype=np.float32), elapsed_ms


def _benchmark(
    args: argparse.Namespace,
    package: Path,
    tensors: dict[str, np.ndarray],
) -> dict[str, Any]:
    import coremltools as ct

    baseline_model = ct.models.MLModel(
        str(args.baseline_package),
        compute_units=_compute_units(ct, args.baseline_compute_units),
    )
    candidate_model = ct.models.MLModel(
        str(package),
        compute_units=_compute_units(ct, args.candidate_compute_units),
    )
    baseline_inputs, candidate_inputs = _inputs(tensors, args.har_time)

    baseline_first, baseline_first_ms = _predict(baseline_model, baseline_inputs)
    candidate_first, candidate_first_ms = _predict(candidate_model, candidate_inputs)

    for _ in range(max(0, args.warmup)):
        _predict(baseline_model, baseline_inputs)
        _predict(candidate_model, candidate_inputs)

    baseline_times: list[float] = []
    candidate_times: list[float] = []
    last_baseline = baseline_first
    last_candidate = candidate_first
    for _ in range(max(1, args.iterations)):
        last_baseline, baseline_ms = _predict(baseline_model, baseline_inputs)
        last_candidate, candidate_ms = _predict(candidate_model, candidate_inputs)
        baseline_times.append(baseline_ms)
        candidate_times.append(candidate_ms)

    trim_len = min(
        int(tensors["waveform"].size),
        int(last_baseline.size),
        int(last_candidate.size),
    )
    dump_trim = tensors["waveform"].reshape(-1)[:trim_len]
    baseline_trim = last_baseline.reshape(-1)[:trim_len]
    candidate_trim = last_candidate.reshape(-1)[:trim_len]

    return {
        "toolchain": _toolchain_report(),
        "baseline_compute_units": args.baseline_compute_units,
        "candidate_compute_units": args.candidate_compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "baseline": float(baseline_first_ms),
            "candidate": float(candidate_first_ms),
        },
        "warm_predict_times_ms": {
            "baseline": baseline_times,
            "candidate": candidate_times,
        },
        "warm_predict_median_ms": {
            "baseline": float(statistics.median(baseline_times)),
            "candidate": float(statistics.median(candidate_times)),
        },
        "metrics": {
            "baseline_vs_dump_trimmed": _metrics(dump_trim, baseline_trim),
            "candidate_vs_dump_trimmed": _metrics(dump_trim, candidate_trim),
            "candidate_vs_baseline_trimmed": _metrics(baseline_trim, candidate_trim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["x_pre_padded", "ref_s", "har_padded", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    label = f"{label}_har{args.har_time}"
    work_dir = args.output_dir / label
    package = work_dir / f"kokoro_generator_har_trim_{label}.mlpackage"
    report_name = Path(args.report_name)
    if report_name.name != str(report_name):
        raise SystemExit(f"--report-name must be a filename, got {args.report_name!r}")
    report_path = work_dir / report_name

    export_report = None
    if args.skip_export:
        if not package.is_dir():
            raise SystemExit(f"--skip-export requested but package is missing: {package}")
    else:
        export_report = _export_package(package, tensors, args.har_time, args.precision)

    benchmark = _benchmark(args, package, tensors)
    metrics = benchmark["metrics"]["candidate_vs_baseline_trimmed"]
    passes = bool(
        metrics["correlation"] is not None
        and metrics["correlation"] >= args.min_corr
        and metrics["snr_db"] >= args.min_snr
        and metrics["max_abs_error"] <= args.max_abs_error
    )
    med = benchmark["warm_predict_median_ms"]
    speedup_vs_baseline_pct = 100.0 * (med["baseline"] - med["candidate"]) / med["baseline"]

    report = {
        "tensor_dump": str(args.tensor_dump),
        "baseline_package": str(args.baseline_package),
        "package": str(package),
        "report": str(report_path),
        "manifest_metadata": manifest.get("metadata", {}),
        "export": export_report,
        "benchmark": benchmark,
        "thresholds": {
            "min_corr": args.min_corr,
            "min_snr": args.min_snr,
            "max_abs_error": args.max_abs_error,
        },
        "speedup_vs_baseline_pct": speedup_vs_baseline_pct,
        "passes": passes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dump", type=Path, default=Path("outputs/generator_isolation/dumps/3s"))
    parser.add_argument("--baseline-package", type=Path, default=Path("coreml/kokoro_decoder_har_post_3s.mlpackage"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generator_har_input_trim"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--har-time", type=int, required=True)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--baseline-compute-units", default="cpuAndGPU")
    parser.add_argument("--candidate-compute-units", default="cpuAndGPU")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--min-corr", type=float, default=0.99)
    parser.add_argument("--min-snr", type=float, default=35.0)
    parser.add_argument("--max-abs-error", type=float, default=1e-2)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args()

    report = run(args)
    med = report["benchmark"]["warm_predict_median_ms"]
    metrics = report["benchmark"]["metrics"]["candidate_vs_baseline_trimmed"]
    print(
        "generator_har_input_trim "
        f"passes={report['passes']} "
        f"label={Path(report['package']).parent.name} "
        f"baseline_median_ms={med['baseline']:.3f} "
        f"candidate_median_ms={med['candidate']:.3f} "
        f"speedup_vs_baseline_pct={report['speedup_vs_baseline_pct']:.2f} "
        f"corr={metrics['correlation']} "
        f"snr_db={metrics['snr_db']:.2f} "
        f"max_abs={metrics['max_abs_error']:.6g} "
        f"report={report['report']}"
    )
    if args.fail_on_difference and not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
