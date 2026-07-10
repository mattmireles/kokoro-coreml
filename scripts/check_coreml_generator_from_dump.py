#!/usr/bin/env python3
"""Run ``GeneratorFromHar`` Core ML on tensors from an audio parity dump."""

from __future__ import annotations

import argparse
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


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    n = min(ref.size, cand.size)
    ref = ref[:n]
    cand = cand[:n]
    diff = ref - cand
    corr = None
    if ref.size > 1 and float(ref.std()) > 0 and float(cand.std()) > 0:
        corr = float(np.corrcoef(ref, cand)[0, 1])
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cosine = float(np.dot(ref, cand) / denom) if denom > 0 else None
    snr = 10.0 * np.log10((np.sum(ref * ref) + 1e-12) / (np.sum(diff * diff) + 1e-12))
    return {
        "samples_compared": int(n),
        "max_abs_error": float(np.max(np.abs(diff))) if n else 0.0,
        "mean_abs_error": float(np.mean(np.abs(diff))) if n else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if n else 0.0,
        "snr_db": float(snr),
        "correlation": corr,
        "cosine_similarity": cosine,
    }


def _compute_unit(coremltools: Any, name: str) -> Any:
    mapping = {
        "all": coremltools.ComputeUnit.ALL,
        "cpuAndGPU": coremltools.ComputeUnit.CPU_AND_GPU,
        "cpuAndNeuralEngine": coremltools.ComputeUnit.CPU_AND_NE,
        "cpuOnly": coremltools.ComputeUnit.CPU_ONLY,
    }
    return mapping[name]


def _timing_summary(times_s: list[float]) -> dict[str, Any]:
    if not times_s:
        return {
            "iterations": 0,
            "times_s": [],
            "min_s": None,
            "max_s": None,
            "mean_s": None,
            "median_s": None,
            "p90_s": None,
        }
    sorted_times = sorted(times_s)
    p90_index = min(len(sorted_times) - 1, int(round(0.9 * (len(sorted_times) - 1))))
    return {
        "iterations": len(times_s),
        "times_s": times_s,
        "min_s": float(min(times_s)),
        "max_s": float(max(times_s)),
        "mean_s": float(statistics.fmean(times_s)),
        "median_s": float(statistics.median(times_s)),
        "p90_s": float(sorted_times[p90_index]),
    }


def _timed_predict(model: Any, inputs: dict[str, np.ndarray]) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    prediction = model.predict(inputs)
    elapsed = time.perf_counter() - start
    return prediction, elapsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["x_pre_padded", "ref_s", "har_padded", "waveform_full", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    inputs = {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }
    model = ct.models.MLModel(str(args.package), compute_units=_compute_unit(ct, args.compute_units))
    prediction, first_prediction_time_s = _timed_predict(model, inputs)
    warmup_times_s = []
    for _ in range(args.warmup):
        prediction, elapsed = _timed_predict(model, inputs)
        warmup_times_s.append(float(elapsed))
    iteration_times_s = []
    for _ in range(args.iterations):
        prediction, elapsed = _timed_predict(model, inputs)
        iteration_times_s.append(float(elapsed))

    key = "waveform" if "waveform" in prediction else next(iter(prediction))
    waveform_full = np.asarray(prediction[key], dtype=np.float32)
    trim_len = int(manifest.get("metadata", {}).get("trim_len") or tensors["waveform"].size)
    waveform = waveform_full.reshape(-1)[:trim_len]

    report = {
        "tensor_dump": str(args.tensor_dump),
        "package": str(args.package),
        "compute_units": args.compute_units,
        "prediction_key": key,
        "first_prediction_time_s": float(first_prediction_time_s),
        "warmup": _timing_summary(warmup_times_s),
        "warmed": _timing_summary(iteration_times_s),
        "reference_metadata": manifest.get("metadata", {}),
        "waveform_full_metrics": _metrics(tensors["waveform_full"], waveform_full),
        "waveform_trimmed_metrics": _metrics(tensors["waveform"], waveform),
    }
    report["passes"] = bool(
        report["waveform_trimmed_metrics"]["correlation"] is not None
        and report["waveform_trimmed_metrics"]["correlation"] >= args.min_corr
        and report["waveform_trimmed_metrics"]["snr_db"] >= args.min_snr
    )
    if args.write_json:
        path = Path(args.write_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dump", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--write-json", default=None)
    parser.add_argument("--min-corr", type=float, default=0.99)
    parser.add_argument("--min-snr", type=float, default=35.0)
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpuAndGPU", "cpuAndNeuralEngine", "cpuOnly"],
        default="cpuAndGPU",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 0:
        parser.error("--warmup and --iterations must be non-negative")

    report = run(args)
    trimmed = report["waveform_trimmed_metrics"]
    warmed = report["warmed"]
    warmed_median = "n/a"
    if warmed["median_s"] is not None:
        warmed_median = f"{warmed['median_s'] * 1000.0:.3f}"
    print(
        "coreml_generator_from_dump "
        f"passes={report['passes']} "
        f"compute_units={report['compute_units']} "
        f"warmed_median_ms={warmed_median} "
        f"corr={trimmed['correlation']} "
        f"snr_db={trimmed['snr_db']:.2f} "
        f"max_abs={trimmed['max_abs_error']:.6g}"
    )
    if args.fail_on_difference and not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
