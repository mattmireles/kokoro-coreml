#!/usr/bin/env python3
"""Benchmark production ``decoder-har`` packages against shipped packages.

This compares two sets of ``kokoro_decoder_har_post_{bucket}.mlpackage`` files
using the saved Swift generator tensor dumps. It is intentionally package-first:
the benchmark proves the exported artifacts, not just the Python wrapper used to
create them.
"""

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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from scripts.probe_generator_exact_geometry import _compute_units, _metrics  # noqa: E402


def _predict(model: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = model.predict(inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    key = "waveform" if "waveform" in out else next(iter(out))
    return np.asarray(out[key], dtype=np.float32), elapsed_ms


def _package(root: Path, bucket: str) -> Path:
    return root / f"kokoro_decoder_har_post_{bucket}.mlpackage"


def _inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _benchmark_bucket(args: argparse.Namespace, bucket: str) -> dict[str, Any]:
    import coremltools as ct

    baseline_package = _package(args.baseline_dir, bucket)
    candidate_package = _package(args.candidate_dir, bucket)
    dump_dir = args.dump_dir / bucket
    if not baseline_package.is_dir():
        raise FileNotFoundError(f"missing baseline package: {baseline_package}")
    if not candidate_package.is_dir():
        raise FileNotFoundError(f"missing candidate package: {candidate_package}")
    if not dump_dir.is_dir():
        raise FileNotFoundError(f"missing tensor dump: {dump_dir}")

    _, tensors = load_tensor_dump(dump_dir)
    inputs = _inputs(tensors)
    compute_units = _compute_units(ct, args.compute_units)
    models = {
        "baseline": ct.models.MLModel(str(baseline_package), compute_units=compute_units),
        "candidate": ct.models.MLModel(str(candidate_package), compute_units=compute_units),
    }

    first_ms: dict[str, float] = {}
    warm_times: dict[str, list[float]] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, model in models.items():
        first_out, first = _predict(model, inputs)
        first_ms[name] = first
        last = first_out
        for _ in range(max(0, args.warmup)):
            last, _ = _predict(model, inputs)
        times: list[float] = []
        for _ in range(max(1, args.iterations)):
            last, elapsed = _predict(model, inputs)
            times.append(elapsed)
        outputs[name] = last
        warm_times[name] = times

    trim_len = int(tensors["waveform"].size)
    baseline_trim = outputs["baseline"].reshape(-1)[:trim_len]
    candidate_trim = outputs["candidate"].reshape(-1)[:trim_len]
    baseline_median = float(statistics.median(warm_times["baseline"]))
    candidate_median = float(statistics.median(warm_times["candidate"]))
    return {
        "bucket": bucket,
        "baseline_package": str(baseline_package),
        "candidate_package": str(candidate_package),
        "tensor_dump": str(dump_dir),
        "compute_units": args.compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": first_ms,
        "warm_predict_times_ms": warm_times,
        "warm_predict_median_ms": {
            "baseline": baseline_median,
            "candidate": candidate_median,
        },
        "speedup_vs_baseline_pct": (
            100.0 * (baseline_median - candidate_median) / baseline_median
            if baseline_median > 0
            else None
        ),
        "metrics_candidate_vs_baseline_trimmed": _metrics(baseline_trim, candidate_trim),
        "metrics_candidate_vs_dump_trimmed": _metrics(tensors["waveform"], candidate_trim),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    buckets = [item.strip() for item in args.buckets.split(",") if item.strip()]
    rows = [_benchmark_bucket(args, bucket) for bucket in buckets]
    payload = {
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "dump_dir": str(args.dump_dir),
        "compute_units": args.compute_units,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("coreml"))
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, default=Path("outputs/generator_isolation/dumps"))
    parser.add_argument("--buckets", default="3s,7s,10s,15s,30s")
    parser.add_argument("--compute-units", default="cpuAndGPU")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = run(args)
    for row in payload["rows"]:
        med = row["warm_predict_median_ms"]
        metrics = row["metrics_candidate_vs_baseline_trimmed"]
        print(
            f"{row['bucket']} baseline_ms={med['baseline']:.3f} "
            f"candidate_ms={med['candidate']:.3f} "
            f"speedup_pct={row['speedup_vs_baseline_pct']:.2f} "
            f"corr={metrics['correlation']} "
            f"snr_db={metrics['snr_db']:.2f} "
            f"max_abs={metrics['max_abs_error']:.6g}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
