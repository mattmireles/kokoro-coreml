#!/usr/bin/env python3
"""Probe splitting HAR noise-source generation out of ``GeneratorFromHar``.

This is closer to the laishere split than a simple final-tail split. The fused
shipping package accepts ``x_pre`` + ``ref_s`` + long HAR features and computes
both:

1. ``noise_convs`` + ``noise_res`` over HAR.
2. The main upsample/resblock/vocoder tail path.

This script exports temporary packages:

- ``noise``: ``ref_s`` + ``har`` -> ``x_source_0`` / ``x_source_1``
- ``body``: ``x_pre`` + ``ref_s`` + ``x_source_*`` -> ``waveform``

It then benchmarks ``noise.predict`` + ``body.predict`` against the current
fused generator package on the exact same Swift tensor dump. Generated packages
and reports stay under ``outputs/``.
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
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _duration_label_from_dump, _precision_arg, _remove_existing_package  # noqa: E402


def _make_noise_module(generator: Any):
    import torch
    import torch.nn as nn

    class _GeneratorNoiseFromHar(nn.Module):
        """HAR noise branch only: ``ref_s`` + ``har`` -> per-upsample sources."""

        def __init__(self, gen: Any):
            super().__init__()
            self.noise_convs = gen.noise_convs
            self.noise_res = gen.noise_res

        def forward(self, ref_s: torch.Tensor, har: torch.Tensor):
            s = ref_s[:, :128]
            outputs = []
            for conv, res in zip(self.noise_convs, self.noise_res):
                x_source = conv(har)
                x_source = res(x_source, s)
                outputs.append(x_source)
            return tuple(outputs)

    return _GeneratorNoiseFromHar(generator).eval()


def _make_body_module(generator: Any, source_count: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _GeneratorBodyFromNoise(nn.Module):
        """Main generator path fed by precomputed noise sources."""

        def __init__(self, gen: Any):
            super().__init__()
            self.generator = gen
            self.source_count = source_count

        def forward(self, x_pre: torch.Tensor, ref_s: torch.Tensor, *noise_sources: torch.Tensor) -> torch.Tensor:
            if len(noise_sources) != self.source_count:
                raise ValueError(f"expected {self.source_count} noise sources, got {len(noise_sources)}")
            s = ref_s[:, :128]
            gen = self.generator
            x = x_pre
            for i in range(gen.num_upsamples):
                x = F.leaky_relu(x, negative_slope=0.1)
                x = gen.ups[i](x)
                if i == gen.num_upsamples - 1:
                    x = gen.reflection_pad(x)
                x_source = noise_sources[i]
                tx = x.size(2)
                ts = x_source.size(2)
                if ts < tx:
                    x_source = F.pad(x_source, (0, tx - ts))
                elif ts > tx:
                    x_source = x_source[:, :, :tx]
                x = x + x_source
                xs = None
                for j in range(gen.num_kernels):
                    y = gen.resblocks[i * gen.num_kernels + j](x, s)
                    xs = y if xs is None else xs + y
                x = xs / gen.num_kernels
            x = F.leaky_relu(x)
            x = gen.conv_post(x)
            spec = torch.exp(x[:, : gen.post_n_fft // 2 + 1, :])
            phase = torch.sin(x[:, gen.post_n_fft // 2 + 1 :, :])
            return gen.stft.inverse(spec, phase)

    return _GeneratorBodyFromNoise(generator).eval()


def _predict_fused(fused: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = fused.predict(inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    key = "waveform" if "waveform" in out else next(iter(out))
    return np.asarray(out[key], dtype=np.float32), elapsed_ms


def _split_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _export_packages(
    noise_package: Path,
    body_package: Path,
    tensors: dict[str, np.ndarray],
    precision: str,
) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import remove_dropout

    kmodel = _load_kmodel()
    gen = kmodel.decoder.generator

    ref_s_shape = tuple(int(v) for v in tensors["ref_s"].shape)
    har_shape = tuple(int(v) for v in tensors["har_padded"].shape)
    x_pre_shape = tuple(int(v) for v in tensors["x_pre_padded"].shape)
    ref_s = torch.zeros(ref_s_shape, dtype=torch.float32)
    har = torch.zeros(har_shape, dtype=torch.float32)
    x_pre = torch.zeros(x_pre_shape, dtype=torch.float32)

    noise = _make_noise_module(gen)
    noise_removed_dropouts = remove_dropout(noise)
    with torch.no_grad():
        traced_noise = torch.jit.trace(noise, (ref_s, har), strict=False, check_trace=False)
        sources = tuple(traced_noise(ref_s, har))
    source_shapes = [tuple(int(v) for v in source.shape) for source in sources]

    noise_outputs = [
        ct.TensorType(name=f"x_source_{idx}") for idx in range(len(source_shapes))
    ]
    noise_model = ct.convert(
        traced_noise,
        inputs=[
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="har", shape=har_shape, dtype=np.float32),
        ],
        outputs=noise_outputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    noise_package.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_package(noise_package)
    noise_model.save(str(noise_package))

    body = _make_body_module(gen, len(source_shapes))
    body_removed_dropouts = remove_dropout(body)
    with torch.no_grad():
        traced_body = torch.jit.trace(
            body,
            (x_pre, ref_s, *sources),
            strict=False,
            check_trace=False,
        )
        body_out = traced_body(x_pre, ref_s, *sources)
    body_samples = int(body_out.shape[-1])

    body_inputs = [
        ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
        ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
    ]
    for idx, shape in enumerate(source_shapes):
        body_inputs.append(ct.TensorType(name=f"x_source_{idx}", shape=shape, dtype=np.float32))

    body_model = ct.convert(
        traced_body,
        inputs=body_inputs,
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    _remove_existing_package(body_package)
    body_model.save(str(body_package))

    return {
        "precision": precision,
        "noise_package": str(noise_package),
        "body_package": str(body_package),
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
        "x_pre_shape": list(x_pre_shape),
        "source_shapes": [list(shape) for shape in source_shapes],
        "body_samples": body_samples,
        "noise_removed_dropouts": noise_removed_dropouts,
        "body_removed_dropouts": body_removed_dropouts,
    }


def _load_models(args: argparse.Namespace, noise_package: Path, body_package: Path):
    import coremltools as ct

    fused = ct.models.MLModel(
        str(args.fused_package),
        compute_units=_compute_units(ct, args.fused_compute_units),
    )
    noise = ct.models.MLModel(
        str(noise_package),
        compute_units=_compute_units(ct, args.noise_compute_units),
    )
    body = ct.models.MLModel(
        str(body_package),
        compute_units=_compute_units(ct, args.body_compute_units),
    )
    return fused, noise, body


def _predict_split(
    noise: Any,
    body: Any,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    noise_out = noise.predict({
        "ref_s": inputs["ref_s"],
        "har": inputs["har"],
    })
    noise_ms = (time.perf_counter() - start) * 1000.0

    body_feed = {
        "x_pre": inputs["x_pre"],
        "ref_s": inputs["ref_s"],
    }
    for key, value in noise_out.items():
        body_feed[key] = np.asarray(value, dtype=np.float32)

    start = time.perf_counter()
    body_out = body.predict(body_feed)
    body_ms = (time.perf_counter() - start) * 1000.0
    waveform_key = "waveform" if "waveform" in body_out else next(iter(body_out))
    waveform = np.asarray(body_out[waveform_key], dtype=np.float32)
    return waveform, {"noise_ms": noise_ms, "body_ms": body_ms, "total_ms": noise_ms + body_ms}


def _benchmark(
    args: argparse.Namespace,
    tensors: dict[str, np.ndarray],
    noise_package: Path,
    body_package: Path,
) -> dict[str, Any]:
    inputs = _split_inputs(tensors)
    fused, noise, body = _load_models(args, noise_package, body_package)

    fused_first, fused_first_ms = _predict_fused(fused, inputs)
    split_first, split_first_times = _predict_split(noise, body, inputs)

    for _ in range(max(0, args.warmup)):
        _predict_fused(fused, inputs)
        _predict_split(noise, body, inputs)

    fused_times: list[float] = []
    split_noise_times: list[float] = []
    split_body_times: list[float] = []
    split_total_times: list[float] = []
    last_fused = fused_first
    last_split = split_first
    for _ in range(max(1, args.iterations)):
        last_fused, fused_ms = _predict_fused(fused, inputs)
        last_split, split_times = _predict_split(noise, body, inputs)
        fused_times.append(fused_ms)
        split_noise_times.append(split_times["noise_ms"])
        split_body_times.append(split_times["body_ms"])
        split_total_times.append(split_times["total_ms"])

    trim_len = int(tensors["waveform"].size)
    fused_trim = last_fused.reshape(-1)[:trim_len]
    split_trim = last_split.reshape(-1)[:trim_len]

    return {
        "fused_compute_units": args.fused_compute_units,
        "noise_compute_units": args.noise_compute_units,
        "body_compute_units": args.body_compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "fused": float(fused_first_ms),
            "split_noise": float(split_first_times["noise_ms"]),
            "split_body": float(split_first_times["body_ms"]),
            "split_total": float(split_first_times["total_ms"]),
        },
        "warm_predict_times_ms": {
            "fused": fused_times,
            "split_noise": split_noise_times,
            "split_body": split_body_times,
            "split_total": split_total_times,
        },
        "warm_predict_median_ms": {
            "fused": float(statistics.median(fused_times)),
            "split_noise": float(statistics.median(split_noise_times)),
            "split_body": float(statistics.median(split_body_times)),
            "split_total": float(statistics.median(split_total_times)),
        },
        "metrics": {
            "fused_vs_dump_full": _metrics(tensors["waveform_full"], last_fused),
            "split_vs_dump_full": _metrics(tensors["waveform_full"], last_split),
            "split_vs_fused_full": _metrics(last_fused, last_split),
            "fused_vs_dump_trimmed": _metrics(tensors["waveform"], fused_trim),
            "split_vs_dump_trimmed": _metrics(tensors["waveform"], split_trim),
            "split_vs_fused_trimmed": _metrics(fused_trim, split_trim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["x_pre_padded", "ref_s", "har_padded", "waveform_full", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    work_dir = args.output_dir / label
    noise_package = work_dir / f"kokoro_generator_noise_from_har_{label}.mlpackage"
    body_package = work_dir / f"kokoro_generator_body_from_noise_{label}.mlpackage"
    report_path = work_dir / "report.json"

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        if not noise_package.is_dir() or not body_package.is_dir():
            raise SystemExit(
                "--skip-export requested but split packages are missing: "
                f"{noise_package}, {body_package}"
            )
    else:
        export_report = _export_packages(noise_package, body_package, tensors, args.precision)

    benchmark = _benchmark(args, tensors, noise_package, body_package)
    split_metrics = benchmark["metrics"]["split_vs_fused_trimmed"]
    passes = bool(
        split_metrics["correlation"] is not None
        and split_metrics["correlation"] >= args.min_corr
        and split_metrics["snr_db"] >= args.min_snr
        and split_metrics["max_abs_error"] <= args.max_abs_error
    )

    med = benchmark["warm_predict_median_ms"]
    speedup_vs_fused_pct = None
    if med["fused"] > 0:
        speedup_vs_fused_pct = 100.0 * (med["fused"] - med["split_total"]) / med["fused"]

    report = {
        "tensor_dump": str(args.tensor_dump),
        "fused_package": str(args.fused_package),
        "noise_package": str(noise_package),
        "body_package": str(body_package),
        "report": str(report_path),
        "manifest_metadata": manifest.get("metadata", {}),
        "export": export_report,
        "benchmark": benchmark,
        "thresholds": {
            "min_corr": args.min_corr,
            "min_snr": args.min_snr,
            "max_abs_error": args.max_abs_error,
        },
        "speedup_vs_fused_pct": speedup_vs_fused_pct,
        "passes": passes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-dump",
        type=Path,
        default=Path("outputs/generator_isolation/dumps/3s"),
        help="Swift generator tensor dump.",
    )
    parser.add_argument(
        "--fused-package",
        type=Path,
        default=Path("coreml/kokoro_decoder_har_post_3s.mlpackage"),
        help="Shipping fused HAR-post package to compare against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generator_noise_split"),
        help="Directory for generated split packages and reports.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "float16", "fp32", "float32"),
        help="Core ML conversion precision for split packages.",
    )
    parser.add_argument("--fused-compute-units", default="cpuAndGPU")
    parser.add_argument("--noise-compute-units", default="cpuAndGPU")
    parser.add_argument("--body-compute-units", default="cpuAndGPU")
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
    split_metrics = report["benchmark"]["metrics"]["split_vs_fused_trimmed"]
    print(
        "generator_noise_split "
        f"passes={report['passes']} "
        f"label={Path(report['noise_package']).parent.name} "
        f"fused_median_ms={med['fused']:.3f} "
        f"split_median_ms={med['split_total']:.3f} "
        f"noise_median_ms={med['split_noise']:.3f} "
        f"body_median_ms={med['split_body']:.3f} "
        f"speedup_vs_fused_pct={report['speedup_vs_fused_pct']:.2f} "
        f"corr={split_metrics['correlation']} "
        f"snr_db={split_metrics['snr_db']:.2f} "
        f"max_abs={split_metrics['max_abs_error']:.6g} "
        f"report={report['report']}"
    )
    if args.fail_on_difference and not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
