#!/usr/bin/env python3
"""Probe per-upsample-stage latency inside the HAR-post generator.

Earlier probes showed that generic generator package boundaries do not beat the
shipping fused HAR-post package. This script asks a narrower question: which
part of the generator body is expensive, and can either upsample/resblock stage
run usefully on a different Core ML compute unit?

The temporary pipeline is:

``noise``: ``ref_s`` + ``har`` -> ``x_source_0`` / ``x_source_1``
``stage0``: ``x_pre`` + ``x_source_0`` -> intermediate feature map
``stage1_tail``: intermediate + ``x_source_1`` -> waveform

Generated packages and reports live under ``outputs/`` and do not modify the
checked-in production packages.
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
        """Static HAR noise branch: ``ref_s`` + ``har`` -> per-stage sources."""

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


def _make_stage0_module(generator: Any):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _GeneratorStage0(nn.Module):
        """First upsample/resblock stage after decoder-pre."""

        def __init__(self, gen: Any):
            super().__init__()
            if gen.num_upsamples != 2:
                raise ValueError(f"expected exactly two generator upsample stages, got {gen.num_upsamples}")
            self.generator = gen

        def forward(
            self,
            x_pre: torch.Tensor,
            ref_s: torch.Tensor,
            x_source_0: torch.Tensor,
        ) -> torch.Tensor:
            s = ref_s[:, :128]
            gen = self.generator
            x = F.leaky_relu(x_pre, negative_slope=0.1)
            x = gen.ups[0](x)
            tx = x.size(2)
            ts = x_source_0.size(2)
            if ts < tx:
                x_source_0 = F.pad(x_source_0, (0, tx - ts))
            elif ts > tx:
                x_source_0 = x_source_0[:, :, :tx]
            x = x + x_source_0
            xs = None
            for j in range(gen.num_kernels):
                y = gen.resblocks[j](x, s)
                xs = y if xs is None else xs + y
            return xs / gen.num_kernels

    return _GeneratorStage0(generator).eval()


def _make_stage1_tail_module(generator: Any):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _GeneratorStage1Tail(nn.Module):
        """Second upsample/resblock stage plus conv_post and iSTFT tail."""

        def __init__(self, gen: Any):
            super().__init__()
            if gen.num_upsamples != 2:
                raise ValueError(f"expected exactly two generator upsample stages, got {gen.num_upsamples}")
            self.generator = gen

        def forward(
            self,
            x_stage0: torch.Tensor,
            ref_s: torch.Tensor,
            x_source_1: torch.Tensor,
        ) -> torch.Tensor:
            s = ref_s[:, :128]
            gen = self.generator
            x = F.leaky_relu(x_stage0, negative_slope=0.1)
            x = gen.ups[1](x)
            x = gen.reflection_pad(x)
            tx = x.size(2)
            ts = x_source_1.size(2)
            if ts < tx:
                x_source_1 = F.pad(x_source_1, (0, tx - ts))
            elif ts > tx:
                x_source_1 = x_source_1[:, :, :tx]
            x = x + x_source_1
            xs = None
            offset = gen.num_kernels
            for j in range(gen.num_kernels):
                y = gen.resblocks[offset + j](x, s)
                xs = y if xs is None else xs + y
            x = xs / gen.num_kernels
            x = F.leaky_relu(x)
            logits = gen.conv_post(x)
            spec = torch.exp(logits[:, : gen.post_n_fft // 2 + 1, :])
            phase = torch.sin(logits[:, gen.post_n_fft // 2 + 1 :, :])
            return gen.stft.inverse(spec, phase)

    return _GeneratorStage1Tail(generator).eval()


def _predict_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _export_packages(
    noise_package: Path,
    stage0_package: Path,
    stage1_tail_package: Path,
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
    if len(source_shapes) != 2:
        raise RuntimeError(f"expected exactly two noise sources, got {len(source_shapes)}")

    noise_model = ct.convert(
        traced_noise,
        inputs=[
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="har", shape=har_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name=f"x_source_{idx}") for idx in range(len(source_shapes))],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    noise_package.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_package(noise_package)
    noise_model.save(str(noise_package))

    stage0 = _make_stage0_module(gen)
    stage0_removed_dropouts = remove_dropout(stage0)
    with torch.no_grad():
        traced_stage0 = torch.jit.trace(
            stage0,
            (x_pre, ref_s, sources[0]),
            strict=False,
            check_trace=False,
        )
        stage0_out = traced_stage0(x_pre, ref_s, sources[0])
    stage0_shape = tuple(int(v) for v in stage0_out.shape)

    stage0_model = ct.convert(
        traced_stage0,
        inputs=[
            ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="x_source_0", shape=source_shapes[0], dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="x_stage0")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    _remove_existing_package(stage0_package)
    stage0_model.save(str(stage0_package))

    stage1_tail = _make_stage1_tail_module(gen)
    stage1_tail_removed_dropouts = remove_dropout(stage1_tail)
    stage0_in = torch.zeros(stage0_shape, dtype=torch.float32)
    with torch.no_grad():
        traced_stage1_tail = torch.jit.trace(
            stage1_tail,
            (stage0_in, ref_s, sources[1]),
            strict=False,
            check_trace=False,
        )
        waveform = traced_stage1_tail(stage0_in, ref_s, sources[1])
    waveform_samples = int(waveform.shape[-1])

    stage1_tail_model = ct.convert(
        traced_stage1_tail,
        inputs=[
            ct.TensorType(name="x_stage0", shape=stage0_shape, dtype=np.float32),
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="x_source_1", shape=source_shapes[1], dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    _remove_existing_package(stage1_tail_package)
    stage1_tail_model.save(str(stage1_tail_package))

    return {
        "precision": precision,
        "noise_package": str(noise_package),
        "stage0_package": str(stage0_package),
        "stage1_tail_package": str(stage1_tail_package),
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
        "x_pre_shape": list(x_pre_shape),
        "source_shapes": [list(shape) for shape in source_shapes],
        "stage0_shape": list(stage0_shape),
        "waveform_samples": waveform_samples,
        "noise_removed_dropouts": noise_removed_dropouts,
        "stage0_removed_dropouts": stage0_removed_dropouts,
        "stage1_tail_removed_dropouts": stage1_tail_removed_dropouts,
    }


def _load_models(
    args: argparse.Namespace,
    noise_package: Path,
    stage0_package: Path,
    stage1_tail_package: Path,
):
    import coremltools as ct

    fused = ct.models.MLModel(
        str(args.fused_package),
        compute_units=_compute_units(ct, args.fused_compute_units),
    )
    noise = ct.models.MLModel(
        str(noise_package),
        compute_units=_compute_units(ct, args.noise_compute_units),
    )
    stage0 = ct.models.MLModel(
        str(stage0_package),
        compute_units=_compute_units(ct, args.stage0_compute_units),
    )
    stage1_tail = ct.models.MLModel(
        str(stage1_tail_package),
        compute_units=_compute_units(ct, args.stage1_compute_units),
    )
    return fused, noise, stage0, stage1_tail


def _predict_fused(fused: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = fused.predict(inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    key = "waveform" if "waveform" in out else next(iter(out))
    return np.asarray(out[key], dtype=np.float32), elapsed_ms


def _predict_split(
    noise: Any,
    stage0: Any,
    stage1_tail: Any,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    noise_out = noise.predict({"ref_s": inputs["ref_s"], "har": inputs["har"]})
    noise_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    stage0_out = stage0.predict(
        {
            "x_pre": inputs["x_pre"],
            "ref_s": inputs["ref_s"],
            "x_source_0": np.asarray(noise_out["x_source_0"], dtype=np.float32),
        }
    )
    stage0_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    stage1_out = stage1_tail.predict(
        {
            "x_stage0": np.asarray(stage0_out["x_stage0"], dtype=np.float32),
            "ref_s": inputs["ref_s"],
            "x_source_1": np.asarray(noise_out["x_source_1"], dtype=np.float32),
        }
    )
    stage1_ms = (time.perf_counter() - start) * 1000.0
    waveform_key = "waveform" if "waveform" in stage1_out else next(iter(stage1_out))
    waveform = np.asarray(stage1_out[waveform_key], dtype=np.float32)
    return waveform, {
        "noise_ms": noise_ms,
        "stage0_ms": stage0_ms,
        "stage1_tail_ms": stage1_ms,
        "total_ms": noise_ms + stage0_ms + stage1_ms,
    }


def _benchmark(
    args: argparse.Namespace,
    tensors: dict[str, np.ndarray],
    noise_package: Path,
    stage0_package: Path,
    stage1_tail_package: Path,
) -> dict[str, Any]:
    inputs = _predict_inputs(tensors)
    fused, noise, stage0, stage1_tail = _load_models(args, noise_package, stage0_package, stage1_tail_package)

    fused_first, fused_first_ms = _predict_fused(fused, inputs)
    split_first, split_first_times = _predict_split(noise, stage0, stage1_tail, inputs)

    for _ in range(max(0, args.warmup)):
        _predict_fused(fused, inputs)
        _predict_split(noise, stage0, stage1_tail, inputs)

    fused_times: list[float] = []
    split_noise_times: list[float] = []
    split_stage0_times: list[float] = []
    split_stage1_times: list[float] = []
    split_total_times: list[float] = []
    last_fused = fused_first
    last_split = split_first
    for _ in range(max(1, args.iterations)):
        last_fused, fused_ms = _predict_fused(fused, inputs)
        last_split, split_times = _predict_split(noise, stage0, stage1_tail, inputs)
        fused_times.append(fused_ms)
        split_noise_times.append(split_times["noise_ms"])
        split_stage0_times.append(split_times["stage0_ms"])
        split_stage1_times.append(split_times["stage1_tail_ms"])
        split_total_times.append(split_times["total_ms"])

    trim_len = int(tensors["waveform"].size)
    fused_trim = last_fused.reshape(-1)[:trim_len]
    split_trim = last_split.reshape(-1)[:trim_len]

    return {
        "fused_compute_units": args.fused_compute_units,
        "noise_compute_units": args.noise_compute_units,
        "stage0_compute_units": args.stage0_compute_units,
        "stage1_compute_units": args.stage1_compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "fused": float(fused_first_ms),
            "split_noise": float(split_first_times["noise_ms"]),
            "split_stage0": float(split_first_times["stage0_ms"]),
            "split_stage1_tail": float(split_first_times["stage1_tail_ms"]),
            "split_total": float(split_first_times["total_ms"]),
        },
        "warm_predict_times_ms": {
            "fused": fused_times,
            "split_noise": split_noise_times,
            "split_stage0": split_stage0_times,
            "split_stage1_tail": split_stage1_times,
            "split_total": split_total_times,
        },
        "warm_predict_median_ms": {
            "fused": float(statistics.median(fused_times)),
            "split_noise": float(statistics.median(split_noise_times)),
            "split_stage0": float(statistics.median(split_stage0_times)),
            "split_stage1_tail": float(statistics.median(split_stage1_times)),
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
    noise_package = work_dir / f"kokoro_generator_stage_noise_{label}.mlpackage"
    stage0_package = work_dir / f"kokoro_generator_stage0_{label}.mlpackage"
    stage1_tail_package = work_dir / f"kokoro_generator_stage1_tail_{label}.mlpackage"
    report_name = Path(args.report_name)
    if report_name.name != str(report_name):
        raise SystemExit(f"--report-name must be a filename, got {args.report_name!r}")
    report_path = work_dir / report_name

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        missing_packages = [
            str(path)
            for path in (noise_package, stage0_package, stage1_tail_package)
            if not path.is_dir()
        ]
        if missing_packages:
            raise SystemExit(f"--skip-export requested but packages are missing: {missing_packages}")
    else:
        export_report = _export_packages(
            noise_package,
            stage0_package,
            stage1_tail_package,
            tensors,
            args.precision,
        )

    benchmark = _benchmark(args, tensors, noise_package, stage0_package, stage1_tail_package)
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
        "stage0_package": str(stage0_package),
        "stage1_tail_package": str(stage1_tail_package),
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
        default=Path("outputs/generator_stage_split"),
        help="Directory for generated stage packages and reports.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "float16", "fp32", "float32"),
        help="Core ML conversion precision for split packages.",
    )
    parser.add_argument("--fused-compute-units", default="cpuAndGPU")
    parser.add_argument("--noise-compute-units", default="cpuAndGPU")
    parser.add_argument("--stage0-compute-units", default="cpuAndGPU")
    parser.add_argument("--stage1-compute-units", default="cpuAndGPU")
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
        "generator_stage_split "
        f"passes={report['passes']} "
        f"label={Path(report['noise_package']).parent.name} "
        f"fused_median_ms={med['fused']:.3f} "
        f"split_median_ms={med['split_total']:.3f} "
        f"noise_median_ms={med['split_noise']:.3f} "
        f"stage0_median_ms={med['split_stage0']:.3f} "
        f"stage1_tail_median_ms={med['split_stage1_tail']:.3f} "
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
