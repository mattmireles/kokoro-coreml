#!/usr/bin/env python3
"""Probe operator rewrites inside the fused HAR-post generator package.

The public laishere exporter rewrites Snake as:

``sin^2(alpha * x) == (1 - cos(2 * alpha * x)) / 2``

This script tests that rewrite, and optionally tests removing explicit AdaIN
``expand`` calls so Core ML can broadcast ``gamma``/``beta`` from ``(B, C, 1)``
instead of materializing tiled ``(B, C, T)`` tensors. It exports a temporary
``GeneratorFromHar`` package using the current Swift tensor dump shape,
benchmarks it against the checked-in fused HAR-post package, and records parity
against both the Swift dump and the shipping fused output.
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
from probe_generator_dual_anchor_split import _patch_cos_snake  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _duration_label_from_dump, _precision_arg, _remove_existing_package  # noqa: E402


def _package_version(package: str) -> str | None:
    """Return the installed package version without importing heavy modules."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _toolchain_report() -> dict[str, Any]:
    """Return conversion/runtime package versions for reproducible probes."""

    return {
        "coremltools": _package_version("coremltools"),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
    }


def _deployment_target(ct: Any, name: str) -> Any:
    """Return a Core ML deployment target enum by stable CLI label."""

    targets = {
        "macos13": ct.target.macOS13,
        "macos14": ct.target.macOS14,
        "macos15": ct.target.macOS15,
        "ios17": ct.target.iOS17,
        "ios18": ct.target.iOS18,
    }
    try:
        return targets[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported deployment target {name!r}") from exc


def _maybe_palettize(mlmodel: Any, enabled: bool) -> Any:
    """Optionally apply 8-bit k-means weight palettization."""

    if not enabled:
        return mlmodel

    import coremltools.optimize.coreml as cto

    pal_config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=8)
    )
    return cto.palettize_weights(mlmodel, pal_config)


def _maybe_linear_quantize(mlmodel: Any, dtype: str | None) -> Any:
    """Optionally apply Core ML linear weight quantization."""

    if not dtype:
        return mlmodel

    import coremltools.optimize.coreml as cto

    quant_config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=dtype,
            granularity="per_channel",
        )
    )
    return cto.linear_quantize_weights(mlmodel, quant_config)


def _patch_broadcast_adain() -> None:
    """Patch AdaIN1d to rely on broadcast instead of explicit expand/tile."""

    import torch

    from export_synth import wrappers

    AdaIN1d = wrappers.kokoro_istftnet.AdaIN1d

    def _broadcast_forward(self: Any, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.shape
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        assert C == self.num_features, f"AdaIN1d channel mismatch: got {C}, expected {self.num_features}"
        h = self.fc(s).view(B, 2 * self.num_features, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1.0 + gamma) * x_norm + beta

    AdaIN1d.forward = _broadcast_forward


def _patch_native_instance_norm_adain(broadcast_adain: bool) -> None:
    """Patch AdaIN1d to use native instance_norm lowering."""

    import torch
    import torch.nn as nn

    from export_synth import wrappers

    AdaIN1d = wrappers.kokoro_istftnet.AdaIN1d

    def _native_instance_norm_init(self: Any, style_dim: int, num_features: int) -> None:
        nn.Module.__init__(self)
        self.num_features = num_features
        self.eps = 1e-5
        self.norm = nn.InstanceNorm1d(num_features, affine=False, track_running_stats=False, eps=self.eps)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def _native_instance_norm_forward(self: Any, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == self.num_features, f"AdaIN1d channel mismatch: got {C}, expected {self.num_features}"
        x_norm = self.norm(x)
        h = self.fc(s).view(B, 2 * self.num_features, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        if broadcast_adain:
            return (1.0 + gamma) * x_norm + beta
        gamma_exp = gamma.expand(B, C, T)
        beta_exp = beta.expand(B, C, T)
        return (1.0 + gamma_exp) * x_norm + beta_exp

    AdaIN1d.__init__ = _native_instance_norm_init
    AdaIN1d.forward = _native_instance_norm_forward


def _predict_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _input_shape(ct: Any, name: str, shape: tuple[int, ...], mode: str) -> Any:
    """Return the Core ML input shape for the requested shape-contract probe."""

    if mode == "fixed":
        return shape
    if mode != "range":
        raise ValueError(f"unsupported input shape mode {mode!r}")

    if name == "x_pre":
        return ct.Shape(
            shape=(
                shape[0],
                shape[1],
                ct.RangeDim(lower_bound=1, upper_bound=2000, default=shape[2]),
            )
        )
    if name == "har":
        return ct.Shape(
            shape=(
                shape[0],
                shape[1],
                ct.RangeDim(lower_bound=1, upper_bound=240001, default=shape[2]),
            )
        )
    return shape


def _input_tensor_types(
    ct: Any,
    x_pre_shape: tuple[int, ...],
    ref_s_shape: tuple[int, ...],
    har_shape: tuple[int, ...],
    shape_mode: str,
) -> list[Any]:
    """Return TensorType declarations for the generator package export."""

    return [
        ct.TensorType(
            name="x_pre",
            shape=_input_shape(ct, "x_pre", x_pre_shape, shape_mode),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="ref_s",
            shape=_input_shape(ct, "ref_s", ref_s_shape, shape_mode),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="har",
            shape=_input_shape(ct, "har", har_shape, shape_mode),
            dtype=np.float32,
        ),
    ]


def _export_package(
    package: Path,
    tensors: dict[str, np.ndarray],
    precision: str,
    cos_snake: bool,
    broadcast_adain: bool,
    deployment_target: str,
    native_instance_norm_adain: bool,
    palettize: bool,
    linear_quantize: str | None,
    input_shape_mode: str,
) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import GeneratorFromHar, remove_dropout

    if cos_snake:
        _patch_cos_snake()
    if native_instance_norm_adain:
        _patch_native_instance_norm_adain(broadcast_adain)
    elif broadcast_adain:
        _patch_broadcast_adain()

    ref_s_shape = tuple(int(v) for v in tensors["ref_s"].shape)
    x_pre_shape = tuple(int(v) for v in tensors["x_pre_padded"].shape)
    har_shape = tuple(int(v) for v in tensors["har_padded"].shape)

    kmodel = _load_kmodel()
    gen_from_har = GeneratorFromHar(kmodel.decoder.generator).eval()
    removed_dropouts = remove_dropout(gen_from_har)

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
    traced_samples = int(traced_out.shape[-1])

    mlmodel = ct.convert(
        traced,
        inputs=_input_tensor_types(ct, x_pre_shape, ref_s_shape, har_shape, input_shape_mode),
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=_deployment_target(ct, deployment_target),
        compute_precision=_precision_arg(ct, precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel = _maybe_palettize(mlmodel, palettize)
    mlmodel = _maybe_linear_quantize(mlmodel, linear_quantize)
    package.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_package(package)
    mlmodel.save(str(package))
    return {
        "package": str(package),
        "toolchain": _toolchain_report(),
        "precision": precision,
        "cos_snake": bool(cos_snake),
        "broadcast_adain": bool(broadcast_adain),
        "native_instance_norm_adain": bool(native_instance_norm_adain),
        "deployment_target": deployment_target,
        "palettize": bool(palettize),
        "linear_quantize": linear_quantize,
        "input_shape_mode": input_shape_mode,
        "removed_dropouts": removed_dropouts,
        "traced_samples": traced_samples,
        "x_pre_shape": list(x_pre_shape),
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
    }


def _predict(model: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = model.predict(inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    key = "waveform" if "waveform" in out else next(iter(out))
    return np.asarray(out[key], dtype=np.float32), elapsed_ms


def _benchmark(
    args: argparse.Namespace,
    tensors: dict[str, np.ndarray],
    cos_package: Path,
) -> dict[str, Any]:
    import coremltools as ct

    inputs = _predict_inputs(tensors)
    fused = ct.models.MLModel(
        str(args.fused_package),
        compute_units=_compute_units(ct, args.compute_units),
    )
    cos_model = ct.models.MLModel(
        str(cos_package),
        compute_units=_compute_units(ct, args.compute_units),
    )

    fused_first, fused_first_ms = _predict(fused, inputs)
    cos_first, cos_first_ms = _predict(cos_model, inputs)

    for _ in range(max(0, args.warmup)):
        _predict(fused, inputs)
        _predict(cos_model, inputs)

    fused_times: list[float] = []
    cos_times: list[float] = []
    last_fused = fused_first
    last_cos = cos_first
    for _ in range(max(1, args.iterations)):
        last_fused, fused_ms = _predict(fused, inputs)
        last_cos, cos_ms = _predict(cos_model, inputs)
        fused_times.append(fused_ms)
        cos_times.append(cos_ms)

    trim_len = int(tensors["waveform"].size)
    fused_trim = last_fused.reshape(-1)[:trim_len]
    cos_trim = last_cos.reshape(-1)[:trim_len]

    fused_median = float(statistics.median(fused_times))
    cos_median = float(statistics.median(cos_times))
    speedup_vs_fused_pct = None
    if fused_median > 0:
        speedup_vs_fused_pct = 100.0 * (fused_median - cos_median) / fused_median

    return {
        "compute_units": args.compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "fused": float(fused_first_ms),
            "cos": float(cos_first_ms),
        },
        "warm_predict_times_ms": {
            "fused": fused_times,
            "cos": cos_times,
        },
        "warm_predict_median_ms": {
            "fused": fused_median,
            "cos": cos_median,
        },
        "speedup_vs_fused_pct": speedup_vs_fused_pct,
        "metrics": {
            "fused_vs_dump_full": _metrics(tensors["waveform_full"], last_fused),
            "cos_vs_dump_full": _metrics(tensors["waveform_full"], last_cos),
            "cos_vs_fused_full": _metrics(last_fused, last_cos),
            "fused_vs_dump_trimmed": _metrics(tensors["waveform"], fused_trim),
            "cos_vs_dump_trimmed": _metrics(tensors["waveform"], cos_trim),
            "cos_vs_fused_trimmed": _metrics(fused_trim, cos_trim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["x_pre_padded", "ref_s", "har_padded", "waveform_full", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    if not args.cos_snake:
        label = f"{label}_plain"
    if args.broadcast_adain:
        label = f"{label}_broadcast_adain"
    if args.native_instance_norm_adain:
        label = f"{label}_native_in"
    if args.palettize:
        label = f"{label}_pal8"
    if args.linear_quantize:
        label = f"{label}_w{args.linear_quantize}"
    if args.input_shape_mode != "fixed":
        label = f"{label}_{args.input_shape_mode}_inputs"
    if args.deployment_target.lower() != "macos13":
        label = f"{label}_{args.deployment_target.lower()}"
    work_dir = args.output_dir / label
    cos_package = work_dir / f"kokoro_generator_cos_snake_{label}.mlpackage"
    report_name = Path(args.report_name)
    if report_name.name != str(report_name):
        raise SystemExit(f"--report-name must be a filename, got {args.report_name!r}")
    report_path = work_dir / report_name

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        if not cos_package.is_dir():
            raise SystemExit(f"--skip-export requested but package is missing: {cos_package}")
    else:
        export_report = _export_package(
            cos_package,
            tensors,
            args.precision,
            args.cos_snake,
            args.broadcast_adain,
            args.deployment_target,
            args.native_instance_norm_adain,
            args.palettize,
            args.linear_quantize,
            args.input_shape_mode,
        )

    benchmark = _benchmark(args, tensors, cos_package)
    metrics = benchmark["metrics"]["cos_vs_fused_trimmed"]
    passes = bool(
        metrics["correlation"] is not None
        and metrics["correlation"] >= args.min_corr
        and metrics["snr_db"] >= args.min_snr
        and metrics["max_abs_error"] <= args.max_abs_error
    )

    report = {
        "tensor_dump": str(args.tensor_dump),
        "fused_package": str(args.fused_package),
        "cos_package": str(cos_package),
        "report": str(report_path),
        "toolchain": _toolchain_report(),
        "manifest_metadata": manifest.get("metadata", {}),
        "cos_snake": bool(args.cos_snake),
        "broadcast_adain": bool(args.broadcast_adain),
        "native_instance_norm_adain": bool(args.native_instance_norm_adain),
        "deployment_target": args.deployment_target,
        "palettize": bool(args.palettize),
        "linear_quantize": args.linear_quantize,
        "input_shape_mode": args.input_shape_mode,
        "export": export_report,
        "benchmark": benchmark,
        "thresholds": {
            "min_corr": args.min_corr,
            "min_snr": args.min_snr,
            "max_abs_error": args.max_abs_error,
        },
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
        default=Path("outputs/generator_cos_snake"),
        help="Directory for generated packages and reports.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--no-cos-snake", dest="cos_snake", action="store_false")
    parser.set_defaults(cos_snake=True)
    parser.add_argument("--broadcast-adain", action="store_true")
    parser.add_argument(
        "--native-instance-norm-adain",
        action="store_true",
        help="Patch AdaIN1d normalization to F.instance_norm before export.",
    )
    parser.add_argument(
        "--deployment-target",
        default="macos13",
        choices=("macos13", "macos14", "macos15", "ios17", "ios18"),
        help="Minimum deployment target for the candidate package.",
    )
    parser.add_argument(
        "--palettize",
        action="store_true",
        help="Apply 8-bit k-means weight palettization to the candidate package.",
    )
    parser.add_argument(
        "--linear-quantize",
        choices=("int8", "uint8", "int4", "uint4"),
        default=None,
        help="Apply Core ML linear weight quantization to the candidate package.",
    )
    parser.add_argument(
        "--input-shape-mode",
        choices=("fixed", "range"),
        default="fixed",
        help="Input shape contract for x_pre/har. 'range' uses bounded RangeDim time axes.",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "float16", "fp32", "float32"),
        help="Core ML conversion precision for the cos-Snake package.",
    )
    parser.add_argument("--compute-units", default="cpuAndGPU")
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
    metrics = report["benchmark"]["metrics"]["cos_vs_fused_trimmed"]
    print(
        "generator_cos_snake "
        f"passes={report['passes']} "
        f"label={Path(report['cos_package']).parent.name} "
        f"fused_median_ms={med['fused']:.3f} "
        f"cos_median_ms={med['cos']:.3f} "
        f"speedup_vs_fused_pct={report['benchmark']['speedup_vs_fused_pct']:.2f} "
        f"corr={metrics['correlation']} "
        f"snr_db={metrics['snr_db']:.2f} "
        f"max_abs={metrics['max_abs_error']:.6g} "
        f"report={report['report']}"
    )
    if args.fail_on_difference and not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
