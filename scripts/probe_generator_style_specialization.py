#!/usr/bin/env python3
"""Probe voice/style-specialized HAR-post generator packages.

The production fused generator accepts ``ref_s`` and computes every AdaIN
``gamma``/``beta`` projection at inference time. The external bakeoff uses one
fixed voice (`af_heart`), so this script tests an aggressive but simple
hypothesis: bake the style vector from a Swift tensor dump into the generator,
replace each ``AdaIN1d`` with precomputed gamma/beta constants, and benchmark
the resulting package against the general shipping fused package.

This is not a production exporter. It writes temporary packages under
``outputs/`` and is intended to decide whether voice-specialized packages are a
real speed path worth product/design discussion.
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


def _make_frozen_adain(original: Any, style: Any):
    import torch
    import torch.nn as nn

    class _FrozenAdaIN1d(nn.Module):
        """AdaIN1d with style projection precomputed for one fixed voice."""

        def __init__(self, source: Any, style_tensor: torch.Tensor):
            super().__init__()
            self.num_features = int(source.num_features)
            self.eps = float(source.eps)
            with torch.no_grad():
                h = source.fc(style_tensor).view(1, 2 * self.num_features, 1)
                gamma, beta = torch.chunk(h, chunks=2, dim=1)
            self.register_buffer("gamma", gamma.detach().clone())
            self.register_buffer("beta", beta.detach().clone())

        def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            B, C, T = x.shape
            mean = x.mean(dim=2, keepdim=True)
            var = x.var(dim=2, unbiased=False, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            assert C == self.num_features, f"AdaIN1d channel mismatch: got {C}, expected {self.num_features}"
            gamma = self.gamma.expand(B, C, T)
            beta = self.beta.expand(B, C, T)
            return (1.0 + gamma) * x_norm + beta

    return _FrozenAdaIN1d(original, style)


def _freeze_adain_modules(module: Any, style: Any) -> int:
    """Replace all AdaIN1d children with fixed-style equivalents."""

    count = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ == "AdaIN1d":
            setattr(module, name, _make_frozen_adain(child, style))
            count += 1
        else:
            count += _freeze_adain_modules(child, style)
    return count


def _make_style_specialized_generator(generator: Any, style: Any):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _StyleSpecializedGeneratorFromHar(nn.Module):
        """Fused HAR-post generator with style vector baked into AdaIN layers."""

        def __init__(self, gen: Any, style_tensor: torch.Tensor):
            super().__init__()
            self.generator = gen
            self.register_buffer("style", style_tensor.detach().clone())

        def forward(self, x_pre: torch.Tensor, har: torch.Tensor) -> torch.Tensor:
            s = self.style
            gen = self.generator
            x = x_pre
            for i in range(gen.num_upsamples):
                x = F.leaky_relu(x, negative_slope=0.1)
                x_source = gen.noise_convs[i](har)
                x_source = gen.noise_res[i](x_source, s)
                x = gen.ups[i](x)
                if i == gen.num_upsamples - 1:
                    x = gen.reflection_pad(x)
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
            logits = gen.conv_post(x)
            spec = torch.exp(logits[:, : gen.post_n_fft // 2 + 1, :])
            phase = torch.sin(logits[:, gen.post_n_fft // 2 + 1 :, :])
            return gen.stft.inverse(spec, phase)

    return _StyleSpecializedGeneratorFromHar(generator, style).eval()


def _predict_fused_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _predict_style_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _export_package(
    package: Path,
    tensors: dict[str, np.ndarray],
    precision: str,
) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import remove_dropout

    ref_s = torch.from_numpy(tensors["ref_s"].astype(np.float32))
    style = ref_s[:, :128].contiguous()

    kmodel = _load_kmodel()
    gen = kmodel.decoder.generator
    frozen_adain_count = _freeze_adain_modules(gen, style)

    x_pre_shape = tuple(int(v) for v in tensors["x_pre_padded"].shape)
    har_shape = tuple(int(v) for v in tensors["har_padded"].shape)
    x_pre = torch.zeros(x_pre_shape, dtype=torch.float32)
    har = torch.zeros(har_shape, dtype=torch.float32)

    model = _make_style_specialized_generator(gen, style)
    removed_dropouts = remove_dropout(model)
    with torch.no_grad():
        traced = torch.jit.trace(model, (x_pre, har), strict=False, check_trace=False)
        traced_out = traced(x_pre, har)
    traced_samples = int(traced_out.shape[-1])

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
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
    mlmodel.save(str(package))

    return {
        "package": str(package),
        "precision": precision,
        "frozen_adain_count": frozen_adain_count,
        "removed_dropouts": removed_dropouts,
        "traced_samples": traced_samples,
        "x_pre_shape": list(x_pre_shape),
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
    style_package: Path,
) -> dict[str, Any]:
    import coremltools as ct

    fused = ct.models.MLModel(
        str(args.fused_package),
        compute_units=_compute_units(ct, args.compute_units),
    )
    style_model = ct.models.MLModel(
        str(style_package),
        compute_units=_compute_units(ct, args.compute_units),
    )
    fused_inputs = _predict_fused_inputs(tensors)
    style_inputs = _predict_style_inputs(tensors)

    fused_first, fused_first_ms = _predict(fused, fused_inputs)
    style_first, style_first_ms = _predict(style_model, style_inputs)

    for _ in range(max(0, args.warmup)):
        _predict(fused, fused_inputs)
        _predict(style_model, style_inputs)

    fused_times: list[float] = []
    style_times: list[float] = []
    last_fused = fused_first
    last_style = style_first
    for _ in range(max(1, args.iterations)):
        last_fused, fused_ms = _predict(fused, fused_inputs)
        last_style, style_ms = _predict(style_model, style_inputs)
        fused_times.append(fused_ms)
        style_times.append(style_ms)

    trim_len = int(tensors["waveform"].size)
    fused_trim = last_fused.reshape(-1)[:trim_len]
    style_trim = last_style.reshape(-1)[:trim_len]

    fused_median = float(statistics.median(fused_times))
    style_median = float(statistics.median(style_times))
    speedup_vs_fused_pct = None
    if fused_median > 0:
        speedup_vs_fused_pct = 100.0 * (fused_median - style_median) / fused_median

    return {
        "compute_units": args.compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "fused": float(fused_first_ms),
            "style_specialized": float(style_first_ms),
        },
        "warm_predict_times_ms": {
            "fused": fused_times,
            "style_specialized": style_times,
        },
        "warm_predict_median_ms": {
            "fused": fused_median,
            "style_specialized": style_median,
        },
        "speedup_vs_fused_pct": speedup_vs_fused_pct,
        "metrics": {
            "fused_vs_dump_full": _metrics(tensors["waveform_full"], last_fused),
            "style_vs_dump_full": _metrics(tensors["waveform_full"], last_style),
            "style_vs_fused_full": _metrics(last_fused, last_style),
            "fused_vs_dump_trimmed": _metrics(tensors["waveform"], fused_trim),
            "style_vs_dump_trimmed": _metrics(tensors["waveform"], style_trim),
            "style_vs_fused_trimmed": _metrics(fused_trim, style_trim),
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
    style_package = work_dir / f"kokoro_generator_style_specialized_{label}.mlpackage"
    report_name = Path(args.report_name)
    if report_name.name != str(report_name):
        raise SystemExit(f"--report-name must be a filename, got {args.report_name!r}")
    report_path = work_dir / report_name

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        if not style_package.is_dir():
            raise SystemExit(f"--skip-export requested but package is missing: {style_package}")
    else:
        export_report = _export_package(style_package, tensors, args.precision)

    benchmark = _benchmark(args, tensors, style_package)
    metrics = benchmark["metrics"]["style_vs_fused_trimmed"]
    passes = bool(
        metrics["correlation"] is not None
        and metrics["correlation"] >= args.min_corr
        and metrics["snr_db"] >= args.min_snr
        and metrics["max_abs_error"] <= args.max_abs_error
    )

    report = {
        "tensor_dump": str(args.tensor_dump),
        "fused_package": str(args.fused_package),
        "style_package": str(style_package),
        "report": str(report_path),
        "manifest_metadata": manifest.get("metadata", {}),
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
        default=Path("outputs/generator_style_specialization"),
        help="Directory for generated packages and reports.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "float16", "fp32", "float32"),
        help="Core ML conversion precision for the style-specialized package.",
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
    metrics = report["benchmark"]["metrics"]["style_vs_fused_trimmed"]
    print(
        "generator_style_specialization "
        f"passes={report['passes']} "
        f"label={Path(report['style_package']).parent.name} "
        f"fused_median_ms={med['fused']:.3f} "
        f"style_median_ms={med['style_specialized']:.3f} "
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
