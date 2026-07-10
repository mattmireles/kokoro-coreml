#!/usr/bin/env python3
"""Probe the laishere dual-output vocoder scheduling trick on our generator.

This is a temporary graph experiment, not a production exporter. It starts from
the same Swift tensor dump used by the generator isolation benchmark and exports
three packages:

- ``noise``: ``ref_s`` + static HAR features -> ``x_source_*``
- ``vocoder``: ``x_pre`` + ``x_source_*`` -> discarded ``anchor`` + ``pre_tail``
- ``tail``: ``pre_tail`` -> waveform via ``conv_post`` + ``exp``/``sin``/iSTFT

The shape is intentionally close to the public laishere implementation: the
vocoder has two outputs so Core ML must materialize a small anchor while also
returning the pre-tail activations for a separate fp32 tail package. Optional
flags let us test the laishere cos-Snake rewrite and int8 palettization without
changing ``kokoro/istftnet.py`` or the checked-in ``coreml/`` packages.
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


def _patch_cos_snake() -> None:
    """Patch AdaINResBlock1 to use laishere's algebraic cos-form Snake."""

    import torch

    from export_synth import wrappers

    AdaINResBlock1 = wrappers.kokoro_istftnet.AdaINResBlock1

    def _cos_forward(self: Any, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
        ):
            xt = n1(x, s)
            cv = torch.cos(xt * (a1 * 2.0))
            xt = xt + (cv * -0.5 + 0.5) * (1.0 / a1)
            xt = c1(xt)
            xt = n2(xt, s)
            cv = torch.cos(xt * (a2 * 2.0))
            xt = xt + (cv * -0.5 + 0.5) * (1.0 / a2)
            xt = c2(xt)
            x = xt + x
        return x

    AdaINResBlock1.forward = _cos_forward


def _maybe_palettize(mlmodel: Any, enabled: bool) -> Any:
    if not enabled:
        return mlmodel

    import coremltools.optimize.coreml as cto

    pal_config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=8)
    )
    return cto.palettize_weights(mlmodel, pal_config)


def _make_noise_module(generator: Any):
    import torch
    import torch.nn as nn

    class _GeneratorNoiseFromHar(nn.Module):
        """Static HAR noise branch: ``ref_s`` + ``har`` -> noise sources."""

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


def _make_vocoder_module(generator: Any, source_count: int, anchor_mode: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _GeneratorDualAnchorVocoder(nn.Module):
        """Generator body with a discarded anchor plus pre-tail output."""

        def __init__(self, gen: Any):
            super().__init__()
            self.generator = gen
            self.source_count = source_count
            self.anchor_mode = anchor_mode

        def forward(self, x_pre: torch.Tensor, ref_s: torch.Tensor, *noise_sources: torch.Tensor):
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

            pre_tail = F.leaky_relu(x)
            if self.anchor_mode == "audio":
                logits = gen.conv_post(pre_tail)
                spec = torch.exp(logits[:, : gen.post_n_fft // 2 + 1, :])
                phase = torch.sin(logits[:, gen.post_n_fft // 2 + 1 :, :])
                anchor = gen.stft.inverse(spec, phase)
            else:
                anchor = pre_tail.mean().unsqueeze(0)
            return anchor, pre_tail

    return _GeneratorDualAnchorVocoder(generator).eval()


def _make_tail_module(generator: Any):
    import torch
    import torch.nn as nn

    class _GeneratorTailFromPreTail(nn.Module):
        """fp32 tail package input: ``pre_tail`` from the dual-output vocoder."""

        def __init__(self, gen: Any):
            super().__init__()
            self.post_n_fft = int(gen.post_n_fft)
            self.conv_post = gen.conv_post
            self.stft = gen.stft

        def forward(self, pre_tail: torch.Tensor) -> torch.Tensor:
            logits = self.conv_post(pre_tail)
            spec = torch.exp(logits[:, : self.post_n_fft // 2 + 1, :])
            phase = torch.sin(logits[:, self.post_n_fft // 2 + 1 :, :])
            return self.stft.inverse(spec, phase)

    return _GeneratorTailFromPreTail(generator).eval()


def _predict_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x_pre": tensors["x_pre_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _export_packages(
    noise_package: Path,
    vocoder_package: Path,
    tail_package: Path,
    tensors: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import remove_dropout

    if args.cos_snake:
        _patch_cos_snake()

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

    noise_model = ct.convert(
        traced_noise,
        inputs=[
            ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
            ct.TensorType(name="har", shape=har_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name=f"x_source_{idx}") for idx in range(len(source_shapes))],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, args.noise_precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    noise_model = _maybe_palettize(noise_model, args.palettize_noise)
    noise_package.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_package(noise_package)
    noise_model.save(str(noise_package))

    vocoder = _make_vocoder_module(gen, len(source_shapes), args.anchor_mode)
    vocoder_removed_dropouts = remove_dropout(vocoder)
    with torch.no_grad():
        traced_vocoder = torch.jit.trace(
            vocoder,
            (x_pre, ref_s, *sources),
            strict=False,
            check_trace=False,
        )
        anchor, pre_tail = traced_vocoder(x_pre, ref_s, *sources)
    anchor_shape = tuple(int(v) for v in anchor.shape)
    pre_tail_shape = tuple(int(v) for v in pre_tail.shape)

    vocoder_inputs = [
        ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
        ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
    ]
    for idx, shape in enumerate(source_shapes):
        vocoder_inputs.append(ct.TensorType(name=f"x_source_{idx}", shape=shape, dtype=np.float32))
    vocoder_model = ct.convert(
        traced_vocoder,
        inputs=vocoder_inputs,
        outputs=[ct.TensorType(name="anchor"), ct.TensorType(name="pre_tail")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, args.vocoder_precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    vocoder_model = _maybe_palettize(vocoder_model, args.palettize_vocoder)
    _remove_existing_package(vocoder_package)
    vocoder_model.save(str(vocoder_package))

    tail = _make_tail_module(gen)
    tail_removed_dropouts = remove_dropout(tail)
    tail_input = torch.zeros(pre_tail_shape, dtype=torch.float32)
    with torch.no_grad():
        traced_tail = torch.jit.trace(tail, (tail_input,), strict=False, check_trace=False)
        tail_out = traced_tail(tail_input)
    tail_samples = int(tail_out.shape[-1])

    tail_model = ct.convert(
        traced_tail,
        inputs=[ct.TensorType(name="pre_tail", shape=pre_tail_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, args.tail_precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    _remove_existing_package(tail_package)
    tail_model.save(str(tail_package))

    return {
        "noise_package": str(noise_package),
        "vocoder_package": str(vocoder_package),
        "tail_package": str(tail_package),
        "anchor_mode": args.anchor_mode,
        "cos_snake": bool(args.cos_snake),
        "palettize_noise": bool(args.palettize_noise),
        "palettize_vocoder": bool(args.palettize_vocoder),
        "noise_precision": args.noise_precision,
        "vocoder_precision": args.vocoder_precision,
        "tail_precision": args.tail_precision,
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
        "x_pre_shape": list(x_pre_shape),
        "source_shapes": [list(shape) for shape in source_shapes],
        "anchor_shape": list(anchor_shape),
        "pre_tail_shape": list(pre_tail_shape),
        "tail_samples": tail_samples,
        "noise_removed_dropouts": noise_removed_dropouts,
        "vocoder_removed_dropouts": vocoder_removed_dropouts,
        "tail_removed_dropouts": tail_removed_dropouts,
    }


def _load_models(
    args: argparse.Namespace,
    noise_package: Path,
    vocoder_package: Path,
    tail_package: Path,
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
    vocoder = ct.models.MLModel(
        str(vocoder_package),
        compute_units=_compute_units(ct, args.vocoder_compute_units),
    )
    tail = ct.models.MLModel(
        str(tail_package),
        compute_units=_compute_units(ct, args.tail_compute_units),
    )
    return fused, noise, vocoder, tail


def _predict_fused(fused: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = fused.predict(inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    key = "waveform" if "waveform" in out else next(iter(out))
    return np.asarray(out[key], dtype=np.float32), elapsed_ms


def _predict_split(
    noise: Any,
    vocoder: Any,
    tail: Any,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    noise_out = noise.predict({"ref_s": inputs["ref_s"], "har": inputs["har"]})
    noise_ms = (time.perf_counter() - start) * 1000.0

    vocoder_feed = {"x_pre": inputs["x_pre"], "ref_s": inputs["ref_s"]}
    for idx in range(len(noise_out)):
        key = f"x_source_{idx}"
        vocoder_feed[key] = np.asarray(noise_out[key], dtype=np.float32)

    start = time.perf_counter()
    vocoder_out = vocoder.predict(vocoder_feed)
    vocoder_ms = (time.perf_counter() - start) * 1000.0
    pre_tail = np.asarray(vocoder_out["pre_tail"], dtype=np.float32)

    start = time.perf_counter()
    tail_out = tail.predict({"pre_tail": pre_tail})
    tail_ms = (time.perf_counter() - start) * 1000.0
    waveform_key = "waveform" if "waveform" in tail_out else next(iter(tail_out))
    waveform = np.asarray(tail_out[waveform_key], dtype=np.float32)
    return waveform, {
        "noise_ms": noise_ms,
        "vocoder_ms": vocoder_ms,
        "tail_ms": tail_ms,
        "total_ms": noise_ms + vocoder_ms + tail_ms,
    }


def _benchmark(
    args: argparse.Namespace,
    tensors: dict[str, np.ndarray],
    noise_package: Path,
    vocoder_package: Path,
    tail_package: Path,
) -> dict[str, Any]:
    inputs = _predict_inputs(tensors)
    fused, noise, vocoder, tail = _load_models(args, noise_package, vocoder_package, tail_package)

    fused_first, fused_first_ms = _predict_fused(fused, inputs)
    split_first, split_first_times = _predict_split(noise, vocoder, tail, inputs)

    for _ in range(max(0, args.warmup)):
        _predict_fused(fused, inputs)
        _predict_split(noise, vocoder, tail, inputs)

    fused_times: list[float] = []
    split_noise_times: list[float] = []
    split_vocoder_times: list[float] = []
    split_tail_times: list[float] = []
    split_total_times: list[float] = []
    last_fused = fused_first
    last_split = split_first
    for _ in range(max(1, args.iterations)):
        last_fused, fused_ms = _predict_fused(fused, inputs)
        last_split, split_times = _predict_split(noise, vocoder, tail, inputs)
        fused_times.append(fused_ms)
        split_noise_times.append(split_times["noise_ms"])
        split_vocoder_times.append(split_times["vocoder_ms"])
        split_tail_times.append(split_times["tail_ms"])
        split_total_times.append(split_times["total_ms"])

    trim_len = int(tensors["waveform"].size)
    fused_trim = last_fused.reshape(-1)[:trim_len]
    split_trim = last_split.reshape(-1)[:trim_len]

    return {
        "fused_compute_units": args.fused_compute_units,
        "noise_compute_units": args.noise_compute_units,
        "vocoder_compute_units": args.vocoder_compute_units,
        "tail_compute_units": args.tail_compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "fused": float(fused_first_ms),
            "split_noise": float(split_first_times["noise_ms"]),
            "split_vocoder": float(split_first_times["vocoder_ms"]),
            "split_tail": float(split_first_times["tail_ms"]),
            "split_total": float(split_first_times["total_ms"]),
        },
        "warm_predict_times_ms": {
            "fused": fused_times,
            "split_noise": split_noise_times,
            "split_vocoder": split_vocoder_times,
            "split_tail": split_tail_times,
            "split_total": split_total_times,
        },
        "warm_predict_median_ms": {
            "fused": float(statistics.median(fused_times)),
            "split_noise": float(statistics.median(split_noise_times)),
            "split_vocoder": float(statistics.median(split_vocoder_times)),
            "split_tail": float(statistics.median(split_tail_times)),
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
    if args.cos_snake:
        label = f"{label}_cos"
    if args.anchor_mode != "mean":
        label = f"{label}_{args.anchor_mode}_anchor"
    if args.palettize_noise:
        label = f"{label}_noise_pal"
    if args.palettize_vocoder:
        label = f"{label}_vocoder_pal"

    work_dir = args.output_dir / label
    noise_package = work_dir / f"kokoro_generator_noise_from_har_{label}.mlpackage"
    vocoder_package = work_dir / f"kokoro_generator_dual_anchor_vocoder_{label}.mlpackage"
    tail_package = work_dir / f"kokoro_generator_pre_tail_tail_{label}.mlpackage"
    report_name = Path(args.report_name)
    if report_name.name != str(report_name):
        raise SystemExit(f"--report-name must be a filename, got {args.report_name!r}")
    report_path = work_dir / report_name

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        missing_packages = [
            str(path)
            for path in (noise_package, vocoder_package, tail_package)
            if not path.is_dir()
        ]
        if missing_packages:
            raise SystemExit(f"--skip-export requested but packages are missing: {missing_packages}")
    else:
        export_report = _export_packages(noise_package, vocoder_package, tail_package, tensors, args)

    benchmark = _benchmark(args, tensors, noise_package, vocoder_package, tail_package)
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
        "vocoder_package": str(vocoder_package),
        "tail_package": str(tail_package),
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
        default=Path("outputs/generator_dual_anchor_split"),
        help="Directory for generated split packages and reports.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--anchor-mode", default="mean", choices=("mean", "audio"))
    parser.add_argument("--cos-snake", action="store_true")
    parser.add_argument("--palettize-noise", action="store_true")
    parser.add_argument("--palettize-vocoder", action="store_true")
    parser.add_argument("--noise-precision", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--vocoder-precision", default="fp16", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--tail-precision", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--fused-compute-units", default="cpuAndGPU")
    parser.add_argument("--noise-compute-units", default="all")
    parser.add_argument("--vocoder-compute-units", default="cpuAndNeuralEngine")
    parser.add_argument("--tail-compute-units", default="all")
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
        "generator_dual_anchor_split "
        f"passes={report['passes']} "
        f"label={Path(report['noise_package']).parent.name} "
        f"fused_median_ms={med['fused']:.3f} "
        f"split_median_ms={med['split_total']:.3f} "
        f"noise_median_ms={med['split_noise']:.3f} "
        f"vocoder_median_ms={med['split_vocoder']:.3f} "
        f"tail_median_ms={med['split_tail']:.3f} "
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
