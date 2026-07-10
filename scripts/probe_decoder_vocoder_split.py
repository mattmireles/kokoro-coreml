#!/usr/bin/env python3
"""Probe a laishere-style decoder-plus-generator split on Swift dumps.

This is an experiment, not a production exporter. It compares two same-input
chains that start from the tensors already emitted by the Swift pipeline:

- baseline: checked-in ``decoder_pre`` package, then checked-in fused
  ``GeneratorFromHar`` package.
- candidate: HAR-noise package, laishere-style dual-output
  decoder-plus-generator body, then fp32 tail.

The goal is to test the boundary difference surfaced by source-auditing
``laishere/kokoro-coreml``: its ``KokoroVocoder`` includes F0/N conv,
``decoder.encode``/``decoder.decode``, and the generator body, then emits a
discarded anchor plus pre-tail activations for a separate tail package.
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
from probe_generator_dual_anchor_split import (  # noqa: E402
    _make_noise_module,
    _make_tail_module,
    _maybe_palettize,
    _patch_cos_snake,
)
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _duration_label_from_dump, _precision_arg, _remove_existing_package  # noqa: E402


def _package_version(package: str) -> str | None:
    """Return the installed package version without importing heavy modules."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _toolchain_report() -> dict[str, str | None]:
    """Return package versions used to export and run this probe."""

    return {
        "coremltools": _package_version("coremltools"),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
    }


def _np_dtype(name: str) -> type[np.floating[Any]]:
    """Return the NumPy floating dtype requested by a probe CLI option."""

    if name in ("fp16", "float16"):
        return np.float16
    if name in ("fp32", "float32"):
        return np.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _patch_resblock_rsqrt() -> None:
    """Patch AdainResBlk1d to match laishere's explicit residual scale."""

    from export_synth import wrappers

    inv_sqrt2 = 2.0 ** -0.5
    AdainResBlk1d = wrappers.kokoro_modules.AdainResBlk1d

    def _patched_forward(self: Any, x: Any, s: Any) -> Any:
        return (self._residual(x, s) + self._shortcut(x)) * inv_sqrt2

    AdainResBlk1d.forward = _patched_forward


def _make_decoder_vocoder_module(decoder: Any, source_count: int, anchor_mode: str):
    """Return laishere-style decoder-plus-generator body module."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DecoderVocoderDualOutput(nn.Module):
        """Decoder pre-body plus generator body, with discarded anchor."""

        def __init__(self, dec: Any):
            super().__init__()
            self.encode = dec.encode
            self.decode = dec.decode
            self.F0_conv = dec.F0_conv
            self.N_conv = dec.N_conv
            self.asr_res = dec.asr_res
            gen = dec.generator
            self.num_kernels = gen.num_kernels
            self.num_upsamples = gen.num_upsamples
            self.ups = gen.ups
            self.resblocks = gen.resblocks
            self.source_count = source_count
            self.anchor_mode = anchor_mode

        def forward(
            self,
            asr: torch.Tensor,
            f0_curve: torch.Tensor,
            n_pred: torch.Tensor,
            style_timbre: torch.Tensor,
            *noise_sources: torch.Tensor,
        ):
            if len(noise_sources) != self.source_count:
                raise ValueError(f"expected {self.source_count} noise sources, got {len(noise_sources)}")
            f0 = self.F0_conv(f0_curve.unsqueeze(1))
            n_feat = self.N_conv(n_pred.unsqueeze(1))
            x = torch.cat([asr, f0, n_feat], dim=1)
            x = self.encode(x, style_timbre)
            asr_res = self.asr_res(asr)
            res = True
            for block in self.decode:
                if res:
                    x = torch.cat([x, asr_res, f0, n_feat], dim=1)
                x = block(x, style_timbre)
                if block.upsample_type != "none":
                    res = False

            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, negative_slope=0.1)
                x = self.ups[i](x)
                if i == self.num_upsamples - 1:
                    x = torch.cat([x[:, :, 1:2], x], dim=2)
                x_source = noise_sources[i]
                tx = x.size(2)
                ts = x_source.size(2)
                if ts < tx:
                    x_source = F.pad(x_source, (0, tx - ts))
                elif ts > tx:
                    x_source = x_source[:, :, :tx]
                x = x + x_source
                xs = None
                for j in range(self.num_kernels):
                    y = self.resblocks[i * self.num_kernels + j](x, style_timbre)
                    xs = y if xs is None else xs + y
                x = xs / self.num_kernels

            pre_tail = F.leaky_relu(x)
            if self.anchor_mode == "mean":
                anchor = pre_tail.mean(dim=(1, 2), keepdim=False).reshape(1, 1)
            else:
                anchor = pre_tail[:, :, :1].mean(dim=(1, 2), keepdim=False).reshape(1, 1)
            return anchor, pre_tail

    return _DecoderVocoderDualOutput(decoder).eval()


def _inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return same-boundary inputs from a Swift generator tensor dump."""

    return {
        "asr": tensors["asr_padded"].astype(np.float32),
        "f0": tensors["f0_padded"].astype(np.float32),
        "n_input": tensors["n_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "style_timbre": tensors["ref_s"][:, :128].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _export_packages(
    noise_package: Path,
    body_package: Path,
    tail_package: Path,
    tensors: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Export temporary noise/body/tail packages for the candidate chain."""

    import coremltools as ct
    import torch

    from export_synth.wrappers import remove_dropout

    if args.cos_snake:
        _patch_cos_snake()
    if args.patch_resblock_scale:
        _patch_resblock_rsqrt()

    kmodel = _load_kmodel()
    decoder = kmodel.decoder
    gen = decoder.generator

    ref_s_shape = tuple(int(v) for v in tensors["ref_s"].shape)
    style_shape = (ref_s_shape[0], 128)
    har_shape = tuple(int(v) for v in tensors["har_padded"].shape)
    asr_shape = tuple(int(v) for v in tensors["asr_padded"].shape)
    f0_shape = tuple(int(v) for v in tensors["f0_padded"].shape)
    n_shape = tuple(int(v) for v in tensors["n_padded"].shape)

    ref_s = torch.zeros(ref_s_shape, dtype=torch.float32)
    style = torch.zeros(style_shape, dtype=torch.float32)
    har = torch.zeros(har_shape, dtype=torch.float32)
    asr = torch.zeros(asr_shape, dtype=torch.float32)
    f0 = torch.zeros(f0_shape, dtype=torch.float32)
    n_pred = torch.zeros(n_shape, dtype=torch.float32)

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

    body = _make_decoder_vocoder_module(decoder, len(source_shapes), args.anchor_mode)
    body_removed_dropouts = remove_dropout(body)
    with torch.no_grad():
        traced_body = torch.jit.trace(
            body,
            (asr, f0, n_pred, style, *sources),
            strict=False,
            check_trace=False,
        )
        anchor, pre_tail = traced_body(asr, f0, n_pred, style, *sources)
    anchor_shape = tuple(int(v) for v in anchor.shape)
    pre_tail_shape = tuple(int(v) for v in pre_tail.shape)

    body_input_dtype = _np_dtype(args.body_input_dtype)
    body_inputs = [
        ct.TensorType(name="asr", shape=asr_shape, dtype=body_input_dtype),
        ct.TensorType(name="f0", shape=f0_shape, dtype=body_input_dtype),
        ct.TensorType(name="n_input", shape=n_shape, dtype=body_input_dtype),
        ct.TensorType(name="style_timbre", shape=style_shape, dtype=body_input_dtype),
    ]
    for idx, shape in enumerate(source_shapes):
        body_inputs.append(ct.TensorType(name=f"x_source_{idx}", shape=shape, dtype=body_input_dtype))
    body_model = ct.convert(
        traced_body,
        inputs=body_inputs,
        outputs=[ct.TensorType(name="anchor"), ct.TensorType(name="pre_tail")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=_precision_arg(ct, args.body_precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    body_model = _maybe_palettize(body_model, args.palettize_body)
    _remove_existing_package(body_package)
    body_model.save(str(body_package))

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
        "toolchain": _toolchain_report(),
        "noise_package": str(noise_package),
        "body_package": str(body_package),
        "tail_package": str(tail_package),
        "cos_snake": bool(args.cos_snake),
        "patch_resblock_scale": bool(args.patch_resblock_scale),
        "palettize_noise": bool(args.palettize_noise),
        "palettize_body": bool(args.palettize_body),
        "noise_precision": args.noise_precision,
        "body_precision": args.body_precision,
        "body_input_dtype": args.body_input_dtype,
        "tail_precision": args.tail_precision,
        "asr_shape": list(asr_shape),
        "f0_shape": list(f0_shape),
        "n_shape": list(n_shape),
        "ref_s_shape": list(ref_s_shape),
        "har_shape": list(har_shape),
        "source_shapes": [list(shape) for shape in source_shapes],
        "anchor_shape": list(anchor_shape),
        "pre_tail_shape": list(pre_tail_shape),
        "tail_samples": tail_samples,
        "noise_removed_dropouts": noise_removed_dropouts,
        "body_removed_dropouts": body_removed_dropouts,
        "tail_removed_dropouts": tail_removed_dropouts,
    }


def _predict(model: Any, feed: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    start = time.perf_counter()
    out = model.predict(feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {key: np.asarray(value) for key, value in out.items()}, elapsed_ms


def _load_models(args: argparse.Namespace, noise_package: Path, body_package: Path, tail_package: Path):
    import coremltools as ct

    decoder_pre = ct.models.MLModel(
        str(args.decoder_pre_package),
        compute_units=_compute_units(ct, args.decoder_pre_compute_units),
    )
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
    tail = ct.models.MLModel(
        str(tail_package),
        compute_units=_compute_units(ct, args.tail_compute_units),
    )
    return decoder_pre, fused, noise, body, tail


def _baseline_predict(decoder_pre: Any, fused: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    dec_feed = {
        "asr": inputs["asr"],
        "f0": inputs["f0"][:, None, :],
        "n_input": inputs["n_input"][:, None, :],
        "ref_s": inputs["ref_s"],
    }
    dec_out, dec_ms = _predict(decoder_pre, dec_feed)
    x_pre = dec_out["x_pre"].astype(np.float32)
    gen_feed = {
        "x_pre": x_pre,
        "ref_s": inputs["ref_s"],
        "har": inputs["har"],
    }
    gen_out, gen_ms = _predict(fused, gen_feed)
    waveform = gen_out.get("waveform", next(iter(gen_out.values()))).astype(np.float32)
    return waveform, {"decoder_pre_ms": dec_ms, "generator_ms": gen_ms, "total_ms": dec_ms + gen_ms}


def _candidate_predict(
    noise: Any,
    body: Any,
    tail: Any,
    inputs: dict[str, np.ndarray],
    body_input_dtype: type[np.floating[Any]],
) -> tuple[np.ndarray, dict[str, float]]:
    noise_out, noise_ms = _predict(noise, {"ref_s": inputs["ref_s"], "har": inputs["har"]})
    body_feed = {
        "asr": inputs["asr"].astype(body_input_dtype),
        "f0": inputs["f0"].astype(body_input_dtype),
        "n_input": inputs["n_input"].astype(body_input_dtype),
        "style_timbre": inputs["style_timbre"].astype(body_input_dtype),
    }
    for idx in range(len(noise_out)):
        key = f"x_source_{idx}"
        body_feed[key] = noise_out[key].astype(body_input_dtype)
    body_out, body_ms = _predict(body, body_feed)
    pre_tail = body_out["pre_tail"].astype(np.float32)
    tail_out, tail_ms = _predict(tail, {"pre_tail": pre_tail})
    waveform = tail_out.get("waveform", next(iter(tail_out.values()))).astype(np.float32)
    return waveform, {"noise_ms": noise_ms, "body_ms": body_ms, "tail_ms": tail_ms, "total_ms": noise_ms + body_ms + tail_ms}


def _benchmark(
    args: argparse.Namespace,
    tensors: dict[str, np.ndarray],
    noise_package: Path,
    body_package: Path,
    tail_package: Path,
) -> dict[str, Any]:
    inputs = _inputs(tensors)
    decoder_pre, fused, noise, body, tail = _load_models(args, noise_package, body_package, tail_package)
    body_input_dtype = _np_dtype(args.body_input_dtype)

    baseline_first, baseline_first_times = _baseline_predict(decoder_pre, fused, inputs)
    candidate_first, candidate_first_times = _candidate_predict(noise, body, tail, inputs, body_input_dtype)

    for _ in range(max(0, args.warmup)):
        _baseline_predict(decoder_pre, fused, inputs)
        _candidate_predict(noise, body, tail, inputs, body_input_dtype)

    baseline_decoder_times: list[float] = []
    baseline_generator_times: list[float] = []
    baseline_total_times: list[float] = []
    candidate_noise_times: list[float] = []
    candidate_body_times: list[float] = []
    candidate_tail_times: list[float] = []
    candidate_total_times: list[float] = []
    last_baseline = baseline_first
    last_candidate = candidate_first
    for _ in range(max(1, args.iterations)):
        last_baseline, baseline_times = _baseline_predict(decoder_pre, fused, inputs)
        last_candidate, candidate_times = _candidate_predict(noise, body, tail, inputs, body_input_dtype)
        baseline_decoder_times.append(baseline_times["decoder_pre_ms"])
        baseline_generator_times.append(baseline_times["generator_ms"])
        baseline_total_times.append(baseline_times["total_ms"])
        candidate_noise_times.append(candidate_times["noise_ms"])
        candidate_body_times.append(candidate_times["body_ms"])
        candidate_tail_times.append(candidate_times["tail_ms"])
        candidate_total_times.append(candidate_times["total_ms"])

    trim_len = int(tensors["waveform"].size)
    baseline_trim = last_baseline.reshape(-1)[:trim_len]
    candidate_trim = last_candidate.reshape(-1)[:trim_len]

    return {
        "toolchain": _toolchain_report(),
        "decoder_pre_compute_units": args.decoder_pre_compute_units,
        "fused_compute_units": args.fused_compute_units,
        "noise_compute_units": args.noise_compute_units,
        "body_compute_units": args.body_compute_units,
        "tail_compute_units": args.tail_compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "baseline_decoder_pre": float(baseline_first_times["decoder_pre_ms"]),
            "baseline_generator": float(baseline_first_times["generator_ms"]),
            "baseline_total": float(baseline_first_times["total_ms"]),
            "candidate_noise": float(candidate_first_times["noise_ms"]),
            "candidate_body": float(candidate_first_times["body_ms"]),
            "candidate_tail": float(candidate_first_times["tail_ms"]),
            "candidate_total": float(candidate_first_times["total_ms"]),
        },
        "warm_predict_times_ms": {
            "baseline_decoder_pre": baseline_decoder_times,
            "baseline_generator": baseline_generator_times,
            "baseline_total": baseline_total_times,
            "candidate_noise": candidate_noise_times,
            "candidate_body": candidate_body_times,
            "candidate_tail": candidate_tail_times,
            "candidate_total": candidate_total_times,
        },
        "warm_predict_median_ms": {
            "baseline_decoder_pre": float(statistics.median(baseline_decoder_times)),
            "baseline_generator": float(statistics.median(baseline_generator_times)),
            "baseline_total": float(statistics.median(baseline_total_times)),
            "candidate_noise": float(statistics.median(candidate_noise_times)),
            "candidate_body": float(statistics.median(candidate_body_times)),
            "candidate_tail": float(statistics.median(candidate_tail_times)),
            "candidate_total": float(statistics.median(candidate_total_times)),
        },
        "metrics": {
            "baseline_vs_dump_full": _metrics(tensors["waveform_full"], last_baseline),
            "candidate_vs_dump_full": _metrics(tensors["waveform_full"], last_candidate),
            "candidate_vs_baseline_full": _metrics(last_baseline, last_candidate),
            "baseline_vs_dump_trimmed": _metrics(tensors["waveform"], baseline_trim),
            "candidate_vs_dump_trimmed": _metrics(tensors["waveform"], candidate_trim),
            "candidate_vs_baseline_trimmed": _metrics(baseline_trim, candidate_trim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = [
        "asr_padded",
        "f0_padded",
        "n_padded",
        "ref_s",
        "har_padded",
        "waveform_full",
        "waveform",
    ]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    if args.cos_snake:
        label = f"{label}_cos"
    if args.patch_resblock_scale:
        label = f"{label}_rsqrt"
    if args.palettize_noise:
        label = f"{label}_noise_pal"
    if args.palettize_body:
        label = f"{label}_body_pal"

    work_dir = args.output_dir / label
    noise_package = work_dir / f"kokoro_decoder_vocoder_noise_{label}.mlpackage"
    body_package = work_dir / f"kokoro_decoder_vocoder_body_{label}.mlpackage"
    tail_package = work_dir / f"kokoro_decoder_vocoder_tail_{label}.mlpackage"
    report_name = Path(args.report_name)
    if report_name.name != str(report_name):
        raise SystemExit(f"--report-name must be a filename, got {args.report_name!r}")
    report_path = work_dir / report_name

    export_report: dict[str, Any] | None = None
    if args.skip_export:
        missing_packages = [
            str(path)
            for path in (noise_package, body_package, tail_package)
            if not path.is_dir()
        ]
        if missing_packages:
            raise SystemExit(f"--skip-export requested but packages are missing: {missing_packages}")
    else:
        export_report = _export_packages(noise_package, body_package, tail_package, tensors, args)

    benchmark = _benchmark(args, tensors, noise_package, body_package, tail_package)
    metrics = benchmark["metrics"]["candidate_vs_baseline_trimmed"]
    passes = bool(
        metrics["correlation"] is not None
        and metrics["correlation"] >= args.min_corr
        and metrics["snr_db"] >= args.min_snr
        and metrics["max_abs_error"] <= args.max_abs_error
    )
    med = benchmark["warm_predict_median_ms"]
    speedup_vs_baseline_pct = None
    if med["baseline_total"] > 0:
        speedup_vs_baseline_pct = 100.0 * (med["baseline_total"] - med["candidate_total"]) / med["baseline_total"]

    report = {
        "tensor_dump": str(args.tensor_dump),
        "decoder_pre_package": str(args.decoder_pre_package),
        "fused_package": str(args.fused_package),
        "noise_package": str(noise_package),
        "body_package": str(body_package),
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
        "speedup_vs_baseline_pct": speedup_vs_baseline_pct,
        "passes": passes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dump", type=Path, default=Path("outputs/generator_isolation/dumps/3s"))
    parser.add_argument("--decoder-pre-package", type=Path, default=Path("coreml/kokoro_decoder_pre_3s.mlpackage"))
    parser.add_argument("--fused-package", type=Path, default=Path("coreml/kokoro_decoder_har_post_3s.mlpackage"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/decoder_vocoder_split"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--anchor-mode", default="mean", choices=("mean", "slice_mean"))
    parser.add_argument("--cos-snake", action="store_true")
    parser.add_argument("--patch-resblock-scale", action="store_true")
    parser.add_argument("--palettize-noise", action="store_true")
    parser.add_argument("--palettize-body", action="store_true")
    parser.add_argument("--noise-precision", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--body-precision", default="fp16", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--body-input-dtype", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--tail-precision", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--decoder-pre-compute-units", default="cpuAndNeuralEngine")
    parser.add_argument("--fused-compute-units", default="cpuAndGPU")
    parser.add_argument("--noise-compute-units", default="all")
    parser.add_argument("--body-compute-units", default="cpuAndNeuralEngine")
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
    metrics = report["benchmark"]["metrics"]["candidate_vs_baseline_trimmed"]
    print(
        "decoder_vocoder_split "
        f"passes={report['passes']} "
        f"label={Path(report['noise_package']).parent.name} "
        f"baseline_median_ms={med['baseline_total']:.3f} "
        f"candidate_median_ms={med['candidate_total']:.3f} "
        f"baseline_decoder_pre_ms={med['baseline_decoder_pre']:.3f} "
        f"baseline_generator_ms={med['baseline_generator']:.3f} "
        f"candidate_noise_ms={med['candidate_noise']:.3f} "
        f"candidate_body_ms={med['candidate_body']:.3f} "
        f"candidate_tail_ms={med['candidate_tail']:.3f} "
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
