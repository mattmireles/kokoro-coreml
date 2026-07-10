#!/usr/bin/env python3
"""Probe ``har_source -> STFT/noise_convs`` as a quality-preserving split.

The F0-source probe is fast but changes the harmonic source enough to fail
strict parity. This experiment keeps the exact dumped Swift ``har_source`` and
moves only STFT + generator noise convolutions into a small Core ML package,
then reuses the decoder-vocoder body/tail split. It measures whether a compact
source boundary is worth implementing in Swift before changing production code.
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
from probe_decoder_vocoder_split import _make_decoder_vocoder_module  # noqa: E402
from probe_generator_dual_anchor_split import _make_tail_module, _maybe_palettize, _patch_cos_snake  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _duration_label_from_dump, _precision_arg, _remove_existing_package  # noqa: E402


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _toolchain_report() -> dict[str, str | None]:
    return {
        "coremltools": _package_version("coremltools"),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
    }


def _make_har_source_noise_module(generator: Any):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from kokoro.custom_stft import CustomSTFT

    class _CoreMLForwardSTFT(nn.Module):
        def __init__(self, original_stft: Any):
            super().__init__()
            self.center = original_stft.center
            self.n_fft = original_stft.n_fft
            self.pad_mode = original_stft.pad_mode
            self.freq_bins = original_stft.freq_bins
            self.conv_real = nn.Conv1d(
                1,
                self.freq_bins,
                self.n_fft,
                stride=original_stft.hop_length,
                padding=0,
                bias=False,
            )
            self.conv_imag = nn.Conv1d(
                1,
                self.freq_bins,
                self.n_fft,
                stride=original_stft.hop_length,
                padding=0,
                bias=False,
            )
            self.conv_real.weight = nn.Parameter(original_stft.weight_forward_real, requires_grad=False)
            self.conv_imag.weight = nn.Parameter(original_stft.weight_forward_imag, requires_grad=False)

        def transform(self, waveform: Any):
            if self.center:
                pad_len = self.n_fft // 2
                waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
            x = waveform.unsqueeze(1)
            real = self.conv_real(x)
            imag = self.conv_imag(x)
            magnitude = torch.sqrt(real**2 + imag**2 + 1e-14)
            phase = torch.atan2(imag, real)
            return magnitude, phase

    class _HarSourceNoiseModel(nn.Module):
        def __init__(self, gen: Any):
            super().__init__()
            fwd_stft = CustomSTFT(
                filter_length=gen.stft.filter_length,
                hop_length=gen.stft.hop_length,
                win_length=gen.stft.win_length,
            )
            self.stft = _CoreMLForwardSTFT(fwd_stft)
            self.noise_convs = gen.noise_convs
            self.noise_res = gen.noise_res

        def forward(self, har_source: Any, style_timbre: Any):
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
            outputs = []
            for conv, res in zip(self.noise_convs, self.noise_res):
                x_source = conv(har)
                x_source = res(x_source, style_timbre)
                outputs.append(x_source)
            return tuple(outputs)

    return _HarSourceNoiseModel(generator).eval()


def _select_inputs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "asr": tensors["asr_padded"].astype(np.float32),
        "f0": tensors["f0_padded"].astype(np.float32),
        "n_input": tensors["n_padded"].astype(np.float32),
        "ref_s": tensors["ref_s"].astype(np.float32),
        "style_timbre": tensors["ref_s"][:, :128].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
        "har_source": tensors["har_source"].astype(np.float32),
    }


def _export_packages(
    noise_package: Path,
    body_package: Path,
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
    decoder = kmodel.decoder
    gen = decoder.generator
    inputs = _select_inputs(tensors)

    har_source_shape = tuple(int(v) for v in inputs["har_source"].shape)
    style_shape = tuple(int(v) for v in inputs["style_timbre"].shape)
    asr_shape = tuple(int(v) for v in inputs["asr"].shape)
    f0_shape = tuple(int(v) for v in inputs["f0"].shape)
    n_shape = tuple(int(v) for v in inputs["n_input"].shape)

    har_source = torch.zeros(har_source_shape, dtype=torch.float32)
    style = torch.zeros(style_shape, dtype=torch.float32)
    noise = _make_har_source_noise_module(gen)
    noise_removed_dropouts = remove_dropout(noise)
    with torch.no_grad():
        traced_noise = torch.jit.trace(noise, (har_source, style), strict=False, check_trace=False)
        sources = tuple(traced_noise(har_source, style))
    source_shapes = [tuple(int(v) for v in source.shape) for source in sources]

    noise_model = ct.convert(
        traced_noise,
        inputs=[
            ct.TensorType(name="har_source", shape=har_source_shape, dtype=np.float32),
            ct.TensorType(name="style_timbre", shape=style_shape, dtype=np.float32),
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
    asr = torch.zeros(asr_shape, dtype=torch.float32)
    f0 = torch.zeros(f0_shape, dtype=torch.float32)
    n_pred = torch.zeros(n_shape, dtype=torch.float32)
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

    body_inputs = [
        ct.TensorType(name="asr", shape=asr_shape, dtype=np.float32),
        ct.TensorType(name="F0_curve", shape=f0_shape, dtype=np.float32),
        ct.TensorType(name="N_pred", shape=n_shape, dtype=np.float32),
        ct.TensorType(name="style_timbre", shape=style_shape, dtype=np.float32),
    ]
    for idx, shape in enumerate(source_shapes):
        body_inputs.append(ct.TensorType(name=f"x_source_{idx}", shape=shape, dtype=np.float32))
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
        "palettize_noise": bool(args.palettize_noise),
        "palettize_body": bool(args.palettize_body),
        "noise_precision": args.noise_precision,
        "body_precision": args.body_precision,
        "tail_precision": args.tail_precision,
        "har_source_shape": list(har_source_shape),
        "asr_shape": list(asr_shape),
        "f0_shape": list(f0_shape),
        "n_shape": list(n_shape),
        "style_shape": list(style_shape),
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
    dec_out, dec_ms = _predict(
        decoder_pre,
        {
            "asr": inputs["asr"],
            "f0": inputs["f0"][:, None, :],
            "n_input": inputs["n_input"][:, None, :],
            "ref_s": inputs["ref_s"],
        },
    )
    x_pre = dec_out["x_pre"].astype(np.float32)
    gen_out, gen_ms = _predict(fused, {"x_pre": x_pre, "ref_s": inputs["ref_s"], "har": inputs["har"]})
    waveform = gen_out.get("waveform", next(iter(gen_out.values()))).astype(np.float32)
    return waveform, {"decoder_pre_ms": dec_ms, "generator_ms": gen_ms, "total_ms": dec_ms + gen_ms}


def _candidate_predict(noise: Any, body: Any, tail: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    noise_out, noise_ms = _predict(
        noise,
        {"har_source": inputs["har_source"], "style_timbre": inputs["style_timbre"]},
    )
    body_feed = {
        "asr": inputs["asr"],
        "F0_curve": inputs["f0"],
        "N_pred": inputs["n_input"],
        "style_timbre": inputs["style_timbre"],
    }
    for idx in range(len(noise_out)):
        key = f"x_source_{idx}"
        body_feed[key] = noise_out[key].astype(np.float32)
    body_out, body_ms = _predict(body, body_feed)
    tail_out, tail_ms = _predict(tail, {"pre_tail": body_out["pre_tail"].astype(np.float32)})
    waveform = tail_out.get("waveform", next(iter(tail_out.values()))).astype(np.float32)
    return waveform, {"noise_ms": noise_ms, "body_ms": body_ms, "tail_ms": tail_ms, "total_ms": noise_ms + body_ms + tail_ms}


def _benchmark(args: argparse.Namespace, tensors: dict[str, np.ndarray], noise_package: Path, body_package: Path, tail_package: Path) -> dict[str, Any]:
    inputs = _select_inputs(tensors)
    decoder_pre, fused, noise, body, tail = _load_models(args, noise_package, body_package, tail_package)

    baseline_first, baseline_first_times = _baseline_predict(decoder_pre, fused, inputs)
    candidate_first, candidate_first_times = _candidate_predict(noise, body, tail, inputs)
    for _ in range(max(0, args.warmup)):
        _baseline_predict(decoder_pre, fused, inputs)
        _candidate_predict(noise, body, tail, inputs)

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
        last_candidate, candidate_times = _candidate_predict(noise, body, tail, inputs)
        baseline_decoder_times.append(baseline_times["decoder_pre_ms"])
        baseline_generator_times.append(baseline_times["generator_ms"])
        baseline_total_times.append(baseline_times["total_ms"])
        candidate_noise_times.append(candidate_times["noise_ms"])
        candidate_body_times.append(candidate_times["body_ms"])
        candidate_tail_times.append(candidate_times["tail_ms"])
        candidate_total_times.append(candidate_times["total_ms"])

    trim_len = min(int(tensors["waveform"].size), int(last_candidate.size), int(last_baseline.size))
    baseline_trim = last_baseline.reshape(-1)[:trim_len]
    candidate_trim = last_candidate.reshape(-1)[:trim_len]
    dump_trim = tensors["waveform"].reshape(-1)[:trim_len]

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
            "baseline_vs_dump_trimmed": _metrics(dump_trim, baseline_trim),
            "candidate_vs_dump_trimmed": _metrics(dump_trim, candidate_trim),
            "candidate_vs_baseline_trimmed": _metrics(baseline_trim, candidate_trim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["asr_padded", "f0_padded", "n_padded", "ref_s", "har_padded", "har_source", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    if args.cos_snake:
        label = f"{label}_cos"
    if args.palettize_noise:
        label = f"{label}_noise_pal"
    if args.palettize_body:
        label = f"{label}_body_pal"

    work_dir = args.output_dir / label
    noise_package = work_dir / f"kokoro_har_source_noise_{label}.mlpackage"
    body_package = work_dir / f"kokoro_har_source_body_{label}.mlpackage"
    tail_package = work_dir / f"kokoro_har_source_tail_{label}.mlpackage"
    report_path = work_dir / args.report_name

    export_report = None
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
    med = benchmark["warm_predict_median_ms"]
    report = {
        "tensor_dump": str(args.tensor_dump),
        "label": label,
        "manifest_metadata": manifest.get("metadata", {}),
        "export": export_report,
        "benchmark": benchmark,
        "speedup_vs_decoder_pre_plus_generator_pct": float(
            (med["baseline_total"] - med["candidate_total"]) / med["baseline_total"] * 100.0
        ),
        "report_path": str(report_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/har_source_noise_split"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report_har_source_noise.json")
    parser.add_argument("--decoder-pre-package", type=Path, default=Path("coreml/kokoro_decoder_pre_3s.mlpackage"))
    parser.add_argument("--fused-package", type=Path, default=Path("coreml/kokoro_decoder_har_post_3s.mlpackage"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--cos-snake", action="store_true")
    parser.add_argument("--palettize-noise", action="store_true")
    parser.add_argument("--palettize-body", action="store_true")
    parser.add_argument("--noise-precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--body-precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--tail-precision", default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--anchor-mode", default="mean", choices=["mean", "sum", "first"])
    parser.add_argument("--decoder-pre-compute-units", default="all")
    parser.add_argument("--fused-compute-units", default="all")
    parser.add_argument("--noise-compute-units", default="all")
    parser.add_argument("--body-compute-units", default="all")
    parser.add_argument("--tail-compute-units", default="cpu_and_gpu")
    args = parser.parse_args()

    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
