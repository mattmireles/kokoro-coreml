#!/usr/bin/env python3
"""Probe a fused ``har_source -> waveform`` generator package.

This keeps the exact Swift harmonic source boundary but avoids the lossy
body/tail split used by ``probe_har_source_noise_split.py``. The exported
temporary package takes ``x_pre``, ``ref_s``, and ``har_source`` and performs
STFT, generator body, and iSTFT tail in one Core ML graph.
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


def _make_har_source_fused_module(generator: Any, phase_mode: str, pad_har_to: int | None):
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
            if phase_mode == "atan2":
                phase = torch.atan2(imag, real)
            elif phase_mode == "acos":
                denom = torch.clamp(magnitude, min=1e-12)
                cos_phase = torch.clamp(real / denom, min=-1.0, max=1.0)
                abs_phase = torch.acos(cos_phase)
                sign = torch.where(imag < 0.0, -torch.ones_like(imag), torch.ones_like(imag))
                phase = abs_phase * sign
            elif phase_mode in {"atan_manual", "atan_swift"}:
                eps = torch.full_like(real, 1e-12)
                safe_real = torch.where(
                    torch.abs(real) < eps,
                    torch.where(real < 0.0, -eps, eps),
                    real,
                )
                base = torch.atan(imag / safe_real)
                phase = torch.where(
                    real < 0.0,
                    torch.where(imag >= 0.0, base + torch.pi, base - torch.pi),
                    base,
                )
                phase = torch.where(
                    torch.abs(real) < eps,
                    torch.where(
                        imag > 0.0,
                        torch.full_like(imag, torch.pi / 2.0),
                        torch.where(imag < 0.0, torch.full_like(imag, -torch.pi / 2.0), torch.zeros_like(imag)),
                    ),
                    phase,
                )
                if phase_mode == "atan_swift":
                    boundary = (real < 0.0) & (torch.abs(imag) < 1e-4)
                    phase = torch.where(
                        boundary,
                        torch.where(imag >= 0.0, torch.full_like(imag, -torch.pi), torch.full_like(imag, torch.pi)),
                        phase,
                    )
            else:
                raise RuntimeError(f"unsupported phase_mode: {phase_mode}")
            return magnitude, phase

    class _HarSourceFusedGenerator(nn.Module):
        def __init__(self, gen: Any):
            super().__init__()
            self.gen = gen
            fwd_stft = CustomSTFT(
                filter_length=gen.stft.filter_length,
                hop_length=gen.stft.hop_length,
                win_length=gen.stft.win_length,
            )
            self.forward_stft = _CoreMLForwardSTFT(fwd_stft)

        def forward(self, x_pre: Any, ref_s: Any, har_source: Any):
            s = ref_s[:, :128]
            gen = self.gen
            har_spec, har_phase = self.forward_stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
            if pad_har_to is not None:
                current = har.size(2)
                if current < pad_har_to:
                    har = F.pad(har, (0, pad_har_to - current))
                elif current > pad_har_to:
                    har = har[:, :, :pad_har_to]
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

    return _HarSourceFusedGenerator(generator).eval()


def _predict(model: Any, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = model.predict(feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    waveform = np.asarray(out.get("waveform", next(iter(out.values())))).astype(np.float32)
    return waveform, elapsed_ms


def run(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct
    import torch

    from export_synth.wrappers import GeneratorFromHar, remove_dropout

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["x_pre_padded", "ref_s", "har_padded", "har_source", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    work_dir = args.output_dir / label
    package = work_dir / f"kokoro_har_source_fused_{label}.mlpackage"
    report_path = work_dir / args.report_name

    kmodel = _load_kmodel()
    gen = kmodel.decoder.generator
    x_pre = tensors["x_pre_padded"].astype(np.float32)
    ref_s = tensors["ref_s"].astype(np.float32)
    har_source = tensors["har_source"].astype(np.float32)
    har = tensors["har_padded"].astype(np.float32)

    if not args.skip_export:
        module = _make_har_source_fused_module(gen, args.phase_mode, args.pad_har_to)
        removed_dropouts = remove_dropout(module)
        with torch.no_grad():
            traced = torch.jit.trace(
                module,
                (
                    torch.zeros(tuple(x_pre.shape), dtype=torch.float32),
                    torch.zeros(tuple(ref_s.shape), dtype=torch.float32),
                    torch.zeros(tuple(har_source.shape), dtype=torch.float32),
                ),
                strict=False,
                check_trace=False,
            )
        model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x_pre", shape=tuple(x_pre.shape), dtype=np.float32),
                ct.TensorType(name="ref_s", shape=tuple(ref_s.shape), dtype=np.float32),
                ct.TensorType(name="har_source", shape=tuple(har_source.shape), dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="waveform")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=_precision_arg(ct, args.precision),
            compute_units=ct.ComputeUnit.ALL,
        )
        package.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_package(package)
        model.save(str(package))
    else:
        removed_dropouts = None
        if not package.is_dir():
            raise SystemExit(f"--skip-export requested but package is missing: {package}")

    fused = ct.models.MLModel(str(args.fused_package), compute_units=_compute_units(ct, args.fused_compute_units))
    candidate = ct.models.MLModel(str(package), compute_units=_compute_units(ct, args.compute_units))

    baseline_first, baseline_first_ms = _predict(fused, {"x_pre": x_pre, "ref_s": ref_s, "har": har})
    candidate_first, candidate_first_ms = _predict(candidate, {"x_pre": x_pre, "ref_s": ref_s, "har_source": har_source})
    for _ in range(max(0, args.warmup)):
        _predict(fused, {"x_pre": x_pre, "ref_s": ref_s, "har": har})
        _predict(candidate, {"x_pre": x_pre, "ref_s": ref_s, "har_source": har_source})

    baseline_times: list[float] = []
    candidate_times: list[float] = []
    last_baseline = baseline_first
    last_candidate = candidate_first
    for _ in range(max(1, args.iterations)):
        last_baseline, baseline_ms = _predict(fused, {"x_pre": x_pre, "ref_s": ref_s, "har": har})
        last_candidate, candidate_ms = _predict(candidate, {"x_pre": x_pre, "ref_s": ref_s, "har_source": har_source})
        baseline_times.append(baseline_ms)
        candidate_times.append(candidate_ms)

    trim_len = min(int(tensors["waveform"].size), int(last_baseline.size), int(last_candidate.size))
    dump = tensors["waveform"].reshape(-1)[:trim_len]
    med_baseline = float(statistics.median(baseline_times))
    med_candidate = float(statistics.median(candidate_times))
    report = {
        "tensor_dump": str(args.tensor_dump),
        "label": label,
        "package": str(package),
        "removed_dropouts": removed_dropouts,
        "manifest_metadata": manifest.get("metadata", {}),
        "precision": args.precision,
        "phase_mode": args.phase_mode,
        "pad_har_to": args.pad_har_to,
        "compute_units": args.compute_units,
        "fused_compute_units": args.fused_compute_units,
        "first_predict_ms": {
            "baseline_generator": float(baseline_first_ms),
            "candidate_har_source_fused": float(candidate_first_ms),
        },
        "warm_predict_times_ms": {
            "baseline_generator": baseline_times,
            "candidate_har_source_fused": candidate_times,
        },
        "warm_predict_median_ms": {
            "baseline_generator": med_baseline,
            "candidate_har_source_fused": med_candidate,
        },
        "speedup_vs_generator_pct": float((med_baseline - med_candidate) / med_baseline * 100.0),
        "metrics": {
            "baseline_vs_dump_trimmed": _metrics(dump, last_baseline.reshape(-1)[:trim_len]),
            "candidate_vs_dump_trimmed": _metrics(dump, last_candidate.reshape(-1)[:trim_len]),
            "candidate_vs_baseline_trimmed": _metrics(
                last_baseline.reshape(-1)[:trim_len],
                last_candidate.reshape(-1)[:trim_len],
            ),
        },
        "report_path": str(report_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/har_source_fused"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report_har_source_fused.json")
    parser.add_argument("--fused-package", type=Path, default=Path("coreml/kokoro_decoder_har_post_3s.mlpackage"))
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--phase-mode", default="atan2", choices=["atan2", "acos", "atan_manual", "atan_swift"])
    parser.add_argument("--pad-har-to", type=int, default=None)
    parser.add_argument("--compute-units", default="all")
    parser.add_argument("--fused-compute-units", default="all")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
