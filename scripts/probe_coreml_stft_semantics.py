#!/usr/bin/env python3
"""Probe Core ML conversion semantics for Kokoro's export-friendly STFT.

The fused ``har_source`` package is speed-positive but fails parity. This script
exports only the forward STFT subgraph used by that package and compares Core ML
``magnitude``/``phase`` tensors against PyTorch and the dumped Swift tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _precision_arg, _remove_existing_package  # noqa: E402


def _make_stft_module(generator: Any, phase_mode: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from kokoro.custom_stft import CustomSTFT

    class _ForwardSTFT(nn.Module):
        def __init__(self, gen: Any):
            super().__init__()
            stft = CustomSTFT(
                filter_length=gen.stft.filter_length,
                hop_length=gen.stft.hop_length,
                win_length=gen.stft.win_length,
            )
            self.center = stft.center
            self.n_fft = stft.n_fft
            self.pad_mode = stft.pad_mode
            self.hop_length = stft.hop_length
            self.register_buffer("weight_forward_real", stft.weight_forward_real)
            self.register_buffer("weight_forward_imag", stft.weight_forward_imag)

        def forward(self, waveform: Any):
            if self.center:
                pad_len = self.n_fft // 2
                waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
            x = waveform.unsqueeze(1)
            real = F.conv1d(x, self.weight_forward_real, stride=self.hop_length)
            imag = F.conv1d(x, self.weight_forward_imag, stride=self.hop_length)
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
            return magnitude, phase, real, imag

    return _ForwardSTFT(generator).eval()


def _phase_wrap_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = ((candidate - reference + np.pi) % (2.0 * np.pi)) - np.pi
    return {
        "wrapped_max_abs_error": float(np.max(np.abs(diff))),
        "wrapped_mean_abs_error": float(np.mean(np.abs(diff))),
        "wrapped_rmse": float(np.sqrt(np.mean(diff**2))),
        "wrapped_mean_cos": float(np.mean(np.cos(diff))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct
    import torch

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["har_source", "har_magnitude", "har_phase"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    package = args.output_dir / args.label / f"kokoro_stft_debug_{args.label}.mlpackage"
    gen = _load_kmodel().decoder.generator
    module = _make_stft_module(gen, args.phase_mode)
    har_source = tensors["har_source"].astype(np.float32)
    dummy = torch.zeros(tuple(har_source.shape), dtype=torch.float32)

    if not args.skip_export:
        with torch.no_grad():
            traced = torch.jit.trace(module, (dummy,), strict=False, check_trace=False)
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="har_source", shape=tuple(har_source.shape), dtype=np.float32)],
            outputs=[
                ct.TensorType(name="magnitude"),
                ct.TensorType(name="phase"),
                ct.TensorType(name="real"),
                ct.TensorType(name="imag"),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=_precision_arg(ct, args.precision),
            compute_units=ct.ComputeUnit.ALL,
        )
        package.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_package(package)
        model.save(str(package))
    elif not package.is_dir():
        raise SystemExit(f"--skip-export requested but package is missing: {package}")

    ml = ct.models.MLModel(str(package), compute_units=_compute_units(ct, args.compute_units))
    coreml_out = {k: np.asarray(v).astype(np.float32) for k, v in ml.predict({"har_source": har_source}).items()}
    with torch.no_grad():
        torch_out = module(torch.from_numpy(har_source))
    torch_names = ["magnitude", "phase", "real", "imag"]
    torch_out_np = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in zip(torch_names, torch_out)}

    report: dict[str, Any] = {
        "tensor_dump": str(args.tensor_dump),
        "label": args.label,
        "package": str(package),
        "precision": args.precision,
        "compute_units": args.compute_units,
        "phase_mode": args.phase_mode,
        "manifest_metadata": manifest.get("metadata", {}),
        "metrics": {},
    }
    metrics = report["metrics"]
    for name in torch_names:
        metrics[f"coreml_vs_torch_{name}"] = _metrics(
            torch_out_np[name].reshape(-1),
            coreml_out[name].reshape(-1),
        )
    metrics["coreml_phase_vs_torch_wrapped"] = _phase_wrap_metrics(
        torch_out_np["phase"].reshape(-1),
        coreml_out["phase"].reshape(-1),
    )
    metrics["torch_magnitude_vs_swift_dump"] = _metrics(
        tensors["har_magnitude"].reshape(-1),
        torch_out_np["magnitude"].reshape(-1),
    )
    metrics["torch_phase_vs_swift_dump"] = _metrics(
        tensors["har_phase"].reshape(-1),
        torch_out_np["phase"].reshape(-1),
    )
    metrics["coreml_magnitude_vs_swift_dump"] = _metrics(
        tensors["har_magnitude"].reshape(-1),
        coreml_out["magnitude"].reshape(-1),
    )
    metrics["coreml_phase_vs_swift_dump"] = _metrics(
        tensors["har_phase"].reshape(-1),
        coreml_out["phase"].reshape(-1),
    )
    metrics["coreml_phase_vs_swift_wrapped"] = _phase_wrap_metrics(
        tensors["har_phase"].reshape(-1),
        coreml_out["phase"].reshape(-1),
    )
    metrics["coreml_phase_range"] = {
        "min": float(np.min(coreml_out["phase"])),
        "max": float(np.max(coreml_out["phase"])),
        "mean": float(np.mean(coreml_out["phase"])),
    }
    report_path = args.output_dir / args.label / args.report_name
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coreml_stft_semantics"))
    parser.add_argument("--label", default="3s")
    parser.add_argument("--report-name", default="report_stft_semantics.json")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--compute-units", default="cpu_only")
    parser.add_argument("--phase-mode", default="atan2", choices=["atan2", "acos", "atan_manual", "atan_swift"])
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
