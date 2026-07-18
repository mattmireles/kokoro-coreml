#!/usr/bin/env python3
"""Bisect the fused ``har_source`` Core ML graph with intermediate outputs.

``probe_coreml_stft_semantics.py`` proved that an fp32 manual-atan STFT subgraph
can match PyTorch. Yet the full fused ``har_source -> waveform`` graph still
misses strict parity. This script exports one debug graph that returns the
intermediate HAR tensor, noise-source tensors, pre-tail logits, and waveform so
the first downstream drift can be measured directly.
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


def _make_debug_module(generator: Any, phase_mode: str, pad_har_to: int | None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from kokoro.custom_stft import CustomSTFT

    class _CoreMLForwardSTFT(nn.Module):
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
            elif phase_mode == "atan_manual":
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
            else:
                raise RuntimeError(f"unsupported phase_mode: {phase_mode}")
            return magnitude, phase

    class _DebugFused(nn.Module):
        def __init__(self, gen: Any):
            super().__init__()
            self.gen = gen
            self.forward_stft = _CoreMLForwardSTFT(gen)

        def forward(self, x_pre: Any, ref_s: Any, har_source: Any):
            s = ref_s[:, :128]
            gen = self.gen
            har_spec, har_phase = self.forward_stft(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
            if pad_har_to is not None:
                current = har.size(2)
                if current < pad_har_to:
                    har = F.pad(har, (0, pad_har_to - current))
                elif current > pad_har_to:
                    har = har[:, :, :pad_har_to]
            x = x_pre
            debug_sources = []
            for i in range(gen.num_upsamples):
                x = F.leaky_relu(x, negative_slope=0.1)
                x_source = gen.noise_convs[i](har)
                x_source = gen.noise_res[i](x_source, s)
                debug_sources.append(x_source)
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
            pre_tail = gen.conv_post(x)
            spec = torch.exp(pre_tail[:, : gen.post_n_fft // 2 + 1, :])
            phase = torch.sin(pre_tail[:, gen.post_n_fft // 2 + 1 :, :])
            waveform = gen.stft.inverse(spec, phase)
            return har, debug_sources[0], debug_sources[1], pre_tail, waveform

    return _DebugFused(generator).eval()


def _reference_outputs(
    generator: Any,
    tensors: dict[str, np.ndarray],
    phase_mode: str,
    pad_har_to: int | None,
) -> dict[str, np.ndarray]:
    import torch

    module = _make_debug_module(generator, phase_mode, pad_har_to)
    with torch.no_grad():
        values = module(
            torch.from_numpy(tensors["x_pre_padded"].astype(np.float32)),
            torch.from_numpy(tensors["ref_s"].astype(np.float32)),
            torch.from_numpy(tensors["har_source"].astype(np.float32)),
        )
    return {
        "har": values[0].detach().cpu().numpy().astype(np.float32),
        "x_source_0": values[1].detach().cpu().numpy().astype(np.float32),
        "x_source_1": values[2].detach().cpu().numpy().astype(np.float32),
        "pre_tail": values[3].detach().cpu().numpy().astype(np.float32),
        "waveform": values[4].detach().cpu().numpy().astype(np.float32),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import coremltools as ct
    import torch

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["x_pre_padded", "ref_s", "har_source", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    package = args.output_dir / args.label / f"kokoro_har_source_fused_debug_{args.label}.mlpackage"
    gen = _load_kmodel().decoder.generator
    reference = _reference_outputs(gen, tensors, args.phase_mode, args.pad_har_to)

    if not args.skip_export:
        module = _make_debug_module(gen, args.phase_mode, args.pad_har_to)
        dummy_inputs = (
            torch.zeros(tuple(tensors["x_pre_padded"].shape), dtype=torch.float32),
            torch.zeros(tuple(tensors["ref_s"].shape), dtype=torch.float32),
            torch.zeros(tuple(tensors["har_source"].shape), dtype=torch.float32),
        )
        with torch.no_grad():
            traced = torch.jit.trace(module, dummy_inputs, strict=False, check_trace=False)
        model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x_pre", shape=tuple(tensors["x_pre_padded"].shape), dtype=np.float32),
                ct.TensorType(name="ref_s", shape=tuple(tensors["ref_s"].shape), dtype=np.float32),
                ct.TensorType(name="har_source", shape=tuple(tensors["har_source"].shape), dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="har"),
                ct.TensorType(name="x_source_0"),
                ct.TensorType(name="x_source_1"),
                ct.TensorType(name="pre_tail"),
                ct.TensorType(name="waveform"),
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
    coreml_out = {
        key: np.asarray(value).astype(np.float32)
        for key, value in ml.predict(
            {
                "x_pre": tensors["x_pre_padded"].astype(np.float32),
                "ref_s": tensors["ref_s"].astype(np.float32),
                "har_source": tensors["har_source"].astype(np.float32),
            }
        ).items()
    }

    metrics: dict[str, Any] = {}
    for name, ref_value in reference.items():
        candidate = coreml_out[name]
        trim_len = min(int(ref_value.size), int(candidate.size))
        metrics[f"coreml_vs_torch_{name}"] = _metrics(
            ref_value.reshape(-1)[:trim_len],
            candidate.reshape(-1)[:trim_len],
        )
    waveform_trim = min(int(tensors["waveform"].size), int(coreml_out["waveform"].size))
    metrics["coreml_waveform_vs_dump"] = _metrics(
        tensors["waveform"].reshape(-1)[:waveform_trim],
        coreml_out["waveform"].reshape(-1)[:waveform_trim],
    )
    metrics["torch_waveform_vs_dump"] = _metrics(
        tensors["waveform"].reshape(-1)[:waveform_trim],
        reference["waveform"].reshape(-1)[:waveform_trim],
    )

    report = {
        "tensor_dump": str(args.tensor_dump),
        "label": args.label,
        "package": str(package),
        "precision": args.precision,
        "phase_mode": args.phase_mode,
        "pad_har_to": args.pad_har_to,
        "compute_units": args.compute_units,
        "manifest_metadata": manifest.get("metadata", {}),
        "metrics": metrics,
    }
    report_path = args.output_dir / args.label / args.report_name
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/har_source_fused_debug"))
    parser.add_argument("--label", default="3s_atan_manual_fp32")
    parser.add_argument("--report-name", default="report_har_source_fused_debug.json")
    parser.add_argument("--precision", default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--phase-mode", default="atan_manual", choices=["atan2", "atan_manual"])
    parser.add_argument("--pad-har-to", type=int, default=None)
    parser.add_argument("--compute-units", default="cpu_only")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
