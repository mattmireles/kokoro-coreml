#!/usr/bin/env python3
"""Probe whether HAR Nyquist phase can be removed from the generator contract.

The compact ``har_source -> waveform`` path is fast, but strict parity fails
because the raw Nyquist phase channel uses a different ``+pi/-pi`` branch than
the current Swift dump. This probe stays in PyTorch and tests the next simple
hypothesis: if the generator barely uses that feature, we can zero or fold it
and keep the shorter HAR source boundary.

Outputs are reports under ``outputs/`` only. Shipping Core ML packages are not
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _load_kmodel, _metrics  # noqa: E402

NYQUIST_BIN = 10
NYQUIST_HAR_CHANNEL = 21


def _manual_stft_har(generator: Any, har_source_np: np.ndarray):
    """Recompute HAR magnitude/phase with CoreML-safe manual atan semantics."""

    import torch
    import torch.nn.functional as F

    from kokoro.custom_stft import CustomSTFT

    stft = CustomSTFT(
        filter_length=generator.stft.filter_length,
        hop_length=generator.stft.hop_length,
        win_length=generator.stft.win_length,
    )
    waveform = torch.from_numpy(har_source_np.astype(np.float32))
    if stft.center:
        waveform = F.pad(waveform, (stft.n_fft // 2, stft.n_fft // 2), mode=stft.pad_mode)
    x = waveform.unsqueeze(1)
    real = F.conv1d(x, stft.weight_forward_real, stride=stft.hop_length)
    imag = F.conv1d(x, stft.weight_forward_imag, stride=stft.hop_length)
    magnitude = torch.sqrt(real**2 + imag**2 + 1e-14)

    eps = torch.full_like(real, 1e-12)
    safe_real = torch.where(torch.abs(real) < eps, torch.where(real < 0.0, -eps, eps), real)
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
    return torch.cat([magnitude, phase], dim=1)


def _run_generator_from_har(generator: Any, tensors: dict[str, np.ndarray], har: Any):
    """Run the generator body from an already-built HAR tensor."""

    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(tensors["x_pre_padded"].astype(np.float32))
    ref_s = torch.from_numpy(tensors["ref_s"].astype(np.float32))
    s = ref_s[:, :128]

    gen = generator
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


def _phase_channel_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    raw = _metrics(reference, candidate)
    wrapped_delta = np.angle(np.exp(1j * (reference.astype(np.float64) - candidate.astype(np.float64))))
    raw["wrapped_mean_abs_error"] = float(np.mean(np.abs(wrapped_delta)))
    raw["wrapped_max_abs_error"] = float(np.max(np.abs(wrapped_delta)))
    raw["two_pi_branch_errors"] = int(np.sum(np.abs(reference - candidate) > math.pi))
    return raw


def _weight_stats(generator: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, conv in enumerate(generator.noise_convs):
        weight = conv.weight.detach().cpu().numpy().astype(np.float64)
        nyq = weight[:, NYQUIST_HAR_CHANNEL, :]
        all_abs = float(np.sum(np.abs(weight)))
        all_l2 = float(np.linalg.norm(weight))
        nyq_abs = float(np.sum(np.abs(nyq)))
        nyq_l2 = float(np.linalg.norm(nyq))
        rows.append(
            {
                "noise_conv_index": index,
                "weight_shape": [int(v) for v in weight.shape],
                "nyquist_abs_fraction": nyq_abs / all_abs if all_abs else None,
                "nyquist_l2_fraction": nyq_l2 / all_l2 if all_l2 else None,
                "nyquist_weight_mean": float(np.mean(nyq)),
                "nyquist_weight_abs_mean": float(np.mean(np.abs(nyq))),
                "nyquist_weight_max_abs": float(np.max(np.abs(nyq))),
            }
        )
    return rows


def _pad_or_trim_har(har: Any, target_time: int | None):
    import torch.nn.functional as F

    if target_time is None:
        return har
    current = int(har.size(2))
    if current < target_time:
        return F.pad(har, (0, target_time - current))
    if current > target_time:
        return har[:, :, :target_time]
    return har


def _make_variants(tensors: dict[str, np.ndarray], recomputed_har: Any, pad_har_to: int | None) -> dict[str, Any]:
    import torch

    dumped_har = torch.from_numpy(tensors["har"].astype(np.float32))
    dumped_phase = torch.from_numpy(tensors["har_phase"].astype(np.float32))
    dumped_nyquist = dumped_phase[:, NYQUIST_BIN, :]
    mean_nyquist = float(dumped_nyquist.mean().item())

    variants: dict[str, Any] = {}
    variants["dumped_har"] = dumped_har
    if "har_padded" in tensors:
        variants["dumped_har_padded"] = torch.from_numpy(tensors["har_padded"].astype(np.float32))
    variants["dumped_har_zero_nyquist"] = dumped_har.clone()
    variants["dumped_har_zero_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = 0.0
    variants["dumped_har_mean_nyquist"] = dumped_har.clone()
    variants["dumped_har_mean_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = mean_nyquist

    variants["recomputed_manual"] = recomputed_har
    variants["recomputed_manual_dumped_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_dumped_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = dumped_nyquist
    variants["recomputed_manual_zero_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_zero_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = 0.0
    variants["recomputed_manual_mean_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_mean_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = mean_nyquist
    variants["recomputed_manual_pos_pi_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_pos_pi_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = math.pi
    variants["recomputed_manual_neg_pi_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_neg_pi_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = -math.pi
    return {name: _pad_or_trim_har(har, pad_har_to) for name, har in variants.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["har_source", "har", "har_phase", "x_pre_padded", "ref_s", "waveform"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    generator = _load_kmodel().decoder.generator.eval()
    recomputed_har = _manual_stft_har(generator, tensors["har_source"])
    variants = _make_variants(tensors, recomputed_har, args.pad_har_to)
    reference_waveform = tensors["waveform"].reshape(-1)

    with torch.no_grad():
        waveform_metrics = {}
        outputs = {}
        for name, har in variants.items():
            waveform = _run_generator_from_har(generator, tensors, har).detach().cpu().numpy().astype(np.float32)
            outputs[name] = waveform
            waveform_metrics[name] = _metrics(reference_waveform, waveform.reshape(-1))

    dumped_har = tensors["har"].astype(np.float32)
    recomputed_np = recomputed_har.detach().cpu().numpy().astype(np.float32)
    feature_metrics = {
        "recomputed_har_vs_dumped_har": _metrics(dumped_har, recomputed_np),
        "recomputed_nyquist_phase_vs_dumped": _phase_channel_metrics(
            dumped_har[:, NYQUIST_HAR_CHANNEL, :].reshape(-1),
            recomputed_np[:, NYQUIST_HAR_CHANNEL, :].reshape(-1),
        ),
    }
    for bin_index in range(NYQUIST_BIN):
        feature_metrics[f"recomputed_phase_bin_{bin_index}_vs_dumped"] = _phase_channel_metrics(
            dumped_har[:, 11 + bin_index, :].reshape(-1),
            recomputed_np[:, 11 + bin_index, :].reshape(-1),
        )

    report = {
        "tensor_dump": str(args.tensor_dump),
        "manifest_metadata": manifest.get("metadata", {}),
        "nyquist_bin": NYQUIST_BIN,
        "nyquist_har_channel": NYQUIST_HAR_CHANNEL,
        "pad_har_to": args.pad_har_to,
        "weight_stats": _weight_stats(generator),
        "feature_metrics": feature_metrics,
        "waveform_metrics_vs_dump": waveform_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/nyquist_phase_contribution/report.json"))
    parser.add_argument(
        "--pad-har-to",
        type=int,
        default=None,
        help="Pad or trim every natural HAR variant to this time length before running the generator.",
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["waveform_metrics_vs_dump"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
