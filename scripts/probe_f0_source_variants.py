#!/usr/bin/env python3
"""Compare F0 source formulations against dumped Swift ``har_source`` tensors.

The F0-noise Core ML probe is speed-positive but quality-negative. This script
keeps the investigation cheap by testing source generation variants in PyTorch
against the tensor dump boundary before exporting any new packages.
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


def _source_current_probe(gen: Any, f0_curve: Any, *, downsample: str, noise_scale: float):
    """Return the deterministic source used by F0 probe variants."""

    import torch
    import torch.nn.functional as F

    f0 = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
    sine = gen.m_source.l_sin_gen
    harmonics = torch.arange(
        1,
        sine.harmonic_num + 2,
        device=f0.device,
        dtype=f0.dtype,
    )
    fn = f0 * harmonics.view(1, 1, -1)
    rad_values = fn / sine.sampling_rate
    rv = rad_values.transpose(1, 2)
    down_len = max(1, int((int(f0.shape[1]) + sine.upsample_scale - 1) // sine.upsample_scale))
    if downsample == "avg_pool":
        rv_down = F.avg_pool1d(
            rv,
            kernel_size=sine.upsample_scale,
            stride=sine.upsample_scale,
        )
    elif downsample == "linear":
        rv_down = F.interpolate(rv, size=down_len, mode="linear")
    else:
        raise ValueError(f"unknown downsample mode: {downsample}")
    rad_down = rv_down.transpose(1, 2)
    phase = torch.cumsum(rad_down, dim=1) * (2.0 * math.pi)
    ph = phase.transpose(1, 2) * sine.upsample_scale
    ph_up = F.interpolate(
        ph,
        size=down_len * sine.upsample_scale,
        mode="linear",
    )
    sines = torch.sin(ph_up.transpose(1, 2)) * sine.sine_amp
    uv = (f0 > sine.voiced_threshold).to(dtype=f0.dtype)
    noise_amp = uv * sine.noise_std + (1.0 - uv) * sine.sine_amp / 3.0
    sine_waves = sines * uv + noise_amp * float(noise_scale)
    merged = gen.m_source.l_tanh(gen.m_source.l_linear(sine_waves))
    return merged.transpose(1, 2).squeeze(1)


def _source_original_seeded(gen: Any, f0_curve: Any, seed: int):
    """Return original PyTorch SourceModuleHnNSF source with a fixed seed."""

    import torch

    torch.manual_seed(seed)
    f0 = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
    har_source, _noise, _uv = gen.m_source(f0)
    return har_source.transpose(1, 2).squeeze(1)


def _xorshift64_values(seed: int, count: int) -> np.ndarray:
    """Return Swift ``SeededRNG`` values for cross-language source parity."""

    state = np.uint64(seed)
    values = np.empty(count, dtype=np.uint64)
    for idx in range(count):
        state = np.uint64(state ^ np.uint64(state << np.uint64(13)))
        state = np.uint64(state ^ np.uint64(state >> np.uint64(7)))
        state = np.uint64(state ^ np.uint64(state << np.uint64(17)))
        values[idx] = state
    return values


def _swift_uniform01(seed: int, count: int) -> np.ndarray:
    """Return Swift-style 24-bit uniform floats in ``[0, 1)``."""

    values = _xorshift64_values(seed, count)
    masked = (values & np.uint64(0xFFFFFF)).astype(np.float32)
    return masked / np.float32(0xFFFFFF)


def _swift_gaussian(seed: int, count: int) -> np.ndarray:
    """Return Gaussian noise matching Swift ``generateGaussianNoise``."""

    pair_count = (count + 1) // 2
    uniforms = _swift_uniform01(seed, pair_count * 2)
    out = np.empty(pair_count * 2, dtype=np.float32)
    tiny = np.finfo(np.float32).eps
    for pair in range(pair_count):
        u1 = max(tiny, float(uniforms[2 * pair]))
        u2 = float(uniforms[2 * pair + 1])
        radius = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out[2 * pair] = np.float32(radius * math.cos(theta))
        out[2 * pair + 1] = np.float32(radius * math.sin(theta))
    return out[:count]


def _linear_interp_np(values: np.ndarray, target_len: int) -> np.ndarray:
    """Match Swift/PyTorch linear interpolation with ``align_corners=False``."""

    src_count = int(values.shape[0])
    if src_count == 0 or target_len <= 0:
        return np.empty((0,), dtype=np.float64)
    if target_len == 1:
        return np.asarray([float(np.mean(values))], dtype=np.float64)
    if src_count == target_len:
        return values.astype(np.float64, copy=True)

    src_len = float(src_count)
    dst_len = float(target_len)
    ratio = src_len / dst_len
    out = np.empty(target_len, dtype=np.float64)
    for idx in range(target_len):
        src_idx = (float(idx) + 0.5) * ratio - 0.5
        src_idx = max(0.0, min(src_idx, src_len - 1.0))
        lo = int(src_idx)
        hi = min(lo + 1, src_count - 1)
        frac = src_idx - float(lo)
        out[idx] = float(values[lo]) * (1.0 - frac) + float(values[hi]) * frac
    return out


def _source_swift_like(gen: Any, f0_curve: Any, seed: int):
    """Return source generated with the Swift HarmonicSource equation."""

    import torch

    f0 = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2).detach().cpu().numpy()
    f0_up = f0.reshape(-1).astype(np.float32)
    sine = gen.m_source.l_sin_gen
    linear = gen.m_source.l_linear
    weights = linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    bias = float(linear.bias.detach().cpu().numpy().reshape(-1)[0])

    length = int(f0_up.shape[0])
    dim = int(sine.harmonic_num + 1)
    scale = int(sine.upsample_scale)
    down_len = max(1, (length + scale - 1) // scale)
    up_len = down_len * scale
    sine_waves = np.empty((dim, length), dtype=np.float32)

    # Swift consumes one random value for each overtone's initial phase before
    # generating the separate Gaussian-noise stream from the same seed.
    initial_uniforms = _swift_uniform01(seed, dim - 1) if dim > 1 else np.empty((0,), dtype=np.float32)
    two_pi_times_scale = 2.0 * math.pi * float(scale)
    for harmonic in range(dim):
        rad = np.remainder(f0_up.astype(np.float64) * float(harmonic + 1) / float(sine.sampling_rate), 1.0)
        if harmonic > 0:
            rad[0] += float(initial_uniforms[harmonic - 1])
        rad_ds = _linear_interp_np(rad, down_len)
        phase_scaled = np.cumsum(rad_ds, dtype=np.float64) * two_pi_times_scale
        phase_up = _linear_interp_np(phase_scaled, up_len)
        sine_waves[harmonic, :] = (np.sin(phase_up[:length]) * float(sine.sine_amp)).astype(np.float32)

    gaussian = _swift_gaussian(seed, dim * length).reshape(dim, length)
    uv = (f0_up > float(sine.voiced_threshold)).astype(np.float32)
    noise_amp = uv * float(sine.noise_std) + (1.0 - uv) * float(sine.sine_amp / 3.0)
    sine_waves = sine_waves * uv[None, :] + gaussian * noise_amp[None, :]

    merged = np.tanh(bias + weights.astype(np.float32) @ sine_waves).astype(np.float32)
    return torch.from_numpy(merged.reshape(1, -1))


def _report_for_dump(tensor_dump: Path, seed_override: int | None) -> dict[str, Any]:
    import torch

    manifest, tensors = load_tensor_dump(tensor_dump)
    if "f0_padded" not in tensors or "har_source" not in tensors or "har_padded" not in tensors:
        raise SystemExit(f"{tensor_dump} must contain f0_padded, har_source, and har_padded")

    metadata = manifest.get("metadata", {})
    seed = int(seed_override if seed_override is not None else metadata.get("seed", 42))
    gen = _load_kmodel().decoder.generator.eval()
    f0 = torch.from_numpy(tensors["f0_padded"].astype(np.float32))
    reference = tensors["har_source"].reshape(-1).astype(np.float32)
    har_reference = tensors["har_padded"].astype(np.float32)

    variants = {
        "original_pytorch_seeded": _source_original_seeded(gen, f0, seed),
        "swift_like_seeded": _source_swift_like(gen, f0, seed),
        "probe_avg_pool_noise_0p01": _source_current_probe(gen, f0, downsample="avg_pool", noise_scale=0.01),
        "probe_avg_pool_noise_0": _source_current_probe(gen, f0, downsample="avg_pool", noise_scale=0.0),
        "linear_interp_noise_0p01": _source_current_probe(gen, f0, downsample="linear", noise_scale=0.01),
        "linear_interp_noise_0": _source_current_probe(gen, f0, downsample="linear", noise_scale=0.0),
    }

    source_rows: dict[str, Any] = {}
    har_rows: dict[str, Any] = {}
    for name, value in variants.items():
        candidate = value.detach().cpu().numpy().reshape(-1).astype(np.float32)
        trim_len = min(int(reference.size), int(candidate.size))
        source_rows[name] = _metrics(reference[:trim_len], candidate[:trim_len])

        with torch.no_grad():
            har_spec, har_phase = gen.stft.transform(value)
            har = torch.cat([har_spec, har_phase], dim=1).detach().cpu().numpy().astype(np.float32)
        har_padded = np.zeros_like(har_reference)
        copy_len = min(int(har.shape[-1]), int(har_padded.shape[-1]))
        har_padded[..., :copy_len] = har[..., :copy_len]
        har_rows[name] = _metrics(har_reference.reshape(-1), har_padded.reshape(-1))

    reference_source = torch.from_numpy(reference.reshape(1, -1))
    with torch.no_grad():
        ref_spec, ref_phase = gen.stft.transform(reference_source)
        ref_har = torch.cat([ref_spec, ref_phase], dim=1).detach().cpu().numpy().astype(np.float32)
    ref_har_padded = np.zeros_like(har_reference)
    ref_copy_len = min(int(ref_har.shape[-1]), int(ref_har_padded.shape[-1]))
    ref_har_padded[..., :ref_copy_len] = ref_har[..., :ref_copy_len]
    har_rows["dump_source_recomputed_stft"] = _metrics(har_reference.reshape(-1), ref_har_padded.reshape(-1))

    return {
        "tensor_dump": str(tensor_dump),
        "producer": metadata.get("producer"),
        "input_key": metadata.get("input_key"),
        "bucket_seconds": metadata.get("bucket_seconds"),
        "seed": seed,
        "reference_samples": int(reference.size),
        "metrics_vs_dump_har_source": source_rows,
        "metrics_vs_dump_har_padded": har_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reports": [
            _report_for_dump(path, args.seed)
            for path in args.tensor_dump
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tensor_dump",
        nargs="+",
        type=Path,
        help="Tensor dump directory containing f0_padded and har_source.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
