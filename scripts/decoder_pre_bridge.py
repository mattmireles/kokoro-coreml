#!/usr/bin/env python3
"""Pre-compute DecoderPre outputs (x_pre) for the bakeoff inputs.

This bridges the gap in the Swift pipeline until Phase 4 (CoreML DecoderPre
export). The Swift benchmark loads these pre-computed tensors from disk instead
of running PyTorch.

Usage::

    uv run python scripts/decoder_pre_bridge.py

Output: outputs/decoder_pre_bridge/{input_key}/x_pre.npy
        outputs/decoder_pre_bridge/{input_key}/metadata.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent

# Same inputs as bakeoff v2
BAKEOFF_INPUTS = {
    "tiny": "Hello world!",
    "short": "The quick brown fox jumps over the dog.",
    "medium": (
        "This is a longer sentence designed to test the performance "
        "of our text to speech system running on the Apple GPU."
    ),
    "long": (
        "This is a longer sentence designed to test the performance "
        "of our text to speech system running on modern Apple Silicon "
        "hardware. A few more words added here."
    ),
}
VOICE = "af_heart"
SPEED = 1.0


def main():
    from kokoro.coreml_pipeline import HybridTTSPipeline
    from kokoro.synthesis_backends import build_decoder_har_post_inputs_np
    from kokoro.conv_length import conv1d_output_length_from_module

    output_dir = _ROOT / "outputs" / "decoder_pre_bridge"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    pipe = HybridTTSPipeline()
    dec = pipe.pytorch_model.decoder
    gen = dec.generator
    f0_samples_per_step = int(round(float(gen.f0_upsamp.scale_factor)))

    for key, text in BAKEOFF_INPUTS.items():
        print(f"\n--- {key}: {text!r:.60} ---")
        case_dir = output_dir / key
        case_dir.mkdir(exist_ok=True)

        # Extract vocoder inputs
        vi = pipe.extract_vocoder_inputs(text, VOICE, SPEED)
        if vi is None:
            print(f"  FAILED to extract vocoder inputs for {key}")
            continue

        T_f0 = int(vi["f0_curve"].shape[-1])
        total_seconds = T_f0 / 80.0

        # Select bucket
        sec = pipe._select_bucket_seconds(total_seconds)
        if sec is None:
            print(f"  No bucket for {key}")
            continue

        # Get bucket geometry
        bucket_samples = sec * 24000
        full_f0_len = int(round(bucket_samples / float(f0_samples_per_step)))
        frame_count = conv1d_output_length_from_module(full_f0_len, dec.F0_conv)

        # Load model spec for expected shapes
        if sec in getattr(pipe, "coreml_decoder_har_post_buckets", {}):
            model = pipe.coreml_decoder_har_post_buckets[sec]
            spec = model.get_spec()
            shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
            asr_len = int(shapes["x_pre"][-1])
            har_t = int(shapes["har"][-1])
        else:
            asr_len = frame_count
            har_t = full_f0_len * f0_samples_per_step // 5 + 1  # approximate

        # Build x_pre via PyTorch decoder pre-processing
        t0 = time.perf_counter()
        x_pre_np, ref_s, har_np, _t_f0, _fc, _mask = build_decoder_har_post_inputs_np(
            dec, vi, sec, asr_len, har_t, warn_geometry=False
        )
        t1 = time.perf_counter()

        # Save outputs
        np.save(case_dir / "x_pre.npy", x_pre_np)
        np.save(case_dir / "ref_s.npy", ref_s)
        np.save(case_dir / "har_reference.npy", har_np)

        # Save all vocoder inputs for Swift pipeline validation
        for k, v in vi.items():
            np.save(case_dir / f"vi_{k}.npy", v)

        # Save metadata
        meta = {
            "input_key": key,
            "text": text,
            "voice": VOICE,
            "speed": SPEED,
            "bucket_seconds": sec,
            "T_f0": T_f0,
            "total_seconds": total_seconds,
            "full_f0_len": full_f0_len,
            "frame_count": frame_count,
            "asr_len": asr_len,
            "har_t": har_t,
            "x_pre_shape": list(x_pre_np.shape),
            "har_shape": list(har_np.shape),
            "decoder_pre_time_ms": (t1 - t0) * 1000,
        }
        with open(case_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Bucket: {sec}s, T_f0: {T_f0}, frame_count: {frame_count}")
        print(f"  x_pre: {x_pre_np.shape}, har: {har_np.shape}")
        print(f"  DecoderPre time: {(t1-t0)*1000:.1f} ms")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
