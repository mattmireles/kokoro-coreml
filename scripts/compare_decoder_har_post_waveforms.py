#!/usr/bin/env python3
"""Compare two decoder-har-post .mlpackage outputs on identical PyTorch-built inputs.

Uses :func:`kokoro.synthesis_backends.build_decoder_har_post_inputs_np` so geometry
matches production ``decoder_har_post_bucket_impl``.

Example::

    uv run python scripts/compare_decoder_har_post_waveforms.py \\
      --baseline /tmp/kokoro_har_post_baseline_3s.mlpackage \\
      --candidate coreml/kokoro_decoder_har_post_3s.mlpackage \\
      --bucket-sec 3
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import coremltools as ct

from kokoro.coreml_pipeline import HybridTTSPipeline
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np


def _predict_waveform(model: ct.models.MLModel, inputs: dict) -> np.ndarray:
    res = model.predict(inputs)
    key = "waveform" if "waveform" in res else list(res.keys())[0]
    return np.asarray(res[key], dtype=np.float32).squeeze()


def _metrics(ref: np.ndarray, cand: np.ndarray):
    ref = ref.astype(np.float64, copy=False)
    cand = cand.astype(np.float64, copy=False)
    n = min(ref.size, cand.size)
    ref = ref.flatten()[:n]
    cand = cand.flatten()[:n]
    max_delta = float(np.max(np.abs(ref - cand)))
    err = ref - cand
    snr = 10.0 * np.log10((np.sum(ref * ref) + 1e-12) / (np.sum(err * err) + 1e-12))
    rms = float(np.sqrt(np.mean(ref * ref)))
    pearson = None
    if rms >= 1e-4:
        pearson = float(np.corrcoef(ref, cand)[0, 1])
    return pearson, float(snr), max_delta, rms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=str, required=True, help="Path to baseline .mlpackage")
    p.add_argument("--candidate", type=str, required=True, help="Path to candidate .mlpackage")
    # Bench-set buckets. The shipped set is a subset; the wider list lets the
    # same comparison cover every bucket an export produces.
    p.add_argument("--bucket-sec", type=int, required=True, choices=(3, 7, 10, 15, 30))
    p.add_argument("--text", type=str, default="Hello from Kokoro.")
    p.add_argument("--voice", type=str, default="af_heart")
    p.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(0)
    pipe = HybridTTSPipeline()
    base = ct.models.MLModel(args.baseline)
    cand = ct.models.MLModel(args.candidate)
    spec_b = base.get_spec()
    shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec_b.description.input}
    asr_len = int(shapes["x_pre"][-1])
    har_t = int(shapes["har"][-1])

    vi = pipe.extract_vocoder_inputs(args.text, args.voice, args.speed)
    if vi is None:
        print("extract_vocoder_inputs failed", file=sys.stderr)
        return 1
    T_f0 = int(vi["f0_curve"].shape[-1])
    total_seconds = T_f0 / 80.0
    selected = pipe._select_bucket_seconds(total_seconds)
    if selected is None or selected not in pipe.coreml_decoder_har_post_buckets:
        print("no decoder_har_post bucket for this utterance / pipeline", file=sys.stderr)
        return 1
    if selected != args.bucket_sec:
        print(
            f"error: pipeline selects {selected}s for this text; use --bucket-sec {selected}",
            file=sys.stderr,
        )
        return 3
    sec = selected

    dec = pipe.pytorch_model.decoder
    x_pre_np, ref_s, har_np, T_f0_b, _fc, mask_np = build_decoder_har_post_inputs_np(
        dec, vi, sec, asr_len, har_t, warn_geometry=True
    )
    assert T_f0_b == T_f0

    # Mask-aware packages declare a `mask` input. Per-spec dispatch keeps the
    # script working against both pre- and post-Phase-3 packages.
    base_inputs = {"x_pre": x_pre_np, "ref_s": ref_s, "har": har_np}
    if "mask" in {i.name for i in base.get_spec().description.input}:
        base_inputs["mask"] = mask_np
    cand_inputs = {"x_pre": x_pre_np, "ref_s": ref_s, "har": har_np}
    if "mask" in {i.name for i in cand.get_spec().description.input}:
        cand_inputs["mask"] = mask_np
    w0 = _predict_waveform(base, base_inputs)
    w1 = _predict_waveform(cand, cand_inputs)
    target_len = int(round((T_f0 / 80.0) * 24000.0))
    w0 = w0[: min(int(w0.shape[-1]), target_len)]
    w1 = w1[: min(int(w1.shape[-1]), target_len)]

    pearson, snr, max_delta, rms = _metrics(w0, w1)
    print(f"ref_rms={rms:.6e} pearson={pearson} snr_db={snr:.2f} max_abs_delta={max_delta:.6e}")
    ok = max_delta <= 1e-2 and snr >= 40.0
    if pearson is not None:
        ok = ok and pearson > 0.99
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
