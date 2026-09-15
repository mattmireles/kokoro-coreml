#!/usr/bin/env python3
"""Measure how bucket padding changes the valid region of Kokoro's output.

Runs the PyTorch fp32 synthesis stack (F0Ntrain, decoder, harmonic source,
generator) for one text twice per bucket: once at the utterance's natural
frame count and once right-padded to the bucket, then compares the valid region
of every stage. Four masking modes are measured so each mask-aware change can
be attributed:

  unmasked    every time-axis statistic sees the padding (the pre-mask model)
  generator   only the generator's AdaIN1d calls masked (the Core ML pipeline
              between the generator and decoder-pre mask exports)
  adain       AdaIN1d statistics confined to valid frames, shared LSTM unmasked
  adain+lstm  AdaIN masked and the F0Ntrain shared BiLSTM masked

Usage::

    uv run python scripts/measure_bucket_contamination.py --text "Hello there." \
        --buckets 3,7,10,15,30 --out outputs/contamination.json

Reports per bucket and mode: fill fraction, SNR (dB) of f0, n, x_pre and the
waveform against the native-shape run, and the valid-region RMS ratio.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

ENUM_TOKEN_SIZES = [32, 64, 128, 256, 512]
F0_FRAMES_PER_SECOND = 40
SAMPLES_PER_FRAME = 24_000 // F0_FRAMES_PER_SECOND


def _snr_db(ref: np.ndarray, cand: np.ndarray) -> float:
    ref = ref.astype(np.float64).ravel()
    cand = cand.astype(np.float64).ravel()[: ref.size]
    den = float(np.linalg.norm(cand - ref))
    return float("inf") if den == 0.0 else 20.0 * math.log10(float(np.linalg.norm(ref)) / den)


def _pad_time(x: torch.Tensor, target: int) -> torch.Tensor:
    if x.shape[-1] >= target:
        return x[..., :target].contiguous()
    return torch.cat([x, torch.zeros(*x.shape[:-1], target - x.shape[-1], dtype=x.dtype)], dim=-1)


def _mask(t_frames: int, valid: int) -> torch.Tensor:
    m = torch.zeros(1, 1, t_frames)
    m[:, :, :valid] = 1.0
    return m


def _alignment(pred_dur: np.ndarray, trace_length: int, frame_count: int) -> torch.Tensor:
    durations = np.zeros(trace_length, dtype=np.int64)
    durations[: pred_dur.size] = pred_dur
    idx = np.repeat(np.arange(trace_length), durations)[:frame_count]
    if idx.size < frame_count:
        idx = np.concatenate([idx, np.full(frame_count - idx.size, idx[-1] if idx.size else 0)])
    mat = np.zeros((trace_length, frame_count), dtype=np.float32)
    mat[idx, np.arange(frame_count)] = 1.0
    return torch.from_numpy(mat)


def prepare(text: str, voice: str, speed: float) -> dict:
    """Tokenise, run the export DurationModel, and align en/asr at natural length."""
    from kokoro import KModel, KPipeline
    from kokoro.pipeline import voice_embedding_for_phoneme_string
    from export_synth.wrappers import DurationModel

    torch.manual_seed(0)
    pipeline = KPipeline(lang_code="a")
    kmodel = KModel().eval()
    phonemes = next(ps for _, ps, _ in pipeline(text, voice, speed))
    ref_s = voice_embedding_for_phoneme_string(pipeline.load_voice(voice), phonemes)
    ids = [0] + [kmodel.vocab[p] for p in phonemes if p in kmodel.vocab] + [0]
    T = next(s for s in ENUM_TOKEN_SIZES if s >= len(ids))
    padded = torch.tensor(ids + [0] * (T - len(ids))).unsqueeze(0)
    attention = torch.tensor([1] * len(ids) + [0] * (T - len(ids))).unsqueeze(0)
    with torch.no_grad():
        pred_dur, d, t_en, s, ref_s_out = DurationModel(kmodel).eval()(
            padded, ref_s.reshape(1, -1).float(), torch.tensor([speed]), attention
        )
    valid_dur = np.maximum(1, pred_dur.numpy().reshape(-1)[: len(ids)])
    frames = int(valid_dur.sum())
    align = _alignment(valid_dur, T, frames)
    return {"kmodel": kmodel, "en": d.transpose(-1, -2) @ align, "asr": t_en @ align, "s": s,
            "ref_s_out": ref_s_out, "frames": frames, "tokens": len(ids), "phonemes": phonemes}


def synthesize(inp: dict, t_frames: int, mode: str) -> dict:
    """Run F0Ntrain -> decoder -> harmonic source -> generator at ``t_frames``."""
    from kokoro.conv_length import conv1d_output_length_from_module
    from export_synth.wrappers import GeneratorFromHar

    torch.manual_seed(0)
    kmodel, V = inp["kmodel"], inp["frames"]
    dec, gen = kmodel.decoder, kmodel.decoder.generator
    en = _pad_time(inp["en"], t_frames)
    m = _mask(t_frames, V) if mode not in ("unmasked", "generator") else None

    if mode == "adain+lstm":
        from export_f0ntrain import F0NtrainWrapper
        wrapper = F0NtrainWrapper(kmodel.predictor).eval()
        if "mask" not in inspect.signature(wrapper.forward).parameters:
            raise RuntimeError("F0NtrainWrapper has no mask input; adain+lstm is unavailable")
        f0, n = wrapper(en, inp["s"], mask=m)
    elif mode == "adain":
        # This diagnostic isolates masked AdaIN while deliberately leaving the
        # shared BiLSTM unchanged. Keep the partial contract local to the
        # experiment instead of exposing a misleading production API.
        predictor = kmodel.predictor
        shared, _ = predictor.shared(en.transpose(-1, -2))

        def masked_branch(blocks, projection):
            value = shared.transpose(-1, -2)
            current = m
            for block in blocks:
                next_mask = (
                    current.repeat_interleave(2, dim=2)
                    if block.upsample_type != "none"
                    else current
                )
                value = block(value, inp["s"], current, next_mask)
                current = next_mask
            return projection(value).squeeze(1)

        f0 = masked_branch(predictor.F0, predictor.F0_proj)
        n = masked_branch(predictor.N, predictor.N_proj)
    else:
        f0, n = kmodel.predictor.F0Ntrain(en, inp["s"])

    frame_count = conv1d_output_length_from_module(f0.shape[-1], dec.F0_conv)
    asr = _pad_time(inp["asr"], frame_count)
    m_dec = _mask(frame_count, V) if m is not None else None
    style = inp["ref_s_out"][:, :128]
    f0_conv, n_conv = dec.F0_conv(f0.unsqueeze(1)), dec.N_conv(n.unsqueeze(1))
    x = dec.encode(torch.cat([asr, f0_conv, n_conv], dim=1), style, m=m_dec, m_up=m_dec)
    asr_res, res, cur = dec.asr_res(asr), True, m_dec
    for block in dec.decode:
        if res:
            x = torch.cat([x, asr_res, f0_conv, n_conv], dim=1)
        m_up = cur.repeat_interleave(2, dim=2) if cur is not None and block.upsample_type != "none" else cur
        x = block(x, style, m=cur, m_up=m_up)
        cur = m_up
        res = res and block.upsample_type == "none"
    x_pre = x

    f0_up = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
    har_source, _, _ = gen.m_source(f0_up)
    har_spec, har_phase = gen.stft.transform(har_source.transpose(1, 2).squeeze(1))
    har = torch.cat([har_spec, har_phase], dim=1)
    if mode == "generator":
        cur = _mask(x_pre.shape[-1], 2 * V)
    waveform = GeneratorFromHar(gen).eval()(x_pre, inp["ref_s_out"], har, mask=cur)
    # F0/N and the decoder's final upsampling block all leave at twice the
    # duration-frame rate.
    return {"f0": f0[..., : 2 * V], "n": n[..., : 2 * V], "x_pre": x_pre[..., : 2 * V], "waveform": waveform.reshape(-1)[: V * SAMPLES_PER_FRAME]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--buckets", default="3,7,10,15,30", help="Bucket seconds; each is padded to seconds * 40 frames")
    parser.add_argument("--modes", default="unmasked,generator,adain,adain+lstm")
    parser.add_argument("--out", type=Path, default=None, help="Write the rows as JSON")
    args = parser.parse_args()

    inp = prepare(args.text, args.voice, args.speed)
    V = inp["frames"]
    print(f"text={args.text!r} tokens={inp['tokens']} frames={V} ({V / F0_FRAMES_PER_SECOND:.2f} s)")
    with torch.no_grad():
        native = synthesize(inp, V, "unmasked")
    rows = []
    for mode in args.modes.split(","):
        for sec in (int(b) for b in args.buckets.split(",")):
            t_frames = sec * F0_FRAMES_PER_SECOND
            if t_frames < V:
                continue
            try:
                with torch.no_grad():
                    out = synthesize(inp, t_frames, mode)
            except RuntimeError as exc:
                print(f"{mode}: {exc}")
                break
            row = {"text": args.text, "mode": mode, "bucket_s": sec, "t_frames": t_frames, "fill": round(V / t_frames, 4)}
            for name in ("f0", "n", "x_pre", "waveform"):
                row[f"{name}_snr_db"] = round(_snr_db(native[name].numpy(), out[name].numpy()), 2)
            row["rms_ratio"] = round(float(out["waveform"].pow(2).mean().sqrt() / native["waveform"].pow(2).mean().sqrt()), 4)
            rows.append(row)
            print(f"{mode:11s} {sec:2d}s fill={row['fill']:.2f}  f0 {row['f0_snr_db']:6.2f}  n {row['n_snr_db']:6.2f}  "
                  f"x_pre {row['x_pre_snr_db']:6.2f}  wave {row['waveform_snr_db']:6.2f} dB  rms x{row['rms_ratio']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"tokens": inp["tokens"], "frames": V, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
