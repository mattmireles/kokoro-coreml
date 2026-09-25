#!/usr/bin/env python3
"""Capture Python-side tensors for the Kokoro audio parity ladder.

This script mirrors the current Swift/Core ML Config F stage boundaries for one
prepared bakeoff input. It writes a language-neutral tensor dump that can be
compared with ``kokoro-bench --dump-tensors`` using
``scripts/compare_audio_parity_tensors.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import TensorDumpWriter, bucket_mask_from_tensors  # noqa: E402
from export_synth.wrappers import DurationModel, GeneratorFromHar  # noqa: E402
from kokoro import KModel  # noqa: E402
from kokoro.conv_length import conv1d_output_length_from_module  # noqa: E402

DEFAULT_BUCKETS = [3, 7, 10, 15, 30]
T_FRAMES_FOR_BUCKET = {3: 120, 7: 280, 10: 400, 15: 600, 30: 1200, 45: 1800}


def _build_alignment_matrix(pred_dur: np.ndarray, trace_length: int, frame_count: int) -> np.ndarray:
    durations = np.zeros((trace_length,), dtype=np.int64)
    copy_len = min(trace_length, pred_dur.shape[-1])
    durations[:copy_len] = pred_dur[:copy_len]

    repeat_idx = np.repeat(np.arange(trace_length, dtype=np.int64), durations)
    if repeat_idx.size > frame_count:
        repeat_idx = repeat_idx[:frame_count]
    elif repeat_idx.size < frame_count:
        last_idx = int(repeat_idx[-1]) if repeat_idx.size else 0
        repeat_idx = np.concatenate(
            [repeat_idx, np.full((frame_count - repeat_idx.size,), last_idx, dtype=np.int64)]
        )
    repeat_idx = np.clip(repeat_idx, 0, trace_length - 1)

    mat = np.zeros((trace_length, frame_count), dtype=np.float32)
    mat[repeat_idx, np.arange(frame_count)] = 1.0
    return mat


def _select_bucket(duration_s: float, buckets: list[int]) -> int:
    threshold = int(math.ceil(duration_s))
    for bucket in sorted(buckets):
        if bucket >= threshold:
            return bucket
    return sorted(buckets)[-1]


def _pad_time(array: np.ndarray, target_time: int) -> np.ndarray:
    out = np.zeros((*array.shape[:-1], target_time), dtype=np.float32)
    copy_time = min(array.shape[-1], target_time)
    out[..., :copy_time] = array[..., :copy_time]
    return out


def _generator_shapes(models_dir: Path, bucket_sec: int, fallback_x_pre: int, fallback_har: int) -> tuple[int, int]:
    package = models_dir / f"kokoro_decoder_har_post_{bucket_sec}s.mlpackage"
    if not package.is_dir():
        return fallback_x_pre, fallback_har
    try:
        import coremltools as ct

        spec = ct.models.MLModel(str(package)).get_spec()
        shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
        return int(shapes["x_pre"][-1]), int(shapes["har"][-1])
    except Exception as exc:  # pragma: no cover - depends on local Core ML runtime
        print(f"WARN: could not inspect {package}: {exc}", file=sys.stderr)
        return fallback_x_pre, fallback_har


def capture(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_json) if args.input_json else Path(args.inputs_dir) / f"{args.input_key}.json"
    data = json.loads(input_path.read_text())
    input_ids = np.asarray(data["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(data["attention_mask"], dtype=np.int64)
    ref_s = np.asarray(data["ref_s"], dtype=np.float32).reshape(1, 256)
    speed = np.asarray([float(data.get("speed", args.speed))], dtype=np.float32)
    num_tokens = int(data.get("num_tokens") or int(attention_mask.sum()))
    text = str(data.get("text", ""))
    voice = str(data.get("voice", args.voice))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kmodel = KModel().eval()
    duration_model = DurationModel(kmodel).eval()

    with torch.no_grad():
        pred_dur_t, d_t, t_en_t, s_t, ref_s_out_t = duration_model(
            torch.from_numpy(input_ids.reshape(1, -1)).long(),
            torch.from_numpy(ref_s),
            torch.from_numpy(speed),
            torch.from_numpy(attention_mask.reshape(1, -1)).long(),
        )

        pred_dur_full = pred_dur_t.detach().cpu().numpy().astype(np.int64)
        d_np = d_t.detach().cpu().numpy().astype(np.float32)
        t_en_np = t_en_t.detach().cpu().numpy().astype(np.float32)
        s_np = s_t.detach().cpu().numpy().astype(np.float32)
        ref_s_out = ref_s_out_t.detach().cpu().numpy().astype(np.float32)

        trace_length = int(pred_dur_full.shape[-1])
        valid_pred_dur = np.maximum(1, pred_dur_full.reshape(-1)[:num_tokens])
        natural_frames = int(valid_pred_dur.sum())
        alignment = _build_alignment_matrix(valid_pred_dur, trace_length, natural_frames)

        d_transposed = np.transpose(d_np, (0, 2, 1))
        en = np.matmul(d_transposed, alignment).astype(np.float32)
        asr = np.matmul(t_en_np, alignment).astype(np.float32)

        canonical_duration_s = data.get("canonical_duration_s")
        duration_for_bucket = float(canonical_duration_s) if canonical_duration_s else float(natural_frames * 2) / 80.0
        buckets = [int(v) for v in args.available_buckets.split(",") if v.strip()]
        bucket_sec = int(args.bucket_sec) if args.bucket_sec else _select_bucket(duration_for_bucket, buckets)
        t_frames = T_FRAMES_FOR_BUCKET[bucket_sec]

        en_padded = _pad_time(en, t_frames)
        f0_t, n_t = kmodel.predictor.F0Ntrain(torch.from_numpy(en_padded), torch.from_numpy(s_np))
        f0 = f0_t.detach().cpu().numpy().astype(np.float32)
        n = n_t.detach().cpu().numpy().astype(np.float32)

        dec = kmodel.decoder
        gen = dec.generator
        f0_samples_per_step = int(round(float(gen.f0_upsamp.scale_factor)))
        full_f0_len = int(round(float(bucket_sec * 24_000) / float(f0_samples_per_step)))
        decoder_frame_count = conv1d_output_length_from_module(full_f0_len, dec.F0_conv)

        f0_padded = _pad_time(f0, full_f0_len)
        n_padded = _pad_time(n, full_f0_len)
        asr_padded = _pad_time(asr, decoder_frame_count)

        baseline_s = torch.from_numpy(ref_s_out[:, :128])
        asr_t = torch.from_numpy(asr_padded)
        f0_conv = dec.F0_conv(torch.from_numpy(f0_padded).unsqueeze(1))
        n_conv = dec.N_conv(torch.from_numpy(n_padded).unsqueeze(1))
        x = torch.cat([asr_t, f0_conv, n_conv], dim=1)
        x = dec.encode(x, baseline_s)
        asr_res = dec.asr_res(asr_t)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, f0_conv, n_conv], dim=1)
            x = block(x, baseline_s)
            if block.upsample_type != "none":
                res = False
        x_pre = x.detach().cpu().numpy().astype(np.float32)

        torch.manual_seed(args.seed)
        f0_up = gen.f0_upsamp(torch.from_numpy(f0_padded)[:, None]).transpose(1, 2)
        har_source, _noise, _uv = gen.m_source(f0_up)
        har_source_2d = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source_2d)
        har = torch.cat([har_spec, har_phase], dim=1).detach().cpu().numpy().astype(np.float32)

        x_pre_time, har_time = _generator_shapes(
            Path(args.models_dir), bucket_sec, int(x_pre.shape[-1]), int(har.shape[-1])
        )
        x_pre_padded = _pad_time(x_pre, x_pre_time)
        har_padded = _pad_time(har, har_time)

        # The runtime feeds mask-aware packages this mask (PR #6); an
        # unmasked reference differs from it by 6-8 dB.
        gen_mask = bucket_mask_from_tensors({"pred_dur_valid": valid_pred_dur}, x_pre_time)
        gen_from_har = GeneratorFromHar(gen).eval()
        waveform_t = gen_from_har(
            torch.from_numpy(x_pre_padded),
            torch.from_numpy(ref_s_out),
            torch.from_numpy(har_padded),
            torch.from_numpy(gen_mask),
        )
        waveform_full = waveform_t.detach().cpu().numpy().astype(np.float32)
        waveform_flat = waveform_full.reshape(-1)
        trim_len = min(int(waveform_flat.shape[-1]), int(round((natural_frames * 2) / 80.0 * 24_000.0)))
        waveform = waveform_flat[:trim_len].astype(np.float32)

    writer = TensorDumpWriter(
        args.out_dir,
        metadata={
            "producer": "python",
            "script": "scripts/capture_audio_parity_tensors.py",
            "input_key": data.get("key", args.input_key),
            "input_json": str(input_path),
            "text": text,
            "voice": voice,
            "speed": float(speed[0]),
            "seed": int(args.seed),
            "bucket_seconds": bucket_sec,
            "trace_length": trace_length,
            "num_tokens": num_tokens,
            "natural_frames": natural_frames,
            "canonical_duration_s": canonical_duration_s,
            "duration_for_bucket_s": duration_for_bucket,
            "t_frames": t_frames,
            "full_f0_len": full_f0_len,
            "decoder_frame_count": decoder_frame_count,
            "x_pre_expected_time": x_pre_time,
            "har_expected_time": har_time,
            "trim_len": trim_len,
            "hnsf_reference_command": "uv run python scripts/validate_hnsf_swift.py generate",
        },
    )
    writer.write("tokens", input_ids.reshape(1, -1).astype(np.int32))
    writer.write("attention_mask", attention_mask.reshape(1, -1).astype(np.int32))
    writer.write("ref_s", ref_s_out)
    writer.write("speed", speed)
    writer.write("pred_dur", pred_dur_full.astype(np.int32))
    writer.write("pred_dur_valid", valid_pred_dur.reshape(1, -1).astype(np.int32))
    writer.write("duration_d", d_np)
    writer.write("duration_t_en", t_en_np)
    writer.write("s", s_np)
    writer.write("d_transposed", d_transposed)
    writer.write("alignment", alignment.reshape(1, trace_length, natural_frames))
    writer.write("en", en)
    writer.write("asr", asr)
    writer.write("en_padded", en_padded)
    writer.write("f0", f0)
    writer.write("n", n)
    writer.write("f0_padded", f0_padded)
    writer.write("n_padded", n_padded)
    writer.write("asr_padded", asr_padded)
    writer.write("x_pre", x_pre)
    writer.write("x_pre_padded", x_pre_padded)
    writer.write("har_source", har_source_2d.detach().cpu().numpy().astype(np.float32))
    writer.write("har_magnitude", har_spec.detach().cpu().numpy().astype(np.float32))
    writer.write("har_phase", har_phase.detach().cpu().numpy().astype(np.float32))
    writer.write("har", har)
    writer.write("har_padded", har_padded)
    writer.write("waveform_full", waveform_full)
    writer.write("waveform", waveform)
    manifest = writer.close()
    print(f"Wrote Python tensor dump: {manifest}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-key", default="3s", help="Prepared input key under --inputs-dir.")
    parser.add_argument("--input-json", default=None, help="Explicit prepared input JSON path.")
    parser.add_argument("--inputs-dir", default=str(_ROOT / "outputs" / "swift_bench_inputs"))
    parser.add_argument("--models-dir", default=str(_ROOT / "coreml"))
    parser.add_argument("--out-dir", default=str(_ROOT / "outputs" / "audio-parity" / "tensors" / "python_3s"))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bucket-sec", type=int, default=None)
    parser.add_argument("--available-buckets", default="3,7,10,15,30")
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
