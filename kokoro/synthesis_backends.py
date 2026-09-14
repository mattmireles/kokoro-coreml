"""Pluggable CoreML / PyTorch synthesis implementations for ``HybridTTSPipeline``.

Each ``*_impl`` function takes ``pipe`` (a :class:`~kokoro.coreml_pipeline.HybridTTSPipeline`)
as the first argument so orchestration stays in one place while new backends can be added by:

- Appending callables to ``DEFAULT_TEXT_BACKENDS`` / ``DEFAULT_VI_BACKENDS``, or
- Passing ``text_backends=`` / ``vi_backends=`` into :class:`~kokoro.coreml_pipeline.HybridTTSPipeline`.

Text backends receive ``(pipe, text, voice, speed)`` and may call ``pipe.extract_vocoder_inputs``.
Vi backends receive ``(pipe, vocoder_inputs)`` where ``vocoder_inputs`` is the dict from
``extract_vocoder_inputs``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch

from kokoro.conv_length import conv1d_output_length_from_module
from kokoro.pipeline import voice_embedding_for_phoneme_string

if TYPE_CHECKING:
    from kokoro.coreml_pipeline import HybridTTSPipeline

# 5 s @ 24 kHz per vocoder window chunk (matches 5s export bucket duration; see export_synth.wrappers.CoreMLExportConstants)
VOCODER_CHUNK_SAMPLES = 5 * 24000

TextBackend = Callable[[Any, str, str, float], np.ndarray | None]
ViBackend = Callable[[Any, dict], np.ndarray | None]


def build_decoder_har_post_inputs_np(
    dec: torch.nn.Module,
    vi: dict,
    sec: int,
    asr_len: int,
    har_t: int,
    *,
    warn_geometry: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Build ``x_pre`` / ``ref_s`` / ``har`` / ``mask`` numpy tensors for ``kokoro_decoder_har_post_*s``.

    Single source of truth for geometry shared with ``decoder_har_post_bucket_impl`` and
    ``scripts/compare_decoder_har_post_waveforms.py``.

    Returns:
        ``x_pre`` (float32), ``ref_s`` (float32), ``har`` (float32), ``T_f0`` (int),
        ``frame_count`` (int; decoder time dim before Core ML padding to ``asr_len``),
        ``mask`` (float32, shape ``(1, 1, asr_len)``; 1.0 on the valid prefix
        ``[0, t_asr)``, 0.0 on bucket padding). Mask-aware packages declare
        ``mask`` as an optional input; packages without it ignore the feature.
    """
    gen = dec.generator
    f0_samples_per_step = int(round(float(gen.f0_upsamp.scale_factor)))
    bucket_samples = sec * 24000
    full_f0_len = int(round(bucket_samples / float(f0_samples_per_step)))
    frame_count = conv1d_output_length_from_module(full_f0_len, dec.F0_conv)
    if warn_geometry and frame_count != asr_len:
        print(
            f"decoder_har_post: bucket geometry frame_count {frame_count} != Core ML x_pre length {asr_len}"
        )

    T_f0 = int(vi["f0_curve"].shape[-1])
    asr = vi["asr"].astype(np.float32)
    f0 = vi["f0_curve"].astype(np.float32)
    n = vi["n"].astype(np.float32)
    ref_s = vi["ref_s"].astype(np.float32)

    asr_pad = np.zeros((1, 512, frame_count), dtype=np.float32)
    t_asr = min(frame_count, asr.shape[-1])
    asr_pad[:, :, :t_asr] = asr[:, :, :t_asr]
    f0_pad = np.zeros((1, full_f0_len), dtype=np.float32)
    n_pad = np.zeros((1, full_f0_len), dtype=np.float32)
    t_f0 = min(full_f0_len, f0.shape[-1])
    f0_pad[:, :t_f0] = f0[:, :t_f0]
    n_pad[:, :t_f0] = n[:, :t_f0]

    with torch.no_grad():
        ref_t = torch.from_numpy(ref_s)
        s = ref_t[:, :128]
        asr_t = torch.from_numpy(asr_pad)
        F0 = dec.F0_conv(torch.from_numpy(f0_pad).unsqueeze(1))
        N = dec.N_conv(torch.from_numpy(n_pad).unsqueeze(1))
        x = torch.cat([asr_t, F0, N], dim=1)
        x = dec.encode(x, s)
        asr_res = dec.asr_res(asr_t)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x_pre = x
        f0_up = gen.f0_upsamp(torch.from_numpy(f0_pad)[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        har_np = har.numpy().astype(np.float32)

    x_pre_np = x_pre.cpu().numpy().astype(np.float32)
    if x_pre_np.shape[-1] != asr_len:
        aligned = np.zeros((x_pre_np.shape[0], x_pre_np.shape[1], asr_len), dtype=np.float32)
        c = min(x_pre_np.shape[-1], asr_len)
        aligned[:, :, :c] = x_pre_np[:, :, :c]
        x_pre_np = aligned

    if har_np.shape[-1] != har_t:
        h_new = np.zeros((har_np.shape[0], har_np.shape[1], har_t), dtype=np.float32)
        cpy = min(har_np.shape[-1], har_t)
        h_new[:, :, :cpy] = har_np[:, :, :cpy]
        har_np = h_new

    # Mask reflects the valid asr prefix in the (possibly Core ML-padded)
    # x_pre tensor. ``t_asr`` is the unpadded source length; ``asr_len`` is
    # the bucket-padded target length the Core ML model declares.
    mask_np = np.zeros((1, 1, asr_len), dtype=np.float32)
    mask_np[:, :, : min(t_asr, asr_len)] = 1.0

    return x_pre_np, ref_s, har_np, T_f0, frame_count, mask_np


def synth_bucket_impl(pipe: HybridTTSPipeline, text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray | None:
    """Single-shot bucketed synthesis using CoreML synthesizer model."""
    if not getattr(pipe, "coreml_synth_buckets", None):
        return None
    vi = pipe.extract_vocoder_inputs(text, voice, speed)
    if vi is None:
        return None
    total_f0_frames = int(vi["f0_curve"].shape[-1])
    total_seconds = total_f0_frames / 80.0
    sec = pipe._select_bucket_seconds(total_seconds)
    if sec is None:
        print("❌ No synthesizer buckets available")
        return None
    model = pipe.coreml_synth_buckets[sec]
    spec = model.get_spec()
    input_shapes = {i.name: i.type.multiArrayType.shape for i in spec.description.input}
    d_shape = input_shapes.get("d") or next(iter(input_shapes.values()))
    trace_length = int(d_shape[-1])
    pred_shape = input_shapes.get("pred_aln_trg")
    frame_count = int(pred_shape[-1]) if pred_shape else sec * 24000

    def pad_time(x, T):
        h = x.shape[1]
        out = np.zeros((1, h, T), dtype=np.float32)
        t = min(T, x.shape[-1])
        out[:, :, :t] = x[:, :, :t]
        return out

    d = pad_time(vi["d"], trace_length)
    t_en = pad_time(vi["t_en"], trace_length)
    s = vi["s"].astype(np.float32)
    ref_s = vi["ref_s"].astype(np.float32)
    pred_aln_trg = pipe._build_alignment_matrix(vi["pred_dur"].reshape(-1), trace_length, frame_count)

    inputs = {
        "d": d,
        "t_en": t_en,
        "s": s,
        "ref_s": ref_s,
        "pred_aln_trg": pred_aln_trg,
    }
    print(f"🍎 Running bucket synthesizer {sec}s: trace={trace_length}, frames={frame_count}")
    res = model.predict(inputs)
    key = list(res.keys())[0]
    audio = res[key].squeeze().astype(np.float32)
    target_len = int(sec * 24000)
    return audio[:target_len]


def decoder_har_post_bucket_impl(
    pipe: HybridTTSPipeline, text: str, voice: str = "af_heart", speed: float = 1.0
) -> np.ndarray | None:
    """PyTorch decoder stack + CPU hn-nsf ``har``; Core ML ``kokoro_decoder_har_post_*s`` → waveform.

    Matches ``export_synth.convert`` ``decoder-har`` geometry for each bucket (seconds).
    Tries before full Decoder_HAR so hn-nsf stays on CPU while conv/iSTFT use Core ML.
    """
    if not getattr(pipe, "coreml_decoder_har_post_buckets", None):
        return None
    vi = pipe.extract_vocoder_inputs(text, voice, speed)
    if vi is None:
        return None
    T_f0 = int(vi["f0_curve"].shape[-1])
    total_seconds = T_f0 / 80.0
    sec = pipe._select_bucket_seconds(total_seconds)
    if sec is None or sec not in pipe.coreml_decoder_har_post_buckets:
        return None
    model = pipe.coreml_decoder_har_post_buckets[sec]
    spec = model.get_spec()
    shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
    x_pre_shape = shapes["x_pre"]
    har_shape = shapes["har"]
    asr_len = int(x_pre_shape[-1])
    har_t = int(har_shape[-1])

    dec = pipe.pytorch_model.decoder
    x_pre_np, ref_s, har_np, _t_check, _fc, mask_np = build_decoder_har_post_inputs_np(
        dec, vi, sec, asr_len, har_t, warn_geometry=True
    )
    assert _t_check == T_f0

    print(
        f"🍎 Decoder_HAR post bucket {sec}s: Core ML post-har "
        f"(x_pre {tuple(x_pre_shape)}, har {tuple(har_shape)})"
    )
    inputs = {
        "x_pre": x_pre_np,
        "ref_s": ref_s,
        "har": har_np,
    }
    if "mask" in {i.name for i in spec.description.input}:
        inputs["mask"] = mask_np
    res = model.predict(inputs)
    key = "waveform" if "waveform" in res else list(res.keys())[0]
    audio = np.asarray(res[key], dtype=np.float32).squeeze()
    # Trim to natural utterance length: F0 curves are 80 Hz; do not derive length from
    # len(audio)/full_f0_len (breaks when Core ML returns fewer samples than a full bucket).
    target_len = int(round((T_f0 / 80.0) * 24000.0))
    return audio[: min(int(audio.shape[-1]), target_len)]


def decoder_har_sliding_impl(pipe: HybridTTSPipeline, vocoder_inputs: dict) -> np.ndarray | None:
    """
    Run CoreML Decoder_HAR (exact hn-nsf parity). PyTorch computes har_spec/har_phase.
    Overlap-add across sliding windows.
    """
    if getattr(pipe, "coreml_decoder_har", None) is None:
        print("❌ CoreML Decoder_HAR not available")
        return None
    print("🍎 Running CoreML Decoder_HAR (exact hn-nsf)...")
    try:
        asr = vocoder_inputs["asr"].astype(np.float32)
        f0 = vocoder_inputs["f0_curve"].astype(np.float32)
        n = vocoder_inputs["n"].astype(np.float32)
        s = vocoder_inputs["s"].astype(np.float32)

        asr_win, f0_win = 200, 400
        T_asr = asr.shape[-1]
        T_f0 = f0.shape[-1]
        hop_f0 = f0_win // 4
        hop_asr = asr_win // 4
        num_windows = int(np.ceil((T_f0 - f0_win) / hop_f0)) + 1 if T_f0 > 0 else 0

        out_audio = None
        acc = None
        hann = None
        chunk_len = None

        dec = pipe.pytorch_model.decoder
        with torch.no_grad():
            for w in range(num_windows):
                f0_start = w * hop_f0
                f0_end = f0_start + f0_win
                asr_start = w * hop_asr
                asr_end = asr_start + asr_win
                f0_slice = np.zeros((1, f0_win), dtype=np.float32)
                n_slice = np.zeros((1, f0_win), dtype=np.float32)
                asr_slice = np.zeros((1, 512, asr_win), dtype=np.float32)
                if f0_start < T_f0:
                    f0_slice_len = max(0, min(f0_end, T_f0) - f0_start)
                    if f0_slice_len > 0:
                        f0_slice[:, :f0_slice_len] = f0[:, f0_start : f0_start + f0_slice_len]
                if f0_start < n.shape[-1]:
                    n_slice_len = max(0, min(f0_end, n.shape[-1]) - f0_start)
                    if n_slice_len > 0:
                        n_slice[:, :n_slice_len] = n[:, f0_start : f0_start + n_slice_len]
                if asr_start < T_asr:
                    asr_slice_len = max(0, min(asr_end, T_asr) - asr_start)
                    if asr_slice_len > 0:
                        asr_slice[:, :, :asr_slice_len] = asr[:, :, asr_start : asr_start + asr_slice_len]

                f0_up = dec.generator.f0_upsamp(torch.from_numpy(f0_slice)[:, None]).transpose(1, 2)
                har_source, _, _ = dec.generator.m_source(f0_up)
                har_source = har_source.transpose(1, 2).squeeze(1)
                har_spec, har_phase = dec.generator.stft.transform(har_source)

                inputs = {
                    "asr": asr_slice.reshape(1, 512, 1, asr_win),
                    "f0_curve": f0_slice.reshape(1, 1, 1, f0_win),
                    "n": n_slice.reshape(1, 1, 1, f0_win),
                    "s": s,
                    "har_spec": har_spec.numpy().astype(np.float32).reshape(1, har_spec.shape[1], 1, har_spec.shape[2]),
                    "har_phase": har_phase.numpy().astype(np.float32).reshape(1, har_phase.shape[1], 1, har_phase.shape[2]),
                }
                res = pipe.coreml_decoder_har.predict(inputs)
                key = list(res.keys())[0]
                x = res[key]
                x_t = torch.from_numpy(x)
                n_fft = dec.generator.post_n_fft
                spec = torch.exp(x_t[:, : n_fft // 2 + 1, :])
                phase = torch.sin(x_t[:, n_fft // 2 + 1 :, :])
                chunk = dec.generator.stft.inverse(spec, phase).squeeze().numpy()

                if chunk_len is None:
                    chunk_len = len(chunk)
                    samples_per_f0_frame = chunk_len // f0_win
                    hop_samples = hop_f0 * samples_per_f0_frame
                    total_len = max(chunk_len, chunk_len + (num_windows - 1) * hop_samples)
                    out_audio = np.zeros((total_len,), dtype=np.float32)
                    acc = np.zeros_like(out_audio)
                    hann = np.hanning(chunk_len).astype(np.float32)

                dst_start = w * hop_samples
                dst_end = dst_start + chunk_len
                if dst_end > out_audio.shape[0]:
                    extend = dst_end - out_audio.shape[0]
                    out_audio = np.concatenate([out_audio, np.zeros((extend,), dtype=np.float32)])
                    acc = np.concatenate([acc, np.zeros((extend,), dtype=np.float32)])
                out_audio[dst_start:dst_end] += chunk * hann
                acc[dst_start:dst_end] += hann

        valid_idx = np.nonzero(acc > 1e-6)[0]
        if valid_idx.size == 0:
            return None
        last = valid_idx.max() + 1
        audio = out_audio[:last] / np.maximum(acc[:last], 1e-6)
        return audio
    except Exception as e:
        print(f"❌ CoreML Decoder_HAR failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def decoder_har_bucket_impl(pipe: HybridTTSPipeline, text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray | None:
    """Single-shot Decoder_HAR bucket: compute har once, call CoreML once, inverse STFT once."""
    if not getattr(pipe, "coreml_decoder_har_buckets", None):
        return None
    vi = pipe.extract_vocoder_inputs(text, voice, speed)
    if vi is None:
        return None
    T_f0 = int(vi["f0_curve"].shape[-1])
    total_seconds = T_f0 / 80.0
    sec = pipe._select_bucket_seconds(total_seconds)
    if sec is None or sec not in pipe.coreml_decoder_har_buckets:
        return None
    model = pipe.coreml_decoder_har_buckets[sec]
    spec = model.get_spec()
    shapes = {i.name: i.type.multiArrayType.shape for i in spec.description.input}
    asr_len = int(shapes["asr"][-1])
    f0_len = int(shapes["f0_curve"][-1])

    def pad_tail(x, T, axis=-1):
        out = np.zeros(list(x.shape[:-1]) + [T], dtype=x.dtype)
        t = min(T, x.shape[axis])
        out[(slice(None),) * (out.ndim - 1) + (slice(0, t),)] = x[(slice(None),) * (x.ndim - 1) + (slice(0, t),)]
        return out

    asr = vi["asr"].astype(np.float32)
    f0 = vi["f0_curve"].astype(np.float32)
    n = vi["n"].astype(np.float32)
    s = vi["s"].astype(np.float32)
    asr_pad = np.zeros((1, 512, asr_len), dtype=np.float32)
    t_asr = min(asr_len, asr.shape[-1])
    asr_pad[:, :, :t_asr] = asr[:, :, :t_asr]
    f0_pad = np.zeros((1, f0_len), dtype=np.float32)
    n_pad = np.zeros((1, f0_len), dtype=np.float32)
    t_f0 = min(f0_len, f0.shape[-1])
    f0_pad[:, :t_f0] = f0[:, :t_f0]
    n_pad[:, :t_f0] = n[:, :t_f0]
    dec = pipe.pytorch_model.decoder
    with torch.no_grad():
        f0_up = dec.generator.f0_upsamp(torch.from_numpy(f0_pad)[:, None]).transpose(1, 2)
        har_source, _, _ = dec.generator.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = dec.generator.stft.transform(har_source)
    inputs = {
        "asr": asr_pad.reshape(1, 512, 1, asr_len),
        "f0_curve": f0_pad.reshape(1, 1, 1, f0_len),
        "n": n_pad.reshape(1, 1, 1, f0_len),
        "s": s,
        "har_spec": har_spec.numpy().astype(np.float32).reshape(1, har_spec.shape[1], 1, har_spec.shape[2]),
        "har_phase": har_phase.numpy().astype(np.float32).reshape(1, har_phase.shape[1], 1, har_phase.shape[2]),
    }
    res = model.predict(inputs)
    key = list(res.keys())[0]
    x = res[key]
    with torch.no_grad():
        n_fft = dec.generator.post_n_fft
        x_t = torch.from_numpy(x)
        spec_t = torch.exp(x_t[:, : n_fft // 2 + 1, :])
        phase_t = torch.sin(x_t[:, n_fft // 2 + 1 :, :])
        audio = dec.generator.stft.inverse(spec_t, phase_t).squeeze().numpy()
    samples_per_f0_frame = len(audio) / float(f0_len)
    target_len = int(round(samples_per_f0_frame * t_f0))
    return audio[:target_len]


def decoder_har_grouped_impl(pipe: HybridTTSPipeline, vocoder_inputs: dict) -> np.ndarray | None:
    """Greedy large-bucket segmentation with minimal calls and seam crossfades."""
    if not getattr(pipe, "coreml_decoder_har_buckets", None):
        return None
    asr = vocoder_inputs["asr"].astype(np.float32)
    f0 = vocoder_inputs["f0_curve"].astype(np.float32)
    n = vocoder_inputs["n"].astype(np.float32)
    s = vocoder_inputs["s"].astype(np.float32)
    T_asr = asr.shape[-1]
    T_f0 = f0.shape[-1]
    bucket_secs = sorted(pipe.coreml_decoder_har_buckets.keys(), reverse=True)
    schedule = []
    f0_pos = 0
    while f0_pos < T_f0:
        chosen = None
        for sec in bucket_secs:
            model = pipe.coreml_decoder_har_buckets[sec]
            spec = model.get_spec()
            shapes = {i.name: i.type.multiArrayType.shape for i in spec.description.input}
            f0_len = int(shapes["f0_curve"][-1])
            if f0_len <= T_f0 - f0_pos or sec == bucket_secs[-1]:
                chosen = (sec, f0_len)
                break
        if chosen is None:
            break
        sec, f0_len = chosen
        overlap = max(0, int(round(0.1 * f0_len)))
        schedule.append((sec, f0_pos, f0_len))
        f0_pos += max(1, f0_len - overlap)
    out_audio = None
    acc = None
    samples_per_f0_frame = None
    dec = pipe.pytorch_model.decoder
    for sec, f0_start, f0_len in schedule:
        model = pipe.coreml_decoder_har_buckets[sec]
        spec = model.get_spec()
        shapes = {i.name: i.type.multiArrayType.shape for i in spec.description.input}
        asr_len = int(shapes["asr"][-1])
        f0_slice = np.zeros((1, f0_len), dtype=np.float32)
        n_slice = np.zeros((1, f0_len), dtype=np.float32)
        asr_slice = np.zeros((1, 512, asr_len), dtype=np.float32)
        asr_start = f0_start // 2
        t_f0_avail = max(0, min(f0_start + f0_len, T_f0) - f0_start)
        t_asr_avail = max(0, min(asr_start + asr_len, T_asr) - asr_start)
        if t_f0_avail > 0:
            f0_slice[:, :t_f0_avail] = f0[:, f0_start : f0_start + t_f0_avail]
            n_slice[:, :t_f0_avail] = n[:, f0_start : f0_start + t_f0_avail]
        if t_asr_avail > 0:
            asr_slice[:, :, :t_asr_avail] = asr[:, :, asr_start : asr_start + t_asr_avail]
        with torch.no_grad():
            f0_up = dec.generator.f0_upsamp(torch.from_numpy(f0_slice)[:, None]).transpose(1, 2)
            har_source, _, _ = dec.generator.m_source(f0_up)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = dec.generator.stft.transform(har_source)
        inputs = {
            "asr": asr_slice.reshape(1, 512, 1, asr_len),
            "f0_curve": f0_slice.reshape(1, 1, 1, f0_len),
            "n": n_slice.reshape(1, 1, 1, f0_len),
            "s": s,
            "har_spec": har_spec.numpy().astype(np.float32).reshape(1, har_spec.shape[1], 1, har_spec.shape[2]),
            "har_phase": har_phase.numpy().astype(np.float32).reshape(1, har_phase.shape[1], 1, har_phase.shape[2]),
        }
        res = model.predict(inputs)
        key = list(res.keys())[0]
        x = res[key]
        with torch.no_grad():
            n_fft = dec.generator.post_n_fft
            x_t = torch.from_numpy(x)
            spec_t = torch.exp(x_t[:, : n_fft // 2 + 1, :])
            phase_t = torch.sin(x_t[:, n_fft // 2 + 1 :, :])
            chunk = dec.generator.stft.inverse(spec_t, phase_t).squeeze().numpy()
        if samples_per_f0_frame is None:
            samples_per_f0_frame = max(1, int(round(len(chunk) / float(f0_len))))
            total_len = samples_per_f0_frame * T_f0
            out_audio = np.zeros((total_len,), dtype=np.float32)
            acc = np.zeros_like(out_audio)
        dst_start = f0_start * samples_per_f0_frame
        dst_end = dst_start + len(chunk)
        end_cap = min(dst_end, out_audio.shape[0])
        cl = end_cap - dst_start
        if cl > 0:
            hann = np.hanning(len(chunk)).astype(np.float32)
            out_audio[dst_start:end_cap] += chunk[:cl] * hann[:cl]
            acc[dst_start:end_cap] += hann[:cl]
    if out_audio is None:
        return None
    valid = acc > 1e-6
    audio = np.zeros_like(out_audio)
    audio[valid] = out_audio[valid] / acc[valid]
    final_len = samples_per_f0_frame * T_f0
    return audio[:final_len]


def vocoder_windows_impl(pipe: HybridTTSPipeline, vocoder_inputs: dict) -> np.ndarray | None:
    """Windowed CoreML vocoder (KokoroVocoder) with overlap-add."""
    if not pipe.use_coreml:
        print("❌ CoreML vocoder not available")
        return None
    if getattr(pipe, "coreml_vocoder", None) is None:
        print("❌ CoreML vocoder model not loaded")
        return None

    print("🍎 Running CoreML vocoder on ANE...")
    try:
        asr = vocoder_inputs["asr"].astype(np.float32)
        f0 = vocoder_inputs["f0_curve"].astype(np.float32)
        n = vocoder_inputs["n"].astype(np.float32)
        s = vocoder_inputs["s"].astype(np.float32)

        asr_win, f0_win = 200, 400
        T_asr = asr.shape[-1]
        T_f0 = f0.shape[-1]
        hop_f0 = f0_win // 4
        hop_asr = asr_win // 4
        num_windows = int(np.ceil((T_f0 - f0_win) / hop_f0)) + 1 if T_f0 > 0 else 0
        chunk_len = VOCODER_CHUNK_SAMPLES
        samples_per_f0_frame = chunk_len // f0_win
        hop_samples = hop_f0 * samples_per_f0_frame
        total_len = max(chunk_len, chunk_len + (num_windows - 1) * hop_samples)
        out_audio = np.zeros((total_len,), dtype=np.float32)
        hann = np.hanning(chunk_len).astype(np.float32)
        acc = np.zeros_like(out_audio)
        start_time = time.time()
        for w in range(num_windows):
            f0_start = w * hop_f0
            f0_end = f0_start + f0_win
            asr_start = w * hop_asr
            asr_end = asr_start + asr_win
            f0_slice = np.zeros((1, f0_win), dtype=np.float32)
            n_slice = np.zeros((1, f0_win), dtype=np.float32)
            asr_slice = np.zeros((1, 512, asr_win), dtype=np.float32)
            if f0_start < T_f0:
                f0_slice_len = max(0, min(f0_end, T_f0) - f0_start)
                if f0_slice_len > 0:
                    f0_slice[:, :f0_slice_len] = f0[:, f0_start : f0_start + f0_slice_len]
            if f0_start < n.shape[-1]:
                n_slice_len = max(0, min(f0_end, n.shape[-1]) - f0_start)
                if n_slice_len > 0:
                    n_slice[:, :n_slice_len] = n[:, f0_start : f0_start + n_slice_len]
            if asr_start < T_asr:
                asr_slice_len = max(0, min(asr_end, T_asr) - asr_start)
                if asr_slice_len > 0:
                    asr_slice[:, :, :asr_slice_len] = asr[:, :, asr_start : asr_start + asr_slice_len]

            cm_inputs = {
                "asr": asr_slice.reshape(1, 512, 1, asr_win),
                "f0_curve": f0_slice.reshape(1, 1, 1, f0_win),
                "n": n_slice.reshape(1, 1, 1, f0_win),
                "s": s,
            }
            result = pipe.coreml_vocoder.predict(cm_inputs)
            audio_key = "waveform" if "waveform" in result else list(result.keys())[0]
            chunk = result[audio_key].squeeze().astype(np.float32)
            dst_start = w * hop_samples
            dst_end = dst_start + chunk_len
            if dst_end > out_audio.shape[0]:
                extend = dst_end - out_audio.shape[0]
                out_audio = np.concatenate([out_audio, np.zeros((extend,), dtype=np.float32)])
                acc = np.concatenate([acc, np.zeros((extend,), dtype=np.float32)])
            out_audio[dst_start:dst_end] += chunk * hann
            acc[dst_start:dst_end] += hann
        end_time = time.time()
        valid_idx = np.nonzero(acc > 1e-6)[0]
        if valid_idx.size == 0:
            audio = out_audio[:0]
        else:
            last = valid_idx.max() + 1
            audio = out_audio[:last] / np.maximum(acc[:last], 1e-6)

        print(f"✅ CoreML vocoder completed in {end_time - start_time:.3f}s")
        print(f"  - Audio shape: {audio.shape}")
        print(f"  - Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio

    except Exception as e:
        print(f"❌ CoreML vocoder failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def pytorch_fallback_impl(pipe: HybridTTSPipeline, text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray | None:
    """Full PyTorch synthesis via KPipeline + KModel."""
    print("🔄 Running PyTorch fallback pipeline...")
    try:
        start_time = time.time()
        for _, phonemes, _ in pipe.pipeline(text, voice, speed):
            voice_pack = pipe.pipeline.load_voice(voice)
            ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)
            audio = pipe.pytorch_model(phonemes, ref_s, speed)
            end_time = time.time()
            print(f"✅ PyTorch fallback completed in {end_time - start_time:.3f}s")
            return audio.numpy()
    except Exception as e:
        print(f"❌ PyTorch fallback failed: {e}")
        import traceback

        traceback.print_exc()
        return None


DEFAULT_TEXT_BACKENDS: tuple[TextBackend, ...] = (
    decoder_har_post_bucket_impl,
    synth_bucket_impl,
    decoder_har_bucket_impl,
)

DEFAULT_VI_BACKENDS: tuple[ViBackend, ...] = (
    decoder_har_sliding_impl,
    vocoder_windows_impl,
)


def run_synthesis_chain(
    pipe: HybridTTSPipeline,
    text: str,
    voice: str,
    speed: float,
    *,
    text_backends: Sequence[TextBackend] | None = None,
    vi_backends: Sequence[ViBackend] | None = None,
) -> tuple[np.ndarray | None, int | None]:
    """Try text backends, then vi backends (after one ``extract_vocoder_inputs``), then PyTorch."""
    text_backends = tuple(text_backends) if text_backends is not None else DEFAULT_TEXT_BACKENDS
    vi_backends = tuple(vi_backends) if vi_backends is not None else DEFAULT_VI_BACKENDS

    print(f"\n🎵 Synthesizing: '{text}' (voice: {voice}, speed: {speed}x)")

    if pipe.use_coreml:
        for fn in text_backends:
            audio = fn(pipe, text, voice, speed)
            if audio is not None:
                return audio, 24000

        vocoder_inputs = pipe.extract_vocoder_inputs(text, voice, speed)
        if vocoder_inputs:
            for fn in vi_backends:
                audio = fn(pipe, vocoder_inputs)
                if audio is not None:
                    return audio, 24000

        print("⚠️ Hybrid pipeline failed, falling back to PyTorch")

    audio = pytorch_fallback_impl(pipe, text, voice, speed)
    if audio is not None:
        return audio, 24000

    print("❌ All synthesis methods failed")
    return None, None


__all__ = [
    "VOCODER_CHUNK_SAMPLES",
    "TextBackend",
    "ViBackend",
    "synth_bucket_impl",
    "decoder_har_post_bucket_impl",
    "decoder_har_sliding_impl",
    "decoder_har_bucket_impl",
    "decoder_har_grouped_impl",
    "vocoder_windows_impl",
    "pytorch_fallback_impl",
    "DEFAULT_TEXT_BACKENDS",
    "DEFAULT_VI_BACKENDS",
    "run_synthesis_chain",
]
