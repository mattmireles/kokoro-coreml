#!/usr/bin/env python3
"""Measure audio energy inside punctuation-owned duration spans, per arm.

Companion to scripts/probe_duration_pause_parity.py. That probe showed the
Core ML duration model matches PyTorch pred_dur to within one frame on the 15s
input, so the 40-110 ms pause elongation reported in
README/Notes/cs1-audio-quality-evaluation-2026-07-14.md cannot come from
duration frames. The suspect is Swift's suppressPunctuationTokenAudio
(swift/Sources/KokoroPipeline/WaveformPostProcess.swift), which hard-zeros
punctuation-owned spans (+adjacent whitespace) with a 5 ms fade, while the
PyTorch reference renders those spans as natural low-level decay.

This script builds token->sample spans from PyTorch pred_dur and reports the
RMS (dBFS, after peak normalization) inside each punctuation-owned span for
both WAVs. If the hypothesis holds, PyTorch shows finite low-level energy and
Config F shows digital zero.

Usage:
  .venv/bin/python scripts/analyze_punctuation_span_energy.py --key 15s
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

SAMPLES_PER_FRAME = 600  # 24 kHz / 40 fps duration frames
SILENT_PUNCT_IDS = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15}
WHITESPACE_ID = 16


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = data.astype(np.float64) / 32768.0
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    r = float(np.sqrt(np.mean(x * x)))
    return 20 * np.log10(max(r, 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="15s")
    ap.add_argument("--coreml-wav", default=None,
                    help="override Config F wav path (default: staged render)")
    args = ap.parse_args()

    data = json.loads((ROOT / "outputs" / "swift_bench_inputs" / f"{args.key}.json").read_text())
    n = int(data["num_tokens"])
    input_ids = data["input_ids"][:n]
    ref_s = np.asarray(data["ref_s"], dtype=np.float32).reshape(1, 256)
    vocab = json.loads((ROOT / "_kokoro_vocab.json").read_text())["vocab"]
    rev = {v: k for k, v in vocab.items()}

    from probe_duration_pause_parity import pytorch_eager_pred_dur
    from kokoro.model import KModel
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
    pred_dur, _ = pytorch_eager_pred_dur(kmodel, input_ids, ref_s, float(data.get("speed", 1.0)))

    ref_wav = read_wav(ROOT / "outputs" / "audio-parity" / "references" / f"pytorch_{args.key}.wav")
    cml_path = Path(args.coreml_wav) if args.coreml_wav else (
        ROOT / "outputs" / "bakeoff" / "listen" / "staged" / f"config_f_staged_{args.key}.wav")
    cml_wav = read_wav(cml_path)
    print(f"ref={len(ref_wav)} samples, coreml={len(cml_wav)} samples ({cml_path.name})")

    # Token spans owned by punctuation or punctuation-adjacent whitespace
    # (mirrors shouldSuppressPunctuationSpan in WaveformPostProcess.swift).
    def suppressed(i: int) -> bool:
        tid = input_ids[i]
        if tid in SILENT_PUNCT_IDS:
            return True
        if tid != WHITESPACE_ID:
            return False
        prev_p = i > 0 and input_ids[i - 1] in SILENT_PUNCT_IDS
        next_p = i + 1 < n and input_ids[i + 1] in SILENT_PUNCT_IDS
        return prev_p or next_p

    frame_start = 0
    print(f"\n{'tok':>4} {'ph':>4} {'span (s)':>16} {'frames':>6} {'ref dBFS':>9} {'coreml dBFS':>11}")
    for i in range(n):
        f = int(pred_dur[i])
        if f > 0 and suppressed(i):
            s0 = frame_start * SAMPLES_PER_FRAME
            s1 = (frame_start + f) * SAMPLES_PER_FRAME
            r = rms_db(ref_wav[s0:s1])
            c = rms_db(cml_wav[s0:s1])
            ph = rev.get(input_ids[i], "?")
            print(f"{i:4d} {ph!r:>4} {s0/24000:7.2f}-{s1/24000:7.2f} {f:6d} {r:9.1f} {c:11.1f}")
        frame_start += f


if __name__ == "__main__":
    main()
