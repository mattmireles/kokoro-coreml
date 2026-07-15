#!/usr/bin/env python3
"""Inspect the PRE-suppression Swift waveform inside punctuation-owned spans.

Reads waveform_raw_trimmed.f32 + pred_dur.i32 + tokens.i32 from a kokoro-bench
--dump-tensors directory and reports, for every span that
suppressPunctuationTokenAudio would silence, the RMS and peak of the raw
Core ML audio next to the PyTorch reference. Answers two questions:

1. Do the 2026-05-26 punctuation clicks still exist in the current pipeline?
   (peak inside punctuation spans >> neighboring speech floor -> clicks)
2. How much real speech energy does the whitespace suppression delete?

FORENSIC SCRIPT — INTENTIONALLY MODELS THE PRE-2026-07-14 BEHAVIOR. The
``suppressed()`` predicate below includes punctuation-adjacent whitespace,
matching the suppression logic that shipped BEFORE the 2026-07-14 fix. That
whitespace suppression was the 15s pause-elongation root cause and was
removed: shouldSuppressPunctuationSpan in
swift/Sources/KokoroPipeline/WaveformPostProcess.swift now suppresses
punctuation tokens only. Do not update this predicate — question 2 above is
precisely about what the OLD logic deleted. See
README/Notes/cs1-audio-quality-evaluation-2026-07-14.md for the fix.

Usage:
  .venv/bin/python scripts/analyze_raw_punct_spans.py \
      --dump outputs/audio-parity/punct-debug/tensors --key 15s
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PER_FRAME = 600
SILENT_PUNCT_IDS = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15}
WHITESPACE_ID = 16


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def read_i32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32)


def read_wav_norm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = data.astype(np.float64) / 32768.0
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def db(v: float) -> float:
    return 20 * np.log10(max(float(v), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--key", default="15s")
    args = ap.parse_args()

    dump = Path(args.dump)
    raw = read_f32(dump / "waveform_raw_trimmed.f32").astype(np.float64)
    peak = np.abs(raw).max()
    raw = raw / peak if peak > 0 else raw
    tokens = read_i32(dump / "tokens.i32")
    pred_dur = read_i32(dump / "pred_dur.i32")

    ref = read_wav_norm(ROOT / "outputs" / "audio-parity" / "references" / f"pytorch_{args.key}.wav")

    vocab = json.loads((ROOT / "_kokoro_vocab.json").read_text())["vocab"]
    rev = {v: k for k, v in vocab.items()}

    n = len(tokens)
    valid = [i for i in range(n) if not (tokens[i] == 0 and i > 0)]  # crude; report all nonzero-dur

    # PRE-2026-07-14 suppression span definition (punctuation + adjacent
    # whitespace) — intentionally NOT the current Swift logic; see docstring.
    def suppressed(i: int) -> bool:
        tid = int(tokens[i])
        if tid in SILENT_PUNCT_IDS:
            return True
        if tid != WHITESPACE_ID:
            return False
        prev_p = i > 0 and int(tokens[i - 1]) in SILENT_PUNCT_IDS
        next_p = i + 1 < n and int(tokens[i + 1]) in SILENT_PUNCT_IDS
        return prev_p or next_p

    print(f"raw={len(raw)} samples, ref={len(ref)} samples, tokens={n}")
    print(f"\n{'tok':>4} {'ph':>4} {'span (s)':>16} "
          f"{'rawRMS':>7} {'rawPK':>7} {'refRMS':>7} {'refPK':>7}  (dBFS)")
    frame_start = 0
    for i in range(n):
        f = int(pred_dur[i]) if i < len(pred_dur) else 0
        if f > 0 and suppressed(i):
            s0 = frame_start * SAMPLES_PER_FRAME
            s1 = min((frame_start + f) * SAMPLES_PER_FRAME, len(raw))
            seg_raw = raw[s0:s1]
            seg_ref = ref[s0:min(s1, len(ref))]
            ph = rev.get(int(tokens[i]), "?")
            print(f"{i:4d} {ph!r:>4} {s0/24000:7.2f}-{s1/24000:7.2f} "
                  f"{db(np.sqrt(np.mean(seg_raw**2))):7.1f} {db(np.abs(seg_raw).max()):7.1f} "
                  f"{db(np.sqrt(np.mean(seg_ref**2))):7.1f} {db(np.abs(seg_ref).max()):7.1f}")
        frame_start += max(0, int(pred_dur[i])) if i < len(pred_dur) else 0


if __name__ == "__main__":
    main()
