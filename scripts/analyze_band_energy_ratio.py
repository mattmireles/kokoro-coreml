#!/usr/bin/env python3
"""Band-energy ratio of a candidate WAV vs the PyTorch reference.

Reproduces the 30s spectral-tilt measurement from
README/Notes/cs1-audio-quality-evaluation-2026-07-14.md: RMS gain-match the
candidate to the reference, then report per-band energy ratios
(candidate / reference) via Welch PSD. A healthy render sits near 1.0x in all
bands; the reported 30s failure was 0.35-0.48x across 1-9 kHz.

Usage:
  .venv/bin/python scripts/analyze_band_energy_ratio.py \
      --reference outputs/audio-parity/references/pytorch_30s.wav \
      --candidate outputs/bakeoff/listen/staged/config_f_staged_30s.wav
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

BANDS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 9000), (9000, 12000)]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float64) / 32768.0, sr


def band_energies(x: np.ndarray, sr: int) -> dict[tuple[int, int], float]:
    # Hann-windowed averaged periodogram (Welch, 50% overlap) — numpy only,
    # scipy is not in the export venv.
    nperseg = 4096
    hop = nperseg // 2
    win = np.hanning(nperseg)
    frames = range(0, len(x) - nperseg + 1, hop)
    psd = np.zeros(nperseg // 2 + 1)
    for i in frames:
        seg = x[i:i + nperseg] * win
        psd += np.abs(np.fft.rfft(seg)) ** 2
    psd /= max(len(list(frames)), 1)
    f = np.fft.rfftfreq(nperseg, 1 / sr)
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[(lo, hi)] = float(psd[m].sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", nargs="+", required=True)
    args = ap.parse_args()

    ref, sr = read_wav(Path(args.reference))
    ref_e = band_energies(ref, sr)

    for cand_path in args.candidate:
        cand, csr = read_wav(Path(cand_path))
        assert csr == sr
        n = min(len(ref), len(cand))
        c = cand[:n].copy()
        r = ref[:n]
        # RMS gain-match candidate to reference (same as the judge script).
        c *= np.sqrt(np.mean(r**2) / max(np.mean(c**2), 1e-12))
        ce = band_energies(c, sr)
        print(f"\n{cand_path}")
        for band in BANDS:
            ratio = ce[band] / max(ref_e[band], 1e-18)
            print(f"  {band[0]/1000:4.0f}-{band[1]/1000:2.0f} kHz: {ratio:5.2f}x")


if __name__ == "__main__":
    main()
