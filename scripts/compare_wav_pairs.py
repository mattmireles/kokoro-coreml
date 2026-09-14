#!/usr/bin/env python3
"""Compare same-named WAVs in two directories: level, sample agreement, spectral agreement.

Sample-level correlation and SNR between the PyTorch and Core ML pipelines are
near zero even when the audio is close, because the Swift harmonic source
accumulates phase in double precision and PyTorch in fp32. The spectral
metrics here are phase-blind and computed on unit-RMS signals so loudness is
reported separately rather than dominating them:

- ``corr`` / ``snr_db``: sample-level, SNR after a least-squares gain match
  (``gain`` is the fitted B-to-A scale)
- ``logmel_corr`` / ``logmel_l1``: 64-band log-mel spectrogram agreement
- ``envelope_corr``: correlation of 12.5 ms frame RMS envelopes

Usage::

    uv run python scripts/compare_wav_pairs.py outputs/audio/pytorch outputs/audio/candidate --json report.json
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz mono")
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        return np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0
    if width == 4:
        return np.frombuffer(raw, np.float32).astype(np.float64)
    raise ValueError(f"{path}: unsupported sample width {width}")


def log_mel(x: np.ndarray, n_fft: int = 1024, hop: int = 256, n_mels: int = 64) -> np.ndarray:
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (n_fft // 2, n_fft // 2)), n_fft)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    mel = 2595 * np.log10(1 + freqs / 700)
    edges = np.linspace(mel.min(), mel.max(), n_mels + 2)
    bank = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, mid, hi = edges[i : i + 3]
        bank[i] = np.clip(np.minimum((mel - lo) / (mid - lo + 1e-9), (hi - mel) / (hi - mid + 1e-9)), 0, 1)
    return np.log10(power @ bank.T + 1e-8)


def stats(x: np.ndarray) -> dict:
    return {"samples": int(x.size), "rms": round(float(np.sqrt(np.mean(x ** 2))), 4),
            "peak": round(float(np.abs(x).max()), 4), "clipped_fraction": round(float(np.mean(np.abs(x) >= 0.999)), 6)}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    gain = float(np.dot(a, b) / (np.dot(b, b) + 1e-12))
    snr = float(10 * np.log10(np.sum(a ** 2) / (np.sum((a - gain * b) ** 2) + 1e-12)))
    an = a / (np.sqrt(np.mean(a ** 2)) + 1e-12)
    bn = b / (np.sqrt(np.mean(b ** 2)) + 1e-12)
    ma, mb = log_mel(an), log_mel(bn)
    frames = n // 300 * 300
    ea = np.sqrt(np.mean(an[:frames].reshape(-1, 300) ** 2, axis=1))
    eb = np.sqrt(np.mean(bn[:frames].reshape(-1, 300) ** 2, axis=1))
    return {"common_samples": n, "corr": round(corr, 5), "snr_db": round(snr, 2), "gain": round(gain, 4),
            "logmel_l1": round(float(np.mean(np.abs(ma - mb))), 4), "logmel_corr": round(float(np.corrcoef(ma.ravel(), mb.ravel())[0, 1]), 4),
            "envelope_corr": round(float(np.corrcoef(ea, eb)[0, 1]), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir_a", type=Path, help="reference directory (A)")
    parser.add_argument("dir_b", type=Path, help="candidate directory (B)")
    parser.add_argument("--json", type=Path, help="write the per-file report here")
    args = parser.parse_args()

    names = sorted(p.name for p in args.dir_a.glob("*.wav") if (args.dir_b / p.name).exists())
    if not names:
        parser.error("no WAVs with matching names in both directories")
    report = {}
    print(f"{'file':14s} {'lenA':>7s} {'lenB':>7s} {'rmsA':>6s} {'rmsB':>6s} {'corr':>7s} {'snr_dB':>7s} {'gain':>6s} {'melcorr':>7s} {'envcorr':>7s}")
    for name in names:
        a, b = read_wav(args.dir_a / name), read_wav(args.dir_b / name)
        sa, sb, c = stats(a), stats(b), compare(a, b)
        report[name] = {"a": sa, "b": sb, "compare": c}
        print(f"{name:14s} {sa['samples']:7d} {sb['samples']:7d} {sa['rms']:6.3f} {sb['rms']:6.3f} {c['corr']:7.3f} {c['snr_db']:7.2f} {c['gain']:6.3f} {c['logmel_corr']:7.4f} {c['envelope_corr']:7.4f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
