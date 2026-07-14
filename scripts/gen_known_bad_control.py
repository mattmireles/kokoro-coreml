#!/usr/bin/env python3
"""Generate a known-bad negative control clip for the audio-judge gate.

Purpose: prove the perceptual gate has discriminative power. If Gemini rates
this clip as unintelligible/corrupt while rating PyTorch and Config F clips as
clean speech, the gate is discriminating, not rubber-stamping.

The control is amplitude-modulated band-limited static shaped to a speech-like
envelope (so it is not trivially silent), 3.0 s, 24 kHz mono 16-bit PCM.
Deterministic (seed=0).

Output: outputs/audio-parity/references/known_bad_static_3s.wav
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_RATE = 24_000
_DURATION_S = 3.0
_SEED = 0


def main() -> int:
    rng = np.random.default_rng(_SEED)
    n = int(_SAMPLE_RATE * _DURATION_S)
    t = np.arange(n) / _SAMPLE_RATE

    # White noise, softened toward pink by a simple 1-pole low-pass.
    noise = rng.standard_normal(n).astype(np.float32)
    lp = np.zeros(n, dtype=np.float32)
    alpha = 0.6
    for i in range(1, n):
        lp[i] = alpha * lp[i - 1] + (1.0 - alpha) * noise[i]

    # Syllable-rate amplitude envelope (~4 Hz) so it reads as "speech-shaped
    # garbage" rather than steady hiss — a harder control for the gate.
    env = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t)).astype(np.float32)
    sig = lp * env
    sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.4  # match reference-ish peak

    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)
    out = _ROOT / "outputs" / "audio-parity" / "references" / "known_bad_static_3s.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    print(f"wrote {out} ({pcm.size} samples, {pcm.size / _SAMPLE_RATE:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
