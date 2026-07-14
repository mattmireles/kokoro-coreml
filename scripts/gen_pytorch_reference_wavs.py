#!/usr/bin/env python3
"""Render PyTorch eager (CPU) reference WAVs for the frozen bakeoff inputs.

This is the QUALITY REFERENCE arm for the Case Study 1 perceptual evaluation
(README/Notes/cs1-audio-quality-evaluation-2026-07-14.md). It reuses the exact
same single-shot inference path as the bakeoff harness Config E (PyTorch CPU
eager): identical phonemes and voice embedding as the Swift Config F path, so
PyTorch vs Config F is apples-to-apples on the same token sequence.

Reuses:
- ``scripts/bakeoff_harness.PyTorchContext("cpu", "e")`` (KModel().to("cpu").eval()).
- ``scripts/bakeoff_harness._run_pytorch`` (kmodel(phonemes, ref_s, speed) single shot).
- ``scripts/bakeoff_harness.BAKEOFF_INPUTS`` (frozen text), VOICE=af_heart, SPEED=1.0.

Output: outputs/audio-parity/references/pytorch_{3s,7s,15s,30s}.wav (24 kHz mono 16-bit PCM).
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bakeoff_harness import (  # noqa: E402
    BAKEOFF_INPUTS,
    SPEED,
    VOICE,
    PyTorchContext,
    _run_pytorch,
)

_SAMPLE_RATE = 24_000


def _write_wav_16bit(path: Path, audio: np.ndarray) -> None:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    # Kokoro output is already ~[-1, 1]; clip defensively without renormalizing.
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    print(
        f"  wrote {path.name}: {pcm.size} samples "
        f"({pcm.size / _SAMPLE_RATE:.3f}s), peak={peak:.4f}"
    )


def main() -> int:
    out_dir = _ROOT / "outputs" / "audio-parity" / "references"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building PyTorch CPU eager context (Config E)...")
    ctx = PyTorchContext("cpu", "e")

    for key, text in BAKEOFF_INPUTS.items():
        print(f"\n--- {key} (voice={VOICE}, speed={SPEED}) ---")
        audio = _run_pytorch(ctx, text)
        if audio is None:
            print(f"  FAILED to synthesize {key}", file=sys.stderr)
            return 1
        _write_wav_16bit(out_dir / f"pytorch_{key}.wav", audio)

    print(f"\nAll PyTorch reference WAVs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
