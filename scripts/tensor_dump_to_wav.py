#!/usr/bin/env python3
"""Write the waveform of a ``kokoro-bench --dump-tensors`` dump as a raw 16-bit WAV.

``kokoro-bench --wav`` peak-normalises before writing, so its WAVs cannot be
compared for level. This writes the pipeline's own waveform unchanged: 24 kHz
mono 16-bit PCM, clipped at +-1 with no renormalisation, the convention
``scripts/gen_pytorch_reference_wavs.py`` uses for the PyTorch reference clips.

Usage::

    uv run python scripts/tensor_dump_to_wav.py outputs/audio-parity/tensors/swift_3s out/3s.wav
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_parity_tensor_io import load_tensor_dump  # noqa: E402

SAMPLE_RATE = 24_000


def write_raw_wav(path: Path, waveform: np.ndarray) -> None:
    pcm = (np.clip(waveform.astype(np.float32).ravel(), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path, help="tensor dump directory written by kokoro-bench --dump-tensors")
    parser.add_argument("wav", type=Path, help="output WAV path")
    parser.add_argument("--name", default="waveform", help="tensor to write (default: waveform)")
    args = parser.parse_args()

    _, tensors = load_tensor_dump(args.dump)
    x = np.asarray(tensors[args.name], np.float32).ravel()
    args.wav.parent.mkdir(parents=True, exist_ok=True)
    write_raw_wav(args.wav, x)
    print(f"{args.wav}: {x.size} samples ({x.size / SAMPLE_RATE:.3f} s), rms {np.sqrt(np.mean(x ** 2)):.4f}, peak {np.abs(x).max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
