#!/usr/bin/env python3
"""Benchmark laishere/kokoro-coreml against the external bakeoff schema."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VOICE,
    error_record,
    result_file_payload,
    result_record,
    sha256_bytes,
    validate_manifest,
    validate_result_payload,
    load_json,
    write_wav_mono16,
    write_json,
)

SR = 24000
MODEL_NAMES = (
    "KokoroAlbert.mlpackage",
    "KokoroPostAlbert.mlpackage",
    "KokoroAlignment.mlpackage",
    "KokoroProsody.mlpackage",
    "KokoroNoise.mlpackage",
    "KokoroVocoder.mlpackage",
    "KokoroTail.mlpackage",
)


def _phonemize(pipe, text: str) -> str:
    _, tokens = pipe.g2p(text)
    pieces: list[str] = []
    for token in tokens:
        if token.phonemes:
            pieces.append(token.phonemes)
            if token.whitespace:
                pieces.append(" ")
    return "".join(pieces).strip()[:510]


def _encode_phonemes(model, phonemes: str):
    import torch

    ids = [model.vocab[p] for p in phonemes if model.vocab.get(p) is not None]
    return torch.LongTensor([[0, *ids, 0]])


class LaishereCoreMLChain:
    def __init__(self, *, models_dir: Path, voice: str, max_frames: int) -> None:
        import coremltools as ct
        from kokoro import KModel
        from kokoro.pipeline import KPipeline

        missing = [name for name in MODEL_NAMES if not (models_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing laishere Core ML packages: {', '.join(missing)}")

        self.models_dir = models_dir
        self.voice = voice
        self.max_frames = max_frames
        self.model = KModel()
        self.model.eval()
        self.pipe = KPipeline(lang_code="a", model=self.model)
        self.voice_pack = self.pipe.load_voice(voice)

        cu_ne = ct.ComputeUnit.CPU_AND_NE
        cu_all = ct.ComputeUnit.ALL
        self.m_albert = ct.models.MLModel(str(models_dir / "KokoroAlbert.mlpackage"), compute_units=cu_ne)
        self.m_post = ct.models.MLModel(str(models_dir / "KokoroPostAlbert.mlpackage"), compute_units=cu_ne)
        self.m_align = ct.models.MLModel(str(models_dir / "KokoroAlignment.mlpackage"), compute_units=cu_ne)
        self.m_pros = ct.models.MLModel(str(models_dir / "KokoroProsody.mlpackage"), compute_units=cu_ne)
        self.m_noise = ct.models.MLModel(str(models_dir / "KokoroNoise.mlpackage"), compute_units=cu_all)
        self.m_voc = ct.models.MLModel(str(models_dir / "KokoroVocoder.mlpackage"), compute_units=cu_ne)
        self.m_tail = ct.models.MLModel(str(models_dir / "KokoroTail.mlpackage"), compute_units=cu_all)

    def prepare(self, text: str) -> dict:
        import torch

        phonemes = _phonemize(self.pipe, text)
        input_ids = _encode_phonemes(self.model, phonemes)
        t_enc = int(input_ids.shape[1])
        ref_idx = max(min(len(phonemes) - 1, self.voice_pack.shape[0] - 1), 0)
        ref_s = self.voice_pack[ref_idx]
        style_s = ref_s[:, 128:]
        style_timbre = ref_s[:, :128]
        attention_mask = torch.ones(1, t_enc, dtype=torch.int32)
        return {
            "phonemes": phonemes,
            "t_enc": t_enc,
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attention_mask.numpy().astype(np.int32),
            "style_s": style_s.numpy().astype(np.float16),
            "style_timbre_f16": style_timbre.numpy().astype(np.float16),
            "style_timbre_f32": style_timbre.numpy().astype(np.float32),
        }

    def synthesize_once(self, prepared: dict) -> tuple[np.ndarray, int]:
        o1 = self.m_albert.predict({
            "input_ids": prepared["input_ids"],
            "attention_mask": prepared["attention_mask"],
        })
        o2 = self.m_post.predict({
            "bert_dur": np.array(o1["bert_dur"]).astype(np.float16),
            "input_ids": prepared["input_ids"],
            "style_s": prepared["style_s"],
            "speed": np.array([1.0], dtype=np.float16),
            "attention_mask": prepared["attention_mask"],
        })
        duration = np.array(o2["duration"]).flatten()
        pred_dur = np.round(duration).clip(min=1).astype(np.int32).reshape(1, -1)
        t_a = int(pred_dur.sum())
        if t_a > self.max_frames:
            raise ValueError(f"T_a={t_a} exceeds converted max_frames={self.max_frames}")
        o3 = self.m_align.predict({
            "pred_dur": pred_dur,
            "d": np.array(o2["d"]).astype(np.float16),
            "t_en": np.array(o2["t_en"]).astype(np.float16),
        })
        o4 = self.m_pros.predict({
            "en": np.array(o3["en"]).astype(np.float16),
            "style_s": prepared["style_s"],
        })
        o5 = self.m_noise.predict({
            "F0_curve": np.array(o4["F0"]).astype(np.float32),
            "style_timbre": prepared["style_timbre_f32"],
        })
        o6 = self.m_voc.predict({
            "asr": np.array(o3["asr"]).astype(np.float16),
            "F0_curve": np.array(o4["F0"]).astype(np.float16),
            "N_pred": np.array(o4["N"]).astype(np.float16),
            "x_source_0": np.array(o5["x_source_0"]).astype(np.float16),
            "x_source_1": np.array(o5["x_source_1"]).astype(np.float16),
            "style_timbre": prepared["style_timbre_f16"],
        })
        o7 = self.m_tail.predict({"x_pre": np.array(o6["x_pre"]).astype(np.float32)})
        return np.array(o7["audio"]).flatten().astype(np.float32), t_a


def _timed_synthesize(chain: LaishereCoreMLChain, prepared: dict) -> tuple[float, np.ndarray, int]:
    start = time.perf_counter()
    audio, t_a = chain.synthesize_once(prepared)
    return time.perf_counter() - start, audio, t_a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--laishere-repo", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--input-key", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--spotcheck-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    keys = args.input_key or list(manifest["inputs"].keys())
    repo = args.laishere_repo.resolve()
    models_dir = (args.models_dir or (repo / "output")).resolve()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    spotcheck_dir = args.spotcheck_dir or (
        DEFAULT_OUTPUT_DIR / "spotcheck_wavs" / f"laishere_kokoro_coreml_{args.machine_id}"
    )

    chain = LaishereCoreMLChain(models_dir=models_dir, voice=args.voice, max_frames=args.max_frames)

    records = []
    for key in keys:
        item = manifest["inputs"][key]
        try:
            prepared = chain.prepare(item["text"])
            cold, audio, t_a = _timed_synthesize(chain, prepared)
            warm_times = []
            last_audio = audio
            last_t_a = t_a
            for _ in range(args.iterations):
                elapsed, last_audio, last_t_a = _timed_synthesize(chain, prepared)
                warm_times.append(elapsed)
            observed = float(last_audio.size) / float(SR)
            spotcheck_wav = spotcheck_dir / f"{key}.wav"
            write_wav_mono16(spotcheck_wav, last_audio, SR)
            records.append(
                result_record(
                    impl="laishere-kokoro-coreml",
                    framework="Core ML Python",
                    hardware_target="ANE/Core ML split pipeline",
                    version=sha,
                    machine_id=args.machine_id,
                    input_key=key,
                    text=item["text"],
                    voice=args.voice,
                    cold_wall_time_s=cold,
                    warm_wall_times_s=warm_times,
                    canonical_audio_duration_s=float(item["canonical_duration_s"]),
                    observed_audio_duration_s=observed,
                    output_sha256=sha256_bytes(last_audio.astype(np.float32).tobytes()),
                    provenance={
                        "laishere_repo": str(repo),
                        "models_dir": str(models_dir),
                        "sample_rate": SR,
                        "spotcheck_wav": str(spotcheck_wav),
                        "max_frames": args.max_frames,
                        "t_enc": prepared["t_enc"],
                        "t_a": last_t_a,
                        "phoneme_count": len(prepared["phonemes"]),
                        "timing_boundary": "Core ML chain only; G2P and feed preparation are outside timed calls, matching laishere benchmark.py.",
                    },
                )
            )
        except Exception as exc:
            records.append(
                error_record(
                    impl="laishere-kokoro-coreml",
                    framework="Core ML Python",
                    hardware_target="ANE/Core ML split pipeline",
                    version=sha,
                    machine_id=args.machine_id,
                    input_key=key,
                    text=item["text"],
                    voice=args.voice,
                    canonical_audio_duration_s=float(item["canonical_duration_s"]),
                    provenance={
                        "laishere_repo": str(repo),
                        "models_dir": str(models_dir),
                        "max_frames": args.max_frames,
                        "spotcheck_dir": str(spotcheck_dir),
                    },
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    payload = result_file_payload(
        impl="laishere-kokoro-coreml",
        machine_id=args.machine_id,
        records=records,
        provenance={
            "laishere_sha": sha,
            "models_dir": str(models_dir),
            "spotcheck_dir": str(spotcheck_dir),
            "max_frames": args.max_frames,
        },
    )
    validate_result_payload(payload)
    output = args.output or (DEFAULT_OUTPUT_DIR / f"results_laishere_kokoro_coreml_{args.machine_id}.json")
    write_json(output, payload)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
