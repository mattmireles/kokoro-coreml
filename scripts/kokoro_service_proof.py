#!/usr/bin/env python3
"""Local text-to-MP3 proof for the web-scraper Kokoro worker plan.

This is a Phase 0 proof command, not the production worker tokenizer. It uses
the Python Kokoro tokenizer/voice loader to prepare the same JSON contract that
the Swift Core ML benchmark already consumes, then invokes the Swift pipeline
and encodes the resulting WAV to MP3 with ffmpeg/libmp3lame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent

BAKEOFF_INPUTS = {
    "3s": "The quick brown fox jumps over the dog.",
    "7s": (
        "The morning sun cast long shadows across the garden as birds began "
        "their chorus in the ancient oak tree."
    ),
    "15s": (
        "The ancient lighthouse stood alone on the rocky cliff, its beam sweeping "
        "across dark waters with the patience of centuries. Ships had come and gone, "
        "storms had battered its walls, yet still it turned, guiding sailors home."
    ),
    "30s": (
        "When the last train departed that evening, the platform fell silent. "
        "The old stationmaster locked the ticket office with trembling hands, "
        "running his fingers along the worn counter where countless journeys "
        "had begun. Outside, autumn wind scattered golden leaves across the "
        "empty tracks. He had spent forty years here, watching the world rush "
        "past in a blur of faces and farewells. The station would stand a while "
        "longer, its clock still ticking, its roof sheltering the pigeons."
    ),
}

ENUM_SIZES = [32, 64, 128, 256, 320, 384, 512]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _run(args: list[str], *, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _resolve_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.text:
        return "custom", args.text
    if args.text_file:
        return args.text_file.stem, args.text_file.read_text(encoding="utf-8")
    if args.preset:
        return args.preset, BAKEOFF_INPUTS[args.preset]
    raise SystemExit("Provide --text, --text-file, or --preset.")


def _prepare_input(input_dir: Path, key: str, text: str, voice: str, speed: float) -> dict[str, Any]:
    from kokoro import KModel, KPipeline
    from kokoro.pipeline import voice_embedding_for_phoneme_string

    pipeline = KPipeline(lang_code="a")
    kmodel = KModel()
    voice_pack = pipeline.load_voice(voice)

    phonemes = None
    for _, ps, _ in pipeline(text, voice, speed):
        phonemes = ps
        break
    if not phonemes:
        raise RuntimeError("Kokoro tokenizer returned no phonemes")

    ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)
    input_ids = list(filter(lambda i: i is not None, map(lambda p: kmodel.vocab.get(p), phonemes)))
    input_ids = [0] + input_ids + [0]
    enum_t = next((size for size in ENUM_SIZES if size >= len(input_ids)), ENUM_SIZES[-1])
    padded_ids = input_ids[:enum_t] + [0] * max(0, enum_t - len(input_ids))
    attention_mask = [1] * min(len(input_ids), enum_t) + [0] * max(0, enum_t - len(input_ids))

    gen = kmodel.decoder.generator
    linear_weights = gen.m_source.l_linear.weight.detach().numpy().flatten().tolist()
    linear_bias = float(gen.m_source.l_linear.bias.detach().numpy().flatten()[0])
    hnsf_weights = {
        "linear_weights": linear_weights,
        "linear_bias": linear_bias,
        "weights_sha256": _json_sha256({"linear_weights": linear_weights, "linear_bias": linear_bias}),
    }
    (input_dir / "hnsf_weights.json").write_text(json.dumps(hnsf_weights, sort_keys=True), encoding="utf-8")

    ref_s_list = ref_s.cpu().numpy().flatten().tolist()
    entry = {
        "key": key,
        "text": text,
        "voice": voice,
        "speed": speed,
        "phonemes": phonemes,
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "ref_s": ref_s_list,
        "canonical_duration_s": None,
        "num_tokens": len(input_ids),
        "hnsf_weights_sha256": hnsf_weights["weights_sha256"],
        "text_sha256": _sha256_bytes(text.encode("utf-8")),
        "voice_embedding_sha256": _json_sha256(ref_s_list),
    }
    (input_dir / f"{key}.json").write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
    return entry


def _ffmpeg_version(ffmpeg: str) -> str:
    proc = _run([ffmpeg, "-hide_banner", "-version"], timeout=20)
    first = proc.stdout.splitlines()[0] if proc.stdout else ""
    if "libmp3lame" not in proc.stdout:
        raise RuntimeError(f"{ffmpeg} does not report libmp3lame support")
    return first


def _encode_mp3(ffmpeg: str, wav_path: Path, mp3_path: Path, bitrate_kbps: int) -> float:
    start = time.perf_counter()
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{bitrate_kbps}k",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(mp3_path),
        ],
        timeout=120,
    )
    return time.perf_counter() - start


def _probe_mp3(ffprobe: str, mp3_path: Path) -> dict[str, Any]:
    proc = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration,bit_rate",
            "-of",
            "json",
            str(mp3_path),
        ],
        timeout=30,
    )
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no audio stream in MP3")
    stream = streams[0]
    duration = float(stream.get("duration") or 0)
    if stream.get("codec_name") != "mp3" or duration <= 0:
        raise RuntimeError(f"invalid MP3 metadata: {stream}")
    return stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--preset", choices=sorted(BAKEOFF_INPUTS))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--models-dir", type=Path, default=_ROOT / "coreml")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "outputs" / "service_proof")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    parser.add_argument("--bitrate-kbps", type=int, default=128)
    parser.add_argument("--compute-units", default="all")
    args = parser.parse_args()

    key, text = _resolve_text(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kokoro-service-proof-") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "inputs"
        input_dir.mkdir()
        prepared = _prepare_input(input_dir, key, text, args.voice, args.speed)

        wav_path = output_dir / f"{key}.wav"
        mp3_path = output_dir / f"{key}.mp3"
        metrics_path = output_dir / f"{key}.swift.json"
        manifest_path = output_dir / f"{key}.manifest.json"

        swift_start = time.perf_counter()
        _run(
            [
                "swift",
                "run",
                "kokoro-bench",
                "--models-dir",
                str(args.models_dir),
                "--inputs-dir",
                str(input_dir),
                "--hnsf-weights",
                str(input_dir / "hnsf_weights.json"),
                "--input-key",
                key,
                "--output",
                str(metrics_path),
                "--wav",
                str(wav_path),
                "--compute-units",
                args.compute_units,
            ],
            cwd=_ROOT / "swift",
            timeout=600,
        )
        swift_wall_s = time.perf_counter() - swift_start

        ffmpeg_version = _ffmpeg_version(args.ffmpeg)
        encode_wall_s = _encode_mp3(args.ffmpeg, wav_path, mp3_path, args.bitrate_kbps)
        mp3_stream = _probe_mp3(args.ffprobe, mp3_path)
        swift_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        manifest = {
            "schemaVersion": 1,
            "producer": "kokoro-service-proof",
            "key": key,
            "textSha256": prepared["text_sha256"],
            "voice": args.voice,
            "speed": args.speed,
            "numTokens": prepared["num_tokens"],
            "phonemeCount": len(prepared["phonemes"]),
            "voiceEmbeddingSha256": prepared["voice_embedding_sha256"],
            "modelBundle": {
                "modelsDirectory": str(args.models_dir),
                "hnsfWeightsSha256": prepared["hnsf_weights_sha256"],
            },
            "swift": {
                "computeUnits": args.compute_units,
                "wallTimeSeconds": round(swift_wall_s, 6),
                "metrics": swift_metrics,
            },
            "encoder": {
                "path": args.ffmpeg,
                "version": ffmpeg_version,
                "codec": "libmp3lame",
                "bitrateKbps": args.bitrate_kbps,
                "sampleRate": 24000,
                "channels": 1,
                "wallTimeSeconds": round(encode_wall_s, 6),
            },
            "outputs": {
                "wav": {
                    "path": str(wav_path),
                    "sha256": _sha256_file(wav_path),
                    "bytes": wav_path.stat().st_size,
                },
                "mp3": {
                    "path": str(mp3_path),
                    "sha256": _sha256_file(mp3_path),
                    "bytes": mp3_path.stat().st_size,
                    "probe": mp3_stream,
                },
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"command failed: {' '.join(exc.cmd)}\n")
        if exc.stdout:
            sys.stderr.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
