#!/usr/bin/env python3
"""Prepare one Kokoro Swift/Core ML input JSON for arbitrary service text.

This script intentionally stops after text/voice preprocessing. The Swift
worker consumes the generated JSON and runs the Core ML synthesis path itself;
normal service requests must not shell out to ``kokoro-bench``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ENUM_SIZES = [32, 64, 128, 256, 320, 384, 512]
# packages/contracts MAX_TTS_CHUNK_TOKENS
MAX_CALLER_CHUNK_TOKENS = 450
REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime_root(arg_root: Path | None) -> Path:
    """Return the Kokoro runtime root for tokenizer/vocab asset lookup.

    Called by ``main`` and service mode before loading vocab or Python Kokoro
    modules. The default is this checkout when it contains repo-local runtime
    assets. ``KOKORO_COREML_ROOT`` and ``--runtime-root`` remain explicit
    overrides for Botnet or generated-runtime layouts.
    """
    if arg_root is not None:
        return arg_root.resolve()
    env_root = os.environ.get("KOKORO_COREML_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    if (REPO_ROOT / "_kokoro_vocab.json").exists():
        return REPO_ROOT
    return Path.cwd().resolve()


def _ensure_python_runtime_path(root: Path) -> None:
    """Put the runtime root on ``sys.path`` for direct script execution.

    Running ``python scripts/kokoro-prepare-input.py`` sets ``sys.path[0]`` to
    ``scripts/``. Without this helper, the local ``kokoro`` package is invisible
    unless the caller has installed it into the active Python environment.
    """
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _language_for_voice(voice: str) -> str:
    return "b" if voice.startswith("b") else "a"


def _load_vocab(root: Path) -> dict[str, int]:
    home_checkout = Path.home() / "Documents" / "GitHub" / "kokoro-coreml"
    candidates = [
        root / "_kokoro_vocab.json",
        root / "checkpoints" / "config.json",
        root / "outputs" / "hnsf_validation" / "config.json",
        home_checkout / "_kokoro_vocab.json",
        home_checkout / "checkpoints" / "config.json",
        home_checkout / "outputs" / "hnsf_validation" / "config.json",
    ]
    env_root = os.environ.get("KOKORO_COREML_ROOT", "").strip()
    if env_root:
        er = Path(env_root)
        candidates.extend(
            [
                er / "checkpoints" / "config.json",
                er / "outputs" / "hnsf_validation" / "config.json",
            ]
        )
    for candidate in candidates:
        try:
            if candidate.exists():
                config = json.loads(candidate.read_text(encoding="utf-8"))
                if "vocab" in config:
                    return config["vocab"]
        except OSError:
            continue

    raise RuntimeError(
        "Kokoro vocab config not found locally; set KOKORO_COREML_ROOT or run from the kokoro-coreml checkout"
    )


def _prepare_entries(
    *,
    texts: list[str],
    key: str,
    voice: str,
    speed: float,
    pipeline: Any,
    voice_pack: Any,
    vocab: dict[str, int],
) -> list[dict[str, Any]]:
    from kokoro.pipeline import voice_embedding_for_phoneme_string

    entries = []
    for index, text in enumerate(texts):
        phonemes = None
        with redirect_stdout(sys.stderr):
            for _, ps, _ in pipeline(text, voice, speed):
                phonemes = ps
                break

        if not phonemes:
            raise RuntimeError("Kokoro tokenizer returned no phonemes")

        with redirect_stdout(sys.stderr):
            ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)

        input_ids = [vocab[p] for p in phonemes if p in vocab]
        input_ids = [0] + input_ids + [0]
        if len(input_ids) > MAX_CALLER_CHUNK_TOKENS:
            raise RuntimeError(
                f"Kokoro chunk exceeds MAX_TTS_CHUNK_TOKENS ({MAX_CALLER_CHUNK_TOKENS}): "
                f"{len(input_ids)} tokens"
            )
        enum_t = next((size for size in ENUM_SIZES if size >= len(input_ids)), ENUM_SIZES[-1])
        padded_ids = input_ids + [0] * max(0, enum_t - len(input_ids))
        attention_mask = [1] * len(input_ids) + [0] * max(0, enum_t - len(input_ids))

        ref_s_list = ref_s.cpu().numpy().flatten().tolist()
        # Emit only what Swift `PreparedKokoroInput` decodes. Phonemes,
        # num_tokens, and the sha256s used to ship as debug metadata;
        # nothing in the worker or runtime read them, so they were dead
        # payload (and phonemes are an unnecessary phonetic mirror of
        # ``text``).
        entries.append({
            "key": key if len(texts) == 1 else f"{key}-{index:03d}",
            "text": text,
            "voice": voice,
            "speed": speed,
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "ref_s": ref_s_list,
            "canonical_duration_s": None,
        })
    return entries


def _run_serve(runtime_root: Path | None) -> int:
    root = _runtime_root(runtime_root)
    _ensure_python_runtime_path(root)

    from kokoro import KPipeline

    vocab = _load_vocab(root)
    pipelines: dict[str, Any] = {}
    voice_packs: dict[str, Any] = {}

    def pipeline_for(voice: str) -> tuple[Any, Any]:
        if voice not in pipelines:
            with redirect_stdout(sys.stderr):
                pipeline = KPipeline(lang_code=_language_for_voice(voice))
                voice_pack = pipeline.load_voice(voice)
            pipelines[voice] = pipeline
            voice_packs[voice] = voice_pack
        return pipelines[voice], voice_packs[voice]

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            request_id = str(request["id"])
            texts = request["texts"]
            voice = str(request["voice"])
            speed = float(request["speed"])
            if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
                raise ValueError("texts must be a string array")
            pipeline, voice_pack = pipeline_for(voice)
            entries = _prepare_entries(
                texts=texts,
                key="service",
                voice=voice,
                speed=speed,
                pipeline=pipeline,
                voice_pack=voice_pack,
                vocab=vocab,
            )
            response = {"id": request_id, "inputs": entries}
        except Exception as exc:  # noqa: BLE001 - service mode must isolate bad slides.
            request_id = locals().get("request_id", None)
            response = {"id": request_id, "error": str(exc)}
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--text-list-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--voice")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    if args.serve:
        return _run_serve(args.runtime_root)
    if (args.text_file is None) == (args.text_list_file is None):
        parser.error("provide exactly one of --text-file or --text-list-file")
    if args.output is None or args.key is None or args.voice is None or args.speed is None:
        parser.error("--output, --key, --voice, and --speed are required outside --serve")

    root = _runtime_root(args.runtime_root)
    _ensure_python_runtime_path(root)

    from kokoro import KPipeline

    texts = (
        [args.text_file.read_text(encoding="utf-8")]
        if args.text_file is not None
        else json.loads(args.text_list_file.read_text(encoding="utf-8"))
    )
    vocab = _load_vocab(root)

    entries = []
    with redirect_stdout(sys.stderr):
        pipeline = KPipeline(lang_code=_language_for_voice(args.voice))
        voice_pack = pipeline.load_voice(args.voice)

    entries = _prepare_entries(
        texts=texts,
        key=args.key,
        voice=args.voice,
        speed=args.speed,
        pipeline=pipeline,
        voice_pack=voice_pack,
        vocab=vocab,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = entries[0] if args.text_file is not None else entries
    args.output.write_text(json.dumps(output, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
