#!/usr/bin/env python3
"""Direct Gemini audio judge — TTS fallback for the audio-judge skill.

The primary path is `llm-workflows` `run-audio-judge.mjs`, which routes clips
through the FFmpeg media-prep worker (MP3 normalization, R2, Gemini Files
upload) and runs `audio_judge_v1`. When that worker is down, this script
preserves the skill's intent — Gemini LISTENS to the clips and applies the
Kokoro TTS rubric — by uploading WAVs straight to the Gemini Files API and
asking for a strict-JSON verdict.

Rubric (from `scripts/audio_quality_probe.py` and bakeoff listening practice):
intelligible speech, natural prosody, no whispering/static/clicks/dropouts,
spoken text matches the input prompt when provided.

Usage:
  uv run --no-sync python scripts/gemini_audio_judge_direct.py \
    --clip pytorch=/abs/path/pytorch_3s.wav \
    --clip coreml=/abs/path/config_f_3s.wav \
    --baseline-label pytorch \
    --prompt "The quick brown fox jumps over the dog." \
    --context-file /abs/path/context.md \
    --output outputs/audio-judge-fallback/my_run.json

Reads GEMINI_API_KEY from the environment or ``LLM_WORKFLOWS_ENV`` (default:
sibling ``../llm-workflows/.env``). WAV inputs are converted to 16-bit PCM via
``afconvert`` before upload (float32 WAVs are common in Swift bench exports).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

BASE = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "models/gemini-3.1-pro-preview"
DEFAULT_EXPECTED_STYLE = (
    "clear intelligible English speech, natural prosody, no whispering or static, "
    "no clicks or dropouts, 24 kHz mono"
)
ALLOWED_CLIP_SUFFIXES = {".wav", ".mp3", ".mpeg", ".mpga"}
MAX_CLIP_BYTES = 50 * 1024 * 1024
MAX_CONTEXT_BYTES = 32_768
HTTP_TIMEOUT_SECONDS = 120
GENERATE_TIMEOUT_SECONDS = 600
FILE_POLL_INTERVAL_SECONDS = 2
MAX_FILE_POLL_ATTEMPTS = 600

PROMPT_TEMPLATE = Template("""You are an expert speech-quality listener judging Kokoro TTS output.

All clips were synthesized from the SAME input text: "$prompt".
Expected style: $expected_style.
The baseline (quality reference) clip is labeled "$baseline".

$context

Listen to each clip fully. Ignore loudness differences. Judge:

1. Per clip: verdict "pass" or "fail" for: intelligible English speech with
   natural prosody; spoken content matches the input text; no whispering,
   static, metallic buzz, or chaotic "broken radio" texture; no periodic
   clicks; no dropouts or long silence gaps; no clipping. Briefly describe
   what you hear.
2. For each non-baseline clip: is it in the same broad quality class as the
   baseline (recognizably the same kind of TTS, acceptable to ship as a
   voice app's output), even if the baseline is somewhat better?
3. Note remaining gaps vs the baseline worth engineering effort.

Reply with STRICT JSON only:
{
  "clips": {
    "<label>": {"verdict": "pass|fail", "description": "...", "artifacts": ["..."]}
  },
  "comparisons": {
    "<label>_vs_$baseline": {"same_quality_class": true, "notes": "..."}
  },
  "overallVerdict": "pass|fail",
  "summary": "..."
}
""")


def llm_workflows_env_path() -> Path:
    """Resolve the llm-workflows ``.env`` path from env or sibling checkout."""
    override = os.environ.get("LLM_WORKFLOWS_ENV", "").strip()
    if override:
        return Path(override).expanduser()
    return _REPO_ROOT.parent / "llm-workflows" / ".env"


def gemini_api_key() -> str:
    """Resolve Gemini API key from env or llm-workflows/.env."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_path = llm_workflows_env_path()
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip()
                if "#" in value:
                    value = value.split("#", 1)[0].strip()
                return value.strip("\"'")
    raise SystemExit(
        "GEMINI_API_KEY not set. Export it or set LLM_WORKFLOWS_ENV to a .env "
        "file containing GEMINI_API_KEY."
    )


def parse_clip_args(items: list[str]) -> dict[str, Path]:
    """Parse ``--clip label=path`` pairs and reject duplicate labels."""
    clips: dict[str, Path] = {}
    for item in items:
        label, sep, raw_path = item.partition("=")
        label = label.strip()
        if not sep or not label or not raw_path.strip():
            raise SystemExit(f"--clip must be label=path, got: {item}")
        if label in clips:
            raise SystemExit(f"duplicate --clip label: {label}")
        clips[label] = Path(raw_path.strip()).expanduser()
    return clips


def validate_clip_path(label: str, path: Path) -> Path:
    """Ensure a clip path exists, is audio, and is within size limits."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"--clip {label}: file not found: {resolved}")
    if resolved.suffix.lower() not in ALLOWED_CLIP_SUFFIXES:
        raise SystemExit(
            f"--clip {label}: unsupported suffix {resolved.suffix!r}; "
            f"allowed: {sorted(ALLOWED_CLIP_SUFFIXES)}"
        )
    size = resolved.stat().st_size
    if size <= 0:
        raise SystemExit(f"--clip {label}: file is empty: {resolved}")
    if size > MAX_CLIP_BYTES:
        raise SystemExit(
            f"--clip {label}: file exceeds {MAX_CLIP_BYTES} bytes: {resolved}"
        )
    return resolved


def read_context_file(path: Path) -> str:
    """Read a bounded context file for the judge prompt."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"--context-file not found: {resolved}")
    data = resolved.read_bytes()
    if len(data) > MAX_CONTEXT_BYTES:
        raise SystemExit(
            f"--context-file exceeds {MAX_CONTEXT_BYTES} bytes: {resolved}"
        )
    return "Run context from the engineer:\n" + data.decode("utf-8")


def resolve_output_path(path: Path) -> Path:
    """Restrict report output to paths inside this repository."""
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (_REPO_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(f"--output must be inside repo root {_REPO_ROOT}") from exc
    return resolved


def build_prompt(*, prompt: str, expected_style: str, baseline: str, context: str) -> str:
    """Build the Gemini judge prompt without interpreting user braces."""
    return PROMPT_TEMPLATE.substitute(
        prompt=prompt,
        expected_style=expected_style,
        baseline=baseline,
        context=context,
    )


def to_pcm16(path: Path, workdir: Path) -> Path:
    """Convert WAV to 16-bit PCM (Gemini-safe; bench tooling may write float32)."""
    if path.suffix.lower() != ".wav":
        return path
    converted = workdir / f"{path.stem}_lei16.wav"
    try:
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(converted)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"afconvert failed for {path}: {exc.stderr or exc.stdout or exc}"
        ) from exc
    return converted


def read_wav_rms(path: Path) -> float:
    """Return RMS of a mono 16-bit PCM WAV."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM WAV for gain match")
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if pcm.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm * pcm)))


def gain_match_clip(path: Path, target_rms: float, workdir: Path) -> Path:
    """Scale one WAV to the target RMS and write a temporary LEI16 file."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise SystemExit(f"{path}: gain match requires 16-bit PCM WAV")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    current_rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
    if current_rms <= 0.0 or target_rms <= 0.0:
        return path
    scaled = np.clip(pcm * (target_rms / current_rms), -32767, 32767).astype(np.int16)
    out = workdir / f"{path.stem}_gainmatched.wav"
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())
    return out


def probe_clips(clips: dict[str, Path], baseline_label: str) -> None:
    """Reject clips that fail the objective speech-health gate."""
    from audio_quality_probe import classify_metrics, compute_metrics, derive_thresholds

    baseline_metrics = compute_metrics(clips[baseline_label])
    thresholds = derive_thresholds([baseline_metrics])
    for label, path in clips.items():
        metrics = compute_metrics(path)
        decision, reasons = classify_metrics(
            metrics,
            thresholds,
            is_reference=label == baseline_label,
        )
        if decision == "reject_without_listening":
            raise SystemExit(
                f"--probe-first rejected {label} ({path}): {', '.join(reasons)}"
            )


def upload(key: str, path: Path, display_name: str) -> tuple[str, str]:
    """Upload one audio file to the Gemini Files API."""
    data = path.read_bytes()
    mime = "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"
    start = urllib.request.Request(
        f"{BASE}/upload/v1beta/files?key={key}",
        method="POST",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(data)),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        },
        data=json.dumps({"file": {"display_name": display_name}}).encode(),
    )
    try:
        with urllib.request.urlopen(start, timeout=HTTP_TIMEOUT_SECONDS) as response:
            upload_url = response.headers["X-Goog-Upload-URL"]
        finalize = urllib.request.Request(
            upload_url,
            method="POST",
            headers={
                "X-Goog-Upload-Command": "upload, finalize",
                "X-Goog-Upload-Offset": "0",
                "Content-Length": str(len(data)),
            },
            data=data,
        )
        with urllib.request.urlopen(finalize, timeout=HTTP_TIMEOUT_SECONDS) as response:
            info = json.load(response)["file"]
        name = info["name"]
        for _ in range(MAX_FILE_POLL_ATTEMPTS):
            state = info.get("state")
            if state != "PROCESSING":
                break
            time.sleep(FILE_POLL_INTERVAL_SECONDS)
            with urllib.request.urlopen(
                f"{BASE}/v1beta/{name}?key={key}",
                timeout=HTTP_TIMEOUT_SECONDS,
            ) as response:
                info = json.load(response)
        else:
            raise RuntimeError(
                f"Gemini file {display_name} stayed PROCESSING after "
                f"{MAX_FILE_POLL_ATTEMPTS * FILE_POLL_INTERVAL_SECONDS}s"
            )
        if info.get("state") == "FAILED":
            raise RuntimeError(f"Gemini file {display_name} failed processing")
        if info.get("state") != "ACTIVE":
            raise RuntimeError(
                f"Gemini file {display_name} state={info.get('state')!r}"
            )
        return info["uri"], mime
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"Gemini upload failed for {display_name} ({path}): HTTP {exc.code}: {body}"
        ) from exc


def extract_gemini_text(result: dict[str, Any]) -> str:
    """Extract strict JSON text from a Gemini generateContent response."""
    feedback = result.get("promptFeedback") or {}
    block_reason = feedback.get("blockReason")
    if block_reason:
        raise SystemExit(f"Gemini blocked the request: {block_reason}")
    candidates = result.get("candidates") or []
    if not candidates:
        raise SystemExit(f"Gemini returned no candidates: {json.dumps(result)[:500]}")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    raise SystemExit(f"Gemini response had no text parts: {json.dumps(result)[:500]}")


def parse_verdict_json(text: str) -> dict[str, Any]:
    """Parse and minimally validate the judge JSON payload."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Gemini returned non-JSON text: {exc}: {text[:500]}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Gemini verdict must be a JSON object")
    if "overallVerdict" not in payload:
        raise SystemExit("Gemini verdict missing overallVerdict")
    return payload


def write_report(
    *,
    output: Path,
    prompt: str,
    expected_style: str,
    baseline_label: str,
    clips: dict[str, Path],
    verdict: dict[str, Any],
    gain_matched: bool,
    probed: bool,
) -> None:
    """Write a structured fallback report envelope."""
    envelope = {
        "source": "kokoro-coreml/gemini_audio_judge_direct",
        "prompt": prompt,
        "expectedStyle": expected_style,
        "baselineLabel": baseline_label,
        "clips": {label: str(path) for label, path in clips.items()},
        "gainMatchedToBaseline": gain_matched,
        "probeFirst": probed,
        "verdict": verdict,
        "comparisonField": "comparisons.<label>_vs_<baseline>.same_quality_class",
        "primaryPathAnalog": "comparison.iphoneAcceptablyCloseToMlx",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")


def main() -> int:
    """Parse CLI args, upload clips, and print/write the Gemini JSON verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Clip as label=/abs/path.wav. Repeatable.",
    )
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expected-style", default=DEFAULT_EXPECTED_STYLE)
    parser.add_argument("--context-file", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        default=None,
        help="Report JSON path under repo root (default: stdout only)",
    )
    parser.add_argument(
        "--probe-first",
        action="store_true",
        help="Run audio_quality_probe and reject unhealthy clips before Gemini",
    )
    parser.add_argument(
        "--no-gain-match",
        action="store_true",
        help="Disable RMS gain-matching non-baseline clips to the baseline",
    )
    args = parser.parse_args()

    clips = parse_clip_args(args.clip)
    if args.baseline_label not in clips:
        raise SystemExit(
            f"--baseline-label {args.baseline_label} is not one of {list(clips)}"
        )
    for label, path in clips.items():
        clips[label] = validate_clip_path(label, path)

    context = ""
    if args.context_file:
        context = read_context_file(Path(args.context_file))

    if args.probe_first:
        probe_clips(clips, args.baseline_label)

    key = gemini_api_key()
    prompt = build_prompt(
        prompt=args.prompt,
        expected_style=args.expected_style,
        baseline=args.baseline_label,
        context=context,
    )

    gain_matched = not args.no_gain_match
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        prepared: dict[str, Path] = {}
        baseline_prepared = to_pcm16(clips[args.baseline_label], work)
        baseline_rms = read_wav_rms(baseline_prepared) if gain_matched else 0.0
        for label, path in clips.items():
            pcm_path = to_pcm16(path, work)
            if gain_matched and label != args.baseline_label and baseline_rms > 0.0:
                pcm_path = gain_match_clip(pcm_path, baseline_rms, work)
            prepared[label] = pcm_path

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for label, path in prepared.items():
            uri, mime = upload(key, path, label)
            print(f"uploaded {label}: {uri}")
            parts.append({"text": f"Clip label: {label}"})
            parts.append({"file_data": {"file_uri": uri, "mime_type": mime}})

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
            },
        }
        request = urllib.request.Request(
            f"{BASE}/v1beta/{args.model}:generateContent?key={key}",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body).encode(),
        )
        try:
            with urllib.request.urlopen(request, timeout=GENERATE_TIMEOUT_SECONDS) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(
                f"Gemini generateContent failed: HTTP {exc.code}: {body_text}"
            ) from exc

    text = extract_gemini_text(result)
    verdict = parse_verdict_json(text)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if args.output:
        output = resolve_output_path(Path(args.output))
        write_report(
            output=output,
            prompt=args.prompt,
            expected_style=args.expected_style,
            baseline_label=args.baseline_label,
            clips=clips,
            verdict=verdict,
            gain_matched=gain_matched,
            probed=args.probe_first,
        )
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
