#!/usr/bin/env python3
"""Create a TTS-only human listening checklist for the external bakeoff."""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import DEFAULT_OUTPUT_DIR, RUNTIME_BUCKETS, load_json


IMPL_LABELS = {
    "config-f-reference": "Config F reference",
    "mlx-audio": "MLX",
    "soniqo-speech-swift-kokoro": "Soniqo",
    "laishere-kokoro-coreml": "laishere",
}

IMPL_ORDER = [
    "config-f-reference",
    "mlx-audio",
    "soniqo-speech-swift-kokoro",
    "laishere-kokoro-coreml",
]
PRESERVED_DECISION_FIELDS = ("human_decision", "notes")


def _quality_index(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Index quality samples by WAV path."""
    indexed: dict[str, dict[str, Any]] = {}
    for path in sorted((results_dir / "quality").glob("*/audio_quality_report.json")):
        report = load_json(path)
        for sample in report.get("samples", []):
            metrics = sample.get("metrics", {})
            wav_path = _normalize_output_path(str(metrics.get("path") or ""))
            if wav_path:
                indexed[wav_path] = sample
    return indexed


def _result_records(results_dir: Path) -> list[dict[str, Any]]:
    """Return result records from primary result JSON files."""
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("results_*.json")):
        if path.name.endswith("_30s_pure.json"):
            continue
        payload = load_json(path)
        records.extend(payload.get("records", []))
    return records


def _manifest_inputs(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Read runtime manifest inputs by bucket key."""
    manifest_path = results_dir / "runtime_input_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    return dict(manifest.get("inputs", {}))


def _caveat(record: dict[str, Any]) -> str:
    """Return the listening caveat for one result record."""
    if record.get("status") != "ok":
        return str(record.get("error") or "missing audio")
    impl = record.get("impl")
    key = record.get("input_key")
    if impl == "soniqo-speech-swift-kokoro" and key != "3s":
        return "public 5s artifact; not long-bucket parity"
    if impl == "laishere-kokoro-coreml":
        return "Core ML chain only; excludes G2P/feed preparation"
    return ""


def _spotcheck_path(record: dict[str, Any]) -> str:
    """Return spot-check WAV path from provenance when present."""
    provenance = record.get("provenance") or {}
    raw = _normalize_output_path(str(provenance.get("spotcheck_wav") or ""))
    if raw and Path(raw).exists():
        return raw
    impl_dir = str(record.get("impl", "")).replace("-", "_")
    machine = str(record.get("machine_id"))
    key = str(record.get("input_key"))
    candidate = Path("outputs/external_bakeoff/spotcheck_wavs") / f"{impl_dir}_{machine}" / f"{key}.wav"
    return candidate.as_posix() if candidate.exists() else raw


def _normalize_output_path(path: str) -> str:
    """Normalize copied-back artifact paths into the local outputs tree."""
    marker = "outputs/external_bakeoff/"
    if marker in path:
        return path[path.index(marker) :]
    return path


def _record_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index result records by machine, input key, and implementation."""
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        indexed[
            (
                str(record.get("machine_id")),
                str(record.get("input_key")),
                str(record.get("impl")),
            )
        ] = record
    return indexed


def _review_rows(
    results_dir: Path,
    records: list[dict[str, Any]],
    quality: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Return one listening-review row per available result record."""
    inputs = _manifest_inputs(results_dir)
    indexed = _record_index(records)
    machines = sorted({str(record.get("machine_id")) for record in records})
    rows: list[dict[str, str]] = []
    for machine in machines:
        for key in RUNTIME_BUCKETS:
            expected = str(inputs.get(key, {}).get("text", ""))
            for impl in IMPL_ORDER:
                record = indexed.get((machine, key, impl))
                if not record:
                    continue
                status = str(record.get("status") or "")
                wav = ""
                duration = ""
                decision = "n/a"
                if status == "ok":
                    wav = _spotcheck_path(record)
                    sample = quality.get(wav, {})
                    metrics = sample.get("metrics", {})
                    raw_duration = metrics.get("duration_s", record.get("observed_audio_duration_s"))
                    duration = f"{float(raw_duration):.3f}" if raw_duration is not None else ""
                    decision = str(sample.get("decision") or "needs_listening")
                rows.append(
                    {
                        "machine_id": machine,
                        "input_key": key,
                        "impl": str(record.get("impl") or ""),
                        "impl_label": IMPL_LABELS.get(impl, impl),
                        "status": status,
                        "wav_path": wav,
                        "duration_s": duration,
                        "waveform_decision": decision,
                        "caveat": _caveat(record),
                        "expected_text": expected,
                        "human_decision": "",
                        "notes": "",
                    }
                )
    return rows


def _markdown(results_dir: Path, records: list[dict[str, Any]], quality: dict[str, dict[str, Any]]) -> str:
    """Render the markdown checklist."""
    inputs = _manifest_inputs(results_dir)
    indexed = _record_index(records)
    machines = sorted({str(record.get("machine_id")) for record in records})
    lines: list[str] = [
        "# External Bakeoff Listening Review",
        "",
        "This generated checklist is for human listening only. It does not use",
        "Whisper, ASR, VAD, transcription, or the Soniqo echo demo. Listen to the",
        "same-machine Config F reference first, then compare each available",
        "candidate for voice, pronunciation, truncation, artifacts, and gross",
        "prosody mismatch.",
        "",
        "A `needs_listening` waveform gate means the sample passed the objective",
        "sanity checks and still needs a human decision. It is not quality parity.",
        "",
    ]
    for machine in machines:
        lines.extend([f"## {machine}", ""])
        for key in RUNTIME_BUCKETS:
            expected = inputs.get(key, {}).get("text", "")
            lines.extend([f"### {key}", ""])
            if expected:
                lines.extend([f"Expected text: {expected}", ""])
            lines.extend(
                [
                    "| Done | Impl | WAV | Duration s | Waveform gate | Caveat |",
                    "| --- | --- | --- | ---: | --- | --- |",
                ]
            )
            for impl in IMPL_ORDER:
                record = indexed.get((machine, key, impl))
                if not record:
                    continue
                label = IMPL_LABELS.get(impl, impl)
                if record.get("status") != "ok":
                    wav_cell = "missing"
                    done_cell = "n/a"
                    duration_cell = "n/a"
                    decision = "n/a"
                else:
                    wav = _spotcheck_path(record)
                    sample = quality.get(wav, {})
                    metrics = sample.get("metrics", {})
                    wav_cell = f"`{wav}`" if wav else "missing"
                    done_cell = "[ ]"
                    duration = metrics.get("duration_s", record.get("observed_audio_duration_s"))
                    duration_cell = f"{float(duration):.3f}" if duration is not None else "n/a"
                    decision = str(sample.get("decision") or "needs_listening")
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            done_cell,
                            label,
                            wav_cell,
                            duration_cell,
                            decision,
                            _caveat(record),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _decision_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    """Return the stable key used to preserve human listening decisions."""
    return (
        row.get("machine_id", ""),
        row.get("input_key", ""),
        row.get("impl", ""),
        row.get("status", ""),
    )


def _existing_decisions(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Load existing human decision fields from a prior decision sheet."""
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        decisions: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for row in reader:
            preserved = {
                field: row.get(field, "")
                for field in PRESERVED_DECISION_FIELDS
                if row.get(field, "")
            }
            if preserved:
                decisions[_decision_key(row)] = preserved
        return decisions


def _merge_existing_decisions(
    rows: list[dict[str, str]],
    existing: dict[tuple[str, str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    """Preserve human decisions while allowing generated metadata to refresh."""
    merged: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item.update(existing.get(_decision_key(row), {}))
        merged.append(item)
    return merged


def _write_decisions_csv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    preserve_existing: bool = True,
) -> None:
    """Write a fillable human listening decision sheet."""
    fieldnames = [
        "machine_id",
        "input_key",
        "impl",
        "impl_label",
        "status",
        "wav_path",
        "duration_s",
        "waveform_decision",
        "caveat",
        "human_decision",
        "notes",
        "expected_text",
    ]
    if preserve_existing:
        rows = _merge_existing_decisions(rows, _existing_decisions(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _html_review(markdown_text: str, output_path: Path) -> str:
    """Render a minimal local HTML review page from generated markdown rows."""
    rows: list[str] = []
    current_heading = ""
    for line in markdown_text.splitlines():
        if line.startswith("## "):
            current_heading = html.escape(line.removeprefix("## "))
            rows.append(f"<h2>{current_heading}</h2>")
        elif line.startswith("### "):
            rows.append(f"<h3>{html.escape(line.removeprefix('### '))}</h3>")
        elif line.startswith("Expected text: "):
            rows.append(f"<p><strong>Expected text:</strong> {html.escape(line.removeprefix('Expected text: '))}</p>")
        elif line.startswith("| [ ] |") or line.startswith("| n/a |"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            impl = html.escape(cells[1])
            wav = cells[2].strip("`")
            duration = html.escape(cells[3])
            decision = html.escape(cells[4])
            caveat = html.escape(cells[5])
            duration_label = f"{duration}s" if duration != "n/a" else duration
            audio = "missing"
            if wav and wav != "missing":
                relative = os.path.relpath(wav, output_path.parent).replace(os.sep, "/")
                audio = f'<audio controls preload="none" src="{html.escape(relative)}"></audio>'
            rows.append(
                "<div class=\"sample\">"
                f"<div><strong>{impl}</strong> <span>{duration_label}</span> <span>{decision}</span></div>"
                f"<div>{audio}</div>"
                f"<div class=\"caveat\">{caveat}</div>"
                "</div>"
            )
    return (
        "<!doctype html>\n"
        "<meta charset=\"utf-8\">\n"
        "<title>External Bakeoff Listening Review</title>\n"
        "<style>"
        "body{font:14px -apple-system,BlinkMacSystemFont,sans-serif;margin:32px;line-height:1.45}"
        ".sample{border-top:1px solid #ddd;padding:10px 0;max-width:900px}"
        "audio{width:520px;max-width:100%;margin-top:6px}"
        ".caveat{color:#666;margin-top:4px}"
        "span{margin-left:10px;color:#555}"
        "</style>\n"
        "<h1>External Bakeoff Listening Review</h1>\n"
        "<p>TTS-only review page. No Whisper, ASR, VAD, or echo-demo dependency is used.</p>\n"
        + "\n".join(rows)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "listening" / "external_bakeoff_listening_review.md",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "listening" / "external_bakeoff_listening_review.html",
    )
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "listening" / "external_bakeoff_listening_decisions.csv",
    )
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--no-decisions-csv", action="store_true")
    parser.add_argument(
        "--reset-decisions",
        action="store_true",
        help="Rewrite the decisions CSV with blank human_decision and notes fields.",
    )
    args = parser.parse_args()

    records = _result_records(args.results_dir)
    quality = _quality_index(args.results_dir)
    markdown_text = _markdown(args.results_dir, records, quality)
    rows = _review_rows(args.results_dir, records, quality)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown_text)
    print(f"Wrote {args.output}")
    if not args.no_decisions_csv:
        _write_decisions_csv(
            args.decisions_output,
            rows,
            preserve_existing=not args.reset_decisions,
        )
        print(f"Wrote {args.decisions_output}")
    if not args.no_html:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(_html_review(markdown_text, args.html_output))
        print(f"Wrote {args.html_output}")


if __name__ == "__main__":
    main()
