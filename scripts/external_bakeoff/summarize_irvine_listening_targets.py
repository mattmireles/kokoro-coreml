#!/usr/bin/env python3
"""Map remaining Irvine speed candidates to no-ASR listening artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_NEXT_TARGETS_JSON = Path("outputs/external_bakeoff/irvine_next_targets.json")
DEFAULT_CANDIDATE_SUMMARY_JSON = Path("outputs/f0_source_listening/f0_source_candidate_summary.json")
DEFAULT_LISTENING_GLOB = "outputs/f0_source_listening/**/provenance.json"
DEFAULT_DECISIONS_GLOB = "outputs/f0_source_listening/**/f0_source_listening_decisions.csv"
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/irvine_listening_targets.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/irvine_listening_targets.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _fmt(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _candidate_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("rows") or []
    return [row for row in rows if row.get("machine") == "irvine-m1"]


def _provenance_index(patterns: list[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match)
            try:
                payload = _load_json(path)
            except Exception:
                continue
            label = str(payload.get("label") or "")
            tensor_dump = str(payload.get("tensor_dump") or "")
            bucket = Path(tensor_dump).name if tensor_dump else label.split("_", 1)[0]
            if label and bucket:
                payload["provenance_path"] = str(path)
                index.setdefault((bucket, label), []).append(payload)
    return index


def _decision_index(patterns: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            with Path(match).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    label = str(row.get("label") or "")
                    source_report = str(row.get("source_report") or "")
                    if label:
                        decisions[(label, source_report)] = row
    return decisions


def _best_candidate_for_bucket(rows: list[dict[str, Any]], bucket: str, preferred_label: str | None) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("bucket") == bucket]
    if preferred_label:
        exact = [row for row in candidates if row.get("label") == preferred_label]
        if exact:
            return exact[0]
    return candidates[0] if candidates else None


def _select_artifact(artifacts: list[dict[str, Any]], timing_source_report: str) -> dict[str, Any] | None:
    for artifact in artifacts:
        if str(artifact.get("source_report") or "") == timing_source_report:
            return artifact
    return artifacts[0] if artifacts else None


def summarize_listening_targets(
    next_targets: dict[str, Any],
    candidate_summary: dict[str, Any],
    provenance: dict[tuple[str, str], list[dict[str, Any]]],
    decisions: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    """Return Irvine cells with speed rows and no-ASR listening links."""

    rows = []
    irvine_candidates = _candidate_rows(candidate_summary)
    for target in next_targets.get("rows") or []:
        bucket = str(target.get("input_key") or "")
        preferred = (target.get("best_quality_fail_signal") or {}).get("label")
        candidate = _best_candidate_for_bucket(irvine_candidates, bucket, preferred)
        if not candidate:
            rows.append({"bucket": bucket, "status": "missing-speed-candidate"})
            continue
        timing_source_report = str(candidate.get("report") or "")
        artifact = _select_artifact(provenance.get((bucket, str(candidate["label"])), []), timing_source_report)
        decision = {}
        if artifact:
            decision = decisions.get((str(candidate["label"]), str(artifact.get("source_report") or "")), {})
        rows.append(
            {
                "bucket": bucket,
                "status": "mapped" if artifact else "missing-listening-artifact",
                "candidate": candidate["label"],
                "speedup_pct": candidate.get("speedup_pct"),
                "candidate_ms": candidate.get("candidate_ms"),
                "baseline_ms": candidate.get("baseline_ms"),
                "corr": candidate.get("corr"),
                "snr_db": candidate.get("snr_db"),
                "timing_source_report": timing_source_report,
                "listening_source_report": None if artifact is None else artifact.get("source_report"),
                "listening_candidate_wav": None if artifact is None else (artifact.get("wavs") or {}).get("candidate"),
                "listening_review": None if artifact is None else artifact.get("review"),
                "waveform_gate_decision": None if artifact is None else artifact.get("waveform_gate_decision"),
                "human_decision": decision.get("human_decision") or "",
                "decision_notes": decision.get("notes") or "",
                "exact_timing_report_has_listening_artifact": (
                    artifact is not None and timing_source_report == str(artifact.get("source_report") or "")
                ),
            }
        )
    return {
        "rows": rows,
        "row_count": len(rows),
        "mapped_count": sum(1 for row in rows if row.get("status") == "mapped"),
        "missing_listening_artifact_count": sum(
            1 for row in rows if row.get("status") == "missing-listening-artifact"
        ),
        "exact_timing_report_listening_artifact_count": sum(
            1 for row in rows if row.get("exact_timing_report_has_listening_artifact")
        ),
        "note": "No ASR/Whisper gate is used. Remote Irvine speed rows are mapped to same-label local WAV artifacts when exact remote-report WAVs are not present.",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Irvine Listening Targets",
        "",
        summary["note"],
        "",
        f"Rows: `{summary['row_count']}`.",
        f"Rows with a no-ASR listening artifact: `{summary['mapped_count']}`.",
        f"Rows where the listening artifact uses the exact Irvine timing report: `{summary['exact_timing_report_listening_artifact_count']}`.",
        "",
        "| Bucket | Candidate | Irvine speedup | Waveform | Listening artifact | Exact timing report? | Human |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        if row.get("status") != "mapped":
            lines.append(f"| {row.get('bucket')} | n/a | n/a | n/a | missing | no | blank |")
            continue
        artifact = row.get("listening_candidate_wav") or "missing"
        exact = "yes" if row.get("exact_timing_report_has_listening_artifact") else "same-label local WAV"
        human = row.get("human_decision") or "blank"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bucket"]),
                    f"`{row['candidate']}`",
                    f"{_fmt(row['speedup_pct'])}% ({_fmt(row['baseline_ms'])} -> {_fmt(row['candidate_ms'])} ms)",
                    f"corr {_fmt(row['corr'], 6)}, SNR {_fmt(row['snr_db'], 2)} dB; gate `{row['waveform_gate_decision']}`",
                    f"`{artifact}`",
                    exact,
                    human,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next-targets-json", type=Path, default=DEFAULT_NEXT_TARGETS_JSON)
    parser.add_argument("--candidate-summary-json", type=Path, default=DEFAULT_CANDIDATE_SUMMARY_JSON)
    parser.add_argument("--listening-glob", action="append", default=[DEFAULT_LISTENING_GLOB])
    parser.add_argument("--decisions-glob", action="append", default=[DEFAULT_DECISIONS_GLOB])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = summarize_listening_targets(
        _load_json(args.next_targets_json),
        _load_json(args.candidate_summary_json),
        _provenance_index(args.listening_glob),
        _decision_index(args.decisions_glob),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": summary["row_count"], "mapped": summary["mapped_count"], "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
