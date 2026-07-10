import csv
import hashlib
import json
from pathlib import Path

from scripts.external_bakeoff.create_listening_review import _write_decisions_csv
from scripts.external_bakeoff.ingest_ios_runner_result import _ingest_records
from scripts.external_bakeoff.schema import (
    RUNTIME_BUCKETS,
    result_file_payload,
    result_record,
    validate_result_payload,
)
from scripts.external_bakeoff.validate_listening_decisions import validate_rows
from scripts.external_bakeoff.verify_external_bakeoff_completion import (
    _check_iphone,
    _check_mac_primary,
    _check_preflight,
    _load_records,
)


def _manifest() -> dict:
    inputs = {}
    for index, key in enumerate(RUNTIME_BUCKETS, start=1):
        text = f"runtime text {key}"
        inputs[key] = {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "voice": "af_heart",
            "speed": 1.0,
            "canonical_duration_s": float(index),
            "expected_bucket_s": int(key.rstrip("s")),
        }
    return {"runtime_buckets": list(RUNTIME_BUCKETS), "inputs": inputs}


def _ios_payload(manifest: dict) -> dict:
    records = []
    for key in RUNTIME_BUCKETS:
        item = manifest["inputs"][key]
        duration = item["canonical_duration_s"]
        warm = [0.1, 0.11, 0.12, 0.13, 0.14]
        records.append(
            {
                "input_key": key,
                "text_sha256": item["text_sha256"],
                "voice": item["voice"],
                "canonical_audio_duration_s": duration,
                "expected_bucket_s": item["expected_bucket_s"],
                "cold_wall_time_s": 0.2,
                "warm_wall_times_s": warm,
                "sample_count": int(duration * 24000),
                "sample_rate": 24000,
                "observed_audio_duration_s": duration,
                "rtf_observed": [value / duration for value in warm],
            }
        )
    return {
        "impl": "soniqo-speech-swift-kokoro-ios",
        "framework": "Swift + Core ML",
        "hardware_target": "ANE/Core ML",
        "compute_units": "all",
        "warm_iterations": 5,
        "records": records,
    }


def test_ingest_ios_runner_result_validates_manifest_and_schema():
    manifest = _manifest()
    records = _ingest_records(
        payload=_ios_payload(manifest),
        manifest=manifest,
        machine_id="iphone-12-pro",
        version="test",
        source_path=Path("ios.json"),
    )

    assert [record["input_key"] for record in records] == list(RUNTIME_BUCKETS)
    assert all(record["machine_id"] == "iphone-12-pro" for record in records)
    assert all(record["status"] == "ok" for record in records)
    assert all(record["output_sha256"] == "" for record in records)
    assert all(record["provenance"]["spotcheck_wav_unavailable"] for record in records)

    validate_result_payload(
        {
            "created_utc": "2026-06-05T00:00:00Z",
            "impl": "soniqo-speech-swift-kokoro-ios",
            "machine_id": "iphone-12-pro",
            "machine": {"machine_id": "iphone-12-pro"},
            "records": records,
            "provenance": {},
        }
    )


def test_listening_decision_validator_blocks_blank_success_rows():
    summary, errors = validate_rows(
        [
            {
                "machine_id": "m2-studio",
                "input_key": "3s",
                "impl": "config-f-reference",
                "status": "ok",
                "wav_path": "sample.wav",
                "human_decision": "",
                "notes": "",
            },
            {
                "machine_id": "m2-studio",
                "input_key": "3s",
                "impl": "mlx-audio",
                "status": "error",
                "wav_path": "",
                "human_decision": "",
                "notes": "",
            },
        ]
    )

    assert not summary["valid"]
    assert summary["ok_rows"] == 1
    assert summary["error_rows"] == 1
    assert errors == ["m2-studio/3s/config-f-reference: missing human_decision"]


def test_listening_review_regeneration_preserves_human_decisions(tmp_path):
    decisions = tmp_path / "decisions.csv"
    original = [
        {
            "machine_id": "m2-studio",
            "input_key": "3s",
            "impl": "config-f-reference",
            "impl_label": "Config F reference",
            "status": "ok",
            "wav_path": "old.wav",
            "duration_s": "2.800",
            "waveform_decision": "reference_pass",
            "caveat": "",
            "human_decision": "pass",
            "notes": "heard cleanly",
            "expected_text": "old text",
        }
    ]
    regenerated = [
        {
            "machine_id": "m2-studio",
            "input_key": "3s",
            "impl": "config-f-reference",
            "impl_label": "Config F reference",
            "status": "ok",
            "wav_path": "new.wav",
            "duration_s": "2.801",
            "waveform_decision": "reference_pass",
            "caveat": "",
            "human_decision": "",
            "notes": "",
            "expected_text": "new text",
        }
    ]

    _write_decisions_csv(decisions, original)
    _write_decisions_csv(decisions, regenerated)

    row = next(csv.DictReader(decisions.open()))
    assert row["wav_path"] == "new.wav"
    assert row["duration_s"] == "2.801"
    assert row["expected_text"] == "new text"
    assert row["human_decision"] == "pass"
    assert row["notes"] == "heard cleanly"


def test_listening_review_reset_decisions_blanks_human_fields(tmp_path):
    decisions = tmp_path / "decisions.csv"
    rows = [
        {
            "machine_id": "m2-studio",
            "input_key": "3s",
            "impl": "config-f-reference",
            "impl_label": "Config F reference",
            "status": "ok",
            "wav_path": "sample.wav",
            "duration_s": "2.800",
            "waveform_decision": "reference_pass",
            "caveat": "",
            "human_decision": "pass",
            "notes": "heard cleanly",
            "expected_text": "text",
        }
    ]

    _write_decisions_csv(decisions, rows)
    reset_rows = [dict(rows[0], human_decision="", notes="")]
    _write_decisions_csv(decisions, reset_rows, preserve_existing=False)

    row = next(csv.DictReader(decisions.open()))
    assert row["human_decision"] == ""
    assert row["notes"] == ""


def test_completion_verifier_allows_mlx_3s_error_but_requires_iphone():
    records = {}
    for machine in ("m2-studio", "m2-air", "irvine-m1"):
        for impl in ("config-f-reference", "mlx-audio", "soniqo-speech-swift-kokoro"):
            for key in RUNTIME_BUCKETS:
                if impl == "mlx-audio" and key == "3s":
                    records[(machine, impl, key)] = {"status": "error"}
                else:
                    records[(machine, impl, key)] = {
                        "status": "ok",
                        "warm_wall_times_s": [0.1] * 5,
                        "cold_wall_time_s": 0.2,
                    }

    assert _check_mac_primary(records) == []
    iphone_errors = _check_iphone(records)
    assert iphone_errors == [
        f"missing signed iPhone result: iphone-12-pro/soniqo-speech-swift-kokoro-ios/{key}"
        for key in RUNTIME_BUCKETS
    ]


def test_completion_verifier_loads_result_payloads_and_preflight(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    record = result_record(
        impl="config-f-reference",
        framework="Swift + Core ML",
        hardware_target="ANE/Core ML",
        version="test",
        machine_id="m2-studio",
        input_key="3s",
        text="sample",
        voice="af_heart",
        cold_wall_time_s=0.2,
        warm_wall_times_s=[0.1] * 5,
        canonical_audio_duration_s=1.0,
        observed_audio_duration_s=1.0,
        output_sha256="abc",
        provenance={},
    )
    payload = result_file_payload(
        impl="config-f-reference",
        machine_id="m2-studio",
        records=[record],
        provenance={},
    )
    (results_dir / "results_config_f_reference_m2-studio.json").write_text(
        json.dumps(payload)
    )
    (results_dir / "ios_runner_preflight_latest.json").write_text(
        json.dumps({"ok": False, "blockers": ["DEVELOPMENT_TEAM is unset"]})
    )

    records, errors = _load_records(results_dir)
    preflight, preflight_errors = _check_preflight(results_dir)

    assert errors == []
    assert ("m2-studio", "config-f-reference", "3s") in records
    assert preflight and not preflight["ok"]
    assert preflight_errors == ["iOS preflight not ready: DEVELOPMENT_TEAM is unset"]
