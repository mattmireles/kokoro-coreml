import json
import subprocess
import sys
from pathlib import Path

from scripts.moe_prefetch.predictors import evaluate_policies
from scripts.moe_prefetch.run_stage0_envelope import fs_usage_command
from scripts.moe_prefetch.schema import (
    estimate_expert_bytes,
    model_inventory_payload,
    thresholds_payload,
    validate_phase0_ready,
)


def test_estimate_expert_bytes_uses_quantization_bits() -> None:
    assert estimate_expert_bytes(176_160_768, 4) == 88_080_384
    assert estimate_expert_bytes(3, 4) == 2


def test_phase0_readiness_requires_inventory_and_threshold_fields() -> None:
    inventory = model_inventory_payload(
        model_id="test/moe",
        quantization_bits=4,
        active_experts_per_token=64,
        target_tokens_per_second=1.0,
        expert_bytes=88_080_384,
        expert_parameters=176_160_768,
        target_device="local",
        estimate_source="unit-test",
        notes="",
        machine={"machine_id": "local"},
    )
    assert validate_phase0_ready(inventory, thresholds_payload()) == []

    broken = dict(inventory)
    broken["expert_bytes"] = 0
    assert "model_inventory.expert_bytes" in validate_phase0_ready(
        broken,
        thresholds_payload(),
    )


def test_model_inventory_cli_writes_inventory_and_thresholds(tmp_path: Path) -> None:
    output = tmp_path / "model_inventory.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/model_inventory.py",
            "--model-id",
            "test/moe",
            "--quantization-bits",
            "4",
            "--active-experts-per-token",
            "64",
            "--target-tokens-per-second",
            "1.0",
            "--expert-parameters",
            "176160768",
            "--target-device",
            "unit-test-mac",
            "--output",
            str(output),
        ],
        check=True,
    )

    inventory = json.loads(output.read_text())
    thresholds = json.loads((tmp_path / "thresholds.json").read_text())
    assert inventory["expert_bytes"] == 88_080_384
    assert inventory["active_expert_bytes_per_token"] == 5_637_144_576
    assert thresholds["speed_win_percent"] == 25.0
    assert thresholds["trivial_margin_percent"] == 10.0


def test_stage0_fs_usage_command_supports_safe_and_interactive_sudo() -> None:
    assert fs_usage_command(pid=123, timeout_s=20, sudo_mode="noninteractive") == [
        "sudo",
        "-n",
        "fs_usage",
        "-w",
        "-f",
        "diskio",
        "-t",
        "20",
        "123",
    ]
    assert fs_usage_command(pid=123, timeout_s=20, sudo_mode="interactive") == [
        "sudo",
        "fs_usage",
        "-w",
        "-f",
        "diskio",
        "-t",
        "20",
        "123",
    ]


def test_stage0_compute_cli_writes_small_cpu_benchmark(tmp_path: Path) -> None:
    inventory = {
        "model_id": "test/moe",
        "expert_bytes": 88_080_384,
        "active_expert_bytes_per_token": 5_637_144_576,
        "target_tokens_per_second": 1.0,
    }
    inventory_path = tmp_path / "model_inventory.json"
    output = tmp_path / "compute.json"
    inventory_path.write_text(json.dumps(inventory))

    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/run_stage0_compute.py",
            "--inventory",
            str(inventory_path),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--hidden-size",
            "8",
            "--intermediate-size",
            "16",
            "--active-experts-per-layer",
            "2",
        ],
        check=True,
    )

    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "synthetic_active_moe_ffn"
    assert payload["config"]["device"] == "cpu"
    assert payload["model_shape"]["hidden_size"] == 8
    assert payload["latency_p50_ms"] > 0


def test_stage0_summarize_kills_on_oracle_bandwidth_ceiling(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    notes = tmp_path / "notes.md"
    notes.write_text(
        "# Results\n\n"
        "## Stage 0: Hardware Envelope\n\n"
        "Pending.\n\n"
        "## Stage 1: Router Trace and Predictor Replay\n\n"
        "Blocked.\n"
    )
    payload = {
        "inventory": {
            "model_id": "test/moe",
            "expert_bytes": 88_080_384,
            "active_expert_bytes_per_token": 5_637_144_576,
            "target_tokens_per_second": 1.0,
        },
        "thresholds": thresholds_payload(),
        "cells": [
            {
                "pattern": "random",
                "returncode": 0,
                "fs_usage_path": str(tmp_path / "fs_usage.txt"),
                "fs_usage_error": "",
                "measurement": {
                    "successful_reads": 1,
                    "failed_reads": 0,
                    "total_bytes_read": 88_080_384,
                    "wall_time_ns": 88_080_384,
                    "latencies_ns": [88_080_384],
                },
            }
        ],
    }
    (tmp_path / "fs_usage.txt").write_text("RdData B=88080384\n")
    results.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/summarize.py",
            "stage0",
            "--input",
            str(results),
            "--notes",
            str(notes),
        ],
        check=True,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["oracle_bandwidth_ceiling_tokens_per_second"] < 1.0
    assert summary["decision"].startswith("KILL: oracle bandwidth ceiling")
    assert "Oracle bandwidth ceiling" in notes.read_text()


def test_stage0_summarize_rejects_empty_fs_usage_capture(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    notes = tmp_path / "notes.md"
    notes.write_text(
        "# Results\n\n"
        "## Stage 0: Hardware Envelope\n\n"
        "Pending.\n\n"
        "## Stage 1: Router Trace and Predictor Replay\n\n"
        "Blocked.\n"
    )
    empty_trace = tmp_path / "fs_usage_empty.txt"
    empty_trace.write_text("")
    payload = {
        "inventory": {
            "model_id": "test/moe",
            "expert_bytes": 88_080_384,
            "active_expert_bytes_per_token": 5_637_144_576,
            "target_tokens_per_second": 1.0,
        },
        "thresholds": thresholds_payload(),
        "cells": [
            {
                "pattern": "random",
                "returncode": 0,
                "fs_usage_path": str(empty_trace),
                "fs_usage_error": "",
                "measurement": {
                    "successful_reads": 1,
                    "failed_reads": 0,
                    "total_bytes_read": 8_000_000_000,
                    "wall_time_ns": 1_000_000_000,
                    "latencies_ns": [1_000_000],
                },
            }
        ],
    }
    results.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/summarize.py",
            "stage0",
            "--input",
            str(results),
            "--notes",
            str(notes),
        ],
        check=True,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["oracle_bandwidth_ceiling_tokens_per_second"] > 1.0
    assert summary["cells"][0]["has_fs_usage"] is False
    assert summary["decision"].startswith("KILL: missing fs_usage")


def test_stage0_summarize_flags_insufficient_hideability_from_compute(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.json"
    compute = tmp_path / "compute.json"
    notes = tmp_path / "notes.md"
    fs_usage = tmp_path / "fs_usage.txt"
    notes.write_text(
        "# Results\n\n"
        "## Stage 0: Hardware Envelope\n\n"
        "Pending.\n\n"
        "## Stage 1: Router Trace and Predictor Replay\n\n"
        "Blocked.\n"
    )
    fs_usage.write_text("RdData B=8000000000 direct_read_benc\n")
    payload = {
        "inventory": {
            "model_id": "test/moe",
            "expert_bytes": 1_000_000_000,
            "active_expert_bytes_per_token": 1_000_000_000,
            "target_tokens_per_second": 1.0,
        },
        "thresholds": thresholds_payload(),
        "config": {"powermetrics_path": ""},
        "cells": [
            {
                "pattern": "random",
                "returncode": 0,
                "fs_usage_path": str(fs_usage),
                "fs_usage_error": "",
                "measurement": {
                    "successful_reads": 1,
                    "failed_reads": 0,
                    "total_bytes_read": 8_000_000_000,
                    "wall_time_ns": 1_000_000_000,
                    "latencies_ns": [10_000_000],
                },
            }
        ],
    }
    compute.write_text(
        json.dumps(
            {
                "benchmark": "unit",
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 1.5,
                "config": {"device": "cpu"},
                "model_shape": {"hidden_size": 8},
            }
        )
    )
    results.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/summarize.py",
            "stage0",
            "--input",
            str(results),
            "--compute",
            str(compute),
            "--notes",
            str(notes),
        ],
        check=True,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["decision"].startswith("FLAG: bandwidth passes")
    assert summary["hideability_ratio"] == 10.0
    note_text = notes.read_text()
    assert "`fs_usage`: present for every accepted read cell." in note_text
    assert "Hideability ratio" in note_text


def test_stage1_policy_eval_zeroes_hideable_recall_when_depth_is_too_small() -> None:
    rows = [
        {
            "request_id": "r0",
            "token_index": 0,
            "layer_index": 0,
            "actual_topk_expert_ids": [1, 2],
        },
        {
            "request_id": "r0",
            "token_index": 0,
            "layer_index": 1,
            "actual_topk_expert_ids": [1, 3],
        },
    ]

    metrics = evaluate_policies(rows, prefetch_depth=1, required_depth=2)

    assert metrics["oracle"]["recall"] == 1.0
    assert metrics["oracle"]["hideable_recall"] == 0.0
    assert metrics["eam"]["recall"] == 0.25
    assert metrics["eam"]["hideable_recall"] == 0.0


def test_stage1_summarize_kills_when_planned_depths_cannot_hide_reads(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    stage0 = tmp_path / "stage0_summary.json"
    output = tmp_path / "predictability.json"
    notes = tmp_path / "notes.md"
    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "request_id": "code-0",
                    "domain": "code",
                    "model_id": "test/moe",
                    "token_index": 0,
                    "layer_index": 0,
                    "actual_topk_expert_ids": [1, 2],
                },
                {
                    "request_id": "code-0",
                    "domain": "code",
                    "model_id": "test/moe",
                    "token_index": 0,
                    "layer_index": 1,
                    "actual_topk_expert_ids": [1, 3],
                },
            ]
        )
        + "\n"
    )
    stage0.write_text(
        json.dumps(
            {
                "hideability_ratio": 4.2,
                "thresholds": {"trivial_margin_percent": 10.0},
            }
        )
    )
    notes.write_text(
        "# Results\n\n"
        "## Stage 1: Router Trace and Predictor Replay\n\n"
        "Pending.\n\n"
        "## Stage 2: Offline Simulator\n\n"
        "Blocked.\n"
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/summarize.py",
            "stage1",
            "--trace",
            str(trace),
            "--stage0",
            str(stage0),
            "--output",
            str(output),
            "--notes",
            str(notes),
            "--prefetch-depths",
            "1,2,3",
        ],
        check=True,
    )

    payload = json.loads(output.read_text())
    assert payload["required_prefetch_depth"] == 5
    assert payload["decision"].startswith("KILL: planned prefetch depths")
    assert "Hideable recall" in notes.read_text()
