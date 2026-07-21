"""Protocol and schema tests for plan 010 Stage 0 tooling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from coremltools.converters.mil.mil.scope import ScopeSource

from scripts.dump_device_compute_plan import (
    model_runner_working_directory,
    signing_build_arguments,
    validated_device_provenance,
)
from scripts.lfm2_surgical.checkpoint import checkpoint_identity, effective_intermediate_size
from scripts.lfm2_surgical.constants import (
    CONV_LAYER_INDEX,
    EXPECTED_LAYER_TYPES,
    GQA_INPUT_LAYER_INDEX,
    GQA_LAYER_INDEX,
    NUMERICS_PROMPT_COUNT,
    PREFILL_BUCKETS,
    REAL_PROMPTS,
    validate_frozen_protocol,
)
from scripts.lfm2_surgical.export_blocks import gqa_mixed_precision_selector


def test_frozen_bucket_and_prompt_protocol() -> None:
    """The executable protocol must match the plan's preregistered constants."""

    validate_frozen_protocol()
    assert PREFILL_BUCKETS == (128, 256, 512, 1024, 2048)
    assert len(REAL_PROMPTS) == NUMERICS_PROMPT_COUNT == 32


def test_frozen_layer_selection_matches_real_350m_config() -> None:
    """The isolated layers must remain one conv and one full-attention block."""

    assert EXPECTED_LAYER_TYPES[CONV_LAYER_INDEX] == "conv"
    assert EXPECTED_LAYER_TYPES[GQA_INPUT_LAYER_INDEX] == "conv"
    assert EXPECTED_LAYER_TYPES[GQA_LAYER_INDEX] == "full_attention"
    assert EXPECTED_LAYER_TYPES == (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )


def test_effective_intermediate_size_matches_liquid_adjustment() -> None:
    """The 350M checkpoint's declared 6656 width must load as trained width 4608."""

    config = {
        "intermediate_size": 6656,
        "block_auto_adjust_ff_dim": True,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256,
    }
    assert effective_intermediate_size(config) == 4608


def test_effective_intermediate_size_can_disable_adjustment() -> None:
    """Explicit checkpoint opt-out must preserve the declared width."""

    config = {
        "intermediate_size": 6656,
        "block_auto_adjust_ff_dim": False,
    }
    assert effective_intermediate_size(config) == 6656


def test_checkpoint_identity_rejects_config_and_tokenizer_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights, graph config, and tokenizer must share one frozen identity."""

    import scripts.lfm2_surgical.checkpoint as checkpoint

    artifacts = {
        "model.safetensors": b"weights",
        "config.json": b'{"norm_eps": 1e-5}',
        "tokenizer.json": b'{"model": "frozen"}',
    }
    metadata_directory = tmp_path / ".cache/huggingface/download"
    metadata_directory.mkdir(parents=True)
    for filename, payload in artifacts.items():
        (tmp_path / filename).write_bytes(payload)
        (metadata_directory / f"{filename}.metadata").write_text(
            f"{checkpoint.CHECKPOINT_REVISION}\nblob\ntimestamp\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        checkpoint,
        "CHECKPOINT_MODEL_SHA256",
        hashlib.sha256(artifacts["model.safetensors"]).hexdigest(),
    )
    monkeypatch.setattr(
        checkpoint,
        "CHECKPOINT_CONFIG_SHA256",
        hashlib.sha256(artifacts["config.json"]).hexdigest(),
    )
    monkeypatch.setattr(
        checkpoint,
        "CHECKPOINT_TOKENIZER_SHA256",
        hashlib.sha256(artifacts["tokenizer.json"]).hexdigest(),
    )
    checkpoint_identity.cache_clear()
    assert checkpoint_identity(tmp_path)["config_sha256"] == hashlib.sha256(
        artifacts["config.json"]
    ).hexdigest()

    (tmp_path / "config.json").write_bytes(b'{"norm_eps": 1e-6}')
    checkpoint_identity.cache_clear()
    with pytest.raises(ValueError, match="config.json digest changed"):
        checkpoint_identity(tmp_path)
    checkpoint_identity.cache_clear()


def test_protocol_rejects_accidental_prompt_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed prompt corpus must fail before any measurement is produced."""

    import scripts.lfm2_surgical.constants as constants

    monkeypatch.setattr(constants, "REAL_PROMPTS", constants.REAL_PROMPTS[:-1])
    with pytest.raises(ValueError, match="expected 32 real prompts"):
        constants.validate_frozen_protocol()


def test_gqa_fp32_selector_uses_named_module_scopes_only() -> None:
    """Precision must not leak from the approved norm/RoPE islands."""

    approved = SimpleNamespace(
        scopes={ScopeSource.TORCHSCRIPT_MODULE_NAME: ["attention_positioning", "42"]}
    )
    residual_add = SimpleNamespace(
        scopes={ScopeSource.TORCHSCRIPT_MODULE_NAME: ["207"]}
    )
    assert gqa_mixed_precision_selector(approved) is False
    assert gqa_mixed_precision_selector(residual_add) is True


def test_mobile_provenance_fails_closed_and_persists_exact_identity() -> None:
    """The stale-tunnel fallback must prove and retain the physical device."""

    device = SimpleNamespace(
        name="iDesk",
        identifier="core-device-id",
        udid="physical-udid",
        type=SimpleNamespace(value="ipad"),
    )
    details = {
        "identifier": "core-device-id",
        "deviceProperties": {
            "name": "iDesk",
            "osVersionNumber": "26.5",
            "osBuildUpdate": "23F77",
            "developerModeStatus": "enabled",
            "ddiServicesAvailable": True,
        },
        "hardwareProperties": {
            "deviceType": "iPad",
            "productType": "iPad14,3",
            "udid": "physical-udid",
        },
        "connectionProperties": {
            "pairingState": "paired",
            "transportType": "wired",
            "tunnelState": "connected",
        },
    }
    provenance = validated_device_provenance(device, details)
    assert provenance["device_udid"] == "physical-udid"
    assert provenance["device_os_build"] == "23F77"

    details["deviceProperties"]["ddiServicesAvailable"] = False
    with pytest.raises(RuntimeError, match="developer_disk_image"):
        validated_device_provenance(device, details)


def test_explicit_profile_disables_automatic_portal_mutations() -> None:
    """Headless signing must not combine a fixed profile with portal updates."""

    profile_uuid = "11111111-2222-3333-4444-555555555555"
    assert signing_build_arguments(profile_uuid) == [
        "CODE_SIGN_STYLE=MANUAL",
        f"PROVISIONING_PROFILE={profile_uuid}"
    ]
    assert signing_build_arguments(None) == [
        "CODE_SIGN_STYLE=AUTOMATIC",
        "-allowProvisioningUpdates",
        "-allowProvisioningDeviceRegistration",
    ]


def test_runner_cache_isolated_by_complete_signing_identity() -> None:
    """A changed profile, team, or bundle must never reuse a stale signed app."""

    first = model_runner_working_directory("TEAMONE", "example.runner", "PROFILE-A")
    assert first == model_runner_working_directory(
        "TEAMONE", "example.runner", "PROFILE-A"
    )
    assert first != model_runner_working_directory(
        "TEAMONE", "example.runner", "PROFILE-B"
    )
    assert first != model_runner_working_directory(
        "TEAMTWO", "example.runner", "PROFILE-A"
    )
    assert first != model_runner_working_directory(
        "TEAMONE", "example.other", "PROFILE-A"
    )
