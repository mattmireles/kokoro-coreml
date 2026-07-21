"""Dump a Core ML compute plan — per-op device placement — on host or mobile.

Phase 2 instrument of README/Plans/007-kokoro-iphone-performance-plan.md: settles
whether decoder-pre's `.cpuAndNeuralEngine` pin is real on the phone and which
generator ops a CPU+GPU plan maps to CPU (the 6.8x-vs-M1 anomaly suspects).
Background: README/Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md
("Verified Diagnostic Tooling").

Usage (from repo root, with the project venv):

    .venv/bin/python scripts/dump_device_compute_plan.py \
        --package coreml/kokoro_decoder_pre_3s.mlpackage \
        --compute-units CPU_AND_NE \
        --device-name "Webcam" \
        --out outputs/iphone_bench/plan_decoder_pre_3s_ne_12pro.json

Omit --device-name to run host-side (Mac) — useful as the M1-generation
reference but NOT as iPhone evidence (the Xcode/host ANE has looser limits;
see the A14 guide's "Do not use host results as device proof").

The package is compiled to a temporary .mlmodelc with `xcrun coremlcompiler`
because MLComputePlan loads compiled models only. On-device loading uses the
coremltools experimental remote-device API (verified against coremltools
8.3.0), which requires the device paired, unlocked, and Developer Mode on.

Output JSON: total op count, per-device preferred-op histogram, per
operator_name placement histogram, the list of ops whose preferred device is
CPU (name + estimated cost weight), and run provenance.
"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

# Maps the CLI flag to the coremltools enum. CPU_AND_NE is the ANE-viability
# probe per the triage guide; ALL is deliberately absent (masks rejections).
COMPUTE_UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
}
MODEL_RUNNER_CACHE = Path("outputs/coreml_model_runner")
CLANG_PROBE_WRAPPER = Path(__file__).with_name("xcode_clang_probe_wrapper.zsh")
DEFAULT_DEVELOPMENT_TEAM = "6ETYBAJKY8"
DEFAULT_BUNDLE_IDENTIFIER = "com.mattmireles.CoreMLModelRunner"


def compile_package(package: Path, workdir: Path) -> Path:
    """Compile a .mlpackage to .mlmodelc; MLComputePlan accepts only compiled models."""
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(workdir)],
        check=True,
        capture_output=True,
    )
    compiled = workdir / (package.stem + ".mlmodelc")
    if not compiled.exists():
        raise FileNotFoundError(f"coremlcompiler produced no {compiled}")
    return compiled


def package_tree_sha256(package: Path) -> str:
    """Hash every package-relative path and byte so provenance is reproducible."""

    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in package.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(package)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as package_file:
            for chunk in iter(lambda: package_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validated_device_provenance(device, details: dict) -> dict:
    """Validate exact live CoreDevice details and return durable provenance."""

    device_properties = details.get("deviceProperties", {})
    hardware_properties = details.get("hardwareProperties", {})
    connection_properties = details.get("connectionProperties", {})
    checks = {
        "identifier": details.get("identifier") == device.identifier,
        "udid": hardware_properties.get("udid") == device.udid,
        "device_type": hardware_properties.get("deviceType", "").lower()
        == device.type.value,
        "developer_mode": device_properties.get("developerModeStatus") == "enabled",
        "developer_disk_image": device_properties.get("ddiServicesAvailable") is True,
        "pairing": connection_properties.get("pairingState") == "paired",
        "tunnel": connection_properties.get("tunnelState") == "connected",
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(
            f"Device {device.name!r} is not debug-ready; failed: {failed_checks}"
        )
    return {
        "device_name": device_properties.get("name"),
        "device_identifier": details.get("identifier"),
        "device_udid": hardware_properties.get("udid"),
        "device_product_type": hardware_properties.get("productType"),
        "device_os_version": device_properties.get("osVersionNumber"),
        "device_os_build": device_properties.get("osBuildUpdate"),
        "developer_mode": device_properties.get("developerModeStatus"),
        "developer_disk_image": device_properties.get("ddiServicesAvailable"),
        "pairing_state": connection_properties.get("pairingState"),
        "transport_type": connection_properties.get("transportType"),
        "tunnel_state": connection_properties.get("tunnelState"),
    }


def refresh_mobile_device_connection(device, device_state_type):
    """Prove debug readiness and return repaired state plus live provenance.

    coremltools 8.3 filters ``Device.get_connected_devices`` using the cached
    ``devicectl list devices`` tunnel state. Xcode 26.6 releases the tunnel as
    soon as a command's usage assertion ends, so a paired, usable device can
    immediately appear ``DISCONNECTED`` in that cache. An identifier-specific
    details request reacquires the tunnel and reports the authoritative DDI,
    pairing, Developer Mode, type, and UDID state. Only after every field is
    verified do we replace the frozen dataclass state needed by
    ``prepare_for_model_debugging``; all subsequent commands still target the
    exact physical identifier and establish their own device connections.
    """

    with tempfile.TemporaryDirectory() as temporary_directory:
        details_path = Path(temporary_directory) / "device-details.json"
        subprocess.run(
            [
                "xcrun",
                "devicectl",
                "-j",
                str(details_path),
                "device",
                "info",
                "details",
                "--device",
                device.identifier,
            ],
            check=True,
            capture_output=True,
        )
        details = json.loads(details_path.read_text())["result"]

    provenance = validated_device_provenance(device, details)
    return replace(device, state=device_state_type.CONNECTED), provenance


def signing_build_arguments(provisioning_profile_uuid: str | None) -> list[str]:
    """Return mutually exclusive Xcode provisioning arguments.

    An explicit profile is the least-privilege headless path after a profile
    has already been created and installed. Without one, Xcode's supported
    automatic-signing path may register the selected device and update the
    managed profile after the user signs into the development team.
    """

    if provisioning_profile_uuid:
        return [
            "CODE_SIGN_STYLE=MANUAL",
            f"PROVISIONING_PROFILE={provisioning_profile_uuid}",
        ]
    return [
        "CODE_SIGN_STYLE=AUTOMATIC",
        "-allowProvisioningUpdates",
        "-allowProvisioningDeviceRegistration",
    ]


def model_runner_working_directory(
    development_team: str,
    bundle_identifier: str,
    provisioning_profile_uuid: str | None,
) -> Path:
    """Return a cache isolated by every signing input that changes the app."""

    signing_identity = "\0".join(
        (
            development_team,
            bundle_identifier,
            provisioning_profile_uuid or "automatic",
        )
    )
    digest = hashlib.sha256(signing_identity.encode("utf-8")).hexdigest()[:16]
    return MODEL_RUNNER_CACHE / f"signing-{digest}"


def ensure_model_runner_app(
    device,
    development_team: str,
    bundle_identifier: str,
    provisioning_profile_uuid: str | None,
) -> Path:
    """Build the exact signed runner and return its isolated working directory."""

    working_directory = model_runner_working_directory(
        development_team,
        bundle_identifier,
        provisioning_profile_uuid,
    )
    build_path = (working_directory / device.udid).resolve()
    if next(build_path.rglob("*.app"), None) is not None:
        return working_directory
    model_runner_root = Path(ct.__file__).resolve().parent / "modelrunner"
    workspace = model_runner_root / "ModelRunner.xcworkspace"
    if not workspace.is_dir():
        raise FileNotFoundError(f"coremltools model runner missing: {workspace}")
    if not CLANG_PROBE_WRAPPER.is_file():
        raise FileNotFoundError(f"Xcode clang probe wrapper missing: {CLANG_PROBE_WRAPPER}")
    build_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "xcodebuild",
            "-workspace",
            str(workspace),
            "-scheme",
            "modelrunner",
            "-sdk",
            "iphoneos",
            "-destination",
            f"platform=iOS,id={device.udid}",
            "-configuration",
            "Release",
            f"SYMROOT={build_path}",
            *signing_build_arguments(provisioning_profile_uuid),
            f"DEVELOPMENT_TEAM={development_team}",
            f"PRODUCT_BUNDLE_IDENTIFIER={bundle_identifier}",
            f"CC={CLANG_PROBE_WRAPPER.resolve()}",
            "ENABLE_ADDRESS_SANITIZER=NO",
            "ENABLE_CODE_COVERAGE=NO",
        ],
        check=True,
    )
    if next(build_path.rglob("*.app"), None) is None:
        raise FileNotFoundError(f"xcodebuild produced no model runner under {build_path}")
    return working_directory


def device_label(device) -> str:
    """MLComputeDevice subclass -> short label (cpu / gpu / neuralEngine)."""
    name = type(device).__name__
    if "NeuralEngine" in name:
        return "neuralEngine"
    if "GPU" in name:
        return "gpu"
    if "CPU" in name:
        return "cpu"
    return name


def walk_operations(block):
    """Yield every operation in a program block, recursing into nested blocks
    (e.g. lstm/while bodies) so placement counts cover the whole graph."""
    for op in block.operations:
        yield op
        for nested in op.blocks:
            yield from walk_operations(nested)


def summarize(plan: MLComputePlan) -> dict:
    """Summarize every costed ML Program operation by preferred compute device."""

    program = plan.model_structure.program
    if program is None:
        raise ValueError("Not an ML Program model")
    preferred_hist: dict = {}
    op_type_hist: dict = {}
    cpu_ops: list = []
    total = 0
    for func_name, func in program.functions.items():
        for op in walk_operations(func.block):
            # const ops carry no compute; usage comes back None for them.
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            total += 1
            label = device_label(usage.preferred_compute_device)
            preferred_hist[label] = preferred_hist.get(label, 0) + 1
            bucket = op_type_hist.setdefault(op.operator_name, {})
            bucket[label] = bucket.get(label, 0) + 1
            if label == "cpu":
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                cpu_ops.append({
                    "op": op.operator_name,
                    "outputs": [o.name for o in op.outputs][:1],
                    "cost_weight": cost.weight if cost else None,
                })
    cpu_ops.sort(key=lambda r: -(r["cost_weight"] or 0))
    preferred_fractions = (
        {label: count / total for label, count in preferred_hist.items()}
        if total
        else {}
    )
    return {
        "total_costed_ops": total,
        "preferred_device_histogram": preferred_hist,
        "preferred_device_fractions": preferred_fractions,
        "per_op_type": op_type_hist,
        "cpu_preferred_ops": cpu_ops,
    }


async def load_plan(
    compiled: Path,
    units: ct.ComputeUnit,
    device_name: str | None,
    device_type_name: str,
    development_team: str,
    bundle_identifier: str,
    provisioning_profile_uuid: str | None,
) -> tuple[MLComputePlan, dict]:
    """Load a host plan or a plan compiled on the selected mobile device."""

    if device_name is None:
        return MLComputePlan.load_from_path(str(compiled), compute_units=units), {"where": "host"}
    from coremltools.models.ml_program.experimental.compute_plan_utils import (
        load_compute_plan_from_path_on_device,
    )
    from coremltools.models.ml_program.experimental.remote_device import (
        AppSigningCredentials,
        Device,
        DeviceState,
        DeviceType,
    )
    # coremltools 8.3 API (newer docs call this get_connected_development_devices).
    device_types = {"iphone": DeviceType.IPHONE, "ipad": DeviceType.IPAD}
    device_type = device_types[device_type_name]
    devices = Device.get_connected_devices(device_type=device_type)
    matches = [d for d in devices if device_name.lower() in d.name.lower()]
    if not matches:
        known_devices = Device.get_devices()
        candidates = [
            device
            for device in known_devices
            if device.type == device_type
            and device_name.lower() in device.name.lower()
        ]
        if not candidates:
            raise SystemExit(
                f"No {device_type_name} matching {device_name!r}; "
                f"saw: {[d.name for d in known_devices]}"
            )
        matches = [candidates[0]]
    device, device_provenance = refresh_mobile_device_connection(
        matches[0], DeviceState
    )
    # prepare_for_model_debugging installs a signed model-runner harness app on
    # the phone; team 6ETYBAJKY8 matches ios-bench/project.yml signing.
    credentials = AppSigningCredentials(
        development_team=development_team,
        bundle_identifier=bundle_identifier,
        provisioning_profile_uuid=provisioning_profile_uuid,
    )
    runner_working_directory = ensure_model_runner_app(
        device,
        development_team,
        bundle_identifier,
        provisioning_profile_uuid,
    )
    device = await device.prepare_for_model_debugging(
        credentials=credentials,
        working_directory=runner_working_directory,
    )
    plan = await load_compute_plan_from_path_on_device(
        path=str(compiled), compute_units=units, device=device
    )
    return plan, {
        "where": "device",
        "device_type": device_type_name,
        "signing_bundle_identifier": bundle_identifier,
        "signing_development_team": development_team,
        "signing_mode": "explicit_profile"
        if provisioning_profile_uuid
        else "automatic",
        "signing_provisioning_profile_uuid": provisioning_profile_uuid,
        **device_provenance,
    }


def main() -> None:
    """Compile one package, load its compute plan, and persist the summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--compute-units", required=True, choices=sorted(COMPUTE_UNITS))
    parser.add_argument("--device-name", default=None,
                        help="Substring of the CoreDevice name (e.g. 'Webcam'); omit for host-side")
    parser.add_argument(
        "--device-type",
        choices=("iphone", "ipad"),
        default="iphone",
        help="Mobile device family used with --device-name",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--development-team",
        default=DEFAULT_DEVELOPMENT_TEAM,
        help="Apple development team used to sign the on-device model runner",
    )
    parser.add_argument(
        "--bundle-identifier",
        default=DEFAULT_BUNDLE_IDENTIFIER,
        help="bundle identifier for the on-device model runner",
    )
    parser.add_argument(
        "--provisioning-profile-uuid",
        default=None,
        help=(
            "installed development profile UUID; bypasses automatic portal "
            "updates and device registration"
        ),
    )
    args = parser.parse_args()

    units = COMPUTE_UNITS[args.compute_units]
    with tempfile.TemporaryDirectory() as tmp:
        compiled = compile_package(args.package, Path(tmp))
        plan, provenance = asyncio.run(
            load_plan(
                compiled,
                units,
                args.device_name,
                args.device_type,
                args.development_team,
                args.bundle_identifier,
                args.provisioning_profile_uuid,
            )
        )
        summary = summarize(plan)

    summary["provenance"] = {
        **provenance,
        "package": str(args.package),
        "package_sha256": package_tree_sha256(args.package),
        "compute_units": args.compute_units,
        "coremltools": ct.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    hist = summary["preferred_device_histogram"]
    print(f"{args.package.name} [{args.compute_units}] {summary['provenance']['where']}: "
          f"{summary['total_costed_ops']} ops, preferred={hist}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
