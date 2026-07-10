"""Dump a Core ML compute plan — per-op device placement — host-side or ON a
connected development iPhone.

Phase 2 instrument of README/Plans/kokoro-iphone-performance-v1.md: settles
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
import json
import subprocess
import sys
import tempfile
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
    return {
        "total_costed_ops": total,
        "preferred_device_histogram": preferred_hist,
        "per_op_type": op_type_hist,
        "cpu_preferred_ops": cpu_ops,
    }


async def load_plan(compiled: Path, units: ct.ComputeUnit, device_name: str | None) -> tuple[MLComputePlan, dict]:
    if device_name is None:
        return MLComputePlan.load_from_path(str(compiled), compute_units=units), {"where": "host"}
    from coremltools.models.ml_program.experimental.compute_plan_utils import (
        load_compute_plan_from_path_on_device,
    )
    from coremltools.models.ml_program.experimental.remote_device import (
        AppSigningCredentials,
        Device,
        DeviceType,
    )
    # coremltools 8.3 API (newer docs call this get_connected_development_devices).
    devices = Device.get_connected_devices(device_type=DeviceType.IPHONE)
    matches = [d for d in devices if device_name.lower() in d.name.lower()]
    if not matches:
        raise SystemExit(
            f"No connected iPhone matching {device_name!r}; saw: {[d.name for d in devices]}"
        )
    # prepare_for_model_debugging installs a signed model-runner harness app on
    # the phone; team 6ETYBAJKY8 matches ios-bench/project.yml signing.
    credentials = AppSigningCredentials(
        development_team="6ETYBAJKY8",
        bundle_identifier="com.mattmireles.CoreMLModelRunner",
    )
    device = await matches[0].prepare_for_model_debugging(credentials=credentials)
    plan = await load_compute_plan_from_path_on_device(
        path=str(compiled), compute_units=units, device=device
    )
    return plan, {"where": "device", "device_name": matches[0].name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--compute-units", required=True, choices=sorted(COMPUTE_UNITS))
    parser.add_argument("--device-name", default=None,
                        help="Substring of the CoreDevice name (e.g. 'Webcam'); omit for host-side")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    units = COMPUTE_UNITS[args.compute_units]
    with tempfile.TemporaryDirectory() as tmp:
        compiled = compile_package(args.package, Path(tmp))
        plan, provenance = asyncio.run(load_plan(compiled, units, args.device_name))
        summary = summarize(plan)

    summary["provenance"] = {
        **provenance,
        "package": str(args.package),
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
