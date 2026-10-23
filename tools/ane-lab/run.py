"""Run one ANE-lab job on a tethered iPhone and pull the result.

Usage:
    python tools/ane-lab/run.py --device UDID --model path/to/x.mlpackage \
        [--units ne,gpu] [--inputs-dir dir_with_bins] [--save-outputs] --out results_dir

Copies the model (and optional <input>.bin files) into the ANELab app's
Documents folder, launches the app with a job file, waits for it to exit, and
copies result.json (plus out_*.bin when --save-outputs) into --out.
The app must already be installed (see tools/ane-lab/project.yml).
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

BUNDLE = "com.mattmireles.ANELab"


def devicectl(*args: str) -> str:
    cmd = ["xcrun", "devicectl", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def copy_to(device: str, src: Path, dest: str) -> None:
    devicectl("device", "copy", "to", "--device", device, "--domain-type", "appDataContainer",
              "--domain-identifier", BUNDLE, "--source", str(src), "--destination", f"Documents/{dest}")


def copy_from(device: str, src: str, dest: Path) -> None:
    devicectl("device", "copy", "from", "--device", device, "--domain-type", "appDataContainer",
              "--domain-identifier", BUNDLE, "--source", f"Documents/{src}", "--destination", str(dest))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--units", default="ne,gpu")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--inputs-dir", type=Path)
    ap.add_argument("--save-outputs", action="store_true")
    ap.add_argument("--function", help="function of a multifunction package")
    ap.add_argument("--io", choices=["pointer", "iosurface"], default="pointer",
                    help="iosurface: fp16 I/O as IOSurface-backed arrays with output backings")
    ap.add_argument("--between-ms", type=float, help="sleep between predictions")
    ap.add_argument("--between-gpu", action="store_true", help="GPU writes the input surfaces before each call")
    ap.add_argument("--rotate", help="comma list of same-shape functions to call round-robin with --function")
    ap.add_argument("--metal-compat", action="store_true", help="create surfaces Metal-compatible, as the pipeline does")
    ap.add_argument("--surface-sets", type=int, help="rotate through N distinct I/O surface sets")
    ap.add_argument("--gpu-outputs", action="store_true", help="with --between-gpu: the GPU pass also touches the output surfaces")
    ap.add_argument("--e5chain", type=int, help="steps of GPU pass + ANE op, sync vs event-chained (E5Chain.swift)")
    ap.add_argument("--e5rt", action="store_true", help="also dispatch through the private e5rt API (E5Direct.swift)")
    ap.add_argument("--no-copy", action="store_true", help="model already on the device")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # Unique on-device name: every probe writes gen_fold.mlpackage, and a copy
    # onto an existing package directory would merge stale files into it.
    name = f"{a.model.parent.name}_{a.model.name}"
    if not a.no_copy:
        copy_to(a.device, a.model, name)
    job = {"model": name, "units": a.units.split(","), "warmup": a.warmup,
           "iterations": a.iterations, "save_outputs": a.save_outputs, "io": a.io}
    if a.function:
        job["function"] = a.function
    if a.rotate:
        job["rotate"] = a.rotate.split(",")
    if a.e5rt:
        job["e5rt"] = True
    if a.e5chain:
        job["e5chain_steps"] = a.e5chain
    if a.surface_sets:
        job["surface_sets"] = a.surface_sets
    if a.between_gpu:
        job["between_gpu"] = True
    if a.between_ms is not None:
        job["between_ms"] = a.between_ms
    if a.inputs_dir:
        copy_to(a.device, a.inputs_dir, "in")
        job["inputs_dir"] = "in"
    with tempfile.TemporaryDirectory() as tmp:
        jp = Path(tmp) / "job.json"
        jp.write_text(json.dumps(job))
        copy_to(a.device, jp, "job.json")

    # --console blocks until the app exits (it calls exit(0) after LABDONE).
    log = devicectl("device", "process", "launch", "--console", "--terminate-existing",
                    "--device", a.device, BUNDLE, "--job", "job.json", *(["--metal-compat"] if a.metal_compat else []), *(["--gpu-outputs"] if a.gpu_outputs else []))
    (a.out / "console.log").write_text(log)
    copy_from(a.device, "result.json", a.out / "result.json")
    result = json.loads((a.out / "result.json").read_text())
    for row in result.get("rows", []):
        for f in row.get("saved", []):
            copy_from(a.device, f, a.out / f)
    for row in result.get("rows", []):
        print(row.get("units"), "median_ms", row.get("median_ms"), "min_ms", row.get("min_ms"),
              "plan", row.get("plan"), "peak_MB", (row.get("footprint_peak") or 0) / 2**20,
              "io", row.get("io"), "err", row.get("error"))
    if "error" in result:
        print("ERROR", result["error"])


if __name__ == "__main__":
    main()
