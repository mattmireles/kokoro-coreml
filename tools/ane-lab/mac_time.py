"""Time one Core ML package on this Mac (the ANE-lab counterpart for Macs).

Usage: python tools/ane-lab/mac_time.py PACKAGE INPUTS_DIR [--units ne|gpu|all] [--iters 10]

Inputs are <name>.bin float32 files matching the model's input shapes. Prints
median/min predict time and the process peak physical footprint.
"""
import argparse
import resource
import time
from pathlib import Path

import coremltools as ct
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("package")
ap.add_argument("inputs_dir")
ap.add_argument("--units", default="ne")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--save", help="write each output as <save>_<name>.bin (float32)")
a = ap.parse_args()
units = {"ne": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL}[a.units]
t = time.time()
m = ct.models.MLModel(a.package, compute_units=units)
load_s = time.time() - t
spec = {i.name: tuple(i.type.multiArrayType.shape) for i in m.get_spec().description.input}
def feed(name, shape):
    f = Path(a.inputs_dir) / f"{name}.bin"
    if f.exists():
        return np.fromfile(f, dtype=np.float32).reshape(shape)
    return np.random.rand(*shape).astype(np.float32) + 0.5  # positive: masks/fills stay sane
feeds = {k: feed(k, s) for k, s in spec.items()}
times = []
for i in range(a.warmup + a.iters):
    t = time.time()
    out = m.predict(feeds)
    if i >= a.warmup:
        times.append(time.time() - t)
times.sort()
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"{a.units}: median {times[len(times)//2]*1000:.1f} ms min {times[0]*1000:.1f} ms load {load_s:.1f} s maxrss {peak_mb:.0f} MB")
if a.save:
    for k, v in out.items():
        np.asarray(v, dtype=np.float32).tofile(f"{a.save}_{k}.bin")
