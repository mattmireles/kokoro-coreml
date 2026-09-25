#!/usr/bin/env python3
"""Merge the bucketed Core ML packages of a model set into one multifunction package per stage.

A multifunction ``mlprogram`` (coremltools 8+, macOS 15 / iOS 18) stores the
weights once and exposes one fixed-shape function per bucket, so the duration,
f0ntrain and decoder-pre sets shrink from 217, 98 and 193 MB to 55, 20 and
64 MB on disk. The Swift loaders (``MultifunctionPackages.swift``) pick a
function by name when the package is present and fall back to the separate
packages otherwise. Function names are the contract shared with Swift:
``t{tokens}`` for duration, ``t{frames}`` for f0ntrain, ``bucket_{N}s`` for
decoder-pre. The saving is disk only: each loaded function is resident on its
own, and a GPU function loads 0.2-0.4 s slower than a separate package.

usage: build_multifunction_packages.py --models-dir coreml [--out coreml]
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import coremltools as ct

STAGES = {
    "kokoro_duration_multifunction.mlpackage": (re.compile(r"^kokoro_duration_t(\d+)\.mlpackage$"), lambda n: f"t{n}"),
    "kokoro_f0ntrain_multifunction.mlpackage": (re.compile(r"^kokoro_f0ntrain_t(\d+)\.mlpackage$"), lambda n: f"t{n}"),
    "kokoro_decoder_pre_multifunction.mlpackage": (re.compile(r"^kokoro_decoder_pre_(\d+)s\.mlpackage$"), lambda n: f"bucket_{n}s"),
}


def size_mb(path: Path) -> float:
    return int(subprocess.run(["du", "-sk", "-L", str(path)], capture_output=True, text=True, check=True).stdout.split()[0]) / 1024


def build(models_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for out_name, (pattern, function_name) in STAGES.items():
        members = sorted(
            ((int(m.group(1)), p) for p in models_dir.iterdir() if (m := pattern.match(p.name))),
        )
        if not members:
            print(f"{out_name}: no packages matched in {models_dir}, skipped")
            continue
        desc = ct.utils.MultiFunctionDescriptor()
        for n, p in members:
            desc.add_function(str(p.resolve()), src_function_name="main", target_function_name=function_name(n))
        desc.default_function_name = function_name(members[0][0])
        out = out_dir / out_name
        ct.utils.save_multifunction(desc, str(out))
        print(f"{out_name}: {[function_name(n) for n, _ in members]}, {sum(size_mb(p) for _, p in members):.0f} MB -> {size_mb(out):.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: the models dir")
    args = parser.parse_args()
    build(args.models_dir, args.out or args.models_dir)


if __name__ == "__main__":
    main()
