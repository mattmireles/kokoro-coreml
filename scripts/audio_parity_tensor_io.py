#!/usr/bin/env python3
"""Shared tensor dump I/O for audio parity debugging.

The format is intentionally small and language-neutral:

``tensor_manifest.json`` records tensor names, dtypes, shapes, and raw file
paths. Tensor payloads are little-endian ``float32`` or ``int32`` binaries so
Swift can write them without a NumPy dependency and Python can load them
directly.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
MANIFEST_NAME = "tensor_manifest.json"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "tensor"


def _summary(array: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(array).reshape(-1)
    if flat.size == 0:
        return {"count": 0}
    if np.issubdtype(flat.dtype, np.floating):
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return {
                "count": int(flat.size),
                "finite_count": 0,
                "nan_count": int(np.isnan(flat).sum()),
                "inf_count": int(np.isinf(flat).sum()),
            }
        return {
            "count": int(flat.size),
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(flat).sum()),
            "inf_count": int(np.isinf(flat).sum()),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "l2": float(np.sqrt(np.mean(finite.astype(np.float64) ** 2))),
        }
    flat64 = flat.astype(np.int64)
    return {
        "count": int(flat.size),
        "min": int(flat64.min()),
        "max": int(flat64.max()),
        "mean": float(flat64.mean()),
    }


def bucket_mask_from_tensors(
    tensors: dict[str, np.ndarray],
    target_frames: int,
    *,
    frame_scale: int = 2,
) -> np.ndarray:
    """Build a padded-stage mask from a dump's valid duration frames.

    Generator inputs follow the decoder's final 2x upsample. Decoder-pre
    callers pass ``frame_scale=1``.
    """
    if "pred_dur_valid" not in tensors:
        raise KeyError("tensor dump has no pred_dur_valid for required mask construction")
    valid_frames = int(np.asarray(tensors["pred_dur_valid"], dtype=np.int64).sum()) * frame_scale
    mask = np.zeros((1, 1, target_frames), dtype=np.float32)
    mask[:, :, : min(max(valid_frames, 0), target_frames)] = 1.0
    return mask


def mask_aware_inputs(
    model: Any,
    inputs: dict[str, np.ndarray],
    tensors: dict[str, np.ndarray],
    *,
    frame_scale: int = 2,
) -> dict[str, np.ndarray]:
    """Attach the required mask when a loaded Core ML artifact declares it."""
    shapes = {
        item.name: tuple(int(value) for value in item.type.multiArrayType.shape)
        for item in model.get_spec().description.input
    }
    mask = bucket_mask_from_tensors(tensors, shapes["mask"][-1], frame_scale=frame_scale) if "mask" in shapes else None
    return inputs_with_mask_if_declared(model, inputs, mask)


def inputs_with_mask_if_declared(
    model: Any,
    inputs: dict[str, np.ndarray],
    mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Add a precomputed mask only to artifacts that declare the input."""
    input_names = {item.name for item in model.get_spec().description.input}
    if "mask" not in input_names:
        return inputs
    if mask is None:
        raise ValueError("mask-aware Core ML artifact requires a validity mask")
    return {**inputs, "mask": mask}


class TensorDumpWriter:
    """Write tensors to a parity dump directory."""

    def __init__(self, directory: Path | str, metadata: dict[str, Any] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[str, Any] = metadata or {}
        self.records: list[dict[str, Any]] = []

    def write(self, name: str, array: Any) -> None:
        arr = np.asarray(array)
        if np.issubdtype(arr.dtype, np.integer):
            dtype = "int32"
            payload = arr.astype("<i4", copy=False)
            suffix = "i32"
        else:
            dtype = "float32"
            payload = arr.astype("<f4", copy=False)
            suffix = "f32"

        filename = f"{_safe_name(name)}.{suffix}"
        payload.tofile(self.directory / filename)
        self.records.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": [int(v) for v in arr.shape],
                "path": filename,
                "summary": _summary(payload),
            }
        )

    def close(self, metadata: dict[str, Any] | None = None) -> Path:
        if metadata:
            self.metadata.update(metadata)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.metadata,
            "tensors": self.records,
        }
        path = self.directory / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return path


def load_tensor_dump(directory: Path | str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load a tensor dump directory created by Python or Swift."""

    root = Path(directory)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported tensor dump schema: {manifest.get('schema_version')!r}")

    tensors: dict[str, np.ndarray] = {}
    for record in manifest.get("tensors", []):
        dtype = record["dtype"]
        if dtype == "float32":
            np_dtype = np.dtype("<f4")
        elif dtype == "int32":
            np_dtype = np.dtype("<i4")
        else:
            raise ValueError(f"unsupported tensor dtype for {record['name']}: {dtype}")

        shape = tuple(int(v) for v in record["shape"])
        arr = np.fromfile(root / record["path"], dtype=np_dtype)
        expected = math.prod(shape) if shape else 1
        if int(arr.size) != int(expected):
            raise ValueError(
                f"tensor {record['name']} has {arr.size} values, expected {expected} for shape {shape}"
            )
        tensors[record["name"]] = arr.reshape(shape)

    return manifest, tensors
