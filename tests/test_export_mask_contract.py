"""Dependency-free source contract for required padded-package masks."""

from __future__ import annotations

import ast
from pathlib import Path


def test_every_exported_bucket_mask_is_required_by_source_contract() -> None:
    """Keep the loud mask break covered when generated packages are absent."""
    root = Path(__file__).resolve().parents[1]
    mask_inputs: list[tuple[Path, int, set[str]]] = []
    for path in (
        root / "export_decoder_pre.py",
        root / "export_f0ntrain.py",
        root / "export_synth" / "convert.py",
    ):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "TensorType":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            name = keywords.get("name")
            if isinstance(name, ast.Constant) and name.value == "mask":
                mask_inputs.append((path, node.lineno, set(keywords)))

    # Four fixed-bucket masks plus the flexible (RangeDim) generator's mask.
    assert len(mask_inputs) == 5, mask_inputs
    assert all("default_value" not in keywords for _, _, keywords in mask_inputs), mask_inputs
