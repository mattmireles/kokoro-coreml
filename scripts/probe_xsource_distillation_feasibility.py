#!/usr/bin/env python3
"""Probe whether cheap source features can predict strict source-side tensors.

This is a fast feasibility check for the highest-priority source-contract
experiment. It does not export Core ML. It asks whether a small, static,
bucket-specific linear/Conv1d-like adapter over cheap Swift-side source features
could approximate pieces of the expensive strict HAR -> noise_convs/noise_res
path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _load_kmodel, _metrics  # noqa: E402


def _resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a 1D array to ``target_len`` points."""

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if flat.size == target_len:
        return flat.astype(np.float32, copy=True)
    if flat.size == 1:
        return np.full((target_len,), float(flat[0]), dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=flat.size, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    return np.interp(dst_x, src_x, flat.astype(np.float64)).astype(np.float32)


def _window_features(series: list[np.ndarray], radius: int) -> np.ndarray:
    """Build local-window features shaped ``[T, F]`` from aligned series."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    normalized = []
    for values in series:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError(f"expected 1D or 2D features, got shape {arr.shape}")
        normalized.append(arr)
    lengths = {int(values.shape[0]) for values in normalized}
    if len(lengths) != 1:
        raise ValueError(f"series lengths differ: {sorted(lengths)}")
    length = next(iter(lengths))
    columns = [np.ones((length, 1), dtype=np.float32)]
    for values in normalized:
        padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
        for offset in range(2 * radius + 1):
            columns.append(padded[offset : offset + length])
    return np.concatenate(columns, axis=1).astype(np.float32)


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, ridge: float) -> np.ndarray:
    """Fit multi-output ridge regression weights."""

    xtx = x_train.T @ x_train
    penalty = np.eye(xtx.shape[0], dtype=np.float32) * np.float32(ridge)
    penalty[0, 0] = 0.0
    xty = x_train.T @ y_train
    return np.linalg.solve((xtx + penalty).astype(np.float64), xty.astype(np.float64)).astype(np.float32)


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    hidden: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit a small per-frame MLP and return train/validation predictions."""

    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden),
        nn.SiLU(),
        nn.Linear(hidden, y_train.shape[1]),
    ).to(device)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0

    xt = torch.from_numpy(((x_train - x_mean) / x_std).astype(np.float32)).to(device)
    yt = torch.from_numpy(((y_train - y_mean) / y_std).astype(np.float32)).to(device)
    xv = torch.from_numpy(((x_val - x_mean) / x_std).astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for step in range(int(steps)):
        pred = model(xt)
        loss = torch.mean((pred - yt) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 5) == 0:
            history.append({"step": float(step + 1), "train_loss": float(loss.detach().cpu())})
    with torch.no_grad():
        train_pred = model(xt).cpu().numpy() * y_std + y_mean
        val_pred = model(xv).cpu().numpy() * y_std + y_mean
    return train_pred.astype(np.float32), val_pred.astype(np.float32), history


def _fit_conv1d(
    features: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    hidden: int,
    steps: int,
    learning_rate: float,
    seed: int,
    conv_kernel: int,
    conv_depth: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit a temporal Conv1d adapter and return train/validation predictions."""

    import torch
    import torch.nn as nn

    if conv_kernel < 1 or conv_kernel % 2 != 1:
        raise ValueError("conv_kernel must be a positive odd integer")
    if conv_depth < 1:
        raise ValueError("conv_depth must be >= 1")

    torch.manual_seed(seed)
    device = torch.device("cpu")
    x_mean = features.mean(axis=0, keepdims=True)
    x_std = features.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = target.mean(axis=0, keepdims=True)
    y_std = target.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0

    x_norm = ((features - x_mean) / x_std).astype(np.float32)
    y_norm = ((target - y_mean) / y_std).astype(np.float32)
    x_tensor = torch.from_numpy(x_norm.T[None, :, :]).to(device)
    y_tensor = torch.from_numpy(y_norm.T[None, :, :]).to(device)
    train_mask = torch.zeros((1, 1, target.shape[0]), dtype=torch.float32, device=device)
    train_mask[:, :, train_idx] = 1.0
    val_mask = torch.zeros((1, 1, target.shape[0]), dtype=torch.float32, device=device)
    val_mask[:, :, val_idx] = 1.0

    layers: list[nn.Module] = [
        nn.Conv1d(features.shape[1], hidden, kernel_size=conv_kernel, padding=conv_kernel // 2),
        nn.SiLU(),
    ]
    for _ in range(conv_depth - 1):
        layers.extend(
            [
                nn.Conv1d(hidden, hidden, kernel_size=conv_kernel, padding=conv_kernel // 2),
                nn.SiLU(),
            ]
        )
    layers.append(nn.Conv1d(hidden, target.shape[1], kernel_size=1))
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []

    train_count = train_mask.sum() * target.shape[1]
    val_count = val_mask.sum() * target.shape[1]
    for step in range(int(steps)):
        pred = model(x_tensor)
        diff = (pred - y_tensor) ** 2
        loss = (diff * train_mask).sum() / train_count
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 5) == 0:
            with torch.no_grad():
                fresh_diff = (model(x_tensor) - y_tensor) ** 2
                val_loss = (fresh_diff * val_mask).sum() / val_count
            history.append(
                {
                    "step": float(step + 1),
                    "train_loss": float(loss.detach().cpu()),
                    "validation_loss": float(val_loss.detach().cpu()),
                }
            )

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy()[0].T * y_std + y_mean
    return pred[train_idx].astype(np.float32), pred[val_idx].astype(np.float32), history


def _split_indices(length: int, holdout_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return train/validation row indices using an interleaved holdout."""

    if holdout_stride < 2:
        raise ValueError("holdout_stride must be >= 2")
    indices = np.arange(length)
    val = indices[(indices % holdout_stride) == (holdout_stride - 1)]
    train = indices[(indices % holdout_stride) != (holdout_stride - 1)]
    return train, val


def _target_source_tensors(tensors: dict[str, np.ndarray], target_mode: str) -> list[np.ndarray]:
    """Compute strict source-side target tensors from dumped HAR and style."""

    import torch

    gen = _load_kmodel().decoder.generator.eval()
    har = torch.from_numpy(tensors["har_padded"].astype(np.float32))
    style = torch.from_numpy(tensors["ref_s"][:, :128].astype(np.float32))
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for conv, res in zip(gen.noise_convs, gen.noise_res):
            pre_residual = conv(har)
            if target_mode == "pre_noise_conv":
                value = pre_residual.detach().cpu().numpy().astype(np.float32)
            elif target_mode == "x_source":
                value = res(pre_residual, style).detach().cpu().numpy().astype(np.float32)
            else:
                raise ValueError(f"unsupported target mode: {target_mode}")
            outputs.append(value)
    return outputs


def _resample_channels(values: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a ``[C,T]`` or ``[B,C,T]`` tensor into ``[T,C]`` features."""

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected channel tensor, got shape {arr.shape}")
    channels = [_resample_1d(arr[idx], target_len) for idx in range(arr.shape[0])]
    return np.stack(channels, axis=1).astype(np.float32)


def _conv_geometry_har_features(tensors: dict[str, np.ndarray], target_index: int, target_len: int) -> np.ndarray:
    """Return exact HAR receptive-field rows for ``noise_convs[target_index]``."""

    gen = _load_kmodel().decoder.generator.eval()
    conv = gen.noise_convs[target_index]
    har = np.asarray(tensors["har_padded"], dtype=np.float32)
    if har.ndim != 3 or har.shape[0] != 1:
        raise ValueError(f"expected har_padded shape [1,C,T], got {har.shape}")
    stride = int(conv.stride[0])
    padding = int(conv.padding[0])
    dilation = int(conv.dilation[0])
    kernel = int(conv.kernel_size[0])
    channels = int(har.shape[1])
    time = int(har.shape[2])
    features = np.empty((target_len, channels * kernel + 1), dtype=np.float32)
    features[:, 0] = 1.0
    for output_index in range(target_len):
        base = output_index * stride - padding
        column = 1
        for kernel_index in range(kernel):
            source_index = base + kernel_index * dilation
            if 0 <= source_index < time:
                features[output_index, column : column + channels] = har[0, :, source_index]
            else:
                features[output_index, column : column + channels] = 0.0
            column += channels
    return features


def _features_for_target(
    tensors: dict[str, np.ndarray],
    target_len: int,
    radius: int,
    feature_set: str,
    target_index: int,
) -> np.ndarray:
    """Return cheap source-side features aligned to a target ``x_source`` length."""

    har_source = _resample_1d(tensors["har_source"], target_len)
    f0 = _resample_1d(tensors["f0_padded"], target_len)
    n = _resample_1d(tensors["n_padded"], target_len)
    voiced = (f0 > 1.0).astype(np.float32)
    # Keep features Core ML-friendly: local source/F0/noise windows plus simple
    # voiced masking. Nonlinear trig/STFT features would defeat this probe's
    # purpose, which is testing a cheap adapter.
    cheap = [har_source, f0, n, voiced]
    if feature_set == "cheap":
        return _window_features(cheap, radius)
    har = _resample_channels(tensors["har_padded"], target_len)
    if feature_set == "har":
        return _window_features([har], radius)
    if feature_set == "cheap_har":
        return _window_features([*cheap, har], radius)
    if feature_set == "har_conv_geometry":
        return _conv_geometry_har_features(tensors, target_index, target_len)
    raise ValueError(f"unsupported feature set: {feature_set}")


def _summarize_target(
    *,
    name: str,
    target: np.ndarray,
    features: np.ndarray,
    ridge: float,
    holdout_stride: int,
    model: str,
    hidden: int,
    steps: int,
    learning_rate: float,
    seed: int,
    conv_kernel: int,
    conv_depth: int,
) -> dict[str, Any]:
    """Fit and score one target tensor."""

    y = np.moveaxis(target[0], 0, -1).reshape(target.shape[-1], target.shape[1]).astype(np.float32)
    if y.shape[0] != features.shape[0]:
        raise ValueError(f"{name}: feature rows {features.shape[0]} != target rows {y.shape[0]}")
    train_idx, val_idx = _split_indices(y.shape[0], holdout_stride)
    history: list[dict[str, float]] = []
    if model == "ridge":
        weights = _fit_ridge(features[train_idx], y[train_idx], ridge)
        train_pred = features[train_idx] @ weights
        val_pred = features[val_idx] @ weights
    elif model == "mlp":
        train_pred, val_pred, history = _fit_mlp(
            features[train_idx],
            y[train_idx],
            features[val_idx],
            hidden=hidden,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed,
        )
    elif model == "conv1d":
        train_pred, val_pred, history = _fit_conv1d(
            features,
            y,
            train_idx,
            val_idx,
            hidden=hidden,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed,
            conv_kernel=conv_kernel,
            conv_depth=conv_depth,
        )
    else:
        raise ValueError(f"unsupported model: {model}")
    train_metrics = _metrics(y[train_idx].reshape(-1), train_pred.reshape(-1))
    val_metrics = _metrics(y[val_idx].reshape(-1), val_pred.reshape(-1))
    return {
        "name": name,
        "target_shape": [int(v) for v in target.shape],
        "feature_shape": [int(v) for v in features.shape],
        "train_rows": int(train_idx.size),
        "validation_rows": int(val_idx.size),
        "model": model,
        "ridge": float(ridge),
        "hidden": int(hidden),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "conv_kernel": int(conv_kernel),
        "conv_depth": int(conv_depth),
        "history": history,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the distillation feasibility probe."""

    _manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["har_source", "f0_padded", "n_padded", "har_padded", "ref_s"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"{args.tensor_dump} missing required tensors: {', '.join(missing)}")
    targets = _target_source_tensors(tensors, args.target_mode)
    rows = []
    for idx, target in enumerate(targets):
        features = _features_for_target(tensors, int(target.shape[-1]), args.radius, args.feature_set, idx)
        rows.append(
            _summarize_target(
                name=f"{args.target_mode}_{idx}",
                target=target,
                features=features,
                ridge=args.ridge,
                holdout_stride=args.holdout_stride,
                model=args.model,
                hidden=args.hidden,
                steps=args.steps,
                learning_rate=args.learning_rate,
                seed=args.seed + idx,
                conv_kernel=args.conv_kernel,
                conv_depth=args.conv_depth,
            )
        )
    payload = {
        "tensor_dump": str(args.tensor_dump),
        "target_mode": args.target_mode,
        "model": args.model,
        "feature_set": args.feature_set,
        "radius": args.radius,
        "holdout_stride": args.holdout_stride,
        "ridge": args.ridge,
        "hidden": args.hidden,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "conv_kernel": args.conv_kernel,
        "conv_depth": args.conv_depth,
        "rows": rows,
        "decision": _decision(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _decision(rows: list[dict[str, Any]]) -> str:
    """Return a conservative interpretation of validation metrics."""

    min_snr = min(float(row["validation_metrics"]["snr_db"]) for row in rows)
    min_corr = min(float(row["validation_metrics"]["correlation"]) for row in rows)
    if min_snr >= 45.0 and min_corr >= 0.999:
        return "promising_adapter"
    if min_snr >= 25.0 and min_corr >= 0.98:
        return "needs_nonlinear_or_multibucket_adapter"
    return "cheap_adapter_not_enough"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact markdown report."""

    lines = [
        "# x_source Distillation Feasibility",
        "",
        f"- Tensor dump: `{payload['tensor_dump']}`.",
        f"- Target mode: `{payload['target_mode']}`.",
        f"- Model: `{payload['model']}`.",
        f"- Feature set: `{payload['feature_set']}`.",
        f"- Radius: `{payload['radius']}`.",
        f"- Holdout stride: `{payload['holdout_stride']}`.",
        f"- Ridge: `{payload['ridge']}`.",
        f"- Hidden: `{payload['hidden']}`.",
        f"- Steps: `{payload['steps']}`.",
        f"- Learning rate: `{payload['learning_rate']}`.",
        f"- Conv kernel: `{payload['conv_kernel']}`.",
        f"- Conv depth: `{payload['conv_depth']}`.",
        f"- Decision: `{payload['decision']}`.",
        "",
        "| Target | Features | Train SNR | Val SNR | Val corr | Val max abs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        train = row["train_metrics"]
        val = row["validation_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    str(row["feature_shape"][1]),
                    f"{float(train['snr_db']):.2f} dB",
                    f"{float(val['snr_db']):.2f} dB",
                    f"{float(val['correlation']):.6f}",
                    f"{float(val['max_abs_error']):.6f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument("--target-mode", choices=("x_source", "pre_noise_conv"), default="x_source")
    parser.add_argument("--model", choices=("ridge", "mlp", "conv1d"), default="ridge")
    parser.add_argument(
        "--feature-set",
        choices=("cheap", "har", "cheap_har", "har_conv_geometry"),
        default="cheap",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--conv-kernel", type=int, default=9)
    parser.add_argument("--conv-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--output", type=Path, default=Path("outputs/xsource_distillation_feasibility/report.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("outputs/xsource_distillation_feasibility/report.md"))
    args = parser.parse_args()

    payload = run(args)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(payload))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown_output": str(args.markdown_output),
                "decision": payload["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
