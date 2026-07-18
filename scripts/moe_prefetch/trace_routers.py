#!/usr/bin/env python3
"""Capture Stage 1 MoE router traces from a Hugging Face causal LM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.moe_prefetch.schema import machine_info, utc_now


def _parse_args() -> argparse.Namespace:
    """Parse Stage 1 router trace CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="hf-internal-testing/tiny-random-MixtralForCausalLM")
    parser.add_argument("--prompt-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def _torch_device(requested: str) -> str:
    """Resolve the requested PyTorch device for router tracing."""
    import torch

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but torch.backends.mps.is_available() is false")
    return requested


def _prompt_files(prompt_suite: Path) -> list[Path]:
    """Return sorted prompt files from the Stage 1 prompt suite."""
    files = sorted(path for path in prompt_suite.glob("*.txt") if path.is_file())
    if not files:
        raise SystemExit(f"no .txt prompts found under {prompt_suite}")
    return files


def _layer_modules(model: Any) -> list[Any]:
    """Return transformer layer modules for Hugging Face MoE causal LMs."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    return list(layers) if layers is not None else []


def main() -> int:
    """Capture router logits and per-layer timing rows for Stage 1 replay."""
    args = _parse_args()
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _torch_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.eval()
    top_k = int(getattr(model.config, "num_experts_per_tok", 2))
    machine = machine_info(None)
    output_rows: list[dict[str, Any]] = []

    for prompt_index, prompt_path in enumerate(_prompt_files(args.prompt_suite)):
        prompt = prompt_path.read_text().strip()
        if not prompt:
            continue
        request_id = f"{prompt_path.stem}-{prompt_index}"
        layer_starts: dict[int, int] = {}
        layer_times: dict[int, int] = {}
        handles = []
        for layer_index, layer in enumerate(_layer_modules(model)):
            handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, _inputs, idx=layer_index: layer_starts.__setitem__(
                        idx,
                        time.perf_counter_ns(),
                    )
                )
            )
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, _output, idx=layer_index: layer_times.__setitem__(
                        idx,
                        time.perf_counter_ns() - layer_starts.get(idx, time.perf_counter_ns()),
                    )
                )
            )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True, use_cache=False)
        for handle in handles:
            handle.remove()

        router_logits = getattr(outputs, "router_logits", None)
        if router_logits is None:
            raise SystemExit(f"{args.model_id} did not return router_logits")

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for layer_index, logits in enumerate(router_logits):
            cpu_logits = logits.detach().float().cpu()
            top = torch.topk(cpu_logits, k=min(top_k, cpu_logits.shape[-1]), dim=-1).indices.tolist()
            for token_index, expert_ids in enumerate(top):
                output_rows.append(
                    {
                        "created_at": utc_now(),
                        "model_id": args.model_id,
                        "request_id": request_id,
                        "domain": prompt_path.stem,
                        "prompt_sha256": prompt_hash,
                        "token_index": token_index,
                        "layer_index": layer_index,
                        "router_input_shape": list(cpu_logits.shape),
                        "actual_topk_expert_ids": [int(v) for v in expert_ids],
                        "layer_compute_ns": int(layer_times.get(layer_index, 0)),
                        "machine": machine,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in output_rows) + "\n")
    print(f"wrote {args.output}")
    print(f"rows={len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
