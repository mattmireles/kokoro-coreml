#!/usr/bin/env python3
"""Predictor baselines for Stage 1 MoE router traces."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TraceRow = dict[str, Any]


def load_trace(path: Path) -> list[TraceRow]:
    """Load JSONL router trace rows from `path`."""
    rows: list[TraceRow] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _row_key(row: TraceRow) -> tuple[str, int, int]:
    return (str(row["request_id"]), int(row["token_index"]), int(row["layer_index"]))


def _actual(row: TraceRow) -> tuple[int, ...]:
    return tuple(int(v) for v in row["actual_topk_expert_ids"])


def evaluate_policies(
    rows: list[TraceRow],
    *,
    prefetch_depth: int,
    required_depth: int,
) -> dict[str, dict[str, float]]:
    """Evaluate deterministic Stage 1 policies against router trace rows.

    `required_depth` comes from Stage 0 hideability: if the configured
    prefetch depth is smaller, correct predictions are still marked as not
    hideable because the SSD read cannot arrive before use.
    """
    ordered = sorted(rows, key=lambda row: (str(row["request_id"]), int(row["token_index"]), int(row["layer_index"])))
    by_key = {_row_key(row): row for row in ordered}
    layer_counts: dict[int, Counter[int]] = defaultdict(Counter)
    previous_by_layer: dict[tuple[str, int], tuple[int, ...]] = {}
    markov_counts: dict[tuple[int, tuple[int, ...]], Counter[int]] = defaultdict(Counter)
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    hideable = prefetch_depth >= required_depth

    for row in ordered:
        request_id = str(row["request_id"])
        token_index = int(row["token_index"])
        layer_index = int(row["layer_index"])
        actual = set(_actual(row))
        topk = len(actual)
        previous_layer_row = by_key.get((request_id, token_index, layer_index - prefetch_depth))
        previous_layer_actual = _actual(previous_layer_row) if previous_layer_row else ()
        previous_token_actual = previous_by_layer.get((request_id, layer_index), ())
        frequency_prediction = tuple(expert for expert, _ in layer_counts[layer_index].most_common(topk))
        markov_prediction = tuple(
            expert
            for expert, _ in markov_counts[(layer_index, previous_token_actual)].most_common(topk)
        )
        predictions: dict[str, tuple[int, ...]] = {
            "demand_lru": (),
            "last_token": previous_token_actual[:topk],
            "frequency": frequency_prediction,
            "markov": markov_prediction,
            "eam": previous_layer_actual[:topk],
            "oracle": tuple(actual),
        }

        for policy, predicted_tuple in predictions.items():
            predicted = set(predicted_tuple)
            hits = len(actual & predicted)
            totals[policy]["rows"] += 1
            totals[policy]["actual"] += len(actual)
            totals[policy]["predicted"] += len(predicted)
            totals[policy]["hits"] += hits
            totals[policy]["hideable_hits"] += hits if hideable else 0
            totals[policy]["wasted"] += len(predicted - actual)

        for expert in actual:
            layer_counts[layer_index][expert] += 1
        if previous_token_actual:
            for expert in actual:
                markov_counts[(layer_index, previous_token_actual)][expert] += 1
        previous_by_layer[(request_id, layer_index)] = tuple(actual)

    metrics: dict[str, dict[str, float]] = {}
    for policy, counter in totals.items():
        actual_total = float(counter["actual"])
        predicted_total = float(counter["predicted"])
        rows_total = float(counter["rows"])
        metrics[policy] = {
            "rows": rows_total,
            "recall": float(counter["hits"]) / actual_total if actual_total else 0.0,
            "precision": float(counter["hits"]) / predicted_total if predicted_total else 0.0,
            "hideable_recall": float(counter["hideable_hits"]) / actual_total if actual_total else 0.0,
            "wasted_prediction_fraction": float(counter["wasted"]) / predicted_total if predicted_total else 0.0,
        }
    return metrics
