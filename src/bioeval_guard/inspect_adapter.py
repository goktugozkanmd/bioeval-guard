"""Drop-in Inspect AI / inspect_evals helpers for dataset sanitization and blind baselines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .hygiene import AuditItem, check_mcq_hygiene, normalize_items
from .shortcuts import compute_option_features


def sanitize_mcq_dataset(
    records: list[dict[str, Any]],
    dataset_name: str = "dataset",
    check_biology: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter out hygiene-broken records (key collisions, literal nulls, comma-shards, PCR sense primers).

    Returns (clean_records, hygiene_report). Designed as a zero-dependency pre-filter
    before converting records into Inspect AI Sample objects.
    """
    items = normalize_items(records, dataset_name=dataset_name)
    report = check_mcq_hygiene(items, check_biology=check_biology)
    bad_ids = set(report["flagged_ids"])
    clean = [rec for rec, item in zip(records, items) if item.item_id not in bad_ids]
    report["retained_items"] = len(clean)
    report["removed_items"] = len(records) - len(clean)
    return clean, report


def question_blind_centroid_baseline(choices: list[str]) -> int:
    """Predict the index of the edit-distance centroid option without inspecting the question.

    Useful as a zero-LLM sanity check inside Inspect AI solvers to detect star-graph distractor leakage.
    """
    if not choices:
        raise ValueError("choices must be non-empty")
    _, mean_sims, _ = compute_option_features([str(c) for c in choices])
    return int(np.argmax(mean_sims))


def make_inspect_sample_filter(
    check_biology: bool = True,
) -> Callable[[AuditItem], bool]:
    """Return a predicate compatible with Inspect AI Dataset.filter() logic."""

    def _keep(item: AuditItem) -> bool:
        rep = check_mcq_hygiene([item], check_biology=check_biology)
        return rep["flagged_items"] == 0

    return _keep
