"""UK AISI Inspect Evals Register-compatible sanitized LAB-Bench tasks and NaN-safe metric."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .grader_audit import nan_safe_selective_precision
from .hygiene import check_mcq_hygiene, normalize_items

LAB_BENCH_DATASET_PATH = "futurehouse/lab-bench"
LAB_BENCH_DATASET_REVISION = "5c77cec648430f30611808808861eb86f81d5eaa"
UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question."

try:
    from inspect_ai import Epochs, Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import NOANSWER, Metric, SampleScore, metric
    from inspect_ai.solver import multiple_choice

    HAS_INSPECT_AI = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_INSPECT_AI = False


def compute_nan_safe_precision_and_coverage(
    score_values: list[str | float | int],
) -> tuple[float, float]:
    """Compute `(precision, coverage)` returning `NaN` when every sample abstained (`NOANSWER` / `-1`)."""
    answered: list[float] = []
    for val in score_values:
        if val in ("N", -1, -1.0):
            continue
        if val == "C":
            answered.append(1.0)
        elif val == "P":
            answered.append(0.5)
        elif val == "I":
            answered.append(0.0)
        elif isinstance(val, int | float | bool):
            answered.append(float(val))
    return nan_safe_selective_precision(answered, len(score_values))


def filter_clean_lab_bench_records(
    records: list[dict[str, Any]],
    subset_name: str,
    check_biology: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return hygiene-screened LAB-Bench records alongside an audit summary."""
    items = normalize_items(records, dataset_name=subset_name)
    report = check_mcq_hygiene(items, check_biology=check_biology)
    bad = set(report["flagged_ids"])
    kept = [rec for rec, it in zip(records, items) if it.item_id not in bad]
    report["dataset_revision"] = LAB_BENCH_DATASET_REVISION
    report["retained_items"] = len(kept)
    report["removed_items"] = len(records) - len(kept)
    return kept, report


if HAS_INSPECT_AI:  # pragma: no cover - requires optional inspect-ai

    @metric
    def nan_safe_precision_metric() -> Metric:
        """Inspect AI metric returning NaN instead of 0.0 when all responses abstain."""

        def _metric(scores: list[SampleScore]) -> float:
            vals = [s.score.value if s.score.value != NOANSWER else -1 for s in scores]
            prec, _ = compute_nan_safe_precision_and_coverage(vals)
            return prec

        return _metric

    def _build_sanitized_task(subset_name: str, check_biology: bool) -> Task:
        from inspect_ai.dataset import hf_dataset

        raw_ds = hf_dataset(
            path=LAB_BENCH_DATASET_PATH,
            name=subset_name,
            split="train",
            revision=LAB_BENCH_DATASET_REVISION,
        )
        raw_records = [
            {
                "id": s.id,
                "question": s.input,
                "ideal": s.choices[0] if s.choices else "",
                "distractors": list(s.choices[1:]) if s.choices else [],
                "subtask": subset_name,
            }
            for s in raw_ds
        ]
        clean_records, _ = filter_clean_lab_bench_records(
            raw_records, subset_name=subset_name, check_biology=check_biology
        )
        samples = [
            Sample(
                id=str(rec["id"]),
                input=str(rec["question"]),
                choices=[str(rec["ideal"]), *(str(d) for d in rec["distractors"]), UNCERTAIN_ANSWER_CHOICE],
                target="A",
            )
            for rec in clean_records
        ]
        return Task(
            dataset=MemoryDataset(samples=samples),
            solver=[multiple_choice(cot=True)],
            epochs=Epochs(1, "mode"),
            metadata={"dataset_revision": LAB_BENCH_REVISION_PIN()},
        )

    @task
    def bioeval_lab_bench_seqqa_sanitized() -> Task:
        """Sanitized LAB-Bench SeqQA task excluding primer/duplicate-key corrupted items."""
        return _build_sanitized_task("SeqQA", check_biology=True)

    @task
    def bioeval_lab_bench_litqa2_sanitized() -> Task:
        """Sanitized LAB-Bench LitQA2 task excluding literal null ideals and comma-sharded distractors."""
        return _build_sanitized_task("LitQA2", check_biology=False)

else:

    def nan_safe_precision_metric() -> Callable[[list[Any]], float]:
        """Fallback precision metric returning NaN on 100%-abstention runs when inspect_ai is absent."""

        def _metric(scores: list[Any]) -> float:
            if not scores:
                return float("nan")
            vals = [s.get("value") if isinstance(s, dict) else s for s in scores]
            prec, _ = compute_nan_safe_precision_and_coverage(vals)
            return prec

        return _metric

    def bioeval_lab_bench_seqqa_sanitized() -> Any:
        """Raise informative error when inspect_ai optional extra is not installed."""
        raise ImportError("Install optional dependency `bioeval-guard[inspect]` to run Inspect AI tasks.")

    def bioeval_lab_bench_litqa2_sanitized() -> Any:
        """Raise informative error when inspect_ai optional extra is not installed."""
        raise ImportError("Install optional dependency `bioeval-guard[inspect]` to run Inspect AI tasks.")


def LAB_BENCH_REVISION_PIN() -> str:
    """Return pinned Hugging Face dataset SHA for reproducible biosecurity audits."""
    return LAB_BENCH_DATASET_REVISION


__all__ = [
    "HAS_INSPECT_AI",
    "LAB_BENCH_DATASET_PATH",
    "LAB_BENCH_DATASET_REVISION",
    "LAB_BENCH_REVISION_PIN",
    "bioeval_lab_bench_litqa2_sanitized",
    "bioeval_lab_bench_seqqa_sanitized",
    "compute_nan_safe_precision_and_coverage",
    "filter_clean_lab_bench_records",
    "math",
    "nan_safe_precision_metric",
]
