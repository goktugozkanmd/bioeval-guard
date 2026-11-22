"""bioeval_guard: Deterministic biosecurity and AIxBio benchmark integrity auditor."""

from .cluster_stats import audit_clustered_benchmark_se
from .grader_audit import (
    audit_answer_tag_parser,
    audit_dna_screener_symmetry,
    audit_numeric_string_verifier,
    audit_safety_judge_callable,
    audit_selective_precision_metric,
    case_safe_judge_label,
    nan_safe_selective_precision,
    parse_tagged_answer_strict,
    run_reference_self_check,
)
from .hygiene import (
    AuditItem,
    check_bixbench_hygiene,
    check_mcq_hygiene,
    check_primer_reverse_complement,
    check_restriction_site_complement_bug,
    complement_dna,
    normalize_items,
    reverse_complement_dna,
)
from .inspect_adapter import (
    question_blind_centroid_baseline,
    sanitize_mcq_dataset,
)
from .inspect_tasks import (
    LAB_BENCH_DATASET_REVISION,
    LAB_BENCH_REVISION_PIN,
    compute_nan_safe_precision_and_coverage,
    filter_clean_lab_bench_records,
    nan_safe_precision_metric,
)
from .shortcuts import (
    audit_question_blind_shortcuts,
    holm_bonferroni,
)

__all__ = [
    "AuditItem",
    "LAB_BENCH_DATASET_REVISION",
    "LAB_BENCH_REVISION_PIN",
    "audit_answer_tag_parser",
    "audit_clustered_benchmark_se",
    "audit_dna_screener_symmetry",
    "audit_numeric_string_verifier",
    "audit_question_blind_shortcuts",
    "audit_safety_judge_callable",
    "audit_selective_precision_metric",
    "case_safe_judge_label",
    "check_bixbench_hygiene",
    "check_mcq_hygiene",
    "check_primer_reverse_complement",
    "check_restriction_site_complement_bug",
    "complement_dna",
    "compute_nan_safe_precision_and_coverage",
    "filter_clean_lab_bench_records",
    "holm_bonferroni",
    "nan_safe_precision_metric",
    "nan_safe_selective_precision",
    "normalize_items",
    "parse_tagged_answer_strict",
    "question_blind_centroid_baseline",
    "reverse_complement_dna",
    "run_reference_self_check",
    "sanitize_mcq_dataset",
]

__version__ = "0.3.0"
