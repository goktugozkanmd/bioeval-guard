"""bioeval_guard: Deterministic biosecurity and AIxBio benchmark integrity auditor."""

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
from .shortcuts import (
    audit_question_blind_shortcuts,
    holm_bonferroni,
)

__all__ = [
    "AuditItem",
    "audit_question_blind_shortcuts",
    "check_bixbench_hygiene",
    "check_mcq_hygiene",
    "check_primer_reverse_complement",
    "check_restriction_site_complement_bug",
    "complement_dna",
    "holm_bonferroni",
    "normalize_items",
    "question_blind_centroid_baseline",
    "reverse_complement_dna",
    "sanitize_mcq_dataset",
]

__version__ = "0.1.0"
