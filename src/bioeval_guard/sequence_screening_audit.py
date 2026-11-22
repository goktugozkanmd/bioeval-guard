"""Defensive nucleic-acid sequence screening invariant auditor.

Uses strictly benign synthetic reporter-linker oligonucleotides (zero pathogen
sequences) to test whether screening pipelines maintain strand, RNA/DNA,
circular-origin, tiled-fragment, IUPAC ambiguity, and FASTA header invariants.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from .hygiene import reverse_complement_dna

# Benign synthetic cloning/reporter linker motif (non-palindromic 20-mer + flanking bases).
SYNTHETIC_CONTROL_KMER = "ATGCTAGTCGATCGATGCAT"
SYNTHETIC_CONTROL_OLIGO = f"AAAGGGCCCTTT{SYNTHETIC_CONTROL_KMER}CCCGGGTTTAAA"
SYNTHETIC_NEGATIVE_OLIGO = "AAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCC"


def normalize_fasta_records(raw: str) -> list[str]:
    """Parse one or more FASTA/raw records while ignoring header and comment lines.

    Whitespace and numeric coordinates are removed, RNA uracil (`U`) is mapped to
    thymine (`T`), and IUPAC ambiguity characters are preserved rather than deleted
    so chimeric k-mers cannot be fabricated across masked positions.
    """
    records: list[str] = []
    current: list[str] = []
    saw_header = False
    for line in (raw or "").replace("\r\n", "\n").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith(">"):
            saw_header = True
            if current:
                records.append("".join(current))
                current = []
            continue
        cleaned = re.sub(r"[\s0-9-]+", "", stripped).upper().replace("U", "T")
        if cleaned:
            current.append(cleaned)
    if current:
        records.append("".join(current))
    if not saw_header and not records:
        return []
    return records


class DefensiveKmerScreener:
    """Reference defensive k-mer screener implementing all 6 screening invariants."""

    def __init__(
        self,
        target_kmers: Sequence[str] = (SYNTHETIC_CONTROL_KMER,),
        circular: bool = True,
        min_overlap: int = 8,
    ) -> None:
        normalized_targets: list[str] = []
        for kmer in target_kmers:
            cleaned = kmer.strip().upper().replace("U", "T")
            if not cleaned:
                continue
            normalized_targets.append(cleaned)
            rc = reverse_complement_dna(cleaned)
            if rc != cleaned:
                normalized_targets.append(rc)
        self.targets = tuple(dict.fromkeys(normalized_targets))
        self.max_k = max((len(t) for t in self.targets), default=0)
        self.circular = circular
        self.min_overlap = max(1, min_overlap)

    def _matches_sequence(self, seq: str) -> bool:
        if not seq or not self.targets:
            return False
        haystack = seq
        if self.circular and len(seq) >= self.max_k and self.max_k > 1:
            haystack = seq + seq[: self.max_k - 1]
        return any(target in haystack for target in self.targets)

    def screen(self, raw: str) -> bool:
        """Screen raw sequence or FASTA text without matching header lines."""
        for rec in normalize_fasta_records(raw):
            if self._matches_sequence(rec):
                return True
        return False

    def screen_batch(self, raw_records: Sequence[str]) -> bool:
        """Screen individual records and greedy suffix-prefix tiled assemblies."""
        seqs: list[str] = []
        for raw in raw_records:
            seqs.extend(normalize_fasta_records(raw))
        if any(self._matches_sequence(s) for s in seqs):
            return True
        if len(seqs) < 2:
            return False

        # Assemble pairs across forward and reverse-complement orientations.
        oriented: list[str] = []
        for s in seqs:
            oriented.append(s)
            if re.fullmatch(r"[ACGT]+", s):
                oriented.append(reverse_complement_dna(s))

        for i, left in enumerate(oriented):
            for j, right in enumerate(oriented):
                if i == j:
                    continue
                max_ov = min(len(left), len(right))
                for ov in range(max_ov, self.min_overlap - 1, -1):
                    if left[-ov:] == right[:ov]:
                        merged = left + right[ov:]
                        if self._matches_sequence(merged):
                            return True
                        break
        return False


def audit_sequence_screener(
    screener_fn: Callable[[str], bool],
    batch_screener_fn: Callable[[list[str]], bool] | None = None,
) -> dict[str, Any]:
    """Audit a nucleic-acid screening callable across 6 defensive invariants.

    Uses only benign synthetic linker sequences (zero pathogen data).
    """
    oligo = SYNTHETIC_CONTROL_OLIGO
    rc_oligo = reverse_complement_dna(oligo)
    rna_wrapped = (
        f"  {oligo[:22].lower().replace('t', 'u')}\n"
        f"  {oligo[22:].lower().replace('t', 'u')}  "
    )
    mid = len(oligo) // 2
    rotated_circular = oligo[mid:] + oligo[:mid]

    frag_a = oligo[: mid + 5]
    frag_b = oligo[mid - 5 :]
    batch_fn = batch_screener_fn or (
        lambda recs: any(bool(screener_fn(r)) for r in recs)
    )

    iupac_flanked = f"AAAGGGNCCTTT{SYNTHETIC_CONTROL_KMER}CCCGGGTYTAAA"
    gapped_chimera = (
        f"AAAGGG{SYNTHETIC_CONTROL_KMER[:10]}N{SYNTHETIC_CONTROL_KMER[10:]}CCCGGG"
    )
    all_degenerate = "NNNNRYKMSWBDHVN" * 3

    header_injection = (
        f">synthetic_header_{SYNTHETIC_CONTROL_KMER}\n{SYNTHETIC_NEGATIVE_OLIGO}\n"
    )
    multirecord_malformed = f"; leading comment\n>empty_first_record\n\n>second_valid\n{oligo[:20]}\n{oligo[20:]}\n"

    invariants: list[dict[str, Any]] = []

    def _record(name: str, passed: bool, detail: str) -> None:
        invariants.append({"invariant": name, "passed": bool(passed), "detail": detail})

    try:
        fwd_hit = bool(screener_fn(oligo))
        rc_hit = bool(screener_fn(rc_oligo))
        neg_hit = bool(screener_fn(SYNTHETIC_NEGATIVE_OLIGO))
        _record(
            "strand_symmetry",
            fwd_hit and rc_hit and not neg_hit,
            f"fwd={fwd_hit}, rc={rc_hit}, neg={neg_hit}",
        )
    except Exception as exc:
        _record("strand_symmetry", False, f"error:{type(exc).__name__}")

    try:
        rna_hit = bool(screener_fn(rna_wrapped))
        _record("rna_dna_and_case_normalization", rna_hit, f"rna_wrapped={rna_hit}")
    except Exception as exc:
        _record("rna_dna_and_case_normalization", False, f"error:{type(exc).__name__}")

    try:
        circ_hit = bool(screener_fn(rotated_circular))
        _record("circular_origin_wrap", circ_hit, f"rotated_circular={circ_hit}")
    except Exception as exc:
        _record("circular_origin_wrap", False, f"error:{type(exc).__name__}")

    try:
        frag_a_alone = bool(screener_fn(frag_a))
        frag_b_alone = bool(screener_fn(frag_b))
        tiled_hit = bool(batch_fn([frag_a, frag_b]))
        _record(
            "tiled_boundary_split",
            (not frag_a_alone) and (not frag_b_alone) and tiled_hit,
            f"frag_a={frag_a_alone}, frag_b={frag_b_alone}, assembled={tiled_hit}",
        )
    except Exception as exc:
        _record("tiled_boundary_split", False, f"error:{type(exc).__name__}")

    try:
        iupac_hit = bool(screener_fn(iupac_flanked))
        gapped_hit = bool(screener_fn(gapped_chimera))
        all_deg_hit = bool(screener_fn(all_degenerate))
        _record(
            "iupac_ambiguity_resilience",
            iupac_hit and (not gapped_hit) and (not all_deg_hit),
            f"iupac_flanked={iupac_hit}, gapped_chimera={gapped_hit}, all_degenerate={all_deg_hit}",
        )
    except Exception as exc:
        _record("iupac_ambiguity_resilience", False, f"error:{type(exc).__name__}")

    try:
        hdr_false_pos = bool(screener_fn(header_injection))
        multi_hit = bool(screener_fn(multirecord_malformed))
        _record(
            "malformed_fasta_header_safety",
            (not hdr_false_pos) and multi_hit,
            f"header_injection={hdr_false_pos}, multirecord={multi_hit}",
        )
    except Exception as exc:
        _record("malformed_fasta_header_safety", False, f"error:{type(exc).__name__}")

    passed_count = sum(1 for inv in invariants if inv["passed"])
    return {
        "auditor": "sequence_screening_invariants",
        "passed": passed_count == len(invariants),
        "passed_count": passed_count,
        "total_invariants": len(invariants),
        "invariants": invariants,
    }


def run_screening_self_check() -> dict[str, Any]:
    """Evaluate the reference DefensiveKmerScreener against a naive linear screener."""
    hardened = DefensiveKmerScreener()
    hardened_report = audit_sequence_screener(hardened.screen, hardened.screen_batch)

    def _naive_screener(raw: str) -> bool:
        return SYNTHETIC_CONTROL_KMER in raw

    naive_report = audit_sequence_screener(_naive_screener)
    naive_failures = [
        inv["invariant"] for inv in naive_report["invariants"] if not inv["passed"]
    ]
    return {
        "hardened_passed": hardened_report["passed"],
        "hardened": hardened_report,
        "naive_passed_count": naive_report["passed_count"],
        "naive_total_invariants": naive_report["total_invariants"],
        "naive_caught_invariants": naive_failures,
    }
