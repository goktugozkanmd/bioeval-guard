"""Defensive pathogen metagenomic surveillance invariant auditor.

Uses strictly benign synthetic reporter/vector control sequences (zero pathogen
sequences) to test whether metagenomic surveillance pipelines preserve chimeric
host-junction retention, strict multiplex barcode validation, fail-closed
sample-sheet checking, LCA alert preservation, spike-in compositional invariance,
and paired-mate strand symmetry.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .hygiene import reverse_complement_dna

# Strictly benign synthetic control motifs (zero pathogen sequences).
SYNTHETIC_HOST_DECOY_KMER = "AAACCCGGGTTTAAACCCGGGTTTAAACCCGG"
SYNTHETIC_VECTOR_TARGET_KMER = "ATGCTAGTCGATCGATGCATCGATCGATAGCTA"
SYNTHETIC_NEGATIVE_READ = "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"

_BARCODE_RE = re.compile(r"^(?:barcode|nb|rlb)(\d{2,})$", flags=re.IGNORECASE)


class DefensiveMetagenomicScreener:
    """Reference metagenomic surveillance screener satisfying all 6 invariants."""

    def __init__(
        self,
        target_kmers: Sequence[str] = (SYNTHETIC_VECTOR_TARGET_KMER,),
        min_non_host_overhang: int = 31,
    ) -> None:
        normalized: list[str] = []
        for kmer in target_kmers:
            cleaned = kmer.strip().upper().replace("U", "T")
            if not cleaned:
                continue
            normalized.append(cleaned)
            rc = reverse_complement_dna(cleaned)
            if rc != cleaned:
                normalized.append(rc)
        self.targets = tuple(dict.fromkeys(normalized))
        self.min_non_host_overhang = max(1, min_non_host_overhang)

    def _matches_seq(self, seq: str) -> bool:
        norm = re.sub(r"[^ACGTU]", "", (seq or "").upper()).replace("U", "T")
        return any(target in norm for target in self.targets)

    def screen_host_depleted_read(
        self, read: str, host_kmers: Sequence[str] = (SYNTHETIC_HOST_DECOY_KMER,)
    ) -> bool:
        """Mask host spans instead of dropping chimeric reads; screen >=31 bp overhangs."""
        norm_read = re.sub(r"[^ACGTU]", "", (read or "").upper()).replace("U", "T")
        if not norm_read:
            return False

        masked = list(norm_read)
        for hk in host_kmers:
            hk_norm = re.sub(r"[^ACGTU]", "", hk.upper()).replace("U", "T")
            if not hk_norm:
                continue
            for pat in (hk_norm, reverse_complement_dna(hk_norm)):
                start = 0
                while True:
                    idx = norm_read.find(pat, start)
                    if idx == -1:
                        break
                    for pos in range(idx, idx + len(pat)):
                        masked[pos] = "N"
                    start = idx + 1

        segments = [
            seg
            for seg in "".join(masked).split("N")
            if len(seg) >= self.min_non_host_overhang
        ]
        return any(self._matches_seq(seg) for seg in segments)

    @staticmethod
    def canonicalize_barcode(barcode: str) -> str:
        """Canonicalize `barcode01`/`NB01`/`RLB01` -> `barcode01`; reject unpadded IDs."""
        raw = (barcode or "").strip()
        m = _BARCODE_RE.fullmatch(raw)
        if not m:
            raise ValueError(
                f"Malformed or unpadded barcode {barcode!r}; expected barcodeNN/NBNN/RLBNN with leading zero."
            )
        return f"barcode{m.group(1)}"

    @staticmethod
    def validate_samplesheet_rows(
        rows: Sequence[Mapping[str, Any]],
        existing_paths: set[str] | frozenset[str],
    ) -> list[dict[str, str]]:
        """Fail closed when sample-sheet entries reference empty or nonexistent FASTQ paths."""
        validated: list[dict[str, str]] = []
        for idx, row in enumerate(rows):
            sample_id = str(row.get("id", "") or "").strip()
            fq1 = str(row.get("fastq_1", "") or "").strip()
            fq2 = str(row.get("fastq_2", "") or "").strip()
            if not sample_id or not fq1 or not fq2:
                raise ValueError(
                    f"Sample row {idx} is missing id, fastq_1, or fastq_2."
                )
            for label, path in (("fastq_1", fq1), ("fastq_2", fq2)):
                if path not in existing_paths:
                    raise FileNotFoundError(
                        f"Missing {label} input file for sample {sample_id!r}: {path}"
                    )
            validated.append({"id": sample_id, "fastq_1": fq1, "fastq_2": fq2})
        return validated

    @staticmethod
    def resolve_lca_candidates(
        hits: Sequence[Mapping[str, Any]],
        parent_taxon: str,
    ) -> dict[str, Any]:
        """Collapse taxonomic rank to LCA while preserving monitored alert candidates."""
        taxa = [str(h.get("taxon", "")) for h in hits if h.get("taxon")]
        if not taxa:
            return {"lca_taxon": None, "alert_candidates": []}
        lca = taxa[0] if len(set(taxa)) == 1 else parent_taxon
        alerts = sorted(
            {str(h["taxon"]) for h in hits if h.get("monitored") and h.get("taxon")}
        )
        return {"lca_taxon": lca, "alert_candidates": alerts}

    @staticmethod
    def normalize_spike_in_abundance(
        target_reads: int | float,
        spike_in_reads: int | float,
        background_reads: int | float = 0,
    ) -> float:
        """Compute spike-in normalized ratio invariant to background microbial bloom."""
        _ = background_reads
        if spike_in_reads <= 0:
            raise ValueError(
                "spike_in_reads must be positive for compositional normalization."
            )
        if target_reads < 0:
            raise ValueError("target_reads must be non-negative.")
        return float(target_reads) / float(spike_in_reads)

    def screen_paired_mates(self, read1: str, read2: str) -> bool:
        """Screen both mates across forward and reverse-complement strands."""
        return self._matches_seq(read1) or self._matches_seq(read2)

    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Any:
        """Dispatch single-callable pipeline interface for invariant auditing."""
        if stage == "host_depletion":
            return self.screen_host_depleted_read(
                payload["read"], payload.get("host_kmers", (SYNTHETIC_HOST_DECOY_KMER,))
            )
        if stage == "canonicalize_barcode":
            return self.canonicalize_barcode(payload["barcode"])
        if stage == "validate_samplesheet":
            return self.validate_samplesheet_rows(
                payload["rows"], set(payload["existing_paths"])
            )
        if stage == "lca_resolve":
            return self.resolve_lca_candidates(payload["hits"], payload["parent_taxon"])
        if stage == "normalize_abundance":
            return self.normalize_spike_in_abundance(
                payload["target_reads"],
                payload["spike_in_reads"],
                payload.get("background_reads", 0),
            )
        if stage == "screen_paired":
            return self.screen_paired_mates(payload["read1"], payload["read2"])
        raise ValueError(f"Unknown surveillance stage: {stage!r}")


def audit_metagenomic_surveillance_pipeline(
    pipeline_fn: Callable[[str, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Audit a metagenomic surveillance pipeline across 6 deterministic invariants."""
    invariants: list[dict[str, Any]] = []

    def _record(name: str, passed: bool, detail: str) -> None:
        invariants.append({"invariant": name, "passed": bool(passed), "detail": detail})

    # 1. Host chimera junction retention (>=31 bp non-host overhang preserved)
    try:
        chimera = SYNTHETIC_HOST_DECOY_KMER + "GG" + SYNTHETIC_VECTOR_TARGET_KMER
        pure_host = SYNTHETIC_HOST_DECOY_KMER + "TTTTAAAACCCC"
        chimera_hit = bool(
            pipeline_fn(
                "host_depletion",
                {"read": chimera, "host_kmers": [SYNTHETIC_HOST_DECOY_KMER]},
            )
        )
        host_hit = bool(
            pipeline_fn(
                "host_depletion",
                {"read": pure_host, "host_kmers": [SYNTHETIC_HOST_DECOY_KMER]},
            )
        )
        _record(
            "host_chimera_junction_retention",
            chimera_hit and not host_hit,
            f"chimera_hit={chimera_hit}, pure_host_hit={host_hit}",
        )
    except Exception as exc:
        _record("host_chimera_junction_retention", False, f"error:{type(exc).__name__}")

    # 2. Multiplex barcode canonicalization + strict zero-padding enforcement
    try:
        ok_b = (
            pipeline_fn("canonicalize_barcode", {"barcode": "barcode01"}) == "barcode01"
        )
        ok_nb = pipeline_fn("canonicalize_barcode", {"barcode": "NB01"}) == "barcode01"
        ok_rlb = (
            pipeline_fn("canonicalize_barcode", {"barcode": "RLB12"}) == "barcode12"
        )
        rejected_bare = False
        rejected_unpadded = False
        for bad, flag_name in (("1", "bare"), ("NB1", "unpadded")):
            try:
                pipeline_fn("canonicalize_barcode", {"barcode": bad})
            except ValueError:
                if flag_name == "bare":
                    rejected_bare = True
                else:
                    rejected_unpadded = True
        _record(
            "barcode_canonicalization",
            ok_b and ok_nb and ok_rlb and rejected_bare and rejected_unpadded,
            f"canon=({ok_b},{ok_nb},{ok_rlb}), rejected=({rejected_bare},{rejected_unpadded})",
        )
    except Exception as exc:
        _record("barcode_canonicalization", False, f"error:{type(exc).__name__}")

    # 3. Sample-sheet missing-file fail-closed check
    try:
        valid_rows = pipeline_fn(
            "validate_samplesheet",
            {
                "rows": [
                    {
                        "id": "s1",
                        "fastq_1": "/data/s1_1.fastq.gz",
                        "fastq_2": "/data/s1_2.fastq.gz",
                    }
                ],
                "existing_paths": {"/data/s1_1.fastq.gz", "/data/s1_2.fastq.gz"},
            },
        )
        raised_missing = False
        try:
            pipeline_fn(
                "validate_samplesheet",
                {
                    "rows": [
                        {
                            "id": "s2",
                            "fastq_1": "/data/missing_1.fastq.gz",
                            "fastq_2": "/data/s2_2.fastq.gz",
                        }
                    ],
                    "existing_paths": {"/data/s2_2.fastq.gz"},
                },
            )
        except (FileNotFoundError, ValueError):
            raised_missing = True

        raised_empty = False
        try:
            pipeline_fn(
                "validate_samplesheet",
                {
                    "rows": [
                        {"id": "s3", "fastq_1": "", "fastq_2": "/data/s3_2.fastq.gz"}
                    ],
                    "existing_paths": {"/data/s3_2.fastq.gz"},
                },
            )
        except (FileNotFoundError, ValueError):
            raised_empty = True

        _record(
            "samplesheet_missing_file_fail_closed",
            len(valid_rows) == 1 and raised_missing and raised_empty,
            f"valid={len(valid_rows)}, raised_missing={raised_missing}, raised_empty={raised_empty}",
        )
    except Exception as exc:
        _record(
            "samplesheet_missing_file_fail_closed", False, f"error:{type(exc).__name__}"
        )

    # 4. LCA tie-breaking preserves monitored alert candidates
    try:
        tie_res = pipeline_fn(
            "lca_resolve",
            {
                "hits": [
                    {
                        "taxon": "synthetic_target_A",
                        "rank": "species",
                        "monitored": True,
                    },
                    {
                        "taxon": "synthetic_neighbor_B",
                        "rank": "species",
                        "monitored": False,
                    },
                ],
                "parent_taxon": "synthetic_genus_X",
            },
        )
        benign_res = pipeline_fn(
            "lca_resolve",
            {
                "hits": [
                    {
                        "taxon": "synthetic_neighbor_B",
                        "rank": "species",
                        "monitored": False,
                    },
                    {
                        "taxon": "synthetic_neighbor_C",
                        "rank": "species",
                        "monitored": False,
                    },
                ],
                "parent_taxon": "synthetic_genus_X",
            },
        )
        passed_lca = (
            tie_res.get("lca_taxon") == "synthetic_genus_X"
            and tie_res.get("alert_candidates") == ["synthetic_target_A"]
            and benign_res.get("alert_candidates") == []
        )
        _record(
            "lca_tie_species_alert_preservation",
            passed_lca,
            f"tie_alerts={tie_res.get('alert_candidates')}, benign_alerts={benign_res.get('alert_candidates')}",
        )
    except Exception as exc:
        _record(
            "lca_tie_species_alert_preservation", False, f"error:{type(exc).__name__}"
        )

    # 5. Spike-in compositional closure invariance under 100x background inflation
    try:
        base_val = float(
            pipeline_fn(
                "normalize_abundance",
                {"target_reads": 50, "spike_in_reads": 500, "background_reads": 10_000},
            )
        )
        bloom_val = float(
            pipeline_fn(
                "normalize_abundance",
                {
                    "target_reads": 50,
                    "spike_in_reads": 500,
                    "background_reads": 1_000_000,
                },
            )
        )
        rel_diff = abs(base_val - bloom_val) / max(abs(base_val), 1e-12)
        _record(
            "spike_in_compositional_closure_invariance",
            base_val > 0.0 and math.isfinite(base_val) and rel_diff < 1e-9,
            f"base={base_val:.4f}, bloom={bloom_val:.4f}, rel_diff={rel_diff:.2e}",
        )
    except Exception as exc:
        _record(
            "spike_in_compositional_closure_invariance",
            False,
            f"error:{type(exc).__name__}",
        )

    # 6. Paired-end mate strand symmetry across R1/R2 forward & reverse-complement
    try:
        rc_target = reverse_complement_dna(SYNTHETIC_VECTOR_TARGET_KMER)
        p_r1 = bool(
            pipeline_fn(
                "screen_paired",
                {
                    "read1": SYNTHETIC_VECTOR_TARGET_KMER,
                    "read2": SYNTHETIC_NEGATIVE_READ,
                },
            )
        )
        p_r2_rc = bool(
            pipeline_fn(
                "screen_paired",
                {"read1": SYNTHETIC_NEGATIVE_READ, "read2": rc_target},
            )
        )
        p_r1_rc = bool(
            pipeline_fn(
                "screen_paired",
                {"read1": rc_target, "read2": SYNTHETIC_NEGATIVE_READ},
            )
        )
        p_neg = bool(
            pipeline_fn(
                "screen_paired",
                {"read1": SYNTHETIC_NEGATIVE_READ, "read2": SYNTHETIC_NEGATIVE_READ},
            )
        )
        _record(
            "paired_mate_strand_concordance",
            p_r1 and p_r2_rc and p_r1_rc and not p_neg,
            f"r1={p_r1}, r2_rc={p_r2_rc}, r1_rc={p_r1_rc}, neg={p_neg}",
        )
    except Exception as exc:
        _record("paired_mate_strand_concordance", False, f"error:{type(exc).__name__}")

    passed_count = sum(1 for inv in invariants if inv["passed"])
    return {
        "auditor": "metagenomic_surveillance_invariants",
        "passed": passed_count == len(invariants),
        "passed_count": passed_count,
        "total_invariants": len(invariants),
        "invariants": invariants,
    }


def run_surveillance_self_check() -> dict[str, Any]:
    """Compare DefensiveMetagenomicScreener against a naive MGS pipeline."""
    hardened = DefensiveMetagenomicScreener()
    hardened_report = audit_metagenomic_surveillance_pipeline(hardened)

    def naive_mgs_pipeline(stage: str, payload: Mapping[str, Any]) -> Any:
        if stage == "host_depletion":
            read = payload["read"]
            if any(hk in read for hk in payload.get("host_kmers", [])):
                return False
            return SYNTHETIC_VECTOR_TARGET_KMER in read
        if stage == "canonicalize_barcode":
            return str(payload["barcode"])
        if stage == "validate_samplesheet":
            existing = set(payload["existing_paths"])
            return [
                r
                for r in payload["rows"]
                if r.get("fastq_1") in existing and r.get("fastq_2") in existing
            ]
        if stage == "lca_resolve":
            taxa = [h["taxon"] for h in payload["hits"]]
            lca = taxa[0] if len(set(taxa)) == 1 else payload["parent_taxon"]
            return {"lca_taxon": lca, "alert_candidates": []}
        if stage == "normalize_abundance":
            total = (
                payload["target_reads"]
                + payload["spike_in_reads"]
                + payload.get("background_reads", 0)
            )
            return (payload["target_reads"] * 1_000_000.0) / max(total, 1)
        if stage == "screen_paired":
            return SYNTHETIC_VECTOR_TARGET_KMER in payload["read1"]
        raise ValueError(stage)

    naive_report = audit_metagenomic_surveillance_pipeline(naive_mgs_pipeline)
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
