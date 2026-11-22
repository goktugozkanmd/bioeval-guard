"""Defensive phylodynamic surveillance invariant auditor.

Uses strictly benign synthetic housekeeping/reporter sequences (zero pathogen
sequences) to test whether genomic surveillance pipelines enforce sequence/metadata
index parity, IUPAC ambiguity masking, incomplete date uncertainty bounds,
frame-preserving codon translation, polytomy transmission tie guarding, and
Horvitz-Thompson sampling-depletion prevalence adjustment.
"""

from __future__ import annotations

import calendar
import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# Strictly benign synthetic housekeeping/reporter template (zero pathogen sequences).
SYNTHETIC_HOUSEKEEPING_REF = (
    "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGAC"
)

_CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

_IUPAC_BASES: dict[str, tuple[str, ...]] = {
    "A": ("A",),
    "C": ("C",),
    "G": ("G",),
    "T": ("T",),
    "R": ("A", "G"),
    "Y": ("C", "T"),
    "S": ("G", "C"),
    "W": ("A", "T"),
    "K": ("G", "T"),
    "M": ("A", "C"),
    "B": ("C", "G", "T"),
    "D": ("A", "G", "T"),
    "H": ("A", "C", "T"),
    "V": ("A", "C", "G"),
    "N": ("A", "C", "G", "T"),
}


class DefensivePhylodynamicScreener:
    """Reference phylodynamic surveillance processor satisfying all 6 invariants."""

    @staticmethod
    def validate_index_parity(
        sequence_ids: Sequence[str],
        metadata_ids: Sequence[str],
        allow_partial: bool = False,
    ) -> dict[str, list[str]]:
        """Fail closed on duplicate strain IDs or silent sequence/metadata mismatches."""
        seq_list = [str(s).strip() for s in sequence_ids]
        meta_list = [str(m).strip() for m in metadata_ids]

        if len(seq_list) != len(set(seq_list)):
            raise ValueError(
                "Duplicate sequence IDs detected in surveillance alignment."
            )
        if len(meta_list) != len(set(meta_list)):
            raise ValueError(
                "Duplicate metadata IDs detected in surveillance metadata."
            )

        seq_set = set(seq_list)
        meta_set = set(meta_list)
        missing_in_meta = sorted(seq_set - meta_set)
        missing_in_seqs = sorted(meta_set - seq_set)
        matched = sorted(seq_set & meta_set)

        if (missing_in_meta or missing_in_seqs) and not allow_partial:
            raise ValueError(
                f"Sequence/metadata index mismatch: {len(missing_in_meta)} sequences lack metadata, "
                f"{len(missing_in_seqs)} metadata rows lack sequences."
            )
        return {
            "matched": matched,
            "missing_in_metadata": missing_in_meta,
            "missing_in_sequences": missing_in_seqs,
        }

    @staticmethod
    def call_nucleotide_mutations(reference: str, query: str) -> list[str]:
        """Call deterministic A/C/G/T point substitutions while masking N, gaps, and IUPAC ambiguity codes."""
        ref = reference.strip().upper().replace("U", "T")
        qry = query.strip().upper().replace("U", "T")
        if len(ref) != len(qry):
            raise ValueError(
                f"Reference length ({len(ref)}) does not match query length ({len(qry)})."
            )
        definite = {"A", "C", "G", "T"}
        muts: list[str] = []
        for idx, (rb, qb) in enumerate(zip(ref, qry), start=1):
            if rb in definite and qb in definite and rb != qb:
                muts.append(f"{rb}{idx}{qb}")
        return muts

    @staticmethod
    def parse_collection_date_bounds(raw_date: str) -> tuple[str, str]:
        """Return explicit `(lower_iso, upper_iso)` uncertainty bounds for partial dates."""
        text = (raw_date or "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            year, month, day = (int(p) for p in text.split("-"))
            max_day = calendar.monthrange(year, month)[1]
            if not (1 <= month <= 12 and 1 <= day <= max_day):
                raise ValueError(f"Invalid ISO calendar date: {raw_date!r}")
            return (text, text)

        if re.fullmatch(r"\d{4}-\d{2}", text):
            year, month = (int(p) for p in text.split("-"))
            if not (1 <= month <= 12):
                raise ValueError(f"Invalid ISO year-month: {raw_date!r}")
            last_day = calendar.monthrange(year, month)[1]
            return (
                f"{year:04d}-{month:02d}-01",
                f"{year:04d}-{month:02d}-{last_day:02d}",
            )

        if re.fullmatch(r"\d{4}", text):
            year = int(text)
            return (f"{year:04d}-01-01", f"{year:04d}-12-31")

        raise ValueError(f"Unsupported or missing ISO collection date: {raw_date!r}")

    @staticmethod
    def translate_codon_aligned(sequence: str) -> str:
        """Translate codon-aligned coding sequences while preserving triplet frame across deletions."""
        seq = (sequence or "").strip().upper().replace("U", "T")
        if len(seq) % 3 != 0:
            raise ValueError(
                f"Codon-aligned sequence length must be divisible by 3 (got {len(seq)})."
            )
        aa_chars: list[str] = []
        for i in range(0, len(seq), 3):
            codon = seq[i : i + 3]
            if codon == "---":
                aa_chars.append("-")
                continue
            if "-" in codon or any(ch not in _IUPAC_BASES for ch in codon):
                aa_chars.append("X")
                continue
            if codon in _CODON_TABLE:
                aa_chars.append(_CODON_TABLE[codon])
                continue
            # Resolve degenerate IUPAC codons only when all expansions yield identical amino acid.
            expansions = [
                _CODON_TABLE["".join(bases)]
                for bases in itertools.product(*(_IUPAC_BASES[b] for b in codon))
            ]
            unique_aas = set(expansions)
            aa_chars.append(next(iter(unique_aas)) if len(unique_aas) == 1 else "X")
        return "".join(aa_chars)

    @staticmethod
    def infer_directed_transmission(branch_length: float) -> dict[str, Any]:
        """Refuse directional transmission inference across zero-length polytomy edges."""
        if branch_length <= 0.0:
            return {"directed_transmission": False, "status": "unresolved_polytomy"}
        return {"directed_transmission": True, "status": "resolved_edge"}

    @staticmethod
    def estimate_regional_prevalence_shares(
        regions: Sequence[Mapping[str, float | int | str]],
    ) -> dict[str, float]:
        """Compute Horvitz-Thompson inverse-probability weighted regional lineage shares."""
        if not regions:
            return {}
        weighted: dict[str, float] = {}
        for row in regions:
            name = str(row["region"])
            hits = float(row["lineage_hits"])
            sequenced = float(row["sequenced_specimens"])
            cases = float(row["total_clinical_cases"])
            if hits < 0 or sequenced <= 0 or cases <= 0 or sequenced > cases:
                raise ValueError(f"Invalid regional surveillance counts for {name!r}.")
            sampling_fraction = sequenced / cases
            weighted[name] = hits / sampling_fraction

        total_est = sum(weighted.values())
        if total_est <= 0.0:
            return {name: 0.0 for name in weighted}
        return {name: val / total_est for name, val in weighted.items()}

    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Any:
        """Dispatch single-callable phylodynamic pipeline interface for auditing."""
        if stage == "index_parity":
            return self.validate_index_parity(
                payload["sequence_ids"],
                payload["metadata_ids"],
                allow_partial=bool(payload.get("allow_partial", False)),
            )
        if stage == "call_mutations":
            return self.call_nucleotide_mutations(
                payload["reference"], payload["query"]
            )
        if stage == "date_bounds":
            return self.parse_collection_date_bounds(payload["raw_date"])
        if stage == "translate_codons":
            return self.translate_codon_aligned(payload["sequence"])
        if stage == "polytomy_guard":
            return self.infer_directed_transmission(float(payload["branch_length"]))
        if stage == "regional_share":
            return self.estimate_regional_prevalence_shares(payload["regions"])
        raise ValueError(f"Unknown phylodynamic surveillance stage: {stage!r}")


def audit_phylodynamic_surveillance_pipeline(
    pipeline_fn: Callable[[str, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Audit a phylodynamic surveillance pipeline across 6 deterministic invariants."""
    invariants: list[dict[str, Any]] = []

    # 1. Sequence/metadata index parity fail-closed check
    try:
        pipeline_fn(
            "index_parity",
            {
                "sequence_ids": ["EPI_01", "EPI_02", "EPI_03"],
                "metadata_ids": ["EPI_01", "EPI_02"],
                "allow_partial": False,
            },
        )
        inv1_ok = False
        inv1_detail = (
            "Silently dropped unmatched outbreak sequence without raising ValueError."
        )
    except ValueError:
        try:
            res_dup = None
            pipeline_fn(
                "index_parity",
                {
                    "sequence_ids": ["EPI_01", "EPI_01"],
                    "metadata_ids": ["EPI_01"],
                    "allow_partial": True,
                },
            )
            inv1_ok = False
            inv1_detail = f"Accepted duplicate sequence ID: {res_dup!r}."
        except ValueError:
            inv1_ok = True
            inv1_detail = (
                "Failed closed on partial index overlap and duplicate strain IDs."
            )
    except Exception as exc:
        inv1_ok = False
        inv1_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "sequence_metadata_index_parity",
            "passed": inv1_ok,
            "detail": inv1_detail,
        }
    )

    # 2. Ambiguous IUPAC and amplicon-dropout N masking
    try:
        ref_chars = list(SYNTHETIC_HOUSEKEEPING_REF)
        qry_chars = ref_chars[:]
        qry_chars[14] = "A" if ref_chars[14] != "A" else "T"  # genuine point SNP (G15A)
        qry_chars[5] = "N"  # amplicon dropout
        qry_chars[10] = "R"  # IUPAC ambiguity
        qry_chars[20] = "Y"  # IUPAC ambiguity
        qry_chars[25] = "-"  # gap
        expected_snp = f"{ref_chars[14]}15{qry_chars[14]}"
        muts = pipeline_fn(
            "call_mutations",
            {"reference": "".join(ref_chars), "query": "".join(qry_chars)},
        )
        inv2_ok = muts == [expected_snp]
        inv2_detail = f"muts={muts!r} expected={[expected_snp]!r}"
    except Exception as exc:
        inv2_ok = False
        inv2_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "ambiguous_iupac_mutation_masking",
            "passed": inv2_ok,
            "detail": inv2_detail,
        }
    )

    # 3. Incomplete collection date bounds (including leap-year Feb)
    try:
        b_nov = tuple(pipeline_fn("date_bounds", {"raw_date": "2026-11"}))
        b_leap = tuple(pipeline_fn("date_bounds", {"raw_date": "2024-02"}))
        b_year = tuple(pipeline_fn("date_bounds", {"raw_date": "2026"}))
        inv3_ok = (
            b_nov == ("2026-11-01", "2026-11-30")
            and b_leap == ("2024-02-01", "2024-02-29")
            and b_year == ("2026-01-01", "2026-12-31")
        )
        inv3_detail = f"nov={b_nov} leap={b_leap} year={b_year}"
    except Exception as exc:
        inv3_ok = False
        inv3_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "incomplete_collection_date_bounds",
            "passed": inv3_ok,
            "detail": inv3_detail,
        }
    )

    # 4. Frame-preserving codon translation across deletions & synonymous IUPAC codons
    try:
        # ATG (M) | --- (-) | ACN (T) | GGN (G) | ATN (X) | TGG (W)
        prot = pipeline_fn("translate_codons", {"sequence": "ATG---ACNGGNATNTGG"})
        inv4_ok = prot == "M-TGXW"
        inv4_detail = f"translated={prot!r} expected='M-TGXW'"
    except Exception as exc:
        inv4_ok = False
        inv4_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "frame_preserving_codon_translation",
            "passed": inv4_ok,
            "detail": inv4_detail,
        }
    )

    # 5. Polytomy transmission tie guard
    try:
        zero_edge = pipeline_fn("polytomy_guard", {"branch_length": 0.0})
        pos_edge = pipeline_fn("polytomy_guard", {"branch_length": 1e-4})
        inv5_ok = (
            isinstance(zero_edge, Mapping)
            and zero_edge.get("directed_transmission") is False
            and zero_edge.get("status") == "unresolved_polytomy"
            and isinstance(pos_edge, Mapping)
            and pos_edge.get("directed_transmission") is True
        )
        inv5_detail = f"zero_edge={zero_edge!r} pos_edge={pos_edge!r}"
    except Exception as exc:
        inv5_ok = False
        inv5_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "polytomy_transmission_tie_guard",
            "passed": inv5_ok,
            "detail": inv5_detail,
        }
    )

    # 6. Horvitz-Thompson sampling-depletion invariance under 50x sequencing effort skew
    try:
        shares = pipeline_fn(
            "regional_share",
            {
                "regions": [
                    {
                        "region": "region_a",
                        "lineage_hits": 10,
                        "sequenced_specimens": 100,
                        "total_clinical_cases": 10_000,
                    },
                    {
                        "region": "region_b",
                        "lineage_hits": 500,
                        "sequenced_specimens": 5_000,
                        "total_clinical_cases": 10_000,
                    },
                ]
            },
        )
        inv6_ok = (
            isinstance(shares, Mapping)
            and abs(float(shares["region_a"]) - 0.5) < 1e-9
            and abs(float(shares["region_b"]) - 0.5) < 1e-9
        )
        inv6_detail = f"shares={shares!r}"
    except Exception as exc:
        inv6_ok = False
        inv6_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "horvitz_thompson_sampling_depletion",
            "passed": inv6_ok,
            "detail": inv6_detail,
        }
    )

    passed_count = sum(1 for inv in invariants if inv["passed"])
    return {
        "passed": passed_count == len(invariants),
        "passed_count": passed_count,
        "total_invariants": len(invariants),
        "invariants": invariants,
    }


def _naive_phylodynamic_pipeline(stage: str, payload: Mapping[str, Any]) -> Any:
    """Reference naive phylodynamic pipeline exhibiting all 6 failure modes."""
    if stage == "index_parity":
        meta_set = set(payload["metadata_ids"])
        return {"matched": [s for s in payload["sequence_ids"] if s in meta_set]}
    if stage == "call_mutations":
        ref = payload["reference"]
        qry = payload["query"]
        return [
            f"{rb}{i}{qb}"
            for i, (rb, qb) in enumerate(zip(ref, qry), start=1)
            if rb != qb and qb != "-"
        ]
    if stage == "date_bounds":
        raw = str(payload["raw_date"])
        if len(raw) == 4:
            return (f"{raw}-01-01", f"{raw}-01-01")
        if len(raw) == 7:
            return (f"{raw}-01", f"{raw}-01")
        return (raw, raw)
    if stage == "translate_codons":
        stripped = str(payload["sequence"]).replace("-", "")
        return "".join(
            _CODON_TABLE.get(stripped[i : i + 3], "X")
            for i in range(0, len(stripped) - (len(stripped) % 3), 3)
        )
    if stage == "polytomy_guard":
        return {"directed_transmission": True, "status": "resolved_edge"}
    if stage == "regional_share":
        regs = payload["regions"]
        tot = sum(float(r["lineage_hits"]) for r in regs) or 1.0
        return {str(r["region"]): float(r["lineage_hits"]) / tot for r in regs}
    raise ValueError(f"Unknown stage: {stage!r}")


def run_phylodynamic_self_check() -> dict[str, Any]:
    """Compare hardened DefensivePhylodynamicScreener against a naive pipeline."""
    hardened = audit_phylodynamic_surveillance_pipeline(DefensivePhylodynamicScreener())
    naive = audit_phylodynamic_surveillance_pipeline(_naive_phylodynamic_pipeline)
    caught = [inv["invariant"] for inv in naive["invariants"] if not inv["passed"]]
    return {
        "hardened_passed": hardened["passed"],
        "hardened": hardened,
        "naive_caught_invariants": caught,
        "naive_total_invariants": naive["total_invariants"],
    }
