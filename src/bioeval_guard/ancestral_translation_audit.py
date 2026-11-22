"""Ancestral translation alignment and codon-frame surveillance invariant auditor.

Uses strictly benign synthetic reporter coding sequences (zero pathogen sequences)
to verify whether phylogenetic translation and ancestral-state pipelines enforce:
1. Equal alignment length diagnostics (modal length + offending sequence IDs/lengths),
2. Duplicate taxon ID fail-closed rejection instead of silent overwrite,
3. CDS-to-protein codon-length parity (3 * len_aa or 3 * (len_aa + 1) with stop),
4. Guard against nucleotide alignments silently passed as amino-acid alignments,
5. Preservation of internal premature stop codons ('*') instead of masking to 'X', and
6. Cross-check consistency between inferred nucleotide codon mutations and AA mutations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# Strictly benign synthetic 60-nt / 20-aa reporter CDS (zero pathogen sequences).
# ATG GCT AGC GAT GCT AGC TAG -> wait: make sure all 20 codons are sense codons!
# Codons (20): ATG GCT AGC GAT GCT AGC GTC GGT ATC GAT CTG GCT AGC ATT GAT CTG GCT ACC GGT GCC (60 nt)
SYNTHETIC_REPORTER_CDS_NT = (
    "ATGGCTAGCGATGCTAGCGTCGGTATCGATCTGGCTAGCATTGATCTGGCTACCGGTGCC"
)
SYNTHETIC_REPORTER_PROTEIN_AA = "MASDASVGIDLASIDLATGA"

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


class TranslationAlignmentError(ValueError):
    """Raised when a translation alignment or codon-frame invariant is violated."""


def _parse_fasta(fasta_text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    cur_id: str | None = None
    buf: list[str] = []
    for raw_line in (fasta_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records.append((cur_id, "".join(buf).upper()))
            cur_id = line[1:].strip().split()[0] if line[1:].strip() else ""
            buf = []
        else:
            if cur_id is None:
                raise TranslationAlignmentError(
                    "Malformed FASTA: sequence data precedes header."
                )
            buf.append(line)
    if cur_id is not None:
        records.append((cur_id, "".join(buf).upper()))
    return records


def _translate_cds_strict(cds_nt: str) -> str:
    seq = cds_nt.strip().upper().replace("U", "T")
    if len(seq) % 3 != 0:
        raise TranslationAlignmentError(f"CDS length {len(seq)} is not divisible by 3.")
    out: list[str] = []
    for idx in range(0, len(seq), 3):
        codon = seq[idx : idx + 3]
        out.append(_CODON_TABLE.get(codon, "X"))
    return "".join(out)


class DefensiveTranslationAlignmentScreener:
    """Reference translation & ancestral-frame screener satisfying all 6 invariants."""

    def load_translation_alignment(
        self, fasta_text: str, cds_nt_length: int | None = None
    ) -> dict[str, Any]:
        """Parse and validate a protein translation FASTA alignment fail-closed."""
        records = _parse_fasta(fasta_text)
        if not records:
            raise TranslationAlignmentError(
                "Translation alignment is empty or contains no valid FASTA records."
            )

        # Invariant 2: reject empty or duplicate taxon identifiers
        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for idx, (seq_id, _seq) in enumerate(records):
            if not seq_id:
                raise TranslationAlignmentError(
                    f"Empty FASTA sequence ID at record index {idx}."
                )
            if seq_id in seen:
                duplicates.append(seq_id)
            seen[seq_id] = seen.get(seq_id, 0) + 1
        if duplicates:
            uniq_dups = sorted(set(duplicates))
            raise TranslationAlignmentError(
                f"Duplicate sequence IDs in translation alignment: {', '.join(uniq_dups)}."
            )

        # Invariant 1: equal alignment length diagnostics with modal length + outliers
        lengths = {seq_id: len(seq) for seq_id, seq in records}
        len_counts = Counter(lengths.values())
        modal_len, modal_count = len_counts.most_common(1)[0]
        if len(len_counts) > 1:
            mismatched = [
                f"{sid}: {slen} aa"
                for sid, slen in lengths.items()
                if slen != modal_len
            ]
            raise TranslationAlignmentError(
                f"Sequences have inconsistent lengths (expected {modal_len} aa across "
                f"{modal_count} sequences): {', '.join(mismatched)}."
            )
        if modal_len == 0:
            raise TranslationAlignmentError("Alignment sequences have length 0.")

        # Invariant 4: nucleotide alignment accidentally passed as protein alignment
        nuc_chars = set("ACGTUN-")
        if all(set(seq) <= nuc_chars for _sid, seq in records):
            looks_like_nuc = (
                cds_nt_length is not None and modal_len == cds_nt_length
            ) or (
                modal_len >= 30
                and all(len(set(seq) - {"-"}) <= 5 for _, seq in records)
            )
            if looks_like_nuc:
                raise TranslationAlignmentError(
                    "Suspected nucleotide alignment passed as amino-acid translation "
                    f"(length={modal_len}, alphabet subset of ACGTUN-)."
                )

        # Invariant 3: CDS-to-protein codon parity
        if cds_nt_length is not None:
            if cds_nt_length % 3 != 0:
                raise TranslationAlignmentError(
                    f"CDS annotation length {cds_nt_length} is not a multiple of 3."
                )
            expected_aa = cds_nt_length // 3
            if modal_len not in (expected_aa, expected_aa - 1):
                raise TranslationAlignmentError(
                    f"Translated alignment length {modal_len} aa (3*len={modal_len * 3}) "
                    f"does not match CDS length {cds_nt_length} nt ({expected_aa} aa)."
                )

        # Invariant 5: distinguish internal premature stop codons from terminal stop
        internal_stops: dict[str, list[int]] = {}
        sequences: dict[str, str] = {}
        for seq_id, seq in records:
            internal_pos = [i + 1 for i, aa in enumerate(seq[:-1]) if aa == "*"]
            if internal_pos:
                internal_stops[seq_id] = internal_pos
            sequences[seq_id] = seq

        return {
            "alignment_length": modal_len,
            "sequences": sequences,
            "internal_premature_stops": internal_stops,
        }

    @staticmethod
    def cross_check_nuc_vs_aa_mutations(
        parent_cds_nt: str,
        child_cds_nt: str,
        reported_aa_muts: Sequence[str],
    ) -> dict[str, Any]:
        """Verify 1-based AA mutations match independently translated parent->child codons."""
        parent_aa = _translate_cds_strict(parent_cds_nt)
        child_aa = _translate_cds_strict(child_cds_nt)
        if len(parent_aa) != len(child_aa):
            raise TranslationAlignmentError(
                "Parent and child CDS sequences differ in codon length."
            )

        expected_muts = [
            f"{pa}{idx}{ca}"
            for idx, (pa, ca) in enumerate(zip(parent_aa, child_aa), start=1)
            if pa != ca
        ]
        reported_norm = [str(m).strip().upper() for m in reported_aa_muts]
        missing_in_reported = sorted(set(expected_muts) - set(reported_norm))
        extra_in_reported = sorted(set(reported_norm) - set(expected_muts))
        consistent = not missing_in_reported and not extra_in_reported
        return {
            "consistent": consistent,
            "expected_aa_muts": expected_muts,
            "reported_aa_muts": reported_norm,
            "missing_in_reported": missing_in_reported,
            "extra_in_reported": extra_in_reported,
        }

    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Any:
        """Dispatch single-callable ancestral translation interface for auditing."""
        if stage == "load_alignment":
            return self.load_translation_alignment(
                payload["fasta_text"],
                cds_nt_length=payload.get("cds_nt_length"),
            )
        if stage == "cross_check_mutations":
            return self.cross_check_nuc_vs_aa_mutations(
                payload["parent_cds_nt"],
                payload["child_cds_nt"],
                payload["reported_aa_muts"],
            )
        raise ValueError(f"Unknown translation audit stage: {stage!r}")


class NaiveTranslationAlignmentLoader:
    """Reference naive loader reproducing augur#1919 / TreeTime / SeqCorrector bugs."""

    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Any:
        if stage == "load_alignment":
            records = _parse_fasta(payload["fasta_text"])
            # Bug 2: dict comprehension silently overwrites duplicate taxon IDs.
            # Bug 1: catches unequal lengths with opaque error missing modal/outlier detail.
            # Bug 3 & 4: skips CDS codon parity and ACGTN protein collision checks.
            # Bug 5: silently masks internal premature '*' stops to 'X'.
            lengths = {len(seq) for _sid, seq in records}
            if len(lengths) > 1:
                raise RuntimeError("ERROR: SequenceData: loading alignment failed")
            seqs = {sid: seq.replace("*", "X") for sid, seq in records}
            return {
                "alignment_length": next(iter(lengths), 0),
                "sequences": seqs,
                "internal_premature_stops": {},
            }
        if stage == "cross_check_mutations":
            # Bug 6: trusts reported AA mutations blindly without checking codon translation.
            reported = [str(m).strip().upper() for m in payload["reported_aa_muts"]]
            return {
                "consistent": True,
                "expected_aa_muts": reported,
                "reported_aa_muts": reported,
                "missing_in_reported": [],
                "extra_in_reported": [],
            }
        raise ValueError(f"Unknown stage: {stage!r}")


def audit_ancestral_translation_pipeline(
    pipeline_fn: Callable[[str, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Audit an ancestral translation pipeline across 6 genomic surveillance invariants."""
    invariants: list[dict[str, Any]] = []
    aa = SYNTHETIC_REPORTER_PROTEIN_AA
    cds = SYNTHETIC_REPORTER_CDS_NT

    # 1. Equal alignment length diagnostics (modal length + offending IDs)
    try:
        unequal_fasta = (
            f">iso_ok1\n{aa}\n>iso_ok2\n{aa}\n>iso_outlier\n{aa + 'GGGAAA'}\n"
        )
        try:
            pipeline_fn("load_alignment", {"fasta_text": unequal_fasta})
            inv1_ok = False
            inv1_detail = "Accepted unaligned translation sequences of unequal length."
        except TranslationAlignmentError as exc:
            msg = str(exc)
            inv1_ok = "20 aa" in msg and "iso_outlier" in msg and "26 aa" in msg
            inv1_detail = f"Diagnostic error: {msg}"
        except Exception as exc:
            inv1_ok = False
            inv1_detail = f"Opaque error without modal/outlier diagnostics: {exc}"
    except Exception as exc:
        inv1_ok = False
        inv1_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "equal_alignment_length_diagnostics",
            "passed": inv1_ok,
            "detail": inv1_detail,
        }
    )

    # 2. Duplicate taxon ID fail-closed rejection
    try:
        dup_fasta = f">iso_dup\n{aa}\n>iso_dup\n{'V' + aa[1:]}\n"
        try:
            pipeline_fn("load_alignment", {"fasta_text": dup_fasta})
            inv2_ok = False
            inv2_detail = "Silently overwrote duplicate sequence ID iso_dup."
        except TranslationAlignmentError as exc:
            inv2_ok = "iso_dup" in str(exc)
            inv2_detail = f"Rejected duplicate ID: {exc}"
    except Exception as exc:
        inv2_ok = False
        inv2_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "duplicate_taxon_id_fail_closed",
            "passed": inv2_ok,
            "detail": inv2_detail,
        }
    )

    # 3. CDS-to-protein codon length parity
    try:
        valid_fasta = f">iso_1\n{aa}\n>iso_2\n{aa}\n"
        try:
            # Pass mismatched CDS length (57 nt vs 60 nt for 20 aa)
            pipeline_fn(
                "load_alignment",
                {"fasta_text": valid_fasta, "cds_nt_length": 57},
            )
            inv3_ok = False
            inv3_detail = "Accepted 20 aa alignment against mismatched 57 nt CDS."
        except TranslationAlignmentError as exc:
            inv3_ok = "57" in str(exc) and "20" in str(exc)
            inv3_detail = f"Rejected CDS mismatch: {exc}"
    except Exception as exc:
        inv3_ok = False
        inv3_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "cds_codon_length_parity",
            "passed": inv3_ok,
            "detail": inv3_detail,
        }
    )

    # 4. Nucleotide alignment passed as protein alignment guard (A/C/G/T are IUPAC AAs!)
    try:
        nuc_as_aa = f">iso_1\n{cds}\n>iso_2\n{cds}\n"
        try:
            pipeline_fn(
                "load_alignment",
                {"fasta_text": nuc_as_aa, "cds_nt_length": len(cds)},
            )
            inv4_ok = False
            inv4_detail = "Accepted raw 60-nt ACGT alignment as amino-acid translation."
        except TranslationAlignmentError as exc:
            inv4_ok = "nucleotide" in str(exc).lower()
            inv4_detail = f"Rejected nucleotide-as-protein: {exc}"
    except Exception as exc:
        inv4_ok = False
        inv4_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "nucleotide_passed_as_protein_guard",
            "passed": inv4_ok,
            "detail": inv4_detail,
        }
    )

    # 5. Internal premature stop preservation ('*' at codon 4, not masked to 'X')
    try:
        stop_aa = aa[:3] + "*" + aa[4:]
        stop_fasta = f">iso_wt\n{aa}\n>iso_stop\n{stop_aa}\n"
        res_stop = pipeline_fn(
            "load_alignment",
            {"fasta_text": stop_fasta, "cds_nt_length": len(cds)},
        )
        seq_out = res_stop["sequences"].get("iso_stop", "")
        prem_stops = res_stop.get("internal_premature_stops", {})
        inv5_ok = (seq_out[3] == "*") and (prem_stops.get("iso_stop") == [4])
        inv5_detail = f"seq[4]={seq_out[3]!r}, internal_premature_stops={prem_stops!r}"
    except Exception as exc:
        inv5_ok = False
        inv5_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "internal_premature_stop_preservation",
            "passed": inv5_ok,
            "detail": inv5_detail,
        }
    )

    # 6. Nucleotide codon vs independent AA mutation consistency cross-check
    try:
        # Mutate codon 2 GCT (A) -> GAT (D), but supply discordant AA report ['A2G']
        child_cds = cds[:3] + "GAT" + cds[6:]
        res_xchk = pipeline_fn(
            "cross_check_mutations",
            {
                "parent_cds_nt": cds,
                "child_cds_nt": child_cds,
                "reported_aa_muts": ["A2G"],
            },
        )
        inv6_ok = (
            res_xchk.get("consistent") is False
            and res_xchk.get("expected_aa_muts") == ["A2D"]
            and res_xchk.get("missing_in_reported") == ["A2D"]
            and res_xchk.get("extra_in_reported") == ["A2G"]
        )
        inv6_detail = f"cross_check={res_xchk!r}"
    except Exception as exc:
        inv6_ok = False
        inv6_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "nuc_vs_aa_mutation_consistency",
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


def run_ancestral_translation_self_check() -> dict[str, Any]:
    """Compare DefensiveTranslationAlignmentScreener against NaiveTranslationAlignmentLoader."""
    hardened = audit_ancestral_translation_pipeline(
        DefensiveTranslationAlignmentScreener()
    )
    naive = audit_ancestral_translation_pipeline(NaiveTranslationAlignmentLoader())
    caught = [inv["invariant"] for inv in naive["invariants"] if not inv["passed"]]
    return {
        "hardened_passed": hardened["passed"],
        "hardened": hardened,
        "naive_caught_invariants": caught,
        "naive_total_invariants": naive["total_invariants"],
    }
