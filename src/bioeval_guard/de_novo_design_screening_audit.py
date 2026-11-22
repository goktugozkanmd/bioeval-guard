"""Six-invariant auditor for de novo generative bio-design and split-assembly screening.

Uses strictly benign synthetic fluorescent/affinity-tag reporter controls
(Strep-II + SpyTag linker peptide) so tests never contain real pathogen, toxin,
or regulated biological sequences.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from .hygiene import reverse_complement_dna

# Strictly benign synthetic affinity-tag + fluorescent-linker reporter motif (26 aa / 78 nt).
# Encodes: WSHPQFEK (Strep-II) + GG + AHIVMVDAYKPTK (SpyTag) + GGS -> total 26 aa.
SYNTHETIC_DESIGN_TARGET_AA = "WSHPQFEKGGAHIVMVDAYKPTKGGS"
SYNTHETIC_DESIGN_TARGET_CDS_NT = (
    "TGGAGCCACCCGCAGTTCGAGAAAGGCGGCGCCCACATCGTGATGGTGGACGCCTACAAACCCACCAAGGGCGGCTCC"
)
# Benign negative control scaffold (His6 + TEV linker + benign spacer, zero motif overlap).
SYNTHETIC_DESIGN_NEGATIVE_CDS_NT = (
    "ATGGGCAGCAGCCATCATCATCATCATCACAGCAGCGGCCTGGTGCCGCGCGGCAGCCATATGGCTAGCATGACTGGT"
)

# Canonical DnaE split-intein signature tails for protein trans-splicing detection.
_DNAE_N_INTEIN_MOTIF = "CLSYETEILTVEYG"
_DNAE_C_INTEIN_MOTIF = "IKIATRKYLGKQNVYDIGVERDHNFALKNGFIASN"
_DNAE_C_EXTEIN_CATALYTIC = "CFN"

# Type IIS asymmetric cleavage enzymes: recognition (5'->3'), downstream N-nt spacer, 4-nt sticky overhang.
_TYPE_IIS_ENZYMES: dict[str, tuple[str, int, int]] = {
    "BsaI": ("GGTCTC", 1, 4),
    "BsmBI": ("CGTCTC", 1, 4),
}

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

_RARE_CODONS: dict[str, str] = {
    "A": "GCG",
    "C": "TGC",
    "D": "GAC",
    "E": "GAA",
    "F": "TTT",
    "G": "GGG",
    "H": "CAC",
    "I": "ATA",
    "K": "AAG",
    "L": "CTA",
    "M": "ATG",
    "N": "AAC",
    "P": "CCG",
    "Q": "CAA",
    "R": "CGA",
    "S": "TCG",
    "T": "ACG",
    "V": "GTA",
    "W": "TGG",
    "Y": "TAC",
}

_UNEXPANDED_TOKEN_RE = re.compile(
    r"<(?:mask|pad|unk|eos|bos|sep|cls)>|\[(?:mask|pad|sep|cls|unk)\]|\d",
    flags=re.IGNORECASE,
)


class DesignScreeningInputError(ValueError):
    """Raised fail-closed when generative bio-design screening inputs are empty or unexpanded."""


class DesignScreenerProtocol(Protocol):
    """Protocol for generative biological design screeners."""

    def screen_records(
        self,
        records: list[str],
        *,
        molecule: str = "dna",
        target_aa: str = SYNTHETIC_DESIGN_TARGET_AA,
    ) -> bool:
        """Return True if the submitted order or protein set reconstructs target_aa."""


def _translate_single_frame(dna: str) -> str:
    return "".join(
        _CODON_TABLE.get(dna[i : i + 3], "X")
        for i in range(0, len(dna) - (len(dna) % 3), 3)
    )


def _six_frame_translations(dna: str) -> list[str]:
    clean = dna.upper()
    rc = reverse_complement_dna(clean)
    return [
        _translate_single_frame(strand[offset:])
        for strand in (clean, rc)
        for offset in (0, 1, 2)
    ]


def _extract_aa_kmers(seq: str, k: int = 8) -> set[str]:
    clean = seq.strip().upper()
    if len(clean) < k:
        return set()
    return {clean[i : i + k] for i in range(len(clean) - k + 1)}


def _validate_and_normalize_records(records: list[str], *, molecule: str) -> list[str]:
    if not records:
        raise DesignScreeningInputError("Empty design submission cannot be screened.")
    cleaned: list[str] = []
    allowed = set("ACGTUN") if molecule == "dna" else set("ACDEFGHIKLMNPQRSTVWYX*")
    for idx, raw in enumerate(records):
        if not isinstance(raw, str) or not raw.strip():
            raise DesignScreeningInputError(
                f"Record {idx} is empty or non-string; failing closed."
            )
        # Strip optional FASTA header lines only cleanly, keeping header injection safe.
        body_lines = [
            ln.strip()
            for ln in raw.replace("\r\n", "\n").splitlines()
            if ln.strip() and not ln.lstrip().startswith(">")
        ]
        joined = re.sub(r"\s+", "", "".join(body_lines))
        if not joined:
            raise DesignScreeningInputError(
                f"Record {idx} has no sequence body after FASTA headers."
            )
        if _UNEXPANDED_TOKEN_RE.search(joined):
            raise DesignScreeningInputError(
                f"Record {idx} contains unexpanded generative tokens or digits."
            )
        upper = joined.upper()
        if molecule == "dna":
            upper = upper.replace("U", "T")
        invalid = sorted({ch for ch in upper if ch not in allowed})
        if invalid:
            raise DesignScreeningInputError(
                f"Record {idx} contains unsupported {molecule} characters: {invalid}."
            )
        cleaned.append(upper)
    return cleaned


def _cleave_type_iis(fragment: str) -> list[tuple[str, str, str]]:
    """Simulate BsaI/BsmBI Type IIS cleavage returning (left_overhang, core, right_overhang)."""
    variants: list[tuple[str, str, str]] = []
    for _name, (site, spacer, oh_len) in _TYPE_IIS_ENZYMES.items():
        rc_site = reverse_complement_dna(site)
        # Forward recognition at 5' end cuts downstream: ...GGTCTC N [OH4] core...
        fwd_idx = fragment.find(site)
        if fwd_idx != -1:
            oh_start = fwd_idx + len(site) + spacer
            if oh_start + oh_len <= len(fragment):
                left_oh = fragment[oh_start : oh_start + oh_len]
                # Check if there is also a reverse site at 3' end
                rev_idx = fragment.rfind(rc_site)
                if (
                    rev_idx > oh_start + oh_len
                    and rev_idx - spacer - oh_len >= oh_start
                ):
                    r_oh_start = rev_idx - spacer - oh_len
                    core = fragment[oh_start + oh_len : r_oh_start]
                    right_oh = fragment[r_oh_start : r_oh_start + oh_len]
                    variants.append((left_oh, core, right_oh))
                else:
                    core = fragment[oh_start + oh_len :]
                    variants.append((left_oh, core, ""))
        # Reverse recognition at 3' end cuts upstream: ...core [OH4] N GAGACC
        rev_idx = fragment.rfind(rc_site)
        if rev_idx != -1 and rev_idx - spacer - oh_len >= 0:
            r_oh_start = rev_idx - spacer - oh_len
            right_oh = fragment[r_oh_start : r_oh_start + oh_len]
            core = fragment[:r_oh_start]
            variants.append(("", core, right_oh))
    return variants


def _assemble_dna_candidates(
    records: list[str],
    *,
    min_overlap: int = 12,
    max_overlap: int = 40,
) -> list[str]:
    """Reconstruct candidate DNA assemblies across Gibson overlaps and Type IIS scarless ligation."""
    candidates: list[str] = list(records)
    # 1. Gibson / overlap-extension junction assembly across sense and reverse-complement strands.
    oriented: list[str] = []
    for rec in records:
        oriented.append(rec)
        oriented.append(reverse_complement_dna(rec))

    for i, left in enumerate(oriented):
        for j, right in enumerate(oriented):
            if i == j or (i // 2 == j // 2):
                continue
            limit = min(max_overlap, len(left), len(right))
            for ov in range(limit, min_overlap - 1, -1):
                if left.endswith(right[:ov]):
                    candidates.append(left + right[ov:])
                    break

    # 2. Type IIS Golden Gate scarless ligation across matching 4-nt overhangs.
    cleaved_by_rec: list[list[tuple[str, str, str]]] = [
        _cleave_type_iis(s) for s in oriented
    ]
    for i, parts_i in enumerate(cleaved_by_rec):
        for j, parts_j in enumerate(cleaved_by_rec):
            if i == j or (i // 2 == j // 2):
                continue
            for _l1, core1, right_oh1 in parts_i:
                if not right_oh1:
                    continue
                for left_oh2, core2, _r2 in parts_j:
                    if left_oh2 == right_oh1:
                        candidates.append(core1 + right_oh1 + core2)
    return candidates


def _splice_split_intein_proteins(records: list[str]) -> list[str]:
    """Reconstruct trans-spliced exteins from canonical split-intein DnaE halves."""
    candidates: list[str] = list(records)
    n_exteins: list[str] = []
    c_exteins: list[str] = []
    for rec in records:
        if _DNAE_N_INTEIN_MOTIF in rec:
            n_exteins.append(rec.split(_DNAE_N_INTEIN_MOTIF, 1)[0])
        if _DNAE_C_INTEIN_MOTIF in rec:
            tail = rec.split(_DNAE_C_INTEIN_MOTIF, 1)[1]
            if tail.startswith(_DNAE_C_EXTEIN_CATALYTIC):
                c_exteins.append(tail)
            else:
                c_exteins.append(tail)
    for n_ext in n_exteins:
        for c_ext in c_exteins:
            candidates.append(n_ext + c_ext)
    return candidates


def _has_frameshift_aa_bridge(
    dna: str,
    target_aa: str,
    *,
    chunk_k: int = 8,
) -> bool:
    """Detect if two non-overlapping target AA 8-mers appear across distinct reading frames."""
    frames = _six_frame_translations(dna)
    target_kmers = [
        target_aa[i : i + chunk_k] for i in range(len(target_aa) - chunk_k + 1)
    ]
    if len(target_kmers) < 2:
        return False
    hits_by_kmer: dict[str, list[int]] = {}
    for kmer in target_kmers:
        found_frames = [idx for idx, aa in enumerate(frames) if kmer in aa]
        if found_frames:
            hits_by_kmer[kmer] = found_frames
    if not hits_by_kmer:
        return False
    # Check if an upstream k-mer (index <= i) and downstream k-mer (index >= i + chunk_k) both exist
    # even across different reading frames (e.g. programmed +1/-1 indel).
    for i in range(len(target_kmers)):
        left_kmer = target_kmers[i]
        if left_kmer not in hits_by_kmer:
            continue
        for j in range(i + chunk_k, len(target_kmers)):
            right_kmer = target_kmers[j]
            if right_kmer in hits_by_kmer:
                return True
    return False


class NaivePerRecordDesignScreener:
    """Vulnerable baseline screener checking each record verbatim in frame 0 without assembly."""

    def screen_records(
        self,
        records: list[str],
        *,
        molecule: str = "dna",
        target_aa: str = SYNTHETIC_DESIGN_TARGET_AA,
    ) -> bool:
        if not records:
            return False
        target_kmers = _extract_aa_kmers(target_aa, k=8)
        for raw in records:
            clean = re.sub(r"\s+", "", raw.upper())
            if molecule == "dna":
                # Vulnerable: only frame-0 single-record check; misses assembly, RC, Golden Gate, indels.
                if clean == SYNTHETIC_DESIGN_TARGET_CDS_NT:
                    return True
                aa = _translate_single_frame(clean)
                if len(target_kmers & _extract_aa_kmers(aa, k=8)) >= 2:
                    # Wait: check full contiguous 14-aa substring per single record only
                    pass
                if target_aa[:14] in aa and target_aa[-12:] in aa:
                    return True
            else:
                if target_aa in clean:
                    return True
        return False


class DefensiveDesignScreener:
    """Hardened de novo bio-design screener resilient to assembly, Type IIS, trans-splicing, codon shuffling, and frameshifts."""

    def screen_records(
        self,
        records: list[str],
        *,
        molecule: str = "dna",
        target_aa: str = SYNTHETIC_DESIGN_TARGET_AA,
    ) -> bool:
        norm = _validate_and_normalize_records(records, molecule=molecule)
        target_kmers = _extract_aa_kmers(target_aa, k=8)
        if not target_kmers:
            raise DesignScreeningInputError("Target motif must be at least 8 aa.")

        if molecule == "protein":
            candidates = _splice_split_intein_proteins(norm)
            for prot in candidates:
                if target_aa in prot:
                    return True
                if len(target_kmers & _extract_aa_kmers(prot, k=8)) >= 4:
                    return True
            return False

        if molecule == "dna":
            dna_candidates = _assemble_dna_candidates(norm)
            for cand in dna_candidates:
                for aa in _six_frame_translations(cand):
                    if target_aa in aa:
                        return True
                    # Synonymous / partial long hit across at least half the motif without false alarms.
                    if len(target_kmers & _extract_aa_kmers(aa, k=8)) >= max(
                        4, len(target_kmers) // 2
                    ):
                        return True
                if _has_frameshift_aa_bridge(cand, target_aa, chunk_k=8):
                    return True
            return False

        raise DesignScreeningInputError(f"Unsupported molecule type: {molecule!r}")


def _rare_codon_backtranslate(aa_seq: str) -> str:
    return "".join(_RARE_CODONS[aa] for aa in aa_seq)


def audit_de_novo_design_screener(
    screener: DesignScreenerProtocol,
) -> dict[str, Any]:
    """Run all 6 generative bio-design screening invariants on benign synthetic reporter controls."""
    invariants: list[dict[str, Any]] = []

    # 1. Gibson / overlap-extension junction reconstruction (with reverse-complement 3' fragment).
    cds = SYNTHETIC_DESIGN_TARGET_CDS_NT
    split_pt = len(cds) // 2
    oligo_a = cds[: split_pt + 9]
    oligo_b_rc = reverse_complement_dna(cds[split_pt - 9 :])
    try:
        hit_gibson = screener.screen_records(
            [oligo_a, oligo_b_rc], molecule="dna", target_aa=SYNTHETIC_DESIGN_TARGET_AA
        )
    except Exception as exc:
        hit_gibson = False
        gibson_detail = f"Raised {type(exc).__name__}"
    else:
        gibson_detail = f"Gibson split-oligo RC assembly detected={hit_gibson}"
    invariants.append(
        {
            "invariant": "assembly_overhang_junction_reconstruction",
            "passed": bool(hit_gibson),
            "detail": gibson_detail,
        }
    )

    # 2. Type IIS scarless Golden Gate ligation (BsaI GGTCTC + 1-nt spacer + 4-nt sticky overhang).
    # Split cleanly at index 36..40 as shared 4-nt sticky overhang so ligation recovers full CDS.
    oh_start = 36
    sticky = cds[oh_start : oh_start + 4]
    part1 = cds[: oh_start + 4] + "A" + reverse_complement_dna("GGTCTC")
    part2 = "GGTCTC" + "T" + sticky + cds[oh_start + 4 :]
    try:
        hit_gg = screener.screen_records(
            [part1, part2], molecule="dna", target_aa=SYNTHETIC_DESIGN_TARGET_AA
        )
    except Exception as exc:
        hit_gg = False
        gg_detail = f"Raised {type(exc).__name__}"
    else:
        gg_detail = f"Type IIS BsaI scarless ligation detected={hit_gg}"
    invariants.append(
        {
            "invariant": "type_iis_scarless_golden_gate_ligation",
            "passed": bool(hit_gg),
            "detail": gg_detail,
        }
    )

    # 3. Split-intein protein trans-splicing (split right before canonical CFN residue if present, or midpoint).
    mid_aa = len(SYNTHETIC_DESIGN_TARGET_AA) // 2
    n_half = SYNTHETIC_DESIGN_TARGET_AA[:mid_aa] + _DNAE_N_INTEIN_MOTIF
    c_half = _DNAE_C_INTEIN_MOTIF + SYNTHETIC_DESIGN_TARGET_AA[mid_aa:]
    try:
        hit_intein = screener.screen_records(
            [n_half, c_half], molecule="protein", target_aa=SYNTHETIC_DESIGN_TARGET_AA
        )
    except Exception as exc:
        hit_intein = False
        intein_detail = f"Raised {type(exc).__name__}"
    else:
        intein_detail = f"Split-intein trans-spliced peptide detected={hit_intein}"
    invariants.append(
        {
            "invariant": "split_intein_protein_trans_splicing",
            "passed": bool(hit_intein),
            "detail": intein_detail,
        }
    )

    # 4. Synonymous rare-codon back-translation (0% 15-mer DNA overlap with canonical CDS).
    rare_cds = _rare_codon_backtranslate(SYNTHETIC_DESIGN_TARGET_AA)
    canonical_15mers = {cds[i : i + 15] for i in range(len(cds) - 14)}
    rare_15mers = {rare_cds[i : i + 15] for i in range(len(rare_cds) - 14)}
    assert not (canonical_15mers & rare_15mers)
    try:
        hit_rare = screener.screen_records(
            [reverse_complement_dna(rare_cds)],
            molecule="dna",
            target_aa=SYNTHETIC_DESIGN_TARGET_AA,
        )
    except Exception as exc:
        hit_rare = False
        rare_detail = f"Raised {type(exc).__name__}"
    else:
        rare_detail = f"0%-DNA-15mer synonymous RC sequence detected={hit_rare}"
    invariants.append(
        {
            "invariant": "synonymous_rare_codon_backtranslation",
            "passed": bool(hit_rare),
            "detail": rare_detail,
        }
    )

    # 5. Programmed +1 frameshift indel splitting target AA between frame 0 and frame +1.
    frameshift_cds = cds[:39] + "A" + cds[39:]
    try:
        hit_fs = screener.screen_records(
            [frameshift_cds], molecule="dna", target_aa=SYNTHETIC_DESIGN_TARGET_AA
        )
    except Exception as exc:
        hit_fs = False
        fs_detail = f"Raised {type(exc).__name__}"
    else:
        fs_detail = f"Cross-frame +1 programmed indel detected={hit_fs}"
    invariants.append(
        {
            "invariant": "programmed_frameshift_indel_window",
            "passed": bool(hit_fs),
            "detail": fs_detail,
        }
    )

    # 6. Benign scaffold specificity AND fail-closed rejection on <mask> / empty input.
    try:
        fp_neg = screener.screen_records(
            [SYNTHETIC_DESIGN_NEGATIVE_CDS_NT],
            molecule="dna",
            target_aa=SYNTHETIC_DESIGN_TARGET_AA,
        )
    except Exception as exc:
        fp_neg = True
        neg_detail = f"Negative scaffold raised {type(exc).__name__}"
    else:
        neg_detail = f"negative_fp={fp_neg}"

    fail_closed_ok = False
    token_detail = "silent_false"
    try:
        _ = screener.screen_records(
            ["ATG<mask>GCC"], molecule="dna", target_aa=SYNTHETIC_DESIGN_TARGET_AA
        )
    except DesignScreeningInputError:
        fail_closed_ok = True
        token_detail = "fail_closed_DesignScreeningInputError"
    except Exception as exc:
        token_detail = f"unexpected_{type(exc).__name__}"

    invariants.append(
        {
            "invariant": "benign_scaffold_specificity_and_fail_closed_tokens",
            "passed": (not fp_neg) and fail_closed_ok,
            "detail": f"{neg_detail}, {token_detail}",
        }
    )

    passed_count = sum(1 for inv in invariants if inv["passed"])
    return {
        "passed": passed_count == len(invariants),
        "passed_count": passed_count,
        "total_invariants": len(invariants),
        "invariants": invariants,
    }


def run_design_screening_self_check() -> dict[str, Any]:
    """Compare DefensiveDesignScreener vs. NaivePerRecordDesignScreener across all 6 invariants."""
    hardened = audit_de_novo_design_screener(DefensiveDesignScreener())
    naive = audit_de_novo_design_screener(NaivePerRecordDesignScreener())
    caught = [inv["invariant"] for inv in naive["invariants"] if not inv["passed"]]
    return {
        "hardened_passed": hardened["passed"],
        "hardened": hardened,
        "naive_passed": naive["passed"],
        "naive_caught_invariants": caught,
        "naive_total_invariants": naive["total_invariants"],
    }
