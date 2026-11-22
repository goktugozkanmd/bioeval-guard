"""Defensive protein and codon-degeneracy screening invariant auditor.

Uses strictly benign synthetic affinity/reporter linker peptides (His6 + Strep-tag II +
GFP N-terminal linker; zero pathogen, toxin, or select-agent sequences) to audit whether
biosecurity sequence-screening pipelines preserve 3-letter IUPAC dealiasing, 6-frame
synonymous nucleotide back-translation coverage, PTM/modified-residue normalization,
circular/split fusion boundary continuity, IUPAC ambiguity expansion with fail-closed
guards, and sliding-window chimeric scaffold localization.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .hygiene import reverse_complement_dna

# Strictly benign synthetic reporter/affinity linker sequences (zero hazardous motifs).
SYNTHETIC_REPORTER_MOTIF = "WSHPQFEKGGSG"
SYNTHETIC_REPORTER_PEPTIDE = f"MGHSHHHHHHGS{SYNTHETIC_REPORTER_MOTIF}SMVSKGEELFT"
SYNTHETIC_NEGATIVE_PEPTIDE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPN"

_THREE_TO_ONE_AA: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "C",
    "PYL": "K",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "XAA": "X",
    "UNK": "X",
}

_MODIFIED_RESIDUE_MAP: dict[str, str] = {
    "SEP": "S",  # O-phosphoserine
    "TPO": "T",  # O-phosphothreonine
    "PTR": "Y",  # O-phosphotyrosine
    "MSE": "M",  # Selenomethionine
    "CSO": "C",  # S-hydroxycysteine
    "HYP": "P",  # 4-hydroxyproline
    "KCX": "K",  # Lysine NZ-carboxylic acid
    "MLY": "K",  # N-dimethyl-lysine
}

_AA_AMBIGUITY: dict[str, tuple[str, ...]] = {
    "B": ("D", "N"),
    "Z": ("E", "Q"),
    "J": ("I", "L"),
    "X": tuple("ACDEFGHIKLMNPQRSTVWY"),
}

_CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

_DNA_CODON_TABLE: dict[str, str] = {
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

_IUPAC_DNA: dict[str, tuple[str, ...]] = {
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


def _validate_brackets(text: str) -> None:
    """Reject empty, unclosed, or mismatched modification brackets."""
    pairs = {"(": ")", "[": "]"}
    closing = {")": "(", "]": "["}
    stack: list[tuple[str, int]] = []
    for idx, ch in enumerate(text):
        if ch in pairs:
            stack.append((ch, idx))
        elif ch in closing:
            if not stack or stack[-1][0] != closing[ch]:
                raise ValueError(f"Mismatched modification bracket at index {idx}.")
            _, start_idx = stack.pop()
            if not text[start_idx + 1 : idx].strip():
                raise ValueError(f"Empty modification tag at index {start_idx}.")
    if stack:
        raise ValueError("Unclosed modification bracket in sequence input.")


def _parse_fasta_bodies(raw: str) -> list[str]:
    """Extract raw sequence bodies per FASTA record while rejecting mid-line `>`."""
    bodies: list[str] = []
    current: list[str] = []
    for line in (raw or "").replace("\r\n", "\n").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith(">"):
            if current:
                bodies.append(" ".join(current))
                current = []
            continue
        if ">" in stripped:
            raise ValueError("Malformed mid-line '>' FASTA header injection detected.")
        current.append(stripped)
    if current:
        bodies.append(" ".join(current))
    return bodies


def _dealias_body(body: str) -> tuple[str, bool]:
    """Normalize a single sequence body to canonical 1-letter AA or nucleotide string.

    Returns `(normalized_sequence, is_nucleotide_candidate)`.
    """
    _validate_brackets(body)
    cleaned_for_tokens = re.sub(r"[0-9*\r\n]+", " ", body).strip()
    if not cleaned_for_tokens:
        return "", False

    # Check if sequence is written in 3-letter IUPAC/PTM blocks (e.g. Trp-Ser-His or TRP SER HIS).
    if re.search(r"[\s\-_,;]", cleaned_for_tokens):
        raw_tokens = [tok for tok in re.split(r"[\s\-_,;]+", cleaned_for_tokens) if tok]
        if len(raw_tokens) >= 3:
            mapped: list[str] = []
            all_three_letter = True
            for tok in raw_tokens:
                inner = tok.strip("()[]").upper()
                if inner in _THREE_TO_ONE_AA:
                    mapped.append(_THREE_TO_ONE_AA[inner])
                elif inner in _MODIFIED_RESIDUE_MAP:
                    mapped.append(_MODIFIED_RESIDUE_MAP[inner])
                else:
                    all_three_letter = False
                    break
            if all_three_letter:
                return "".join(mapped), False

    # Replace inline parenthesized/bracketed modification tokens such as (SEP), [TPO].
    def _replace_mod(match: re.Match[str]) -> str:
        tag = match.group(1).strip().upper()
        if tag in _MODIFIED_RESIDUE_MAP:
            return _MODIFIED_RESIDUE_MAP[tag]
        if tag in _THREE_TO_ONE_AA:
            return _THREE_TO_ONE_AA[tag]
        raise ValueError(f"Unsupported chemical/PTM modification tag: {tag!r}")

    dealiased = re.sub(r"\(([^()]+)\)", _replace_mod, body)
    dealiased = re.sub(r"\[([^\[\]]+)\]", _replace_mod, dealiased)
    compact = re.sub(r"[\s0-9*\-]+", "", dealiased).upper()
    if not compact:
        return "", False

    # Distinguish nucleotide back-translations (U->T) from selenocysteine proteins (U->C).
    if len(compact) >= 36 and set(compact) <= set("ACGTUN"):
        return compact.replace("U", "T"), True

    protein = compact.replace("U", "C").replace("O", "K")
    return protein, False


def _translate_dna_frame(dna_seq: str, offset: int) -> str:
    """Translate one reading frame into canonical AA symbols (`*` for stops)."""
    aa: list[str] = []
    limit = len(dna_seq) - ((len(dna_seq) - offset) % 3)
    for i in range(offset, limit, 3):
        codon = dna_seq[i : i + 3]
        if codon in _DNA_CODON_TABLE:
            aa.append(_DNA_CODON_TABLE[codon])
            continue
        if any(ch not in _IUPAC_DNA for ch in codon):
            aa.append("X")
            continue
        expansions = [
            _DNA_CODON_TABLE["".join(bases)]
            for bases in itertools.product(*(_IUPAC_DNA[b] for b in codon))
        ]
        uniq = set(expansions)
        aa.append(next(iter(uniq)) if len(uniq) == 1 else "X")
    return "".join(aa)


def _six_frame_translations(dna_seq: str) -> list[str]:
    """Return all 6 reading-frame amino acid translations (`+1/+2/+3` and `-1/-2/-3`)."""
    rc = reverse_complement_dna(dna_seq)
    frames: list[str] = []
    for strand in (dna_seq, rc):
        for offset in (0, 1, 2):
            translated = _translate_dna_frame(strand, offset)
            if translated:
                frames.append(translated)
    return frames


class DefensiveProteinScreener:
    """Reference defensive protein and 6-frame screening processor."""

    def __init__(
        self,
        target_motifs: Sequence[str] = (SYNTHETIC_REPORTER_MOTIF,),
        circular: bool = False,
        concat_segments: bool = True,
        max_ambiguous_expansions: int = 20,
        max_unknown_fraction: float = 0.20,
    ) -> None:
        self.targets = tuple(
            dict.fromkeys(
                t.strip().upper().replace("U", "C").replace("O", "K")
                for t in target_motifs
                if t and t.strip()
            )
        )
        self.target_lengths = sorted({len(t) for t in self.targets})
        self.targets_by_len: dict[int, set[str]] = {}
        for t in self.targets:
            self.targets_by_len.setdefault(len(t), set()).add(t)
        self.circular = circular
        self.concat_segments = concat_segments
        self.max_ambiguous_expansions = max(1, int(max_ambiguous_expansions))
        self.max_unknown_fraction = float(max_unknown_fraction)

    def _scan_protein_seq(self, seq: str, circular: bool) -> dict[str, Any]:
        if not seq or not self.targets:
            return {"flagged": False, "reason": "clean", "matched_windows": []}

        unknown_count = sum(
            1
            for ch in seq
            if ch == "X" or ch not in (_CANONICAL_AA | set(_AA_AMBIGUITY) | {"*"})
        )
        unknown_fraction = unknown_count / len(seq)
        if unknown_fraction > self.max_unknown_fraction:
            return {
                "flagged": True,
                "reason": "fail_closed_excessive_ambiguity",
                "unknown_fraction": round(unknown_fraction, 4),
                "matched_windows": [],
            }

        matches: list[dict[str, Any]] = []
        max_k = max(self.target_lengths, default=0)
        haystack = seq
        if circular and max_k > 1 and len(seq) >= max_k:
            haystack = seq + seq[: max_k - 1]

        for k in self.target_lengths:
            target_set = self.targets_by_len[k]
            if len(haystack) < k:
                continue
            limit = len(seq) if circular else (len(seq) - k + 1)
            for start in range(max(0, limit)):
                win = haystack[start : start + k]
                if len(win) < k:
                    break
                if "*" in win:
                    continue
                if set(win) <= _CANONICAL_AA:
                    if win in target_set:
                        matches.append({"start": start, "end": start + k, "motif": win})
                    continue

                options: list[tuple[str, ...]] = []
                exp_count = 1
                valid_chars = True
                for ch in win:
                    if ch in _CANONICAL_AA:
                        options.append((ch,))
                    elif ch in _AA_AMBIGUITY:
                        alts = _AA_AMBIGUITY[ch]
                        options.append(alts)
                        exp_count *= len(alts)
                    else:
                        valid_chars = False
                        break

                if not valid_chars:
                    return {
                        "flagged": True,
                        "reason": "fail_closed_excessive_ambiguity",
                        "matched_windows": [],
                    }
                if exp_count > self.max_ambiguous_expansions:
                    return {
                        "flagged": True,
                        "reason": "fail_closed_excessive_ambiguity",
                        "matched_windows": [],
                    }
                for combo in itertools.product(*options):
                    cand = "".join(combo)
                    if cand in target_set:
                        matches.append(
                            {
                                "start": start,
                                "end": start + k,
                                "motif": cand,
                                "raw_window": win,
                            }
                        )
                        break

        if matches:
            return {
                "flagged": True,
                "reason": "motif_match",
                "matched_windows": matches,
            }
        return {"flagged": False, "reason": "clean", "matched_windows": []}

    def _screen_single_body(self, body: str, circular: bool) -> dict[str, Any]:
        norm_seq, is_nuc = _dealias_body(body)
        if is_nuc:
            for frame_idx, aa_seq in enumerate(_six_frame_translations(norm_seq)):
                res = self._scan_protein_seq(aa_seq, circular=False)
                if res["flagged"]:
                    return {
                        **res,
                        "source_mode": "six_frame_nucleotide",
                        "frame_index": frame_idx,
                    }
        res = self._scan_protein_seq(norm_seq, circular=circular)
        return {**res, "source_mode": "amino_acid"}

    def screen(self, raw: str, circular: bool | None = None) -> dict[str, Any]:
        """Screen protein/FASTA/3-letter/PTM or back-translated nucleotide text."""
        use_circular = self.circular if circular is None else circular
        bodies = _parse_fasta_bodies(raw)
        for idx, body in enumerate(bodies):
            res = self._screen_single_body(body, circular=use_circular)
            if res["flagged"]:
                return {**res, "record_index": idx}
        return {
            "flagged": False,
            "reason": "clean",
            "matched_windows": [],
            "records_checked": len(bodies),
        }

    def screen_segments(
        self, segments: Sequence[str], circular: bool | None = None
    ) -> dict[str, Any]:
        """Screen split fusion or pro-peptide fragments individually and across junctions."""
        use_circular = self.circular if circular is None else circular
        norm_segments: list[str] = []
        for seg in segments:
            for body in _parse_fasta_bodies(seg):
                res = self._screen_single_body(body, circular=False)
                if res["flagged"]:
                    return res
                norm_seq, is_nuc = _dealias_body(body)
                if not is_nuc and norm_seq:
                    norm_segments.append(norm_seq)

        if self.concat_segments and len(norm_segments) >= 2:
            k_overlap = max((k - 1 for k in self.target_lengths), default=0)
            for i in range(len(norm_segments) - 1):
                left = norm_segments[i]
                right = norm_segments[i + 1]
                junction = (left[-k_overlap:] if k_overlap else "") + right[:k_overlap]
                res = self._scan_protein_seq(junction, circular=False)
                if res["flagged"]:
                    return {**res, "source_mode": "split_segment_junction"}
            joined = "".join(norm_segments)
            if use_circular:
                res = self._scan_protein_seq(joined, circular=True)
                if res["flagged"]:
                    return {**res, "source_mode": "circular_fusion"}
        return {"flagged": False, "reason": "clean", "matched_windows": []}


def _extract_flagged(result: bool | Mapping[str, Any]) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("flagged"))
    return bool(result)


def audit_protein_screener(
    screener_fn: Callable[..., bool | Mapping[str, Any]],
    segment_screener_fn: Callable[[Sequence[str]], bool | Mapping[str, Any]]
    | None = None,
    circular_screener_fn: Callable[[str], bool | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit a protein/codon screening callable across 6 strictly benign invariants."""
    three_letter = "Trp-Ser-His-Pro-Gln-Phe-Glu-Lys-Gly-Gly-Ser-Gly"
    # Synonymous back-translation of WSHPQFEKGGSG using varied degenerate codons.
    fwd_synonymous_dna = "AAACCCGGGTGGAGTCATCCACAGTTCGAAAAAGGTGGTAGCGGTAAACCCGGG"
    rc_synonymous_dna = "TTT" + reverse_complement_dna(fwd_synonymous_dna)
    ptm_encoded = "MGHSHHHHHHGSW(SEP)HPQFEKGG(SEP)GSMVSKGEELF(TPO)"

    mid = len(SYNTHETIC_REPORTER_MOTIF) // 2
    rotated_circular = (
        SYNTHETIC_REPORTER_MOTIF[mid:] + "AAATTTGGG" + SYNTHETIC_REPORTER_MOTIF[:mid]
    )
    split_left = "MGHSHHHHHHGS" + SYNTHETIC_REPORTER_MOTIF[:mid]
    split_right = SYNTHETIC_REPORTER_MOTIF[mid:] + "SMVSKGEELFT"

    z_ambig = f"MGHSHHHHHHGS{SYNTHETIC_REPORTER_MOTIF.replace('Q', 'Z', 1)}SMVSKGEELFT"
    excessive_x = "MGHSHHHHHH" + ("X" * 16) + "SMVSKGEELFT"

    benign_pad_left = (
        "AGSGAGSGAG" * 11 + "AGSGAG"
    )  # 116 aa (splits motif across 60-col wrap)
    benign_pad_right = "AGSGAGSGAG" * 13  # 130 aa benign linker scaffold
    chimeric_body = (
        benign_pad_left + SYNTHETIC_REPORTER_MOTIF + benign_pad_right
    )  # 258 aa (<5% target span)
    wrapped_chimera = ">synthetic_benign_scaffold_258aa\n" + "\n".join(
        chimeric_body[i : i + 60] for i in range(0, len(chimeric_body), 60)
    )

    invariants: list[dict[str, Any]] = []

    def _record(name: str, passed: bool, detail: str) -> None:
        invariants.append({"invariant": name, "passed": bool(passed), "detail": detail})

    # 1. 3-letter IUPAC dealiasing
    try:
        hit_3 = _extract_flagged(screener_fn(three_letter))
        neg_1 = _extract_flagged(screener_fn(SYNTHETIC_NEGATIVE_PEPTIDE))
        _record(
            "three_letter_iupac_dealiasing",
            hit_3 and not neg_1,
            f"three_letter={hit_3}, neg={neg_1}",
        )
    except Exception as exc:
        _record(
            "three_letter_iupac_dealiasing",
            False,
            f"error:{type(exc).__name__}",
        )

    # 2. 6-frame synonymous back-translation (forward and reverse-complement DNA/RNA)
    try:
        hit_fwd = _extract_flagged(screener_fn(fwd_synonymous_dna))
        hit_rc = _extract_flagged(screener_fn(rc_synonymous_dna))
        _record(
            "six_frame_synonymous_backtranslation",
            hit_fwd and hit_rc,
            f"fwd_dna={hit_fwd}, rc_dna={hit_rc}",
        )
    except Exception as exc:
        _record(
            "six_frame_synonymous_backtranslation",
            False,
            f"error:{type(exc).__name__}",
        )

    # 3. Modified-residue / PTM token normalization
    try:
        hit_ptm = _extract_flagged(screener_fn(ptm_encoded))
        _record(
            "modified_residue_token_normalization",
            hit_ptm,
            f"ptm_encoded={hit_ptm}",
        )
    except Exception as exc:
        _record(
            "modified_residue_token_normalization",
            False,
            f"error:{type(exc).__name__}",
        )

    # 4. Circular origin wraparound and split fusion boundary continuity
    try:
        circ_fn = circular_screener_fn or screener_fn
        seg_fn = segment_screener_fn or (
            lambda segs: _extract_flagged(screener_fn("".join(segs)))
        )
        hit_circ = _extract_flagged(circ_fn(rotated_circular))
        hit_split = _extract_flagged(seg_fn([split_left, split_right]))
        _record(
            "circular_and_split_fusion_boundary",
            hit_circ and hit_split,
            f"circular={hit_circ}, split_fusion={hit_split}",
        )
    except Exception as exc:
        _record(
            "circular_and_split_fusion_boundary",
            False,
            f"error:{type(exc).__name__}",
        )

    # 5. Ambiguous B/Z/J/X expansion and excessive-X fail-closed guard
    try:
        hit_z = _extract_flagged(screener_fn(z_ambig))
        x_res = screener_fn(excessive_x)
        hit_x_fail_closed = _extract_flagged(x_res)
        if isinstance(x_res, Mapping):
            reason_ok = x_res.get("reason") == "fail_closed_excessive_ambiguity"
        else:
            reason_ok = hit_x_fail_closed
        _record(
            "ambiguous_bzj_expansion_and_x_fail_closed",
            hit_z and hit_x_fail_closed and reason_ok,
            f"z_ambig={hit_z}, excessive_x={hit_x_fail_closed}, reason_ok={reason_ok}",
        )
    except Exception as exc:
        _record(
            "ambiguous_bzj_expansion_and_x_fail_closed",
            False,
            f"error:{type(exc).__name__}",
        )

    # 6. Chimeric scaffold localization (<5% span) & negative control specificity
    try:
        chim_res = screener_fn(wrapped_chimera)
        hit_chim = _extract_flagged(chim_res)
        has_windows = (
            bool(chim_res.get("matched_windows"))
            if isinstance(chim_res, Mapping)
            else hit_chim
        )
        neg_clean = not _extract_flagged(screener_fn(SYNTHETIC_NEGATIVE_PEPTIDE))
        _record(
            "chimeric_scaffold_localization_and_specificity",
            hit_chim and has_windows and neg_clean,
            f"chimeric_hit={hit_chim}, localized={has_windows}, neg_clean={neg_clean}",
        )
    except Exception as exc:
        _record(
            "chimeric_scaffold_localization_and_specificity",
            False,
            f"error:{type(exc).__name__}",
        )

    passed_count = sum(1 for inv in invariants if inv["passed"])
    return {
        "auditor": "protein_screening_invariants",
        "passed": passed_count == len(invariants),
        "passed_count": passed_count,
        "total_invariants": len(invariants),
        "invariants": invariants,
    }


def run_protein_screening_self_check() -> dict[str, Any]:
    """Evaluate DefensiveProteinScreener against a naive 1-letter substring screener."""
    hardened = DefensiveProteinScreener()
    hardened_report = audit_protein_screener(
        screener_fn=hardened.screen,
        segment_screener_fn=hardened.screen_segments,
        circular_screener_fn=lambda s: hardened.screen(s, circular=True),
    )

    def _naive_protein_screener(raw: str) -> bool:
        return SYNTHETIC_REPORTER_MOTIF in raw

    naive_report = audit_protein_screener(_naive_protein_screener)
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
