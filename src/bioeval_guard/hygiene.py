"""Deterministic hygiene, sequence-biology, and schema checks for AIxBio evaluations."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

META_OPTION_RE = re.compile(
    r"\b(all of the above|none of the above|both [a-d] and [a-d]|neither)\b",
    re.IGNORECASE,
)
SCI_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$")
DEC_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
RANGE_RE = re.compile(r"^\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)$")

RESTRICTION_SITES: dict[str, str] = {
    "BamHI": "GGATCC",
    "EcoRI": "GAATTC",
    "HindIII": "AAGCTT",
    "KpnI": "GGTACC",
    "PstI": "CTGCAG",
    "SacI": "GAGCTC",
    "SstI": "GAGCTC",
    "SalI": "GTCGAC",
    "SmaI": "CCCGGG",
    "XmaI": "CCCGGG",
    "SphI": "GCATGC",
    "XbaI": "TCTAGA",
    "NotI": "GCGGCCGC",
    "XhoI": "CTCGAG",
    "NcoI": "CCATGG",
    "AvrII": "CCTAGG",
    "AflII": "CTTAAG",
    "BstBI": "TTCGAA",
    "AatII": "GACGTC",
    "PvuII": "CAGCTG",
    "ApaI": "GGGCCC",
    "BsiWI": "CGTACG",
    "BglII": "AGATCT",
}

CANONICAL_ISOSCHIZOMERS: dict[str, str] = {
    "SstI": "SacI",
    "XmaI": "SmaI",
}

_DNA_TRANS = str.maketrans("ACGTacgt", "TGCAtgca")


@dataclass
class AuditItem:
    """Normalized multiple-choice evaluation record."""

    item_id: str
    question: str
    options: list[str]
    ideal_idx: int
    subtask: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ideal(self) -> str:
        return self.options[self.ideal_idx]

    @property
    def distractors(self) -> list[str]:
        return [opt for idx, opt in enumerate(self.options) if idx != self.ideal_idx]


def complement_dna(seq: str) -> str:
    """Return 5'->3' strand-inverted base complement (without reversing)."""
    return seq.translate(_DNA_TRANS)


def reverse_complement_dna(seq: str) -> str:
    """Return true biological 5'->3' reverse complement of a DNA sequence."""
    return complement_dna(seq)[::-1]


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a Parquet, JSONL, or JSON file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pq.read_table(p).to_pylist()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Expected list JSON in {p}")
    raise ValueError(f"Unsupported dataset extension: {p.suffix}")


def normalize_items(
    records: list[dict[str, Any]], dataset_name: str = "dataset"
) -> list[AuditItem]:
    """Normalize LAB-Bench, BixBench, and WMDP record schemas into AuditItem objects."""
    items: list[AuditItem] = []
    for idx, rec in enumerate(records):
        if "ideal" in rec and "distractors" in rec:
            ideal = str(rec["ideal"])
            dists = [str(d) for d in rec["distractors"]]
            opts = [ideal] + dists
            ideal_idx = 0
            item_id = str(
                rec.get("question_id") or rec.get("id") or f"{dataset_name}-{idx}"
            )
            subtask = str(rec.get("subtask") or rec.get("eval_mode") or dataset_name)
        elif "choices" in rec and "answer" in rec:
            opts = [str(c) for c in rec["choices"]]
            ans = rec["answer"]
            if isinstance(ans, str) and ans.strip().upper() in {
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
            }:
                ideal_idx = ord(ans.strip().upper()) - ord("A")
            else:
                ideal_idx = int(ans)
            item_id = str(rec.get("id") or f"{dataset_name}-{idx}")
            subtask = str(rec.get("subtask") or dataset_name)
        else:
            raise ValueError(
                f"Unrecognized MCQ schema keys at row {idx}: {sorted(rec.keys())}"
            )

        meta = {
            k: v
            for k, v in rec.items()
            if k not in {"question", "ideal", "distractors", "choices", "answer"}
        }
        items.append(
            AuditItem(
                item_id=item_id,
                question=str(rec.get("question", "")),
                options=opts,
                ideal_idx=ideal_idx,
                subtask=subtask,
                metadata=meta,
            )
        )
    return items


def check_primer_reverse_complement(
    template_dna: str, reverse_primer: str, anchor_len: int = 15
) -> bool:
    """Return True if reverse_primer anneals to coding template (matches reverse complement, not sense strand).

    Returns False when the primer copies the raw 5'->3' sense tail verbatim (PCR-incompatible).
    """
    dna = re.sub(r"[^ACGTacgt]", "", template_dna).upper()
    primer = re.sub(r"[^ACGTacgt]", "", reverse_primer).upper()
    if len(dna) < anchor_len or len(primer) < anchor_len:
        return True
    sense_tail = dna[-anchor_len:]
    rc_tail = reverse_complement_dna(dna)[:anchor_len]
    return not (sense_tail in primer and rc_tail not in primer)


def check_restriction_site_complement_bug(enzyme_name: str, primer_seq: str) -> bool:
    """Return True if a primer's 5' overhang contains comp(site) instead of the true recognition site."""
    site = RESTRICTION_SITES.get(enzyme_name)
    if not site:
        return False
    mutated = complement_dna(site)
    seq = primer_seq.upper()
    return not seq.startswith(site) and mutated in seq[: max(12, len(site) + 4)]


def _canonicalize_enzyme_tuple(option: str) -> tuple[str, ...]:
    parts = [p.strip() for p in option.split(",") if p.strip()]
    return tuple(sorted(CANONICAL_ISOSCHIZOMERS.get(p, p) for p in parts))


def _is_comma_split_shard(ideal: str, distractors: list[str]) -> bool:
    if len(distractors) <= 3:
        return False
    if "," in ideal and any(re.fullmatch(r"\d{1,3}", d.strip()) for d in distractors):
        return True
    return any(len(d.split()) <= 2 for d in distractors) and any(
        d.strip().lower().startswith("and ") for d in distractors
    )


def check_mcq_hygiene(
    items: list[AuditItem], check_biology: bool = True
) -> dict[str, Any]:
    """Run deterministic key-collision, null-value, duplicate, comma-split, and biology checks."""
    ideal_in_dist_exact: list[str] = []
    ideal_in_dist_norm: list[str] = []
    duplicate_option_items: list[str] = []
    literal_null_items: list[str] = []
    empty_option_items: list[str] = []
    comma_split_items: list[str] = []
    meta_ideal_only_items: list[str] = []
    sense_reverse_primer_bugs: list[str] = []
    comp_restriction_bugs: list[str] = []
    isoschizomer_collisions: list[str] = []
    question_counts = Counter(
        it.question.strip().lower() for it in items if it.question.strip()
    )

    for it in items:
        ideal = it.ideal
        dists = it.distractors
        if ideal in dists:
            ideal_in_dist_exact.append(it.item_id)
        if ideal.strip().lower() in [d.strip().lower() for d in dists]:
            ideal_in_dist_norm.append(it.item_id)

        norm_opts = [o.strip().lower() for o in it.options]
        if len(set(norm_opts)) < len(norm_opts):
            duplicate_option_items.append(it.item_id)

        if any(o is None or o.strip() == "" for o in it.options):
            empty_option_items.append(it.item_id)
        if ideal.strip().lower() in {"null", "nan"}:
            literal_null_items.append(it.item_id)

        if _is_comma_split_shard(ideal, dists):
            comma_split_items.append(it.item_id)

        has_meta_i = bool(META_OPTION_RE.search(ideal))
        has_meta_d = any(bool(META_OPTION_RE.search(d)) for d in dists)
        if has_meta_i and not has_meta_d:
            meta_ideal_only_items.append(it.item_id)

        if check_biology:
            m_dna = re.search(r"([ACGT]{30,})", it.question)
            if m_dna and ("PCR" in it.subtask or "primer" in it.question.lower()):
                dna = m_dna.group(1)
                txt = it.question if "geneprimers-enz" in it.subtask else ideal
                seqs = re.findall(r"[ACGT]{18,}", txt)
                if seqs and not check_primer_reverse_complement(dna, seqs[-1]):
                    sense_reverse_primer_bugs.append(it.item_id)
                if "enzprimers" in it.subtask and len(seqs) >= 2:
                    enzymes = re.findall(
                        r"\b(" + "|".join(RESTRICTION_SITES.keys()) + r")\b",
                        it.question,
                    )
                    if len(enzymes) >= 2 and check_restriction_site_complement_bug(
                        enzymes[1], seqs[-1]
                    ):
                        comp_restriction_bugs.append(it.item_id)
            if "geneprimers-enz" in it.subtask:
                canon = [_canonicalize_enzyme_tuple(o) for o in it.options]
                if len(set(canon)) < len(canon):
                    isoschizomer_collisions.append(it.item_id)

    flagged_ids = set(
        ideal_in_dist_norm
        + duplicate_option_items
        + literal_null_items
        + empty_option_items
        + comma_split_items
        + sense_reverse_primer_bugs
        + comp_restriction_bugs
        + isoschizomer_collisions
    )

    return {
        "total_items": len(items),
        "flagged_items": len(flagged_ids),
        "flagged_ids": sorted(flagged_ids),
        "ideal_in_distractors_exact": len(ideal_in_dist_exact),
        "ideal_in_distractors_norm": len(ideal_in_dist_norm),
        "duplicate_option_items": len(duplicate_option_items),
        "literal_null_ideal_items": len(literal_null_items),
        "literal_null_ideal_ids": literal_null_items,
        "empty_option_items": len(empty_option_items),
        "comma_split_distractor_items": len(comma_split_items),
        "comma_split_ids": comma_split_items,
        "duplicate_question_texts": sum(
            c - 1 for c in question_counts.values() if c > 1
        ),
        "meta_phrase_ideal_only_items": len(meta_ideal_only_items),
        "sense_reverse_primer_bugs": len(sense_reverse_primer_bugs),
        "comp_restriction_bugs": len(comp_restriction_bugs),
        "isoschizomer_collisions": len(isoschizomer_collisions),
    }


def check_bixbench_hygiene(items: list[AuditItem]) -> dict[str, Any]:
    """Audit BixBench eval_mode consistency, scientific-notation mismatches, and range overlaps."""
    notation_mismatches: list[str] = []
    overlapping_ranges: list[str] = []
    eval_modes = Counter(str(it.metadata.get("eval_mode", "unknown")) for it in items)

    for it in items:
        mode = str(it.metadata.get("eval_mode", ""))
        ideal = it.ideal.strip()
        dists = [d.strip() for d in it.distractors]
        if mode == "str_verifier":
            i_sci = bool(SCI_NUM_RE.match(ideal))
            d_sci = [bool(SCI_NUM_RE.match(d)) for d in dists]
            i_dec = bool(DEC_NUM_RE.match(ideal))
            d_dec = [bool(DEC_NUM_RE.match(d)) for d in dists]
            if (i_dec and any(d_sci)) or (i_sci and any(d_dec)):
                notation_mismatches.append(it.item_id)
        elif mode == "range_verifier":
            parsed: list[tuple[float, float]] = []
            for opt in [ideal] + dists:
                m = RANGE_RE.match(opt)
                if m:
                    parsed.append((float(m.group(1)), float(m.group(2))))
            if len(parsed) == len(it.options) and len(parsed) >= 2:
                i_lo, i_hi = parsed[0]
                for d_lo, d_hi in parsed[1:]:
                    if max(i_lo, d_lo) <= min(i_hi, d_hi):
                        overlapping_ranges.append(it.item_id)
                        break

    return {
        "eval_mode_counts": dict(eval_modes),
        "str_verifier_notation_mismatches": len(notation_mismatches),
        "str_verifier_notation_ids": notation_mismatches,
        "range_verifier_overlapping_intervals": len(overlapping_ranges),
        "range_verifier_overlap_ids": overlapping_ranges,
    }
