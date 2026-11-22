"""Pathogen-surveillance read deduplication and index-hopping invariant auditor.

Uses strictly benign synthetic bacteriophage MS2 / ribosomal spike-in reporter
sequences (zero pathogen sequences) to audit whether deduplication and demux QC
pipelines enforce:
1. Paired-end reverse-complement orientation symmetry (`(R1, R2)` vs `(revcomp(R2), revcomp(R1))`),
2. UMI + 5' coordinate disentanglement (retain `Hamming >= 2`, collapse `<= 1` edit),
3. Phred quality probability-space consensus (`p = 10 ** (-Q / 10)`),
4. Cross-library patterned-flowcell index-hopping quarantine,
5. Salt-hashed `(kmer_hash, pos_bin)` minimizer homopolymer resilience under bucket caps, and
6. Fail-closed desynced FASTQ mate-pair / sequence-quality length guards.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# Strictly benign synthetic bacteriophage MS2 / ribosomal spike-in reporter tags (zero pathogen sequences).
SYNTHETIC_MS2_SPIKE_R1_NT = "ATGGCTAGCGATGCTAGCTAGTCGGATCGATCGATCGA"
SYNTHETIC_MS2_SPIKE_R2_NT = "GGAATTCCGGAATTCCGGTTAACCGATCGATGCTAGCA"
SYNTHETIC_HOST_RIBO_NT = "TGCATCGATCGATCGATCGAATTCCGGAATTCCGTTAA"

_DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMPLEMENT)[::-1]


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b)) + 1
    return sum(1 for x, y in zip(a, b) if x != y)


def _phred_to_prob(q: int) -> float:
    return 10.0 ** (-float(q) / 10.0)


def _prob_to_phred(p: float) -> float:
    bounded = max(1e-12, min(1.0, float(p)))
    return -10.0 * math.log10(bounded)


class ReadDedupIntegrityError(ValueError):
    """Raised when FASTQ mate pairs or sequence/quality lengths fail validation."""


class NaiveReadDedupCaller:
    """Deliberately flawed reference implementation capturing all 6 dedup/index bugs."""

    def deduplicate_pairs(
        self,
        reads: Sequence[Mapping[str, Any]],
        max_bucket_size: int = 2,
    ) -> list[dict[str, Any]]:
        # Bug 1: orientation-sensitive key; Bug 2: drops UMI or collapses all at same coord
        seen: set[tuple[str, str, int]] = set()
        buckets: dict[str, int] = {}
        kept: list[dict[str, Any]] = []
        for rec in reads:
            r1 = str(rec["r1"]).upper()
            r2 = str(rec["r2"]).upper()
            coord = int(rec.get("coord", 0))
            # Bug 6: never checks r1_id/r2_id or len(seq) != len(qual)
            # Bug 5: unhashed lexicographical minimizer -> homopolymer collapse
            kmer = min(r1[i : i + 8] for i in range(max(1, len(r1) - 7)))
            buckets[kmer] = buckets.get(kmer, 0) + 1
            if buckets[kmer] > max_bucket_size:
                continue
            key = (r1, r2, coord)
            if key not in seen:
                seen.add(key)
                kept.append(dict(rec))
        return kept

    def summarize_phred_quality(self, phred_scores: Sequence[int]) -> float:
        # Bug 3: raw arithmetic mean in log/Phred space instead of error-probability space
        return sum(float(q) for q in phred_scores) / len(phred_scores)

    def screen_index_hopping(
        self,
        target_reads: Sequence[Mapping[str, Any]],
        colibraries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        _ = colibraries
        # Bug 4: trusts every target read without checking patterned-flowcell index hopping
        return {
            "clean_reads": list(target_reads),
            "quarantined_reads": [],
        }


class DefensiveReadDedupAuditor:
    """Reference read deduplication and index-hopping auditor satisfying all 6 invariants."""

    def _canonical_insert(self, r1: str, r2: str) -> tuple[str, str]:
        fwd = (r1, r2)
        rev = (_revcomp(r2), _revcomp(r1))
        return fwd if fwd <= rev else rev

    def _validate_record(self, rec: Mapping[str, Any]) -> tuple[str, str]:
        r1 = str(rec.get("r1", "")).strip().upper()
        r2 = str(rec.get("r2", "")).strip().upper()
        if not r1 or not r2:
            raise ReadDedupIntegrityError("Empty R1 or R2 read sequence.")

        r1_id = rec.get("r1_id")
        r2_id = rec.get("r2_id")
        if r1_id is not None and r2_id is not None:
            core1 = str(r1_id).split()[0].removesuffix("/1")
            core2 = str(r2_id).split()[0].removesuffix("/2")
            if core1 != core2:
                raise ReadDedupIntegrityError(
                    f"Desynced mate pair IDs: '{r1_id}' vs '{r2_id}'."
                )

        q1 = rec.get("q1")
        q2 = rec.get("q2")
        if q1 is not None and len(q1) != len(r1):
            raise ReadDedupIntegrityError(
                f"R1 sequence/quality length mismatch: {len(r1)} vs {len(q1)}."
            )
        if q2 is not None and len(q2) != len(r2):
            raise ReadDedupIntegrityError(
                f"R2 sequence/quality length mismatch: {len(r2)} vs {len(q2)}."
            )
        return r1, r2

    def _salt_hashed_minimizer(
        self, seq: str, k: int = 8, salt: str = "bioeval-v130", pos_bin: int = 4
    ) -> tuple[int, int]:
        if len(seq) < k:
            digest = hashlib.blake2b(f"{salt}:{seq}".encode(), digest_size=8).digest()
            return (int.from_bytes(digest, "big"), 0)
        best: tuple[int, int] | None = None
        for idx in range(len(seq) - k + 1):
            kmer = seq[idx : idx + k]
            digest = hashlib.blake2b(f"{salt}:{kmer}".encode(), digest_size=8).digest()
            val = int.from_bytes(digest, "big")
            cand = (val, idx // max(1, pos_bin))
            if best is None or cand < best:
                best = cand
        assert best is not None
        return best

    def deduplicate_pairs(
        self,
        reads: Sequence[Mapping[str, Any]],
        max_bucket_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Deduplicate paired reads enforcing mate symmetry, UMI tolerance, salted minimizers, and FASTQ integrity."""
        parsed: list[tuple[Mapping[str, Any], str, str, tuple[str, str], int, str]] = []
        for rec in reads:
            r1, r2 = self._validate_record(rec)
            canon = self._canonical_insert(r1, r2)
            coord = int(rec.get("coord", 0))
            umi = str(rec.get("umi", "")).strip().upper()
            parsed.append((rec, r1, r2, canon, coord, umi))

        # Keep records partitioned by salted (kmer_hash, pos_bin) bucket with bounded comparison cap.
        buckets: dict[
            tuple[int, int], list[tuple[Mapping[str, Any], tuple[str, str], int, str]]
        ] = {}
        kept: list[dict[str, Any]] = []

        for rec, _r1, _r2, canon, coord, umi in parsed:
            mkey = self._salt_hashed_minimizer(canon[0])
            bucket = buckets.setdefault(mkey, [])
            is_dup = False
            for _prev_rec, prev_canon, prev_coord, prev_umi in bucket[:max_bucket_size]:
                if prev_canon == canon and prev_coord == coord:
                    if not umi and not prev_umi:
                        is_dup = True
                        break
                    if umi and prev_umi and _hamming(umi, prev_umi) <= 1:
                        is_dup = True
                        break
            if not is_dup and len(bucket) < max_bucket_size:
                bucket.append((rec, canon, coord, umi))
                kept.append(dict(rec))
            elif not is_dup and not any(
                c == canon and co == coord and _hamming(umi, u) <= 1
                for _, c, co, u in bucket
            ):
                # Distinct molecule in an active non-homopolymer bucket is never dropped silently.
                kept.append(dict(rec))
        return kept

    def summarize_phred_quality(self, phred_scores: Sequence[int]) -> float:
        """Compute mean base-error probability in linear probability space and map back to Phred."""
        if not phred_scores:
            raise ReadDedupIntegrityError("Phred quality sequence must be non-empty.")
        mean_err = sum(_phred_to_prob(q) for q in phred_scores) / len(phred_scores)
        return _prob_to_phred(mean_err)

    def screen_index_hopping(
        self,
        target_reads: Sequence[Mapping[str, Any]],
        colibraries: Sequence[Mapping[str, Any]],
        hop_ratio_threshold: float = 100.0,
    ) -> dict[str, Any]:
        """Quarantine low-count target reads matching high-titer co-library inserts sharing i7 or i5."""
        # Index colibraries by (canonical_insert, umi) -> list[(i7, i5, count)]
        co_index: dict[tuple[tuple[str, str], str], list[tuple[str, str, int]]] = {}
        for col in colibraries:
            r1, r2 = self._validate_record(col)
            canon = self._canonical_insert(r1, r2)
            umi = str(col.get("umi", "")).strip().upper()
            i7 = str(col.get("i7", "")).strip().upper()
            i5 = str(col.get("i5", "")).strip().upper()
            cnt = int(col.get("count", 1))
            co_index.setdefault((canon, umi), []).append((i7, i5, cnt))

        clean: list[dict[str, Any]] = []
        quarantined: list[dict[str, Any]] = []
        for rec in target_reads:
            r1, r2 = self._validate_record(rec)
            canon = self._canonical_insert(r1, r2)
            umi = str(rec.get("umi", "")).strip().upper()
            i7 = str(rec.get("i7", "")).strip().upper()
            i5 = str(rec.get("i5", "")).strip().upper()
            t_cnt = max(1, int(rec.get("count", 1)))

            hopped = False
            for co_i7, co_i5, co_cnt in co_index.get((canon, umi), []):
                shares_one_index = (bool(i7) and i7 == co_i7) or (
                    bool(i5) and i5 == co_i5
                )
                dual_unique = (i7 != co_i7) and (i5 != co_i5)
                if (
                    shares_one_index
                    and not dual_unique
                    and (co_cnt / t_cnt) >= hop_ratio_threshold
                ):
                    hopped = True
                    break
            item = dict(rec)
            if hopped:
                item["quarantine_reason"] = "index_hopping_single_shared_index"
                quarantined.append(item)
            else:
                clean.append(item)
        return {
            "clean_reads": clean,
            "quarantined_reads": quarantined,
        }


def audit_read_dedup_hopping_pipeline(auditor: Any) -> dict[str, Any]:
    """Run all 6 deterministic read-dedup & index-hopping invariants on benign controls."""
    results: list[dict[str, Any]] = []

    # 1. Paired-end reverse-complement orientation symmetry
    pair_fwd = {
        "r1_id": "READ_001/1",
        "r2_id": "READ_001/2",
        "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
        "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
        "coord": 100,
        "umi": "ACGTACGT",
    }
    pair_rev = {
        "r1_id": "READ_002/1",
        "r2_id": "READ_002/2",
        "r1": _revcomp(SYNTHETIC_MS2_SPIKE_R2_NT),
        "r2": _revcomp(SYNTHETIC_MS2_SPIKE_R1_NT),
        "coord": 100,
        "umi": "ACGTACGT",
    }
    try:
        dedup_orient = auditor.deduplicate_pairs([pair_fwd, pair_rev])
        inv1 = len(dedup_orient) == 1
        d1 = f"kept={len(dedup_orient)} (expected 1 canonical molecule)"
    except Exception as exc:
        inv1, d1 = False, f"error: {exc}"
    results.append(
        {
            "invariant": "paired_revcomp_orientation_symmetry",
            "passed": inv1,
            "detail": d1,
        }
    )

    # 2. UMI + coordinate disentanglement (Hamming >= 2 kept; Hamming <= 1 collapsed)
    u_ref = {
        "r1_id": "UMI_01/1",
        "r2_id": "UMI_01/2",
        "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
        "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
        "coord": 250,
        "umi": "ACGTACGT",
    }
    u_1err = {
        "r1_id": "UMI_02/1",
        "r2_id": "UMI_02/2",
        "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
        "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
        "coord": 250,
        "umi": "TCGTACGT",  # Hamming 1 -> PCR error, collapse
    }
    u_2diff = {
        "r1_id": "UMI_03/1",
        "r2_id": "UMI_03/2",
        "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
        "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
        "coord": 250,
        "umi": "TTGATCGT",  # Hamming >= 2 -> distinct molecule, retain
    }
    try:
        dedup_umi = auditor.deduplicate_pairs([u_ref, u_1err, u_2diff])
        umis = {r.get("umi") for r in dedup_umi}
        inv2 = len(dedup_umi) == 2 and "ACGTACGT" in umis and "TTGATCGT" in umis
        d2 = f"kept={len(dedup_umi)} umis={sorted(str(u) for u in umis)}"
    except Exception as exc:
        inv2, d2 = False, f"error: {exc}"
    results.append(
        {
            "invariant": "umi_coordinate_disentanglement",
            "passed": inv2,
            "detail": d2,
        }
    )

    # 3. Phred probability-space consensus (Phred 10 + Phred 40 -> mean err=0.05005 -> Q~13.01, NOT 25.0)
    try:
        q_summary = float(auditor.summarize_phred_quality([10, 40]))
        expected_q = _prob_to_phred((_phred_to_prob(10) + _phred_to_prob(40)) / 2.0)
        inv3 = abs(q_summary - expected_q) < 0.05 and q_summary < 15.0
        d3 = f"q_summary={q_summary:.2f} (expected {expected_q:.2f}, naive=25.00)"
    except Exception as exc:
        inv3, d3 = False, f"error: {exc}"
    results.append(
        {
            "invariant": "phred_probability_space_consensus",
            "passed": inv3,
            "detail": d3,
        }
    )

    # 4. Cross-library patterned-flowcell index-hopping quarantine
    hop_candidate = {
        "r1_id": "HOP_1/1",
        "r2_id": "HOP_1/2",
        "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
        "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
        "umi": "GGCCAATT",
        "i7": "ATTACTCG",  # shared with high-titer colibrary!
        "i5": "TATAGCCT",  # different i5 -> single-index hop
        "count": 1,
    }
    dual_unique_true = {
        "r1_id": "TRUE_2/1",
        "r2_id": "TRUE_2/2",
        "r1": SYNTHETIC_HOST_RIBO_NT,
        "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
        "umi": "CCGGTTAA",
        "i7": "TCCGGAGA",  # unique i7
        "i5": "TATAGCCT",  # unique i5 vs colibrary below
        "count": 3,
    }
    colib = {
        "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
        "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
        "umi": "GGCCAATT",
        "i7": "ATTACTCG",
        "i5": "GGCTCTGA",
        "count": 10000,
    }
    try:
        hop_res = auditor.screen_index_hopping(
            [hop_candidate, dual_unique_true], [colib]
        )
        clean_ids = {r.get("r1_id") for r in hop_res.get("clean_reads", [])}
        quar_ids = {r.get("r1_id") for r in hop_res.get("quarantined_reads", [])}
        inv4 = clean_ids == {"TRUE_2/1"} and quar_ids == {"HOP_1/1"}
        d4 = f"clean={sorted(clean_ids)} quarantined={sorted(quar_ids)}"
    except Exception as exc:
        inv4, d4 = False, f"error: {exc}"
    results.append(
        {
            "invariant": "cross_library_index_hopping_quarantine",
            "passed": inv4,
            "detail": d4,
        }
    )

    # 5. Salt-hashed minimizer homopolymer resilience under bounded bucket caps
    poly_reads = [
        {
            "r1_id": f"POLY_{i}/1",
            "r2_id": f"POLY_{i}/2",
            "r1": "AAAAAAAAAAAA" + body,
            "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
            "coord": 400 + i * 20,
            "umi": f"AA{i}CGTAC",
        }
        for i, body in enumerate(
            [
                "TCGAACTGGCTAGCATGCTAGCTA",
                "GGTTAACCGGATTCCGATCGGCTA",
                "CCGGTTAAGGCCAATTGGCCATAG",
                "GCTAGCTAGTCAGTCAGTCATGCA",
            ]
        )
    ]
    try:
        kept_poly = auditor.deduplicate_pairs(poly_reads, max_bucket_size=2)
        inv5 = len(kept_poly) == 4
        d5 = f"kept={len(kept_poly)}/4 distinct poly-A prefixed templates"
    except Exception as exc:
        inv5, d5 = False, f"error: {exc}"
    results.append(
        {
            "invariant": "salt_hashed_minimizer_homopolymer_resilience",
            "passed": inv5,
            "detail": d5,
        }
    )

    # 6. Fail-closed desynced FASTQ mate-pair / length mismatch guard
    desynced = [
        {
            "r1_id": "MATE_A/1",
            "r2_id": "MATE_B/2",
            "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
            "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
            "q1": "I" * (len(SYNTHETIC_MS2_SPIKE_R1_NT) - 3),
            "coord": 500,
        }
    ]
    inv6 = False
    d6 = "desynced mate pair accepted silently"
    try:
        auditor.deduplicate_pairs(desynced)
    except ReadDedupIntegrityError as exc:
        inv6 = True
        d6 = f"rejected desynced mate pair: {exc}"
    except Exception as exc:
        inv6 = True
        d6 = f"rejected with {type(exc).__name__}: {exc}"
    results.append(
        {
            "invariant": "fail_closed_desynced_fastq_pair_guard",
            "passed": inv6,
            "detail": d6,
        }
    )

    return {
        "invariants": results,
        "all_passed": all(r["passed"] for r in results),
        "passed_count": sum(1 for r in results if r["passed"]),
        "total_invariants": len(results),
    }


def run_read_dedup_self_check(
    naive_auditor_fn: Callable[[], Any] = NaiveReadDedupCaller,
) -> dict[str, Any]:
    """Verify DefensiveReadDedupAuditor passes 6/6 and NaiveReadDedupCaller fails all 6."""
    hardened = audit_read_dedup_hopping_pipeline(DefensiveReadDedupAuditor())
    naive = audit_read_dedup_hopping_pipeline(naive_auditor_fn())
    naive_caught = [r["invariant"] for r in naive["invariants"] if not r["passed"]]
    return {
        "hardened": hardened,
        "hardened_passed": hardened["all_passed"],
        "naive_caught_invariants": naive_caught,
        "naive_total_invariants": naive["total_invariants"],
    }
