"""Tiled-amplicon and wastewater pathogen surveillance invariant auditor.

Uses strictly benign synthetic GFP-derived reporter templates (zero pathogen
sequences) to verify whether amplicon and wastewater surveillance pipelines
enforce:
1. 5' and 3' primer-binding-site clipping before variant calling and coverage accounting,
2. Short-insert / deletion 3' primer read-through trimming symmetrically on both strands,
3. Duplicate reference handling parity (`augur#1892`: fail-closed helpful error unless
   explicitly deduplicated when `--remove-reference` is set, and always rejecting
   conflicting duplicate sequences),
4. Low-depth site masking (`N` / `NaN`) without silent coercion to `0.0` wild-type frequency,
5. Non-negative simplex wastewater lineage deconvolution with explicit unexplained
   cryptic residual reporting, and
6. Negative-template control (NTC) contamination fail-closed gating.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# Strictly benign synthetic GFP-derived housekeeping reporter sequence (72 nt; zero pathogen sequences).
SYNTHETIC_AMPLICON_REF_NT = (
    "ATGGCTAGCGATGCTAGCTAGTCGGATCGATCGATCGATCAATTGGCCGGTTAACCGGAATTCCGGAATTCC"
)
SYNTHETIC_FWD_PRIMER_NT = SYNTHETIC_AMPLICON_REF_NT[:18]
_DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMPLEMENT)[::-1]


SYNTHETIC_REV_PRIMER_NT = _revcomp(SYNTHETIC_AMPLICON_REF_NT[-18:])


class AmpliconSurveillanceError(ValueError):
    """Raised when amplicon inputs, duplicate references, or NTC gates fail closed."""


class DefensiveAmpliconSurveillanceAuditor:
    """Reference tiled-amplicon & wastewater auditor satisfying all 6 invariants."""

    def call_amplicon_variants(
        self,
        reads: Sequence[Mapping[str, Any]],
        ref_seq: str = SYNTHETIC_AMPLICON_REF_NT,
        primers: Sequence[Mapping[str, Any]] = (),
        min_depth: int = 10,
        min_af: float = 0.5,
    ) -> dict[str, Any]:
        """Clip primers and trim 3' read-through before calling site allele frequencies."""
        ref = ref_seq.strip().upper()
        n = len(ref)
        if n == 0:
            raise AmpliconSurveillanceError("Reference sequence must be non-empty.")

        counts: list[dict[str, int]] = [
            {"A": 0, "C": 0, "G": 0, "T": 0} for _ in range(n)
        ]

        # Build reverse-primer RC tokens for 3' short-insert read-through trimming.
        rev_primer_rcs: list[str] = []
        for p in primers:
            r_start = int(p.get("rev_start", n))
            r_end = int(p.get("rev_end", n))
            if r_end > r_start:
                r_seq = str(p.get("rev_seq", _revcomp(ref[r_start:r_end]))).upper()
                rev_primer_rcs.append(_revcomp(r_seq))

        for read in reads:
            seq = str(read.get("seq", "")).strip().upper()
            start = int(read.get("start", 0))
            strand = str(read.get("strand", "+"))
            if not seq:
                continue

            # Invariant 2: trim 3' reverse-primer read-through on short/deletion inserts.
            # On '-' strand reads provided in read orientation, reverse-complement to ref orientation first.
            if strand == "-":
                seq_ref = _revcomp(seq)
            else:
                seq_ref = seq

            for rc_tok in rev_primer_rcs:
                if rc_tok and rc_tok in seq_ref:
                    idx = seq_ref.rfind(rc_tok)
                    if idx >= max(1, len(seq_ref) - len(rc_tok) - 6):
                        seq_ref = seq_ref[:idx]

            # Invariant 1: mask/clip any bases falling inside primer-binding regions of THIS amplicon
            amplicon_id = read.get("amplicon_id")
            active_primers = (
                [p for p in primers if p.get("amplicon_id") == amplicon_id]
                if amplicon_id is not None
                else list(primers)
            )
            masked_positions: set[int] = set()
            for p in active_primers:
                fs = int(p.get("fwd_start", 0))
                fe = int(p.get("fwd_end", 0))
                rs = int(p.get("rev_start", n))
                re_ = int(p.get("rev_end", n))
                masked_positions.update(range(fs, fe))
                masked_positions.update(range(rs, re_))

            for offset, base in enumerate(seq_ref):
                pos = start + offset
                if pos < 0 or pos >= n:
                    continue
                if pos in masked_positions:
                    continue
                if base in counts[pos]:
                    counts[pos][base] += 1

        consensus_chars: list[str] = []
        alt_frequencies: dict[int, float] = {}
        depths: list[int] = []
        masked_low_depth_sites: list[int] = []

        for pos in range(n):
            site_counts = counts[pos]
            depth = sum(site_counts.values())
            depths.append(depth)
            # Invariant 4: low-depth site masking (never coerce depth < min_depth to 0.0 wild-type)
            if depth < min_depth:
                consensus_chars.append("N")
                alt_frequencies[pos] = float("nan")
                masked_low_depth_sites.append(pos)
                continue
            ref_base = ref[pos]
            top_base = max("ACGT", key=lambda b: (site_counts[b], b == ref_base))
            top_af = site_counts[top_base] / depth
            non_ref_af = (depth - site_counts.get(ref_base, 0)) / depth
            alt_frequencies[pos] = non_ref_af
            if top_base != ref_base and top_af >= min_af:
                consensus_chars.append(top_base)
            elif non_ref_af >= min_af:
                # Ambiguous / mixed alt calls without single majority non-ref base
                consensus_chars.append(top_base if top_af >= min_af else "N")
            else:
                consensus_chars.append(ref_base)

        return {
            "consensus": "".join(consensus_chars),
            "depths": depths,
            "alt_frequencies": alt_frequencies,
            "masked_low_depth_sites": masked_low_depth_sites,
        }

    def prepare_reference_alignment_inputs(
        self,
        sample_sequences: Sequence[tuple[str, str]],
        ref_id: str,
        ref_seq: str,
        remove_reference: bool = False,
        reference_name: str | None = None,
    ) -> list[tuple[str, str]]:
        """Deduplicate identical reference only under remove_reference=True; else fail closed."""
        ref_norm = ref_seq.strip().upper()
        seen: dict[str, str] = {}
        for sid, sseq in sample_sequences:
            norm = sseq.strip().upper()
            if sid in seen and seen[sid] != norm:
                raise AmpliconSurveillanceError(
                    f"Detected duplicate input strain {sid!r} with conflicting sequences."
                )
            seen[sid] = norm

        if reference_name is not None:
            if reference_name not in seen:
                raise AmpliconSurveillanceError(
                    f"Specified --reference-name {reference_name!r} not present in sample sequences."
                )
            items = list(seen.items())
            return (
                [(sid, s) for sid, s in items if sid != reference_name]
                if remove_reference
                else items
            )

        if ref_id in seen:
            if seen[ref_id] != ref_norm:
                raise AmpliconSurveillanceError(
                    f"Duplicate reference sequence {ref_id!r} differs from --reference-sequence."
                )
            if not remove_reference:
                raise AmpliconSurveillanceError(
                    f"Duplicate reference sequence {ref_id!r} detected in both --sequences and "
                    "--reference-sequence. Use --reference-name or pass --remove-reference."
                )
            # remove_reference=True: exclude duplicate from input samples and return post-stripped output
            return [(sid, s) for sid, s in seen.items() if sid != ref_id]

        items = [(ref_id, ref_norm), *seen.items()]
        if remove_reference:
            return [(sid, s) for sid, s in items if sid != ref_id]
        return items

    def deconvolve_wastewater(
        self,
        site_afs: Mapping[int, float],
        lineage_signatures: Mapping[str, set[int]],
        min_cryptic_af: float = 0.15,
    ) -> dict[str, Any]:
        """Simplex non-negative deconvolution over non-NaN sites with cryptic residual check."""
        lineages = sorted(lineage_signatures.keys())
        observed = {
            pos: af
            for pos, af in site_afs.items()
            if isinstance(af, (int, float)) and not math.isnan(af)
        }
        signature_union = (
            set().union(*lineage_signatures.values()) if lineages else set()
        )

        # Find unique marker sites per lineage (1-to-1 clean deconvolution when available)
        abundances: dict[str, float] = {}
        for lin in lineages:
            others = set().union(*(lineage_signatures[o] for o in lineages if o != lin))
            unique_sites = sorted(lineage_signatures[lin] - others)
            candidate_sites = [s for s in unique_sites if s in observed]
            if not candidate_sites:
                candidate_sites = [
                    s for s in sorted(lineage_signatures[lin]) if s in observed
                ]
            if not candidate_sites:
                abundances[lin] = 0.0
            else:
                mean_af = sum(observed[s] for s in candidate_sites) / len(
                    candidate_sites
                )
                abundances[lin] = max(0.0, min(1.0, mean_af))

        # Simplex projection if modeled lineages exceed 1.0 slightly due to noise
        total_modeled = sum(abundances.values())
        if total_modeled > 1.0 and total_modeled > 0.0:
            for lin in lineages:
                abundances[lin] /= total_modeled
            total_modeled = 1.0

        # Cryptic residual: unexplained high-AF variants outside known signatures
        # plus uncovered simplex mass when unexplained signals exist.
        unmodeled_sites = sorted(
            pos
            for pos, af in observed.items()
            if pos not in signature_union and af >= min_cryptic_af
        )
        unexplained_mass = max(0.0, 1.0 - total_modeled)
        cryptic_fraction = (
            max((observed[p] for p in unmodeled_sites), default=0.0)
            if unmodeled_sites
            else 0.0
        )
        if unmodeled_sites and unexplained_mass > 0.0:
            cryptic_fraction = max(
                cryptic_fraction,
                min(unexplained_mass, max(observed[p] for p in unmodeled_sites)),
            )

        return {
            "abundances": abundances,
            "modeled_sum": round(total_modeled, 6),
            "cryptic_residual": round(cryptic_fraction, 6),
            "unmodeled_high_af_sites": unmodeled_sites,
            "informative_sites_used": len(observed),
        }

    def validate_ntc_batch(
        self,
        ntc_reads: Sequence[Mapping[str, Any]],
        sample_reads: Sequence[Mapping[str, Any]],
        ref_len: int = len(SYNTHETIC_AMPLICON_REF_NT),
        max_ntc_reads: int = 5,
        max_ntc_coverage_frac: float = 0.15,
    ) -> dict[str, Any]:
        """Fail closed when negative-template control (NTC) indicates cross-contamination."""
        covered: set[int] = set()
        for r in ntc_reads:
            st = int(r.get("start", 0))
            ln = len(str(r.get("seq", "")))
            covered.update(p for p in range(st, st + ln) if 0 <= p < ref_len)
        cov_frac = (len(covered) / ref_len) if ref_len > 0 else 0.0
        if len(ntc_reads) > max_ntc_reads or cov_frac > max_ntc_coverage_frac:
            raise AmpliconSurveillanceError(
                f"CONTAMINATED_FAIL_CLOSED: NTC detected {len(ntc_reads)} reads "
                f"({cov_frac:.2%} genome coverage)."
            )
        return {
            "status": "PASS",
            "ntc_reads": len(ntc_reads),
            "ntc_coverage_frac": cov_frac,
            "sample_reads": len(sample_reads),
        }

    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Any:
        if stage == "call_variants":
            return self.call_amplicon_variants(
                reads=payload["reads"],
                ref_seq=payload.get("ref_seq", SYNTHETIC_AMPLICON_REF_NT),
                primers=payload.get("primers", ()),
                min_depth=int(payload.get("min_depth", 10)),
                min_af=float(payload.get("min_af", 0.5)),
            )
        if stage == "prepare_reference":
            return self.prepare_reference_alignment_inputs(
                sample_sequences=payload["sample_sequences"],
                ref_id=str(payload["ref_id"]),
                ref_seq=str(payload["ref_seq"]),
                remove_reference=bool(payload.get("remove_reference", False)),
                reference_name=payload.get("reference_name"),
            )
        if stage == "deconvolve_wastewater":
            return self.deconvolve_wastewater(
                site_afs=payload["site_afs"],
                lineage_signatures=payload["lineage_signatures"],
                min_cryptic_af=float(payload.get("min_cryptic_af", 0.15)),
            )
        if stage == "validate_ntc":
            return self.validate_ntc_batch(
                ntc_reads=payload["ntc_reads"],
                sample_reads=payload.get("sample_reads", ()),
                ref_len=int(payload.get("ref_len", len(SYNTHETIC_AMPLICON_REF_NT))),
            )
        raise ValueError(f"Unknown stage: {stage!r}")


class NaiveAmpliconConsensusCaller:
    """Reference naive amplicon pipeline exhibiting all 6 surveillance defects."""

    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Any:
        ref = str(payload.get("ref_seq", SYNTHETIC_AMPLICON_REF_NT)).strip().upper()
        n = len(ref)

        if stage == "call_variants":
            # Defects 1, 2, 4:
            # - does NOT clip primer coordinates (primer reads dilute interior mutations)
            # - does NOT trim 3' reverse-primer read-through on '-' or '+' strands
            # - coerces low-depth (< min_depth) sites to ref base / 0.0 AF instead of N / NaN
            counts = [{"A": 0, "C": 0, "G": 0, "T": 0} for _ in range(n)]
            for read in payload["reads"]:
                seq = str(read.get("seq", "")).strip().upper()
                start = int(read.get("start", 0))
                if str(read.get("strand", "+")) == "-":
                    seq = _revcomp(seq)
                for off, b in enumerate(seq):
                    pos = start + off
                    if 0 <= pos < n and b in counts[pos]:
                        counts[pos][b] += 1
            chars: list[str] = []
            afs: dict[int, float] = {}
            for pos in range(n):
                depth = sum(counts[pos].values())
                if depth == 0:
                    chars.append(ref[pos])
                    afs[pos] = 0.0
                else:
                    top = max("ACGT", key=lambda b: counts[pos][b])
                    non_ref = (depth - counts[pos].get(ref[pos], 0)) / depth
                    afs[pos] = non_ref
                    chars.append(
                        top
                        if non_ref >= float(payload.get("min_af", 0.5))
                        else ref[pos]
                    )
            return {
                "consensus": "".join(chars),
                "depths": [sum(c.values()) for c in counts],
                "alt_frequencies": afs,
                "masked_low_depth_sites": [],
            }

        if stage == "prepare_reference":
            # Defect 3: blindly accepts conflicting duplicates or ignores --remove-reference
            return list(payload["sample_sequences"])

        if stage == "deconvolve_wastewater":
            # Defect 5: renormalizes modeled abundances to sum=1.0, erasing cryptic variant signal
            obs = {
                pos: (
                    0.0
                    if (not isinstance(af, (int, float)) or math.isnan(af))
                    else float(af)
                )
                for pos, af in payload["site_afs"].items()
            }
            raw: dict[str, float] = {}
            for lin, sig in payload["lineage_signatures"].items():
                vals = [obs.get(s, 0.0) for s in sig]
                raw[lin] = sum(vals) / len(vals) if vals else 0.0
            tot = sum(raw.values())
            norm = {k: (v / tot if tot > 0 else 0.0) for k, v in raw.items()}
            return {
                "abundances": norm,
                "modeled_sum": 1.0 if tot > 0 else 0.0,
                "cryptic_residual": 0.0,
                "unmodeled_high_af_sites": [],
            }

        if stage == "validate_ntc":
            # Defect 6: silently ignores contaminated NTC and returns PASS
            return {"status": "PASS", "ntc_reads": len(payload["ntc_reads"])}

        raise ValueError(f"Unknown stage: {stage!r}")


def _mutate_base(base: str) -> str:
    return "C" if base != "C" else "G"


def audit_amplicon_surveillance_pipeline(
    auditor: Callable[[str, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Run all 6 tiled-amplicon & wastewater surveillance invariants on synthetic controls."""
    ref = SYNTHETIC_AMPLICON_REF_NT
    n = len(ref)
    invariants: list[dict[str, Any]] = []

    # Define two overlapping tiled amplicons over the 72-nt benign synthetic template:
    # Amplicon 1: [0, 44), fwd primer [0, 18), rev primer [26, 44) -> mutation at pos 30 sits in AMP1 rev primer!
    # Amplicon 2: [20, 72), fwd primer [20, 38), rev primer [54, 72) -> covers pos 30? Wait: put mutation at pos 32:
    # Actually let AMP1 = [0, 46), fwd [0,18), rev [28,46); AMP2 = [22, 72), fwd [22,40), rev [54,72)
    # Wait: pos 25 is interior of AMP1 (18..28), and sits INSIDE AMP2's fwd primer [22,40)!
    # So 20 AMP2 primer-carrying reads have WT at pos 25 while 12 AMP1 interior reads have ALT at pos 25!
    mut_pos1 = 25
    alt_base1 = _mutate_base(ref[mut_pos1])
    primers = [
        {
            "amplicon_id": "amp1",
            "fwd_start": 0,
            "fwd_end": 18,
            "rev_start": 28,
            "rev_end": 46,
        },
        {
            "amplicon_id": "amp2",
            "fwd_start": 22,
            "fwd_end": 40,
            "rev_start": 54,
            "rev_end": 72,
        },
    ]

    # 1. Primer-binding-site clipping before variant calling
    try:
        amp1_seq = list(ref[:46])
        amp1_seq[mut_pos1] = alt_base1
        amp1_read = {
            "amplicon_id": "amp1",
            "start": 0,
            "seq": "".join(amp1_seq),
            "strand": "+",
        }
        # AMP2 has wild-type synthetic primer at [22, 40) overlapping mut_pos1=25, plus mutant or WT elsewhere
        amp2_read = {
            "amplicon_id": "amp2",
            "start": 22,
            "seq": ref[22:72],
            "strand": "+",
        }
        reads1 = [amp1_read] * 12 + [amp2_read] * 25
        res1 = auditor(
            "call_variants",
            {
                "reads": reads1,
                "ref_seq": ref,
                "primers": primers,
                "min_depth": 10,
                "min_af": 0.5,
            },
        )
        cons1 = str(res1["consensus"])
        af1 = float(res1["alt_frequencies"][mut_pos1])
        inv1_ok = cons1[mut_pos1] == alt_base1 and abs(af1 - 1.0) < 1e-6
        inv1_detail = f"pos25_call={cons1[mut_pos1]} af={af1:.3f}"
    except Exception as exc:
        inv1_ok = False
        inv1_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "primer_binding_site_clipping_before_calling",
            "passed": inv1_ok,
            "detail": inv1_detail,
        }
    )

    # 2. Short-insert 3' primer/adapter read-through trimming across strands
    try:
        mut_pos2 = 24
        alt_base2 = _mutate_base(ref[mut_pos2])
        custom_ref = list(ref)
        if custom_ref[25] == custom_ref[28]:
            custom_ref[25] = _mutate_base(custom_ref[28])
        cref = "".join(custom_ref)
        rev_rc_tok = cref[28:46]
        short_read_ref = cref[:18] + cref[18:24] + alt_base2 + rev_rc_tok
        short_primers = [
            {
                "amplicon_id": "short_amp",
                "fwd_start": 0,
                "fwd_end": 18,
                "rev_start": 28,
                "rev_end": 46,
                "fwd_seq": cref[0:18],
                "rev_seq": _revcomp(rev_rc_tok),
            }
        ]
        # Plus 12 full-length WT reads covering [18:28] with depth=12 at pos 25: if short reads are NOT trimmed,
        # pos 25 receives 12 bogus bases from rev_rc_tok[0] -> 50% false alt call at pos 25!
        full_wt = cref[:46]
        reads2 = (
            [
                {
                    "amplicon_id": "short_amp",
                    "start": 0,
                    "seq": short_read_ref,
                    "strand": "+",
                }
            ]
            * 6
            + [
                {
                    "amplicon_id": "short_amp",
                    "start": 0,
                    "seq": _revcomp(short_read_ref),
                    "strand": "-",
                }
            ]
            * 6
            + [{"amplicon_id": "short_amp", "start": 0, "seq": full_wt, "strand": "+"}]
            * 12
        )
        res2 = auditor(
            "call_variants",
            {
                "reads": reads2,
                "ref_seq": cref,
                "primers": short_primers,
                "min_depth": 10,
                "min_af": 0.45,
            },
        )
        # At pos 25, trimmed short reads stop at 25 (trimmed before rev_rc_tok), so ONLY full_wt contributes -> alt_af[25] == 0.0!
        # Meanwhile pos 24 has 12 alt / 24 total = 0.50 -> called as alt_base2!
        af25 = float(res2["alt_frequencies"][25])
        af24 = float(res2["alt_frequencies"][24])
        inv2_ok = abs(af25 - 0.0) < 1e-6 and abs(af24 - 0.5) < 1e-6
        inv2_detail = f"trimmed_pos25_false_af={af25:.3f} pos24_af={af24:.3f}"
    except Exception as exc:
        inv2_ok = False
        inv2_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "short_insert_3prime_adapter_primer_trimming",
            "passed": inv2_ok,
            "detail": inv2_detail,
        }
    )

    # 3. Duplicate reference deduplication parity (augur#1892)
    try:
        samples = [("REF_CTRL", ref), ("SAMPLE_1", "C" + ref[1:])]
        raised_helpful = False
        try:
            auditor(
                "prepare_reference",
                {
                    "sample_sequences": samples,
                    "ref_id": "REF_CTRL",
                    "ref_seq": ref,
                    "remove_reference": False,
                },
            )
        except AmpliconSurveillanceError as err:
            msg = str(err)
            raised_helpful = "--reference-name" in msg and "--remove-reference" in msg

        deduped = auditor(
            "prepare_reference",
            {
                "sample_sequences": samples,
                "ref_id": "REF_CTRL",
                "ref_seq": ref,
                "remove_reference": True,
            },
        )
        conflicting_blocked = False
        try:
            auditor(
                "prepare_reference",
                {
                    "sample_sequences": [("REF_CTRL", _mutate_base(ref[0]) + ref[1:])],
                    "ref_id": "REF_CTRL",
                    "ref_seq": ref,
                    "remove_reference": True,
                },
            )
        except AmpliconSurveillanceError:
            conflicting_blocked = True

        inv3_ok = (
            raised_helpful
            and conflicting_blocked
            and [sid for sid, _ in deduped] == ["SAMPLE_1"]
        )
        inv3_detail = f"helpful_err={raised_helpful} conflict_blocked={conflicting_blocked} out={[s[0] for s in deduped]}"
    except Exception as exc:
        inv3_ok = False
        inv3_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "duplicate_reference_deduplication_parity",
            "passed": inv3_ok,
            "detail": inv3_detail,
        }
    )

    # 4. Low-depth dropout masking (N / NaN instead of silent 0.0 wild-type coercion)
    try:
        # 20 reads only covering [18, 28); positions [46, 54) have 0 reads (< min_depth=10)
        reads4 = [
            {"amplicon_id": "amp1", "start": 0, "seq": ref[:46], "strand": "+"}
        ] * 20
        res4 = auditor(
            "call_variants",
            {
                "reads": reads4,
                "ref_seq": ref,
                "primers": [primers[0]],
                "min_depth": 10,
                "min_af": 0.5,
            },
        )
        cons4 = str(res4["consensus"])
        af50 = res4["alt_frequencies"][50]
        inv4_ok = (
            cons4[50] == "N"
            and isinstance(af50, float)
            and math.isnan(af50)
            and 50 in res4["masked_low_depth_sites"]
        )
        inv4_detail = f"pos50_base={cons4[50]} pos50_af={af50}"
    except Exception as exc:
        inv4_ok = False
        inv4_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "low_depth_site_masking_no_zero_coercion",
            "passed": inv4_ok,
            "detail": inv4_detail,
        }
    )

    # 5. Simplex wastewater deconvolution preserving unexplained cryptic variant residual
    try:
        sigs = {"Lineage_A": {20, 21}, "Lineage_B": {23, 24}}
        # Lineage A at 30%, Lineage B at 20%, plus an unmodeled cryptic lineage at 50% on site 26
        # plus dropout NaN at site 21 (should NOT pull Lineage A down toward 0!)
        site_afs = {
            20: 0.30,
            21: float("nan"),
            23: 0.20,
            24: 0.20,
            26: 0.50,
        }
        res5 = auditor(
            "deconvolve_wastewater",
            {"site_afs": site_afs, "lineage_signatures": sigs, "min_cryptic_af": 0.15},
        )
        ab = res5["abundances"]
        inv5_ok = (
            abs(float(ab["Lineage_A"]) - 0.30) < 1e-6
            and abs(float(ab["Lineage_B"]) - 0.20) < 1e-6
            and float(res5["cryptic_residual"]) >= 0.45
            and 26 in res5["unmodeled_high_af_sites"]
        )
        inv5_detail = f"abundances={ab} cryptic={res5['cryptic_residual']}"
    except Exception as exc:
        inv5_ok = False
        inv5_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "simplex_wastewater_deconvolution_with_cryptic_residual",
            "passed": inv5_ok,
            "detail": inv5_detail,
        }
    )

    # 6. NTC contamination fail-closed gate
    try:
        ntc_reads = [{"start": 10, "seq": ref[10:40]}] * 12
        inv6_ok = False
        inv6_detail = "Contaminated NTC was accepted without failing closed."
        try:
            res6 = auditor(
                "validate_ntc",
                {"ntc_reads": ntc_reads, "sample_reads": [], "ref_len": n},
            )
            inv6_detail = f"Accepted contaminated NTC: {res6!r}"
        except AmpliconSurveillanceError as err:
            inv6_ok = "CONTAMINATED" in str(err)
            inv6_detail = f"Fail-closed triggered: {err}"
    except Exception as exc:
        inv6_ok = False
        inv6_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "ntc_contamination_fail_closed_gate",
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


def run_amplicon_surveillance_self_check() -> dict[str, Any]:
    """Compare DefensiveAmpliconSurveillanceAuditor against NaiveAmpliconConsensusCaller."""
    hardened = audit_amplicon_surveillance_pipeline(
        DefensiveAmpliconSurveillanceAuditor()
    )
    naive = audit_amplicon_surveillance_pipeline(NaiveAmpliconConsensusCaller())
    caught = [inv["invariant"] for inv in naive["invariants"] if not inv["passed"]]
    return {
        "hardened_passed": hardened["passed"],
        "hardened": hardened,
        "naive_caught_invariants": caught,
        "naive_total_invariants": naive["total_invariants"],
        "caught_all_naive_bugs": len(caught) == naive["total_invariants"],
    }
