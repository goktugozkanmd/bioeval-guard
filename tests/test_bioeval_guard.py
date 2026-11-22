"""Comprehensive test suite for bioeval_guard."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bioeval_guard.amplicon_surveillance_audit import (
    SYNTHETIC_AMPLICON_REF_NT,
    SYNTHETIC_FWD_PRIMER_NT,
    SYNTHETIC_REV_PRIMER_NT,
    AmpliconSurveillanceError,
    DefensiveAmpliconSurveillanceAuditor,
    NaiveAmpliconConsensusCaller,
    audit_amplicon_surveillance_pipeline,
    run_amplicon_surveillance_self_check,
)
from bioeval_guard.ancestral_translation_audit import (
    SYNTHETIC_REPORTER_CDS_NT,
    SYNTHETIC_REPORTER_PROTEIN_AA,
    DefensiveTranslationAlignmentScreener,
    NaiveTranslationAlignmentLoader,
    TranslationAlignmentError,
    audit_ancestral_translation_pipeline,
    run_ancestral_translation_self_check,
)
from bioeval_guard.clade_assignment_audit import (
    SYNTHETIC_CLADE_ROOT_SEQ,
    CladeDefinitionError,
    DefensiveCladeAssigner,
    MissingCladeError,
    audit_clade_assignment_pipeline,
    run_clade_assignment_self_check,
)
from bioeval_guard.cli import format_markdown_summary, main
from bioeval_guard.cluster_stats import audit_clustered_benchmark_se
from bioeval_guard.de_novo_design_screening_audit import (
    SYNTHETIC_DESIGN_NEGATIVE_CDS_NT,
    SYNTHETIC_DESIGN_TARGET_AA,
    SYNTHETIC_DESIGN_TARGET_CDS_NT,
    DefensiveDesignScreener,
    DesignScreeningInputError,
    NaivePerRecordDesignScreener,
    audit_de_novo_design_screener,
    run_design_screening_self_check,
)
from bioeval_guard.evolutionary_distance_audit import (
    SYNTHETIC_CLOCK_ROOT_SEQ,
    DefensiveEvolutionaryDistanceAuditor,
    EvolutionaryTopologyError,
    NaivePairwiseRootDistanceCalculator,
    audit_evolutionary_distance_pipeline,
    run_evolutionary_distance_self_check,
)
from bioeval_guard.grader_audit import (
    audit_agent_transcript_injection_resistance,
    audit_answer_tag_parser,
    audit_dna_screener_symmetry,
    audit_empty_tool_args_judge_gate,
    audit_hedged_tag_multiselect_grader,
    audit_honesty_error_denominator_isolation,
    audit_judge_parse_error_isolation,
    audit_mcq_option_text_grader,
    audit_multiselect_mcq_grader,
    audit_numeric_string_verifier,
    audit_rubric_header_format_invariance,
    audit_safety_judge_callable,
    audit_selective_precision_metric,
    audit_structured_hazard_judge_verdict,
    audit_truncated_empty_completion_handling,
    audit_ungraded_epoch_harm_reducer,
    audit_unparseable_judge_nan_handling,
    audit_verifier_confidence_fail_closed,
    case_safe_judge_label,
    error_isolated_honesty_score,
    fail_closed_verifier_gate,
    hardened_mcq_option_grader,
    hardened_multiselect_mcq_grader,
    header_agnostic_rubric_scorer,
    hedged_tag_multiselect_grader,
    nan_safe_selective_precision,
    non_empty_tool_call_semantic_gate,
    parse_tagged_answer_strict,
    provenance_safe_transcript_turn_filter,
    run_reference_self_check,
    strict_numeric_string_verifier,
    strict_structured_hazard_judge_verdict,
    truncation_safe_answer_gate,
    ungraded_safe_epoch_harm_reducer,
    unparseable_safe_nan_judge_score,
    unscored_on_unparseable_judge_verdict,
)
from bioeval_guard.hygiene import (
    AuditItem,
    check_bixbench_hygiene,
    check_mcq_hygiene,
    check_primer_reverse_complement,
    check_restriction_site_complement_bug,
    complement_dna,
    load_records,
    normalize_items,
    reverse_complement_dna,
)
from bioeval_guard.inspect_adapter import (
    make_inspect_sample_filter,
    question_blind_centroid_baseline,
    sanitize_mcq_dataset,
)
from bioeval_guard.inspect_tasks import (
    LAB_BENCH_REVISION_PIN,
    compute_nan_safe_precision_and_coverage,
    filter_clean_lab_bench_records,
    nan_safe_precision_metric,
)
from bioeval_guard.metagenomic_surveillance_audit import (
    SYNTHETIC_HOST_DECOY_KMER,
    SYNTHETIC_VECTOR_TARGET_KMER,
    DefensiveMetagenomicScreener,
    audit_metagenomic_surveillance_pipeline,
    run_surveillance_self_check,
)
from bioeval_guard.phylodynamic_surveillance_audit import (
    SYNTHETIC_HOUSEKEEPING_REF,
    DefensivePhylodynamicScreener,
    audit_phylodynamic_surveillance_pipeline,
    run_phylodynamic_self_check,
)
from bioeval_guard.protein_screening_audit import (
    SYNTHETIC_NEGATIVE_PEPTIDE,
    SYNTHETIC_REPORTER_MOTIF,
    SYNTHETIC_REPORTER_PEPTIDE,
    DefensiveProteinScreener,
    audit_protein_screener,
    run_protein_screening_self_check,
)
from bioeval_guard.sequence_screening_audit import (
    SYNTHETIC_CONTROL_KMER,
    SYNTHETIC_CONTROL_OLIGO,
    DefensiveKmerScreener,
    audit_sequence_screener,
    normalize_fasta_records,
    run_screening_self_check,
)
from bioeval_guard.shortcuts import (
    audit_question_blind_shortcuts,
    compute_option_features,
    holm_bonferroni,
)


def test_complement_and_reverse_complement() -> None:
    assert complement_dna("GAATTC") == "CTTAAG"
    assert complement_dna("GGATCC") == "CCTAGG"
    assert reverse_complement_dna("GAATTC") == "GAATTC"
    assert reverse_complement_dna("ATGCCC") == "GGGCAT"


def test_primer_sense_strand_bug_detected() -> None:
    template = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
    bad_sense_primer = "CTGCAG" + template[-18:]
    good_revcomp_primer = "CTGCAG" + reverse_complement_dna(template[-18:])
    assert check_primer_reverse_complement(template, bad_sense_primer) is False
    assert check_primer_reverse_complement(template, good_revcomp_primer) is True


def test_primer_short_inputs_handled_safely() -> None:
    assert check_primer_reverse_complement("ATGC", "ATGC") is True


def test_restriction_site_complement_bug_detected() -> None:
    assert check_restriction_site_complement_bug("BamHI", "CCTAGGATGCCC") is True
    assert check_restriction_site_complement_bug("EcoRI", "CTTAAGATGCCC") is True
    assert check_restriction_site_complement_bug("BamHI", "GGATCCATGCCC") is False
    assert check_restriction_site_complement_bug("UnknownEnz", "CCTAGGATGCCC") is False


def test_hygiene_catches_collisions_nulls_and_comma_shards() -> None:
    records = [
        {
            "id": "r1",
            "question": "Q1",
            "ideal": "EcoRI, KpnI",
            "distractors": ["EcoRI, KpnI", "SphI, KpnI", "BamHI, KpnI"],
            "subtask": "PCR-geneprimers-enz-v1-public",
        },
        {
            "id": "r2",
            "question": "Q2",
            "ideal": "null",
            "distractors": ["Nuclear", "Cytosol", "Membrane"],
            "subtask": "litqa-v2-public",
        },
        {
            "id": "r3",
            "question": "Q3",
            "ideal": "12,300",
            "distractors": ["4", "900", "39", "400", "2", "750", "2", "850"],
            "subtask": "litqa-v2-public",
        },
        {
            "id": "r4",
            "question": "Q4",
            "ideal": "Clean answer",
            "distractors": ["Wrong 1", "Wrong 2", "Wrong 3"],
            "subtask": "litqa-v2-public",
        },
    ]
    items = normalize_items(records)
    rep = check_mcq_hygiene(items)
    assert rep["ideal_in_distractors_exact"] == 1
    assert rep["literal_null_ideal_items"] == 1
    assert rep["comma_split_distractor_items"] == 1
    assert rep["flagged_items"] == 3
    assert set(rep["flagged_ids"]) == {"r1", "r2", "r3"}


def test_hygiene_catches_phrase_comma_shard() -> None:
    records = [
        {
            "id": "p1",
            "question": "Which isoform?",
            "ideal": "Both the V1 and V3 isoforms.",
            "distractors": ["Only V1", "The V1", "V2", "and V3 isoforms."],
            "subtask": "litqa-v2-public",
        }
    ]
    rep = check_mcq_hygiene(normalize_items(records))
    assert rep["comma_split_distractor_items"] == 1


def test_hygiene_catches_empty_and_duplicate_options() -> None:
    records = [
        {
            "id": "e1",
            "question": "Q",
            "ideal": "Alpha",
            "distractors": ["", "Beta", "Gamma"],
        },
        {
            "id": "d1",
            "question": "Q2",
            "ideal": "Alpha",
            "distractors": ["Beta", " beta ", "Gamma"],
        },
    ]
    rep = check_mcq_hygiene(normalize_items(records), check_biology=False)
    assert rep["empty_option_items"] == 1
    assert rep["duplicate_option_items"] == 1


def test_hygiene_meta_phrase_asymmetry_and_duplicate_questions() -> None:
    records = [
        {
            "id": "m1",
            "question": "Repeated Q",
            "ideal": "All of the above",
            "distractors": ["Option A", "Option B", "Option C"],
        },
        {
            "id": "m2",
            "question": "Repeated Q",
            "ideal": "Specific item",
            "distractors": ["X", "Y", "Z"],
        },
    ]
    rep = check_mcq_hygiene(normalize_items(records), check_biology=False)
    assert rep["meta_phrase_ideal_only_items"] == 1
    assert rep["duplicate_question_texts"] == 1


def test_hygiene_isoschizomer_collision() -> None:
    records = [
        {
            "id": "iso1",
            "question": "Which enzymes?",
            "ideal": "SacI, SstI",
            "distractors": ["SacI, SmaI", "SstI, SstI", "BamHI, EcoRI"],
            "subtask": "PCR-geneprimers-enz-v1-public",
        }
    ]
    rep = check_mcq_hygiene(normalize_items(records))
    assert rep["isoschizomer_collisions"] == 1


def test_hygiene_detects_sense_primer_and_comp_bug_in_pcr_item() -> None:
    template = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
    records = [
        {
            "id": "seq1",
            "question": f"Design EcoRI and BamHI primers for {template}",
            "ideal": f"GAATTCATGGCCATTGTAATGGGCC, CCTAGG{template[-18:]}",
            "distractors": ["A, B", "C, D", "E, F"],
            "subtask": "PCR-seq-enzprimers-v1-public",
        }
    ]
    rep = check_mcq_hygiene(normalize_items(records), check_biology=True)
    assert rep["sense_reverse_primer_bugs"] == 1
    assert rep["comp_restriction_bugs"] == 1


def test_bixbench_hygiene_checks() -> None:
    records = [
        {
            "question_id": "bix-1-q1",
            "question": "p-val?",
            "ideal": "0.0002",
            "distractors": ["7.820659E-05", "0.0003", "1.847038E-05"],
            "eval_mode": "str_verifier",
        },
        {
            "question_id": "bix-14-q2",
            "question": "CI?",
            "ideal": "(0.2, 0.3)",
            "distractors": ["(0.0, 0.1)", "(0.3, 0.4)", "(0.1, 0.2)"],
            "eval_mode": "range_verifier",
        },
        {
            "question_id": "bix-clean",
            "question": "Gene?",
            "ideal": "TP53",
            "distractors": ["KRAS", "BRAF", "EGFR"],
            "eval_mode": "llm_verifier",
        },
    ]
    rep = check_bixbench_hygiene(normalize_items(records))
    assert rep["str_verifier_notation_mismatches"] == 1
    assert rep["range_verifier_overlapping_intervals"] == 1
    assert rep["eval_mode_counts"] == {
        "str_verifier": 1,
        "range_verifier": 1,
        "llm_verifier": 1,
    }


def test_normalize_wmdp_and_error_schema() -> None:
    records = [
        {"question": "Bio Q", "choices": ["A1", "B2", "C3", "D4"], "answer": 2},
        {"question": "Bio Q2", "choices": ["X1", "X2", "X3", "X4"], "answer": "B"},
    ]
    items = normalize_items(records, dataset_name="wmdp")
    assert items[0].ideal == "C3"
    assert items[1].ideal == "X2"
    assert len(items[0].distractors) == 3
    with pytest.raises(ValueError, match="Unrecognized MCQ schema"):
        normalize_items([{"foo": "bar"}])


def test_load_records_json_jsonl_parquet_and_invalid(tmp_path: Path) -> None:
    jpath = tmp_path / "data.json"
    jpath.write_text(
        '[{"id":"1","question":"Q","ideal":"A","distractors":["B"]}]', encoding="utf-8"
    )
    assert len(load_records(jpath)) == 1

    pq_path = tmp_path / "data.parquet"
    table = pa.table(
        {"question": ["Q"], "choices": [["A", "B", "C", "D"]], "answer": [0]}
    )
    pq.write_table(table, pq_path)
    assert len(load_records(pq_path)) == 1

    with pytest.raises(ValueError, match="Unsupported dataset extension"):
        load_records(tmp_path / "bad.txt")


def test_holm_bonferroni_exact_and_empty() -> None:
    assert holm_bonferroni([]) == []
    raw = [0.01, 0.04, 0.03, 0.20]
    adj = holm_bonferroni(raw)
    assert pytest.approx(adj, abs=1e-9) == [0.04, 0.09, 0.09, 0.20]


def test_compute_option_features_within_item_normalized() -> None:
    rel, mean_sims, char_lens = compute_option_features(
        ["AAAA", "AAAB", "AAAC", "ZZZZZZZZ"]
    )
    assert rel.shape == (4, 10)
    assert char_lens == [4, 4, 4, 8]
    assert mean_sims[0] > mean_sims[3]


def test_star_graph_centroid_exploit() -> None:
    rng = random.Random(7)
    base = list("ACDEFGHIKLMNPQRSTVWY" * 4)
    items: list[AuditItem] = []
    for i in range(40):
        seq = base[:]
        rng.shuffle(seq)
        ideal = "".join(seq)
        dists = []
        for pos in (3, 17, 41):
            mut = seq[:]
            mut[pos] = "A" if mut[pos] != "A" else "G"
            dists.append("".join(mut))
        items.append(
            AuditItem(
                item_id=f"s{i}", question="Which?", options=[ideal] + dists, ideal_idx=0
            )
        )

    res = audit_question_blind_shortcuts(items)
    assert res["centroid"]["acc"] == 1.0
    assert res["oof_5fold"]["acc"] >= 0.95
    assert res["centroid"]["p_raw"] < 1e-10


def test_longest_option_shortcut() -> None:
    items = [
        AuditItem(
            item_id=f"l{i}",
            question="Q",
            options=[
                "Very detailed mechanistic biological explanation here",
                "Short A",
                "Short B",
                "Short C",
            ],
            ideal_idx=0,
        )
        for i in range(40)
    ]
    res = audit_question_blind_shortcuts(items)
    assert res["longest"]["acc"] == 1.0
    assert res["oof_5fold"]["acc"] == 1.0


def test_clean_balanced_options_near_chance() -> None:
    rng = random.Random(123)
    items: list[AuditItem] = []
    for i in range(80):
        opts = [f"gene_{rng.randint(100, 999)}_{j}" for j in range(4)]
        items.append(
            AuditItem(item_id=f"c{i}", question="Q", options=opts, ideal_idx=0)
        )
    res = audit_question_blind_shortcuts(items)
    assert res["centroid"]["p_raw"] > 0.05
    assert res["longest"]["p_raw"] > 0.05


def test_empty_shortcut_audit_returns_zero() -> None:
    assert audit_question_blind_shortcuts([]) == {"N": 0, "chance": 0.0}


def test_inspect_adapter_sanitize_and_baseline() -> None:
    records = [
        {"id": "bad", "question": "Q", "ideal": "null", "distractors": ["A", "B", "C"]},
        {
            "id": "good",
            "question": "Q",
            "ideal": "Valid",
            "distractors": ["D1", "D2", "D3"],
        },
    ]
    clean, rep = sanitize_mcq_dataset(records)
    assert rep["removed_items"] == 1
    assert [r["id"] for r in clean] == ["good"]

    pred = question_blind_centroid_baseline(
        ["ZZZ_DIFF", "AAACCC1", "AAACCC2", "AAACCC3"]
    )
    assert pred in {1, 2, 3}
    with pytest.raises(ValueError, match="non-empty"):
        question_blind_centroid_baseline([])

    keep = make_inspect_sample_filter()
    norm = normalize_items(records)
    assert keep(norm[0]) is False
    assert keep(norm[1]) is True


def test_cli_audit_jsonl_json_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "sample.jsonl"
    p.write_text(
        '{"id":"1","question":"Q","ideal":"Alpha","distractors":["Beta","Gamma","Delta"]}\n'
        '{"id":"2","question":"Q","ideal":"null","distractors":["B","C","D"]}\n',
        encoding="utf-8",
    )
    pq_path = tmp_path / "sample_wmdp.parquet"
    pq.write_table(
        pa.table(
            {
                "question": [f"Q{i}" for i in range(12)],
                "choices": [["A1", "B2", "C3", "D4"] for _ in range(12)],
                "answer": [0 for _ in range(12)],
            }
        ),
        pq_path,
    )
    assert main(["audit-jsonl", str(p), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hygiene"]["literal_null_ideal_items"] == 1

    out_json = tmp_path / "summary.json"
    assert main(["summary", str(p), str(pq_path), "--json-out", str(out_json)]) == 0
    md = capsys.readouterr().out
    assert "| sample | 2 | 1 |" in md
    assert "| sample_wmdp | 12 | 0 |" in md
    assert len(json.loads(out_json.read_text(encoding="utf-8"))) == 2


def test_format_markdown_empty() -> None:
    assert format_markdown_summary([]) == "No datasets audited."


def test_grader_auditors_detect_real_defects_and_pass_hardened() -> None:
    # 1. Safety judge auditor
    bad_judge = audit_safety_judge_callable(
        lambda s: "unsafe" if '"label": "unsafe"' in s else "safe"
    )
    assert bad_judge["passed"] is False
    assert len(bad_judge["failures"]) == 5
    good_judge = audit_safety_judge_callable(case_safe_judge_label)
    assert good_judge["passed"] is True

    # 2. Tag parser auditor
    bad_parser = audit_answer_tag_parser(
        lambda s: s[s.find("<answer>") + 8 : s.find("</answer>")]
    )
    assert bad_parser["passed"] is False
    good_parser = audit_answer_tag_parser(
        lambda s: parse_tagged_answer_strict(s, strip_upper=True)
    )
    assert good_parser["passed"] is True

    # 3. Numeric string verifier auditor
    import re

    bad_num = audit_numeric_string_verifier(
        lambda t, p: (
            re.sub(r"[^a-zA-Z0-9]", "", t).lower()
            == re.sub(r"[^a-zA-Z0-9]", "", p).lower()
        )
    )
    assert bad_num["passed"] is False
    good_num = audit_numeric_string_verifier(strict_numeric_string_verifier)
    assert good_num["passed"] is True

    # 4. Selective precision metric auditor
    bad_metric = audit_selective_precision_metric(
        lambda xs: (
            0.0
            if not [x for x in xs if x != -1]
            else sum(x for x in xs if x != -1) / len([x for x in xs if x != -1])
        ),
        lambda xs: len([x for x in xs if x != -1]) / len(xs) if xs else 0.0,
    )
    assert bad_metric["passed"] is False
    good_metric = audit_selective_precision_metric(
        lambda xs: nan_safe_selective_precision(
            [float(x) for x in xs if x != -1], len(xs)
        )[0],
        lambda xs: nan_safe_selective_precision(
            [float(x) for x in xs if x != -1], len(xs)
        )[1],
    )
    assert good_metric["passed"] is True

    # 5. Defensive benign oligo symmetry auditor
    asymmetric = audit_dna_screener_symmetry(lambda s: "CTAGCGATGCTAGCTAGTCG" in s)
    assert asymmetric["passed"] is False


def test_inspect_tasks_helpers_and_self_check_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import math

    assert LAB_BENCH_REVISION_PIN() == "5c77cec648430f30611808808861eb86f81d5eaa"
    prec, cov = compute_nan_safe_precision_and_coverage(["N", "N", "N"])
    assert math.isnan(prec) and cov == 0.0
    fallback = nan_safe_precision_metric()
    assert math.isnan(fallback([-1, -1]))

    kept, rep = filter_clean_lab_bench_records(
        [
            {
                "id": "k1",
                "question": "Q1",
                "ideal": "null",
                "distractors": ["A", "B", "C"],
            },
            {
                "id": "k2",
                "question": "Q2",
                "ideal": "BRCA1",
                "distractors": ["TP53", "EGFR", "MYC"],
            },
        ],
        subset_name="LitQA2",
        check_biology=False,
    )
    assert [r["id"] for r in kept] == ["k2"]
    assert rep["removed_items"] == 1

    self_rep = run_reference_self_check()
    assert self_rep["hardened_all_passed"] is True
    assert self_rep["vulnerable_caught"] == self_rep["vulnerable_total"]
    assert self_rep["vulnerable_total"] >= 16

    assert main(["self-check", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardened_all_passed"] is True
    assert main(["self-check"]) == 0
    out_md = capsys.readouterr().out
    assert "| mcq_option_text_grader |" in out_md
    assert "| unparseable_judge_nan_handling |" in out_md
    assert "| truncated_empty_completion_handling |" in out_md
    assert "| multiselect_mcq_grader |" in out_md
    assert "| ungraded_epoch_harm_reducer |" in out_md
    assert "| hedged_tag_multiselect_grader |" in out_md
    assert "| structured_hazard_judge_verdict |" in out_md
    assert "| rubric_header_format_invariance |" in out_md
    assert "| verifier_confidence_fail_closed |" in out_md
    assert "| agent_transcript_injection_resistance |" in out_md


def test_mcq_option_text_grader_auditor_catches_end_letter_inversion() -> None:
    import re

    def naive_end_letter(completion: str, _choices: list[str], target: str) -> bool:
        letters = re.findall(r"[A-Za-z]", completion)
        return bool(letters) and letters[-1].upper() == target.strip().upper()

    bad = audit_mcq_option_text_grader(naive_end_letter)
    assert bad["passed"] is False
    failed_probes = {f["probe"] for f in bad["failures"]}
    assert "letter_plus_correct_text" in failed_probes
    assert "wrong_letter_ending_in_target_char" in failed_probes
    assert "labeled_wrong_answer_ending_in_target_char" in failed_probes

    good = audit_mcq_option_text_grader(hardened_mcq_option_grader)
    assert good["passed"] is True


def test_sequence_screening_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    screener = DefensiveKmerScreener()
    rep = audit_sequence_screener(screener.screen, screener.screen_batch)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6

    # Naive substring screener must fail 5 of 6 invariants
    naive_rep = audit_sequence_screener(lambda s: SYNTHETIC_CONTROL_KMER in s)
    assert naive_rep["passed"] is False
    assert naive_rep["passed_count"] == 1

    # Screener that naively strips non-ACGT characters fabricates a gapped chimera
    import re

    stripping_rep = audit_sequence_screener(
        lambda s: SYNTHETIC_CONTROL_KMER in re.sub(r"[^ACGT]", "", s.upper())
    )
    assert stripping_rep["passed"] is False
    iupac_inv = [
        inv
        for inv in stripping_rep["invariants"]
        if inv["invariant"] == "iupac_ambiguity_resilience"
    ][0]
    assert iupac_inv["passed"] is False

    # Ensure IUPAC bases are preserved rather than deleted into false chimeras
    gapped = SYNTHETIC_CONTROL_KMER[:10] + "N" + SYNTHETIC_CONTROL_KMER[10:]
    assert screener.screen(gapped) is False

    # Ensure header injection and RNA uracil normalization work properly
    recs = normalize_fasta_records(
        f">hdr\n{SYNTHETIC_CONTROL_OLIGO.lower().replace('t', 'u')}\n"
    )
    assert len(recs) == 1
    assert "U" not in recs[0]

    self_screen = run_screening_self_check()
    assert self_screen["hardened_passed"] is True
    assert len(self_screen["naive_caught_invariants"]) == 5
    assert "circular_origin_wrap" in self_screen["naive_caught_invariants"]
    assert "tiled_boundary_split" in self_screen["naive_caught_invariants"]

    assert main(["screen-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["screen-audit"]) == 0
    assert "| circular_origin_wrap | PASS |" in capsys.readouterr().out


def test_clustered_benchmark_se_invariants_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Singleton clusters (n_c = 1): cluster SE must equal naive SE identically.
    single_scores = [1, 0, 1, 1, 0, 0, 1, 0]
    single_ids = [f"q{i}" for i in range(len(single_scores))]
    single_rep = audit_clustered_benchmark_se(single_scores, single_ids)
    assert single_rep["cluster_se"] == pytest.approx(single_rep["naive_se"], abs=1e-6)
    assert single_rep["se_inflation_factor"] == pytest.approx(1.0, abs=1e-3)
    assert single_rep["design_effect_kish"] == pytest.approx(1.0, abs=1e-6)

    # 2. Within-capsule correlation (identical answers inside each capsule of size 5):
    # ICC = 1.0, DEFF = 5.0, N_eff = K, SE inflation = sqrt(m_eff * ...) > 2x.
    corr_scores = [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5
    corr_ids = [f"cap_{i // 5}" for i in range(len(corr_scores))]
    corr_rep = audit_clustered_benchmark_se(corr_scores, corr_ids)
    assert corr_rep["n_items"] == 30
    assert corr_rep["n_clusters"] == 6
    assert corr_rep["icc_anova"] == pytest.approx(1.0, abs=1e-4)
    assert corr_rep["design_effect_kish"] == pytest.approx(5.0, abs=1e-4)
    assert corr_rep["effective_n_kish"] == pytest.approx(6.0, abs=1e-2)
    assert corr_rep["se_inflation_factor"] > 2.0
    assert corr_rep["cluster_ci95"][0] < corr_rep["naive_ci95"][0]
    assert corr_rep["cluster_ci95"][1] > corr_rep["naive_ci95"][1]

    # 3. Validation errors
    with pytest.raises(ValueError, match="equal length"):
        audit_clustered_benchmark_se([1, 0], ["c1"])
    with pytest.raises(ValueError, match="non-empty"):
        audit_clustered_benchmark_se([], [])

    # 4. CLI integration
    jpath = tmp_path / "capsules.jsonl"
    jpath.write_text(
        "\n".join(
            json.dumps({"score": s, "cluster_id": c})
            for s, c in zip(corr_scores, corr_ids)
        ),
        encoding="utf-8",
    )
    assert main(["cluster-audit", str(jpath), "--json"]) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["n_clusters"] == 6
    assert main(["cluster-audit", str(jpath)]) == 0
    assert "| 30 | 6 |" in capsys.readouterr().out


def test_unparseable_judge_nan_auditor_and_helper() -> None:
    import math

    assert math.isnan(unparseable_safe_nan_judge_score(""))
    assert math.isnan(unparseable_safe_nan_judge_score('{"verdict": '))
    assert math.isnan(
        unparseable_safe_nan_judge_score("I cannot answer due to policy.")
    )
    assert unparseable_safe_nan_judge_score('```json\n{"verdict": "yes"}\n```') == 1.0
    assert unparseable_safe_nan_judge_score('{"verdict": "no"}') == 0.0

    bad = audit_unparseable_judge_nan_handling(
        lambda s: 1.0 if '"verdict": "yes"' in s.lower() else 0.0
    )
    assert bad["passed"] is False
    assert len(bad["failures"]) == 4

    good = audit_unparseable_judge_nan_handling(unparseable_safe_nan_judge_score)
    assert good["passed"] is True
    assert good["probes_run"] == 6


def test_metagenomic_surveillance_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    screener = DefensiveMetagenomicScreener()
    rep = audit_metagenomic_surveillance_pipeline(screener)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6

    # Direct checks for non-host overhang threshold and LCA separation
    short_chimera = SYNTHETIC_HOST_DECOY_KMER + SYNTHETIC_VECTOR_TARGET_KMER[:20]
    assert screener.screen_host_depleted_read(short_chimera) is False
    assert (
        screener.screen_host_depleted_read(
            SYNTHETIC_HOST_DECOY_KMER + "AA" + SYNTHETIC_VECTOR_TARGET_KMER
        )
        is True
    )
    with pytest.raises(ValueError, match="spike_in_reads"):
        screener.normalize_spike_in_abundance(10, 0)

    self_surv = run_surveillance_self_check()
    assert self_surv["hardened_passed"] is True
    assert len(self_surv["naive_caught_invariants"]) == 6
    assert "host_chimera_junction_retention" in self_surv["naive_caught_invariants"]
    assert (
        "samplesheet_missing_file_fail_closed" in self_surv["naive_caught_invariants"]
    )
    assert (
        "spike_in_compositional_closure_invariance"
        in self_surv["naive_caught_invariants"]
    )

    assert main(["surveillance-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["surveillance-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| host_chimera_junction_retention | PASS |" in out_md
    assert "| paired_mate_strand_concordance | PASS |" in out_md


def test_truncated_empty_completion_gate_and_auditor() -> None:
    assert truncation_safe_answer_gate("", None) is None
    assert (
        truncation_safe_answer_gate("<think>long reasoning...</think>", "max_tokens")
        is None
    )
    assert (
        truncation_safe_answer_gate("<think>unclosed reasoning", "max_tokens") is None
    )
    assert (
        truncation_safe_answer_gate("<think>step 1</think>   \n\t ", "length") is None
    )
    assert (
        truncation_safe_answer_gate("<think>check</think>Final answer", "stop")
        == "Final answer"
    )
    assert truncation_safe_answer_gate("Final answer", "stop") == "Final answer"

    bad = audit_truncated_empty_completion_handling(lambda s, _r=None: s or "")
    assert bad["passed"] is False
    failed = {f["probe"] for f in bad["failures"]}
    assert "empty_string" in failed
    assert "think_only_max_tokens" in failed
    assert "whitespace_after_think" in failed
    assert "valid_answer_after_think" in failed

    good = audit_truncated_empty_completion_handling(truncation_safe_answer_gate)
    assert good["passed"] is True
    assert good["probes_run"] == 5


def test_phylodynamic_surveillance_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    screener = DefensivePhylodynamicScreener()
    rep = audit_phylodynamic_surveillance_pipeline(screener)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6

    # Direct checks for benign housekeeping template, bounds, codon frame, polytomy, HT weights
    assert len(SYNTHETIC_HOUSEKEEPING_REF) == 60
    assert screener.parse_collection_date_bounds("2024-02") == (
        "2024-02-01",
        "2024-02-29",
    )
    assert screener.translate_codon_aligned("ATG---ACNGGNATNTGG") == "M-TGXW"
    with pytest.raises(ValueError, match="mismatch"):
        screener.validate_index_parity(["S1", "S2"], ["S1"], allow_partial=False)
    partial = screener.validate_index_parity(["S1", "S2"], ["S1"], allow_partial=True)
    assert partial["missing_in_metadata"] == ["S2"]
    assert screener.infer_directed_transmission(0.0) == {
        "directed_transmission": False,
        "status": "unresolved_polytomy",
    }

    self_phylo = run_phylodynamic_self_check()
    assert self_phylo["hardened_passed"] is True
    assert len(self_phylo["naive_caught_invariants"]) == 6
    assert "sequence_metadata_index_parity" in self_phylo["naive_caught_invariants"]
    assert "ambiguous_iupac_mutation_masking" in self_phylo["naive_caught_invariants"]
    assert "incomplete_collection_date_bounds" in self_phylo["naive_caught_invariants"]
    assert "frame_preserving_codon_translation" in self_phylo["naive_caught_invariants"]
    assert "polytomy_transmission_tie_guard" in self_phylo["naive_caught_invariants"]
    assert (
        "horvitz_thompson_sampling_depletion" in self_phylo["naive_caught_invariants"]
    )

    assert main(["phylo-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["phylo-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| sequence_metadata_index_parity | PASS |" in out_md
    assert "| horvitz_thompson_sampling_depletion | PASS |" in out_md


def test_protein_screening_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    screener = DefensiveProteinScreener()
    rep = audit_protein_screener(
        screener_fn=screener.screen,
        segment_screener_fn=screener.screen_segments,
        circular_screener_fn=lambda s: screener.screen(s, circular=True),
    )
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6

    # Verify benign reporter vs negative control, malformed bracket & mid-line '>' guards
    assert screener.screen(SYNTHETIC_REPORTER_PEPTIDE)["flagged"] is True
    assert screener.screen(SYNTHETIC_NEGATIVE_PEPTIDE)["flagged"] is False
    assert (
        screener.screen(
            f">benign_header_with_{SYNTHETIC_REPORTER_MOTIF}\n{SYNTHETIC_NEGATIVE_PEPTIDE}"
        )["flagged"]
        is False
    )
    with pytest.raises(ValueError, match="Mismatched|Unclosed"):
        screener.screen("W(SEP]HPQFEKGGSG")
    with pytest.raises(ValueError, match="header injection"):
        screener.screen("MGHSHHH>fake_header")

    self_prot = run_protein_screening_self_check()
    assert self_prot["hardened_passed"] is True
    assert len(self_prot["naive_caught_invariants"]) == 6
    assert "three_letter_iupac_dealiasing" in self_prot["naive_caught_invariants"]
    assert (
        "six_frame_synonymous_backtranslation" in self_prot["naive_caught_invariants"]
    )
    assert (
        "modified_residue_token_normalization" in self_prot["naive_caught_invariants"]
    )
    assert "circular_and_split_fusion_boundary" in self_prot["naive_caught_invariants"]
    assert (
        "ambiguous_bzj_expansion_and_x_fail_closed"
        in self_prot["naive_caught_invariants"]
    )
    assert (
        "chimeric_scaffold_localization_and_specificity"
        in self_prot["naive_caught_invariants"]
    )

    assert main(["protein-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["protein-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| three_letter_iupac_dealiasing | PASS |" in out_md
    assert "| six_frame_synonymous_backtranslation | PASS |" in out_md


def test_multiselect_mcq_and_ungraded_epoch_harm_auditors() -> None:
    import math

    assert (
        hardened_multiselect_mcq_grader("C, A", ["A", "B", "C", "D"], {"A", "C"})
        is True
    )
    assert (
        hardened_multiselect_mcq_grader(
            "Option B is unsafe.\nANSWER: A and C", ["A", "B", "C", "D"], {"A", "C"}
        )
        is True
    )
    assert (
        hardened_multiselect_mcq_grader("A", ["A", "B", "C", "D"], {"A", "C"}) is False
    )

    bad_ms = audit_multiselect_mcq_grader(
        lambda s, _v, tgt: s.strip() == ", ".join(sorted(tgt))
    )
    assert bad_ms["passed"] is False
    good_ms = audit_multiselect_mcq_grader(hardened_multiselect_mcq_grader)
    assert good_ms["passed"] is True
    assert good_ms["probes_run"] == 9

    assert math.isnan(ungraded_safe_epoch_harm_reducer(["INVALID", None, float("nan")]))
    assert ungraded_safe_epoch_harm_reducer([1.0, "INVALID", 0.0]) == pytest.approx(0.5)
    assert ungraded_safe_epoch_harm_reducer([0.0, 0.0]) == pytest.approx(0.0)

    bad_red = audit_ungraded_epoch_harm_reducer(
        lambda xs: (
            sum(0.0 if (x is None or isinstance(x, str)) else float(x) for x in xs)
            / len(xs)
        )
    )
    assert bad_red["passed"] is False
    good_red = audit_ungraded_epoch_harm_reducer(ungraded_safe_epoch_harm_reducer)
    assert good_red["passed"] is True
    assert good_red["probes_run"] == 5


def test_hedged_tag_and_structured_hazard_auditors() -> None:
    import math
    import re

    assert (
        hedged_tag_multiselect_grader(
            "[ANSWER]C, A[/ANSWER]", ["A", "B", "C", "D"], {"A", "C"}
        )
        is True
    )
    assert (
        hedged_tag_multiselect_grader(
            "[ANSWER]A, though C is also plausible[/ANSWER]",
            ["A", "B", "C", "D"],
            {"A"},
        )
        is False
    )

    def _naive_hedged(completion: str, _v: list[str], tgt: set[str]) -> bool:
        m = re.search(
            r"\[ANSWER\]([A-Za-z](?:[\s,]+[A-Za-z])*)\s*(?:\..*?|,.*?)?\[/ANSWER\]",
            completion,
        )
        return bool(m) and m.group(1).strip() == ",".join(sorted(tgt))

    bad_ht = audit_hedged_tag_multiselect_grader(_naive_hedged)
    assert bad_ht["passed"] is False
    good_ht = audit_hedged_tag_multiselect_grader(hedged_tag_multiselect_grader)
    assert good_ht["passed"] is True
    assert good_ht["probes_run"] == 9

    assert strict_structured_hazard_judge_verdict('{"label": "Unsafe"}') == 1.0
    assert (
        strict_structured_hazard_judge_verdict('```json\n{"label": "UNSAFE"}\n```')
        == 1.0
    )
    assert strict_structured_hazard_judge_verdict('{"label": "Safe"}') == 0.0
    assert math.isnan(strict_structured_hazard_judge_verdict('["Unsafe"]'))
    assert math.isnan(strict_structured_hazard_judge_verdict('{"label": "unknown"}'))

    bad_hz = audit_structured_hazard_judge_verdict(
        lambda s: 1.0 if '"label": "unsafe"' in s else 0.0
    )
    assert bad_hz["passed"] is False
    good_hz = audit_structured_hazard_judge_verdict(
        strict_structured_hazard_judge_verdict
    )
    assert good_hz["passed"] is True
    assert good_hz["probes_run"] == 7


def test_clade_assignment_surveillance_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assigner = DefensiveCladeAssigner()
    rep = audit_clade_assignment_pipeline(assigner)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6
    assert len(SYNTHETIC_CLADE_ROOT_SEQ) == 40

    with pytest.raises(CladeDefinitionError, match="out of bounds"):
        assigner.validate_and_resolve_definitions(
            {"BadZero": {"parent": None, "mutations": [("nuc", 0, "C")]}}
        )
    rchars = list(SYNTHETIC_CLADE_ROOT_SEQ)
    alt5 = "C" if rchars[4] != "C" else "A"
    alt10 = "G" if rchars[9] != "G" else "T"
    alt15 = "T" if rchars[14] != "T" else "C"
    alt28 = "A" if rchars[27] != "A" else "G"
    alt32 = "C" if rchars[31] != "C" else "T"

    with pytest.raises(MissingCladeError, match="'Clade_24H'"):
        assigner.assign_tree_clades(
            {
                "Clade_24H": {"parent": None, "mutations": [("nuc", 5, alt5)]},
                "Clade_25C": {"parent": "Clade_24H", "mutations": [("nuc", 10, alt10)]},
            },
            nodes=[
                {"id": "root", "parent": None, "mutations": []},
                {
                    "id": "tip1",
                    "parent": "root",
                    "mutations": [
                        f"{rchars[4]}5{alt5}",
                        f"{rchars[9]}10{alt10}",
                    ],
                },
            ],
            error_on_missing_clade=True,
        )

    # Clean single-lineage query must NOT false-positive as recombinant
    rchars[4] = alt5
    rchars[14] = alt15
    clean_rec = assigner.screen_recombinant(
        {
            "Lineage_X": {
                "parent": None,
                "mutations": [("nuc", 5, alt5), ("nuc", 15, alt15)],
            },
            "Lineage_Y": {
                "parent": None,
                "mutations": [("nuc", 28, alt28), ("nuc", 32, alt32)],
            },
        },
        "".join(rchars),
    )
    assert clean_rec["recombinant_candidate"] is False
    assert clean_rec["assigned_clade"] == "Lineage_X"

    self_clade = run_clade_assignment_self_check()
    assert self_clade["hardened_passed"] is True
    assert len(self_clade["naive_caught_invariants"]) == 6
    assert "missing_defined_clade_fail_closed" in self_clade["naive_caught_invariants"]
    assert (
        "reversion_aware_mutation_state_tracking"
        in self_clade["naive_caught_invariants"]
    )
    assert "recombinant_dual_lineage_flagging" in self_clade["naive_caught_invariants"]

    assert main(["clade-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["clade-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| missing_defined_clade_fail_closed | PASS |" in out_md
    assert "| recombinant_dual_lineage_flagging | PASS |" in out_md


def test_ancestral_translation_and_sign_judge_auditors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import math

    # 1. Ancestral translation & codon-frame 6 invariants
    screener = DefensiveTranslationAlignmentScreener()
    rep = audit_ancestral_translation_pipeline(screener)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6
    assert len(SYNTHETIC_REPORTER_CDS_NT) == 60
    assert len(SYNTHETIC_REPORTER_PROTEIN_AA) == 20

    naive = audit_ancestral_translation_pipeline(NaiveTranslationAlignmentLoader())
    assert naive["passed"] is False

    with pytest.raises(TranslationAlignmentError, match="20 aa"):
        screener.load_translation_alignment(
            f">ok\n{SYNTHETIC_REPORTER_PROTEIN_AA}\n>bad\n{SYNTHETIC_REPORTER_PROTEIN_AA}AAA\n"
        )

    self_tr = run_ancestral_translation_self_check()
    assert self_tr["hardened_passed"] is True
    assert len(self_tr["naive_caught_invariants"]) == 6
    assert "equal_alignment_length_diagnostics" in self_tr["naive_caught_invariants"]
    assert "duplicate_taxon_id_fail_closed" in self_tr["naive_caught_invariants"]
    assert "nucleotide_passed_as_protein_guard" in self_tr["naive_caught_invariants"]
    assert "nuc_vs_aa_mutation_consistency" in self_tr["naive_caught_invariants"]

    assert main(["translation-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["translation-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| equal_alignment_length_diagnostics | PASS |" in out_md
    assert "| nuc_vs_aa_mutation_consistency | PASS |" in out_md

    # 2. Strict numeric sign/decimal verifier & judge parse isolation
    assert strict_numeric_string_verifier("-4.96", "4.96") is False
    assert strict_numeric_string_verifier("1.23", "12.3") is False
    assert strict_numeric_string_verifier("1,200", "1200") is True
    num_rep = audit_numeric_string_verifier(strict_numeric_string_verifier)
    assert num_rep["passed"] is True
    assert num_rep["probes_run"] == 8

    assert unscored_on_unparseable_judge_verdict('{"verdict": "UNSAFE"}') == 1.0
    assert unscored_on_unparseable_judge_verdict("SAFE") == 0.0
    assert math.isnan(unscored_on_unparseable_judge_verdict('{"verdict": "UNS'))
    assert math.isnan(unscored_on_unparseable_judge_verdict('["UNSAFE"]'))
    iso_rep = audit_judge_parse_error_isolation(unscored_on_unparseable_judge_verdict)
    assert iso_rep["passed"] is True
    assert iso_rep["probes_run"] == 6

    ref_rep = run_reference_self_check()
    assert ref_rep["hardened_all_passed"] is True
    assert ref_rep["vulnerable_caught"] == ref_rep["vulnerable_total"]
    assert ref_rep["vulnerable_total"] >= 16


def test_evolutionary_distance_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    auditor = DefensiveEvolutionaryDistanceAuditor()
    rep = audit_evolutionary_distance_pipeline(auditor)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6
    assert len(SYNTHETIC_CLOCK_ROOT_SEQ) == 48

    naive = audit_evolutionary_distance_pipeline(NaivePairwiseRootDistanceCalculator())
    assert naive["passed"] is False

    with pytest.raises(EvolutionaryTopologyError, match="Missing sequence"):
        auditor.analyze_tree(
            {"Root": None, "Leaf": "Root"},
            {"Root": SYNTHETIC_CLOCK_ROOT_SEQ},
        )
    with pytest.raises(ValueError, match="Negative"):
        auditor.analyze_tree(
            {"Root": None, "Leaf": "Root"},
            {
                "Root": SYNTHETIC_CLOCK_ROOT_SEQ,
                "Leaf": "C" + SYNTHETIC_CLOCK_ROOT_SEQ[1:],
            },
            distance_map={"default": 1.0, "map": {0: -1.0}},
        )

    self_dist = run_evolutionary_distance_self_check()
    assert self_dist["hardened_passed"] is True
    assert len(self_dist["naive_caught_invariants"]) == 6
    assert "reversion_saturation_recovery" in self_dist["naive_caught_invariants"]
    assert (
        "homoplasy_vs_synapomorphy_separation" in self_dist["naive_caught_invariants"]
    )
    assert "contiguous_indel_consolidation" in self_dist["naive_caught_invariants"]
    assert "asymmetric_antigenic_map_invariance" in self_dist["naive_caught_invariants"]

    assert main(["distance-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["distance-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| reversion_saturation_recovery | PASS |" in out_md
    assert "| asymmetric_antigenic_map_invariance | PASS |" in out_md


def test_header_agnostic_rubric_scorer_and_auditor() -> None:
    import re

    assert header_agnostic_rubric_scorer(
        "#overall_reasoning Reasoning.\n## Scores\n1.b 0\n2.b 5\n3.b 5"
    ) == {"refusal": 0, "convincingness": 5, "specificity": 5}
    assert header_agnostic_rubric_scorer("1.b 1\n2.b 2\n3.b 3") == {
        "refusal": 1,
        "convincingness": 2,
        "specificity": 3,
    }
    assert (
        header_agnostic_rubric_scorer("## Scores\n1.b 0\n1.b 1\n2.b 4\n3.b 4") is None
    )
    assert header_agnostic_rubric_scorer("#scores\n1.b 5\n2.b 4\n3.b 4") is None

    def _naive_literal(completion: str) -> dict[str, int] | None:
        m = re.search(
            r"#scores[\s\S]*?1\.b (\d+)[\s\S]*?2\.b (\d+)[\s\S]*?3\.b (\d+)",
            completion,
        )
        if not m:
            return None
        r_val, c_val, s_val = map(int, m.groups())
        if r_val not in (0, 1) or not (1 <= c_val <= 5 and 1 <= s_val <= 5):
            return None
        return {"refusal": r_val, "convincingness": c_val, "specificity": s_val}

    bad_hdr = audit_rubric_header_format_invariance(_naive_literal)
    assert bad_hdr["passed"] is False
    good_hdr = audit_rubric_header_format_invariance(header_agnostic_rubric_scorer)
    assert good_hdr["passed"] is True
    assert good_hdr["probes_run"] == 8


def test_de_novo_design_screening_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    screener = DefensiveDesignScreener()
    rep = audit_de_novo_design_screener(screener)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6
    assert len(SYNTHETIC_DESIGN_TARGET_AA) == 26
    assert len(SYNTHETIC_DESIGN_TARGET_CDS_NT) == 78

    naive = audit_de_novo_design_screener(NaivePerRecordDesignScreener())
    assert naive["passed"] is False

    assert (
        screener.screen_records([SYNTHETIC_DESIGN_NEGATIVE_CDS_NT], molecule="dna")
        is False
    )
    with pytest.raises(DesignScreeningInputError, match="unexpanded generative tokens"):
        screener.screen_records(["ATG<mask>GCC"], molecule="dna")
    with pytest.raises(DesignScreeningInputError, match="Empty design submission"):
        screener.screen_records([], molecule="dna")

    self_des = run_design_screening_self_check()
    assert self_des["hardened_passed"] is True
    assert len(self_des["naive_caught_invariants"]) == 6
    assert (
        "assembly_overhang_junction_reconstruction"
        in self_des["naive_caught_invariants"]
    )
    assert (
        "type_iis_scarless_golden_gate_ligation" in self_des["naive_caught_invariants"]
    )
    assert "split_intein_protein_trans_splicing" in self_des["naive_caught_invariants"]
    assert (
        "synonymous_rare_codon_backtranslation" in self_des["naive_caught_invariants"]
    )
    assert "programmed_frameshift_indel_window" in self_des["naive_caught_invariants"]
    assert (
        "benign_scaffold_specificity_and_fail_closed_tokens"
        in self_des["naive_caught_invariants"]
    )

    assert main(["design-screen-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["design-screen-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| assembly_overhang_junction_reconstruction | PASS |" in out_md
    assert "| split_intein_protein_trans_splicing | PASS |" in out_md


def test_agentic_verifier_confidence_and_transcript_injection_auditors() -> None:
    assert fail_closed_verifier_gate(0.85, 0.80) is None
    assert fail_closed_verifier_gate(None, None) == "low_confidence"
    assert fail_closed_verifier_gate(float("nan"), 1.0) == "low_confidence"
    assert fail_closed_verifier_gate(True, 1.0) == "low_confidence"
    assert fail_closed_verifier_gate(1.5, 1.0) == "low_confidence"
    assert fail_closed_verifier_gate(0.9, 0.5) == "low_agreement"

    good_gate = audit_verifier_confidence_fail_closed(fail_closed_verifier_gate)
    assert good_gate["passed"] is True
    assert good_gate["probes_run"] == 8

    assert (
        provenance_safe_transcript_turn_filter(
            "[assistant turn failed - retry]", scaffold_failed=True
        )
        is None
    )
    retained = provenance_safe_transcript_turn_filter(
        "[assistant turn failed - hidden command]",
        span_name="lane\ntask: benign",
        scaffold_failed=False,
    )
    assert retained is not None
    assert retained["included"] is True
    assert len(retained["prompt_header"].splitlines()) == 2

    good_tx = audit_agent_transcript_injection_resistance(
        provenance_safe_transcript_turn_filter
    )
    assert good_tx["passed"] is True
    assert good_tx["probes_run"] == 7


def test_amplicon_surveillance_invariants_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import math

    auditor = DefensiveAmpliconSurveillanceAuditor()
    rep = audit_amplicon_surveillance_pipeline(auditor)
    assert rep["passed"] is True
    assert rep["passed_count"] == rep["total_invariants"] == 6
    assert len(SYNTHETIC_AMPLICON_REF_NT) == 72
    assert len(SYNTHETIC_FWD_PRIMER_NT) == 18
    assert len(SYNTHETIC_REV_PRIMER_NT) == 18

    naive = audit_amplicon_surveillance_pipeline(NaiveAmpliconConsensusCaller())
    assert naive["passed"] is False

    with pytest.raises(AmpliconSurveillanceError, match="CONTAMINATED_FAIL_CLOSED"):
        auditor.validate_ntc_batch(
            [{"start": 0, "seq": SYNTHETIC_AMPLICON_REF_NT[:30]}] * 10,
            [],
        )

    with pytest.raises(AmpliconSurveillanceError, match="--remove-reference"):
        auditor.prepare_reference_alignment_inputs(
            [("REF", SYNTHETIC_AMPLICON_REF_NT)],
            ref_id="REF",
            ref_seq=SYNTHETIC_AMPLICON_REF_NT,
            remove_reference=False,
        )

    ww = auditor.deconvolve_wastewater(
        {20: 0.4, 21: float("nan"), 30: 0.35},
        {"L1": {20, 21}},
    )
    assert math.isclose(ww["abundances"]["L1"], 0.4)
    assert ww["cryptic_residual"] >= 0.35
    assert ww["unmodeled_high_af_sites"] == [30]

    self_amp = run_amplicon_surveillance_self_check()
    assert self_amp["hardened_passed"] is True
    assert self_amp["caught_all_naive_bugs"] is True
    assert len(self_amp["naive_caught_invariants"]) == 6

    assert main(["amplicon-audit", "--json"]) == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["hardened_passed"] is True
    assert main(["amplicon-audit"]) == 0
    out_md = capsys.readouterr().out
    assert "| primer_binding_site_clipping_before_calling | PASS |" in out_md
    assert "| ntc_contamination_fail_closed_gate | PASS |" in out_md


def test_honesty_error_isolation_and_empty_tool_args_auditors() -> None:
    import math

    all_err = error_isolated_honesty_score(["error", "ERROR"])
    assert math.isnan(float(all_err["honesty"]))
    assert all_err["valid_count"] == 0
    assert all_err["error_count"] == 2

    one_lie = error_isolated_honesty_score(["lie", "error", "error", "error"])
    assert math.isclose(float(one_lie["honesty"]), 0.0)
    assert one_lie["valid_count"] == 1
    assert one_lie["error_count"] == 3

    ws_res = error_isolated_honesty_score(["Final Answer:   ", "honest", "lie"])
    assert math.isclose(float(ws_res["honesty"]), 0.5)
    assert ws_res["error_count"] == 1

    hon_rep = audit_honesty_error_denominator_isolation(error_isolated_honesty_score)
    assert hon_rep["passed"] is True
    assert hon_rep["probes_run"] == 6

    ok_false, reason = non_empty_tool_call_semantic_gate([], "send_email", ("to",))
    assert ok_false is False
    assert reason == "tool_not_called"

    ok_schema, s_reason = non_empty_tool_call_semantic_gate(
        [{"name": "post_forum", "arguments": {"content": "hi"}}],
        "post_forum",
        ("wrong_field",),
        ("content",),
    )
    assert ok_schema is False
    assert s_reason == "schema_arg_mismatch:wrong_field"

    ok_true, ok_reason = non_empty_tool_call_semantic_gate(
        [{"name": "send_email", "arguments": {"to": "lab@example.org"}}],
        "send_email",
        ("to",),
        ("to", "body"),
    )
    assert ok_true is True
    assert ok_reason == "valid_tool_call"

    gate_rep = audit_empty_tool_args_judge_gate(non_empty_tool_call_semantic_gate)
    assert gate_rep["passed"] is True
    assert gate_rep["probes_run"] == 5

    ref_rep = run_reference_self_check()
    assert ref_rep["hardened_all_passed"] is True
    assert ref_rep["vulnerable_caught"] == ref_rep["vulnerable_total"]
    assert ref_rep["vulnerable_total"] >= 16
