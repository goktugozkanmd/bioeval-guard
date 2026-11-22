"""Comprehensive test suite for bioeval_guard."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bioeval_guard.cli import format_markdown_summary, main
from bioeval_guard.cluster_stats import audit_clustered_benchmark_se
from bioeval_guard.grader_audit import (
    audit_answer_tag_parser,
    audit_dna_screener_symmetry,
    audit_mcq_option_text_grader,
    audit_numeric_string_verifier,
    audit_safety_judge_callable,
    audit_selective_precision_metric,
    audit_truncated_empty_completion_handling,
    audit_unparseable_judge_nan_handling,
    case_safe_judge_label,
    hardened_mcq_option_grader,
    nan_safe_selective_precision,
    parse_tagged_answer_strict,
    run_reference_self_check,
    truncation_safe_answer_gate,
    unparseable_safe_nan_judge_score,
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
    good_num = audit_numeric_string_verifier(
        lambda t, p: t.strip() == p.strip() and t.strip() != ""
    )
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
    assert self_rep["vulnerable_caught"] == self_rep["vulnerable_total"] == 6

    assert main(["self-check", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardened_all_passed"] is True
    assert main(["self-check"]) == 0
    out_md = capsys.readouterr().out
    assert "| mcq_option_text_grader |" in out_md
    assert "| unparseable_judge_nan_handling |" in out_md
    assert "| truncated_empty_completion_handling |" in out_md


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
