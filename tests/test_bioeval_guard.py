"""Comprehensive test suite for bioeval_guard."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bioeval_guard.cli import format_markdown_summary, main
from bioeval_guard.grader_audit import (
    audit_answer_tag_parser,
    audit_dna_screener_symmetry,
    audit_numeric_string_verifier,
    audit_safety_judge_callable,
    audit_selective_precision_metric,
    case_safe_judge_label,
    nan_safe_selective_precision,
    parse_tagged_answer_strict,
    run_reference_self_check,
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
    assert rep["eval_mode_counts"] == {"str_verifier": 1, "range_verifier": 1, "llm_verifier": 1}


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
    jpath.write_text('[{"id":"1","question":"Q","ideal":"A","distractors":["B"]}]', encoding="utf-8")
    assert len(load_records(jpath)) == 1

    pq_path = tmp_path / "data.parquet"
    table = pa.table({"question": ["Q"], "choices": [["A", "B", "C", "D"]], "answer": [0]})
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
    rel, mean_sims, char_lens = compute_option_features(["AAAA", "AAAB", "AAAC", "ZZZZZZZZ"])
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
        items.append(AuditItem(item_id=f"s{i}", question="Which?", options=[ideal] + dists, ideal_idx=0))

    res = audit_question_blind_shortcuts(items)
    assert res["centroid"]["acc"] == 1.0
    assert res["oof_5fold"]["acc"] >= 0.95
    assert res["centroid"]["p_raw"] < 1e-10


def test_longest_option_shortcut() -> None:
    items = [
        AuditItem(
            item_id=f"l{i}",
            question="Q",
            options=["Very detailed mechanistic biological explanation here", "Short A", "Short B", "Short C"],
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
        items.append(AuditItem(item_id=f"c{i}", question="Q", options=opts, ideal_idx=0))
    res = audit_question_blind_shortcuts(items)
    assert res["centroid"]["p_raw"] > 0.05
    assert res["longest"]["p_raw"] > 0.05


def test_empty_shortcut_audit_returns_zero() -> None:
    assert audit_question_blind_shortcuts([]) == {"N": 0, "chance": 0.0}


def test_inspect_adapter_sanitize_and_baseline() -> None:
    records = [
        {"id": "bad", "question": "Q", "ideal": "null", "distractors": ["A", "B", "C"]},
        {"id": "good", "question": "Q", "ideal": "Valid", "distractors": ["D1", "D2", "D3"]},
    ]
    clean, rep = sanitize_mcq_dataset(records)
    assert rep["removed_items"] == 1
    assert [r["id"] for r in clean] == ["good"]

    pred = question_blind_centroid_baseline(["ZZZ_DIFF", "AAACCC1", "AAACCC2", "AAACCC3"])
    assert pred in {1, 2, 3}
    with pytest.raises(ValueError, match="non-empty"):
        question_blind_centroid_baseline([])

    keep = make_inspect_sample_filter()
    norm = normalize_items(records)
    assert keep(norm[0]) is False
    assert keep(norm[1]) is True


def test_cli_audit_jsonl_json_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "sample.jsonl"
    p.write_text(
        '{"id":"1","question":"Q","ideal":"Alpha","distractors":["Beta","Gamma","Delta"]}\n'
        '{"id":"2","question":"Q","ideal":"null","distractors":["B","C","D"]}\n',
        encoding="utf-8",
    )
    pq_path = tmp_path / "sample_wmdp.parquet"
    pq.write_table(
        pa.table({
            "question": [f"Q{i}" for i in range(12)],
            "choices": [["A1", "B2", "C3", "D4"] for _ in range(12)],
            "answer": [0 for _ in range(12)],
        }),
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
    bad_judge = audit_safety_judge_callable(lambda s: "unsafe" if '"label": "unsafe"' in s else "safe")
    assert bad_judge["passed"] is False
    assert len(bad_judge["failures"]) == 5
    good_judge = audit_safety_judge_callable(case_safe_judge_label)
    assert good_judge["passed"] is True

    # 2. Tag parser auditor
    bad_parser = audit_answer_tag_parser(lambda s: s[s.find("<answer>") + 8 : s.find("</answer>")])
    assert bad_parser["passed"] is False
    good_parser = audit_answer_tag_parser(lambda s: parse_tagged_answer_strict(s, strip_upper=True))
    assert good_parser["passed"] is True

    # 3. Numeric string verifier auditor
    import re

    bad_num = audit_numeric_string_verifier(
        lambda t, p: re.sub(r"[^a-zA-Z0-9]", "", t).lower() == re.sub(r"[^a-zA-Z0-9]", "", p).lower()
    )
    assert bad_num["passed"] is False
    good_num = audit_numeric_string_verifier(lambda t, p: t.strip() == p.strip() and t.strip() != "")
    assert good_num["passed"] is True

    # 4. Selective precision metric auditor
    bad_metric = audit_selective_precision_metric(
        lambda xs: 0.0 if not [x for x in xs if x != -1] else sum(x for x in xs if x != -1) / len([x for x in xs if x != -1]),
        lambda xs: len([x for x in xs if x != -1]) / len(xs) if xs else 0.0,
    )
    assert bad_metric["passed"] is False
    good_metric = audit_selective_precision_metric(
        lambda xs: nan_safe_selective_precision([float(x) for x in xs if x != -1], len(xs))[0],
        lambda xs: nan_safe_selective_precision([float(x) for x in xs if x != -1], len(xs))[1],
    )
    assert good_metric["passed"] is True

    # 5. Defensive benign oligo symmetry auditor
    asymmetric = audit_dna_screener_symmetry(lambda s: "CTAGCGATGCTAGCTAGTCG" in s)
    assert asymmetric["passed"] is False


def test_inspect_tasks_helpers_and_self_check_cli(capsys: pytest.CaptureFixture[str]) -> None:
    import math

    assert LAB_BENCH_REVISION_PIN() == "5c77cec648430f30611808808861eb86f81d5eaa"
    prec, cov = compute_nan_safe_precision_and_coverage(["N", "N", "N"])
    assert math.isnan(prec) and cov == 0.0
    fallback = nan_safe_precision_metric()
    assert math.isnan(fallback([-1, -1]))

    kept, rep = filter_clean_lab_bench_records(
        [
            {"id": "k1", "question": "Q1", "ideal": "null", "distractors": ["A", "B", "C"]},
            {"id": "k2", "question": "Q2", "ideal": "BRCA1", "distractors": ["TP53", "EGFR", "MYC"]},
        ],
        subset_name="LitQA2",
        check_biology=False,
    )
    assert [r["id"] for r in kept] == ["k2"]
    assert rep["removed_items"] == 1

    self_rep = run_reference_self_check()
    assert self_rep["hardened_all_passed"] is True
    assert self_rep["vulnerable_caught"] == self_rep["vulnerable_total"] == 3

    assert main(["self-check", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardened_all_passed"] is True
    assert main(["self-check"]) == 0
    assert "| dna_screener_symmetry |" in capsys.readouterr().out
