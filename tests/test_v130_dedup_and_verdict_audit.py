"""Tests for bioeval-guard v1.3.0 read dedup/index-hopping and judge guards."""

from __future__ import annotations

import json
import math

import pytest

from bioeval_guard import (
    SYNTHETIC_HOST_RIBO_NT,
    SYNTHETIC_MS2_SPIKE_R1_NT,
    SYNTHETIC_MS2_SPIKE_R2_NT,
    DefensiveReadDedupAuditor,
    NaiveReadDedupCaller,
    ReadDedupIntegrityError,
    audit_last_line_research_verdict_parser,
    audit_read_dedup_hopping_pipeline,
    audit_string_boolean_judge_coercion,
    last_line_verdict_research_scorer,
    run_read_dedup_self_check,
    run_reference_self_check,
    strict_boolean_autopatch_scorer,
)
from bioeval_guard.cli import main as cli_main


def test_read_dedup_hopping_6_invariants_pass_and_catch_naive() -> None:
    report = run_read_dedup_self_check()
    assert report["hardened_passed"] is True
    assert report["hardened"]["passed_count"] == 6
    assert report["hardened"]["total_invariants"] == 6
    assert len(report["naive_caught_invariants"]) == 6


def test_read_dedup_individual_controls_and_fail_closed() -> None:
    auditor = DefensiveReadDedupAuditor()
    naive = NaiveReadDedupCaller()

    # Phred Q10 + Q40 in probability space is ~13.01, never arithmetic 25.0
    q_prob = auditor.summarize_phred_quality([10, 40])
    assert 12.9 <= q_prob <= 13.1
    assert naive.summarize_phred_quality([10, 40]) == 25.0

    # Desynced mate IDs fail closed
    with pytest.raises(ReadDedupIntegrityError):
        auditor.deduplicate_pairs(
            [
                {
                    "r1_id": "M1/1",
                    "r2_id": "M2/2",
                    "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
                    "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
                }
            ]
        )

    # Quality length mismatch fails closed
    with pytest.raises(ReadDedupIntegrityError):
        auditor.deduplicate_pairs(
            [
                {
                    "r1_id": "M1/1",
                    "r2_id": "M1/2",
                    "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
                    "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
                    "q1": "IIII",
                }
            ]
        )

    # Index hopping quarantine isolates single-shared-index bleed vs dual-unique read
    hop_res = auditor.screen_index_hopping(
        [
            {
                "r1_id": "HOP/1",
                "r2_id": "HOP/2",
                "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
                "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
                "umi": "AAAACCCC",
                "i7": "SHARED7",
                "i5": "UNIQUE5A",
                "count": 1,
            },
            {
                "r1_id": "TRUE/1",
                "r2_id": "TRUE/2",
                "r1": SYNTHETIC_HOST_RIBO_NT,
                "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
                "umi": "TTTTGGGG",
                "i7": "UNIQUE7B",
                "i5": "UNIQUE5B",
                "count": 2,
            },
        ],
        [
            {
                "r1": SYNTHETIC_MS2_SPIKE_R1_NT,
                "r2": SYNTHETIC_MS2_SPIKE_R2_NT,
                "umi": "AAAACCCC",
                "i7": "SHARED7",
                "i5": "COLIB5",
                "count": 10000,
            }
        ],
    )
    assert [r["r1_id"] for r in hop_res["quarantined_reads"]] == ["HOP/1"]
    assert [r["r1_id"] for r in hop_res["clean_reads"]] == ["TRUE/1"]


def test_last_line_research_verdict_and_boolean_autopatch_auditors() -> None:
    assert (
        last_line_verdict_research_scorer(
            "Initial draft VERDICT: 2.0\nRevised calculation\n**VERDICT: 8.5**"
        )
        == 0.85
    )
    assert last_line_verdict_research_scorer("Missing verdict tag") is None
    assert last_line_verdict_research_scorer("VERDICT: 1.0.1") is None
    assert last_line_verdict_research_scorer("> VERDICT: 7.0") is None

    v_audit = audit_last_line_research_verdict_parser(last_line_verdict_research_scorer)
    assert v_audit["passed"] is True

    parsed = strict_boolean_autopatch_scorer(
        '```json\n{"patched": "false", "preserved": true}\n```'
    )
    assert parsed == {"patched": False, "preserved": True}
    assert strict_boolean_autopatch_scorer("Unparseable judge text") is None

    b_audit = audit_string_boolean_judge_coercion(strict_boolean_autopatch_scorer)
    assert b_audit["passed"] is True

    ref = run_reference_self_check()
    assert ref["hardened_all_passed"] is True
    assert len(ref["hardened"]) == 20
    assert ref["vulnerable_caught"] == ref["vulnerable_total"] == 18


def test_cli_dedup_audit_and_self_check_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli_main(["dedup-audit", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardened_passed"] is True
    assert len(payload["naive_caught_invariants"]) == 6

    rc2 = cli_main(["self-check", "--json"])
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["hardened_all_passed"] is True
    assert len(payload2["hardened"]) == 20
    assert payload2["vulnerable_caught"] == 18

    # Sanity check unused import silence: direct audit call works cleanly
    direct = audit_read_dedup_hopping_pipeline(DefensiveReadDedupAuditor())
    assert not math.isnan(direct["passed_count"]) and direct["all_passed"] is True
