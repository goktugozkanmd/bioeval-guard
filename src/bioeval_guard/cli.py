"""Command-line interface for bioeval-guard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cluster_stats import audit_clustered_benchmark_se
from .grader_audit import run_reference_self_check
from .hygiene import (
    check_bixbench_hygiene,
    check_mcq_hygiene,
    load_records,
    normalize_items,
)
from .metagenomic_surveillance_audit import run_surveillance_self_check
from .sequence_screening_audit import run_screening_self_check
from .shortcuts import audit_question_blind_shortcuts, holm_bonferroni


def audit_file(
    path: Path, name: str | None = None, check_biology: bool = True
) -> dict[str, Any]:
    """Audit a single Parquet, JSONL, or JSON evaluation dataset file."""
    ds_name = name or path.stem
    records = load_records(path)
    items = normalize_items(records, dataset_name=ds_name)
    hygiene = check_mcq_hygiene(items, check_biology=check_biology)
    if any("eval_mode" in rec for rec in records):
        hygiene["bixbench"] = check_bixbench_hygiene(items)
    shortcuts = audit_question_blind_shortcuts(items)
    return {
        "dataset": ds_name,
        "path": str(path),
        "hygiene": hygiene,
        "shortcuts": shortcuts,
    }


def format_markdown_summary(results: list[dict[str, Any]]) -> str:
    """Format multi-dataset audit results as a Markdown table with Holm-adjusted p-values."""
    if not results:
        return "No datasets audited."
    for method in ("centroid", "longest", "oof_5fold"):
        raw_ps = [res["shortcuts"].get(method, {}).get("p_raw", 1.0) for res in results]
        adj_ps = holm_bonferroni(raw_ps)
        for res, ap in zip(results, adj_ps):
            if method in res["shortcuts"]:
                res["shortcuts"][method]["p_holm"] = ap

    lines = [
        "| Dataset | N | Flagged | Chance | Centroid | Longest | 5-Fold OOF | Min p_holm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for res in results:
        h = res["hygiene"]
        s = res["shortcuts"]
        p_min = min(
            s.get(m, {}).get("p_holm", 1.0)
            for m in ("centroid", "longest", "oof_5fold")
        )
        lines.append(
            f"| {res['dataset']} | {h['total_items']} | {h['flagged_items']} | "
            f"{s['chance'] * 100:.1f}% | {s['centroid']['acc'] * 100:.1f}% | "
            f"{s['longest']['acc'] * 100:.1f}% | {s['oof_5fold']['acc'] * 100:.1f}% | {p_min:.2e} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioeval-guard", description="Audit AIxBio benchmarks."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("audit-parquet", "audit-jsonl"):
        p = sub.add_parser(cmd, help="Audit one dataset file.")
        p.add_argument("path", type=Path, help="Path to dataset file.")
        p.add_argument("--name", type=str, default=None, help="Display name.")
        p.add_argument("--no-bio", action="store_true", help="Skip DNA/primer checks.")
        p.add_argument(
            "--json", action="store_true", help="Emit JSON instead of Markdown."
        )

    ps = sub.add_parser(
        "summary", help="Audit multiple dataset files and emit Markdown table."
    )
    ps.add_argument("paths", nargs="+", type=Path, help="Paths to Parquet/JSONL files.")
    ps.add_argument(
        "--json-out", type=Path, default=None, help="Optional JSON output path."
    )

    psc = sub.add_parser(
        "self-check",
        help="Verify reference hardened vs vulnerable grader/screener checks.",
    )
    psc.add_argument("--json", action="store_true", help="Emit JSON output.")

    pca = sub.add_parser(
        "cluster-audit",
        help="Compute cluster-robust SE, ICC, DEFF, and effective N on grouped eval records.",
    )
    pca.add_argument(
        "path", type=Path, help="Path to scored JSON, JSONL, or Parquet file."
    )
    pca.add_argument(
        "--score-field", type=str, default="score", help="Numeric score field name."
    )
    pca.add_argument(
        "--cluster-field",
        type=str,
        default="cluster_id",
        help="Cluster identifier field name.",
    )
    pca.add_argument(
        "--json", action="store_true", help="Emit JSON instead of Markdown."
    )

    psa = sub.add_parser(
        "screen-audit",
        help="Audit defensive DNA/RNA screening invariants on benign synthetic reporter oligos.",
    )
    psa.add_argument("--json", action="store_true", help="Emit JSON output.")

    psv = sub.add_parser(
        "surveillance-audit",
        help="Audit pathogen metagenomic surveillance invariants on benign synthetic controls.",
    )
    psv.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser


def format_cluster_markdown(report: dict[str, Any]) -> str:
    """Format cluster-robust audit metrics as a single-row Markdown table."""
    lines = [
        "| N | Clusters | Mean | Naive SE | Cluster SE | SE Inflation | ICC | DEFF | N_eff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {report['n_items']} | {report['n_clusters']} | {report['mean']:.4f} | "
        f"{report['naive_se']:.4f} | {report['cluster_se'] if report['cluster_se'] is not None else 'NaN'} | "
        f"{report['se_inflation_factor']:.2f}x | {report['icc_anova']:.3f} | "
        f"{report['design_effect_kish']:.2f} | {report['effective_n_cluster_se']:.1f} |",
    ]
    return "\n".join(lines)


def format_self_check_markdown(report: dict[str, Any]) -> str:
    """Format self-check auditor summary as Markdown."""
    lines = [
        "| Auditor | Probes | Passed | Status |",
        "|---|---:|---:|---|",
    ]
    for row in report["hardened"]:
        lines.append(
            f"| {row['auditor']} | {row['probes_run']} | {'PASS' if row['passed'] else 'FAIL'} | hardened |"
        )
    lines.append(
        f"| vulnerable_regression_probes | {report['vulnerable_total']} | "
        f"{report['vulnerable_caught']}/{report['vulnerable_total']} caught | PASS |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in ("audit-parquet", "audit-jsonl"):
        res = audit_file(args.path, name=args.name, check_biology=not args.no_bio)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(format_markdown_summary([res]))
        return 0

    if args.command == "summary":
        results = [audit_file(p) for p in args.paths]
        print(format_markdown_summary(results))
        if args.json_out:
            args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return 0

    if args.command == "self-check":
        report = run_reference_self_check()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(format_self_check_markdown(report))
        return 0 if report["hardened_all_passed"] else 1

    if args.command == "cluster-audit":
        records = load_records(args.path)
        scores = [float(rec[args.score_field]) for rec in records]
        clusters = [str(rec[args.cluster_field]) for rec in records]
        report = audit_clustered_benchmark_se(scores, clusters)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(format_cluster_markdown(report))
        return 0

    if args.command == "screen-audit":
        report = run_screening_self_check()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            lines = ["| Invariant | Status | Detail |", "|---|---|---|"]
            for inv in report["hardened"]["invariants"]:
                lines.append(
                    f"| {inv['invariant']} | {'PASS' if inv['passed'] else 'FAIL'} | {inv['detail']} |"
                )
            lines.append(
                f"| naive_screener_caught | {len(report['naive_caught_invariants'])}/{report['naive_total_invariants']} | "
                f"{', '.join(report['naive_caught_invariants'])} |"
            )
            print("\n".join(lines))
        return 0 if report["hardened_passed"] else 1

    if args.command == "surveillance-audit":
        report = run_surveillance_self_check()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            lines = ["| Invariant | Status | Detail |", "|---|---|---|"]
            for inv in report["hardened"]["invariants"]:
                lines.append(
                    f"| {inv['invariant']} | {'PASS' if inv['passed'] else 'FAIL'} | {inv['detail']} |"
                )
            lines.append(
                f"| naive_mgs_caught | {len(report['naive_caught_invariants'])}/{report['naive_total_invariants']} | "
                f"{', '.join(report['naive_caught_invariants'])} |"
            )
            print("\n".join(lines))
        return 0 if report["hardened_passed"] else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
