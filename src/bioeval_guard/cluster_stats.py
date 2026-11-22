"""Cluster-robust standard error and effective sample size auditing for AIxBio evaluations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from scipy.stats import t as student_t


def audit_clustered_benchmark_se(
    scores: Sequence[float | int | bool],
    cluster_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute naive vs. cluster-robust standard errors, ICC, and design effect.

    Many frontier biology and biosecurity benchmarks (e.g. BixBench capsules,
    LAB-Bench FigQA/SuppQA/TableQA papers, and multi-primer SeqQA sets) contain
    multiple correlated questions sharing an underlying artifact or stimulus.
    Treating items as i.i.d. inflates statistical precision and produces false
    significance when comparing frontier models.

    Uses the cluster-total residual variance estimator (Liang-Zeger sandwich
    with small-sample `K / (K - 1)` correction), which reduces identically to the
    ordinary sample mean SE when every cluster has size 1.
    """
    if len(scores) != len(cluster_ids):
        raise ValueError(
            f"scores and cluster_ids must have equal length ({len(scores)} != {len(cluster_ids)})"
        )
    n_items = len(scores)
    if n_items == 0:
        raise ValueError("scores must be non-empty")

    vals = [float(x) for x in scores]
    clusters: dict[str, list[float]] = defaultdict(list)
    for val, cid in zip(vals, cluster_ids):
        clusters[str(cid)].append(val)

    k_clusters = len(clusters)
    mean_val = sum(vals) / n_items

    if n_items > 1:
        ss_tot = sum((v - mean_val) ** 2 for v in vals)
        naive_var = (ss_tot / (n_items - 1)) / n_items
        naive_se = math.sqrt(naive_var)
    else:
        ss_tot = 0.0
        naive_se = 0.0

    cluster_sizes = [len(g) for g in clusters.values()]
    max_cluster_size = max(cluster_sizes)
    mean_cluster_size = n_items / k_clusters

    if k_clusters >= 2:
        dof_scale = k_clusters / (k_clusters - 1)
        sum_sq_cluster_resid = sum(
            (sum(g) - len(g) * mean_val) ** 2 for g in clusters.values()
        )
        cluster_var = dof_scale * sum_sq_cluster_resid / (n_items**2)
        cluster_se = math.sqrt(cluster_var)
    else:
        cluster_se = float("nan")

    # One-way random-effects ANOVA intraclass correlation (ICC)
    if k_clusters >= 2 and n_items > k_clusters:
        ss_between = sum(
            len(g) * ((sum(g) / len(g)) - mean_val) ** 2 for g in clusters.values()
        )
        ss_within = sum(
            (x - (sum(g) / len(g))) ** 2 for g in clusters.values() for x in g
        )
        ms_between = ss_between / (k_clusters - 1)
        ms_within = ss_within / (n_items - k_clusters)
        sum_sq_sizes = sum(sz**2 for sz in cluster_sizes)
        n0 = (n_items - (sum_sq_sizes / n_items)) / (k_clusters - 1)
        denom = ms_between + (n0 - 1.0) * ms_within
        icc_raw = ((ms_between - ms_within) / denom) if denom > 1e-15 else 0.0
        icc = max(0.0, min(1.0, icc_raw))
        m_eff = sum_sq_sizes / n_items
        design_effect_kish = 1.0 + max(0.0, m_eff - 1.0) * icc
    else:
        icc_raw = 0.0
        icc = 0.0
        design_effect_kish = 1.0

    if not math.isnan(cluster_se) and naive_se > 1e-15:
        se_inflation = cluster_se / naive_se
        variance_inflation = se_inflation**2
        effective_n_cluster_se = n_items / max(1e-12, variance_inflation)
    else:
        se_inflation = 1.0
        variance_inflation = 1.0
        effective_n_cluster_se = float(n_items)

    effective_n_kish = n_items / max(1.0, design_effect_kish)

    z_95 = 1.959963984540054
    naive_ci95 = (mean_val - z_95 * naive_se, mean_val + z_95 * naive_se)
    if k_clusters >= 2 and not math.isnan(cluster_se):
        t_crit = float(student_t.ppf(0.975, df=k_clusters - 1))
        cluster_ci95 = (
            mean_val - t_crit * cluster_se,
            mean_val + t_crit * cluster_se,
        )
    else:
        cluster_ci95 = (float("nan"), float("nan"))

    return {
        "n_items": n_items,
        "n_clusters": k_clusters,
        "mean_cluster_size": round(mean_cluster_size, 4),
        "max_cluster_size": max_cluster_size,
        "mean": round(mean_val, 6),
        "naive_se": round(naive_se, 6),
        "cluster_se": None if math.isnan(cluster_se) else round(cluster_se, 6),
        "se_inflation_factor": round(se_inflation, 4),
        "variance_inflation_ratio": round(variance_inflation, 4),
        "icc_anova": round(icc, 4),
        "icc_raw": round(icc_raw, 4),
        "design_effect_kish": round(design_effect_kish, 4),
        "effective_n_kish": round(effective_n_kish, 2),
        "effective_n_cluster_se": round(effective_n_cluster_se, 2),
        "naive_ci95": [round(naive_ci95[0], 6), round(naive_ci95[1], 6)],
        "cluster_ci95": (
            None
            if math.isnan(cluster_ci95[0])
            else [round(cluster_ci95[0], 6), round(cluster_ci95[1], 6)]
        ),
    }
