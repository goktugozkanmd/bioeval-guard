"""Deterministic zero-question option-geometry and 5-fold OOF shortcut auditing."""

from __future__ import annotations

import math
import random
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .hygiene import DEC_NUM_RE, META_OPTION_RE, SCI_NUM_RE, AuditItem


def _sim_3gram(a: str, b: str) -> float:
    a = a.strip().lower()
    b = b.strip().lower()
    if a == b:
        return 1.0
    if len(a) < 3 or len(b) < 3:
        return SequenceMatcher(None, a, b).ratio()
    sa = Counter(a[i : i + 3] for i in range(len(a) - 2))
    sb = Counter(b[i : i + 3] for i in range(len(b) - 2))
    inter = sum((sa & sb).values())
    total = sum(sa.values()) + sum(sb.values())
    return (2.0 * inter / total) if total else 0.0


def compute_option_features(options: list[str]) -> tuple[np.ndarray, list[float], list[int]]:
    """Compute strictly within-item relative option features without accessing the question."""
    k = len(options)
    sim_mat = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            if len(options[i]) <= 120 and len(options[j]) <= 120:
                sim = SequenceMatcher(None, options[i].lower(), options[j].lower()).ratio()
            else:
                sim = _sim_3gram(options[i], options[j])
            sim_mat[i][j] = sim_mat[j][i] = sim

    raw_rows: list[list[float]] = []
    mean_sims: list[float] = []
    char_lens: list[int] = []
    for i, opt in enumerate(options):
        s = opt.strip()
        clen = len(s)
        wcnt = len(s.split())
        digits = sum(c.isdigit() for c in s) / max(1, clen)
        upper_punct = sum(c.isupper() or (not c.isalnum() and not c.isspace()) for c in s) / max(1, clen)
        dna_chars = sum(c in "ACGTacgt" for c in s)
        gc = (s.upper().count("G") + s.upper().count("C")) / dna_chars if dna_chars >= 8 else 0.0
        others = [sim_mat[i][j] for j in range(k) if j != i]
        mean_sim = sum(others) / max(1, len(others))
        max_sim = max(others) if others else 0.0
        meta = 1.0 if META_OPTION_RE.search(s) else 0.0
        has_sci = 1.0 if SCI_NUM_RE.match(s) else 0.0
        has_dec = 1.0 if DEC_NUM_RE.match(s) else 0.0
        raw_rows.append([clen, wcnt, digits, upper_punct, gc, mean_sim, max_sim, meta, has_sci, has_dec])
        mean_sims.append(mean_sim)
        char_lens.append(clen)

    arr = np.array(raw_rows, dtype=float)
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    rel = (arr - mu) / sd
    return rel, mean_sims, char_lens


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Adjust p-values using Holm-Bonferroni step-down procedure."""
    m = len(pvals)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda pair: pair[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def _binomial_summary(hits: int, total: int, chance: float) -> dict[str, Any]:
    acc = hits / total if total else 0.0
    se = math.sqrt(chance * (1.0 - chance) / total) if total and 0.0 < chance < 1.0 else 0.0
    z = (acc - chance) / se if se > 0 else 0.0
    p_raw = float(binomtest(hits, total, chance, alternative="two-sided").pvalue) if total else 1.0
    return {
        "hits": int(hits),
        "N": int(total),
        "acc": round(acc, 4),
        "chance": round(chance, 4),
        "lift_pp": round((acc - chance) * 100.0, 2),
        "z": round(z, 2),
        "p_raw": p_raw,
    }


def audit_question_blind_shortcuts(items: list[AuditItem], seed: int = 42, n_folds: int = 5) -> dict[str, Any]:
    """Measure question-blind single-rule heuristics and 5-fold stratified OOF exploitability."""
    if not items:
        return {"N": 0, "chance": 0.0}

    rng = random.Random(seed)
    prepared: list[dict[str, Any]] = []
    for it in items:
        indexed = list(enumerate(it.options))
        rng.shuffle(indexed)
        shuf_opts = [opt for _, opt in indexed]
        shuf_target = next(pos for pos, (orig_idx, _) in enumerate(indexed) if orig_idx == it.ideal_idx)
        rel_feats, mean_sims, char_lens = compute_option_features(shuf_opts)
        prepared.append({
            "subtask": it.subtask,
            "k": len(shuf_opts),
            "target": shuf_target,
            "mean_sim": mean_sims,
            "char_lens": char_lens,
            "rel_feats": rel_feats,
        })

    n = len(prepared)
    chance = sum(1.0 / row["k"] for row in prepared) / n
    centroid_hits = sum(1 for row in prepared if int(np.argmax(row["mean_sim"])) == row["target"])
    longest_hits = sum(1 for row in prepared if int(np.argmax(row["char_lens"])) == row["target"])
    shortest_hits = sum(1 for row in prepared if int(np.argmin(row["char_lens"])) == row["target"])

    strat_labels = [row["subtask"] for row in prepared]
    counts = Counter(strat_labels)
    strat_labels = [lbl if counts[lbl] >= n_folds else "OTHER" for lbl in strat_labels]
    if not strat_labels or Counter(strat_labels).most_common()[-1][1] < n_folds:
        strat_labels = ["ALL"] * n

    effective_folds = min(n_folds, n)
    oof_stats: dict[str, Any]
    if effective_folds >= 2 and n >= 10:
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        oof_hits = 0
        for tr_idx, te_idx in skf.split(np.zeros(n), strat_labels):
            x_tr_rows: list[np.ndarray] = []
            y_tr_rows: list[int] = []
            for idx in tr_idx:
                row = prepared[idx]
                for opt_j in range(row["k"]):
                    x_tr_rows.append(row["rel_feats"][opt_j])
                    y_tr_rows.append(1 if opt_j == row["target"] else 0)
            x_tr = np.vstack(x_tr_rows)
            y_tr = np.array(y_tr_rows)
            scaler = StandardScaler()
            x_tr_scaled = scaler.fit_transform(x_tr)
            clf = LogisticRegression(C=0.25, max_iter=500, random_state=seed)
            clf.fit(x_tr_scaled, y_tr)
            for idx in te_idx:
                row = prepared[idx]
                x_te_scaled = scaler.transform(row["rel_feats"])
                scores = clf.decision_function(x_te_scaled)
                if int(np.argmax(scores)) == row["target"]:
                    oof_hits += 1
        oof_stats = _binomial_summary(oof_hits, n, chance)
    else:
        oof_stats = _binomial_summary(0, 0, chance)

    return {
        "N": n,
        "chance": round(chance, 4),
        "centroid": _binomial_summary(centroid_hits, n, chance),
        "longest": _binomial_summary(longest_hits, n, chance),
        "shortest": _binomial_summary(shortest_hits, n, chance),
        "oof_5fold": oof_stats,
    }
