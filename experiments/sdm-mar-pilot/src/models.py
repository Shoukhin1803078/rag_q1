"""Model A (hop count), Model B (retrieval-score), Model C (structural) with
grouped cross-validation and clustered statistics (Section 12.5 of the proposal).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "A": ["H"],
    "B": ["top_score", "margin", "gold_rank"],
    "C": ["H", "CSL", "MS", "T->I", "I->T", "I->I", "n_images", "HxCSL", "RC"],
}


def build_feature_table(questions, retriever_docs: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for q in questions:
        d = {
            "qid": q.qid,
            "H": q.hops.__len__(),
            "CSL": q.CSL,
            "MS": q.MS,
            "n_images": q.n_images,
            "RC": 0.0,
            "HxCSL": q.hops.__len__() * q.CSL,
            "question_words": len(q.text.split()),
            **q.edge_types,
            **retriever_docs.get(q.qid, {}),
        }
        rows.append(d)
    df = pd.DataFrame(rows)
    for c in ["top_score", "margin", "gold_rank"]:
        if c in df:
            df[c] = df[c].fillna(df[c].median())
    return df


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def clustered_bootstrap_delta_auc(
    y: np.ndarray, p_c: np.ndarray, p_a: np.ndarray, groups: np.ndarray,
    n_boot: int = 2000, seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap ΔAUC = AUC(C) - AUC(A), resampling whole questions (clusters)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in gs])
        yy, cc, aa = y[idx], p_c[idx], p_a[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(roc_auc_score(yy, cc) - roc_auc_score(yy, aa))
    deltas = np.array(deltas)
    return {
        "delta_auc": float(roc_auc_score(y, p_c) - roc_auc_score(y, p_a)),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "n_boot_ok": int(len(deltas)),
    }


def _cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    return oof


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
           "brier": float(brier_score_loss(y, p)),
           "logloss": float(log_loss(y, p))}
    # calibration slope / intercept
    z = logit(p).reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000)
    cal.fit(z, y)
    out["cal_slope"] = float(cal.coef_[0][0])
    out["cal_intercept"] = float(cal.intercept_[0])
    return out


def fit_all(table: pd.DataFrame, runs: pd.DataFrame, results: Dict, seed: int = 0) -> Dict:
    """Fit Models A/B/C with grouped CV; return metrics and OOF predictions."""
    df = runs.merge(table, on="qid", how="left")
    y = df["success"].astype(float).values
    groups = df["qid"].values
    metrics, oof = {}, {}
    for name, cols in FEATURES.items():
        X = df[cols].astype(float).values
        p = _cv_predict(X, y, groups, seed)
        oof[name] = p
        metrics[name] = _metrics(y, p)
    results["model_metrics"] = metrics
    delta = clustered_bootstrap_delta_auc(y, oof["C"], oof["A"], groups, seed=seed)
    results["delta_auc_C_minus_A"] = delta
    return {"oof": oof, "metrics": metrics, "delta": delta, "y": y, "groups": groups}


def ablation_table(table: pd.DataFrame, runs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Leave-one-factor-out ablations around Model C (Section 12.5)."""
    base = ["H", "CSL", "MS", "T->I", "I->T", "I->I", "n_images", "HxCSL", "RC"]
    variants = {
        "full": base,
        "hop_only": ["H"],
        "hop+csl": ["H", "CSL"],
        "hop+csl+ms": ["H", "CSL", "MS"],
        "minus_H": [c for c in base if c != "H"],
        "minus_CSL": [c for c in base if c != "CSL"],
        "minus_MS": [c for c in base if c != "MS"],
        "minus_RC": [c for c in base if c != "RC"],
    }
    df = runs.merge(table, on="qid", how="left")
    y = df["success"].astype(float).values
    groups = df["qid"].values
    rows = []
    for name, cols in variants.items():
        X = df[cols].astype(float).values
        p = _cv_predict(X, y, groups, seed)
        rows.append({"variant": name, "n_features": len(cols), **_metrics(y, p)})
    return pd.DataFrame(rows)
