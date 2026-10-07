"""Phase B: build a tidy table from the grid cache and evaluate retrieval
policies entirely offline (no new LLM calls).

Every policy yields a chosen retrieval depth k per question; quality and cost
are then read back from the cache, so all policies are compared at the same
realized budget definition.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import cost_from_timings


def build_frame(records: List[dict], kmax: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for r in records:
        costs = cost_from_timings(r)
        rows.append(
            {
                "qid": r["qid"],
                "k": r["k"],
                "question": r["question"],
                "gold": r["gold"],
                "qtype": r["qtype"],
                "level": r["level"],
                "em": r["em"],
                "f1": r["f1"],
                "mean_logprob": r.get("mean_logprob"),
                "scores": r.get("retrieved_scores", []),
                "n_gold_retrieved": len(set(r.get("retrieved_titles", [])) & set(r.get("gold_titles", []))),
                "context_chars": r.get("context_chars", 0),
                **costs,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["query_len"] = df["question"].str.split().str.len()

    # Retriever-signal features (constant across k for a question).
    df["max_score"] = df["scores"].apply(lambda s: s[0] if len(s) else np.nan)
    df["mean_score"] = df["scores"].apply(lambda s: float(np.mean(s)) if len(s) else np.nan)
    df["score_gap"] = df["scores"].apply(
        lambda s: (s[0] - s[1]) if len(s) > 1 else (s[0] if len(s) == 1 else np.nan)
    )

    # No-retrieval baseline per question.
    base = df[df["k"] == 0].set_index("qid")
    df["f1_0"] = df["qid"].map(base["f1"])
    df["em_0"] = df["qid"].map(base["em"])
    df["unc0"] = df["qid"].map(-base["mean_logprob"]) if "mean_logprob" in base else np.nan
    df["delta_f1"] = df["f1"] - df["f1_0"]
    df["delta_em"] = df["em"] - df["em_0"]
    df["harm_f1"] = np.maximum(0.0, -df["delta_f1"])

    # Normalise cost per question by the cost of the largest k available.
    kmax = kmax if kmax is not None else int(df["k"].max())
    top = df[df["k"] == kmax].set_index("qid")
    df["cost_norm"] = df["total_tokens"] / df["qid"].map(top["total_tokens"])
    df["cost_norm_time"] = df["total_ms"] / df["qid"].map(top["total_ms"])
    return df


def questions(frame: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(frame["qid"].tolist()))


def question_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per question with signals that are constant across k.

    Uses groupby-max because retriever scores are empty at k=0.
    """
    g = frame.groupby("qid")
    return pd.DataFrame(
        {
            "max_score": g["max_score"].max(),
            "mean_score": g["mean_score"].max(),
            "score_gap": g["score_gap"].max(),
            "unc0": g["unc0"].max(),
            "query_len": g["query_len"].max(),
        }
    )


def split_questions(frame: pd.DataFrame, cfg: dict):
    """Deterministic calibration / evaluation split of question ids."""
    rng = np.random.default_rng(cfg["seed"])
    qids = questions(frame)
    rng.shuffle(qids)
    n = int(len(qids) * cfg["splits"]["calib_frac"])
    return qids[:n], qids[n:]


def evaluate(frame: pd.DataFrame, choices: Dict[str, int]) -> dict:
    """Score a policy's per-question depth choices against the cached results."""
    idx = frame.set_index(["qid", "k"])
    f1s, ems, costs, costs_t, harms, ks, missing = [], [], [], [], [], [], 0
    for qid, k in choices.items():
        if (qid, k) not in idx.index:
            missing += 1
            continue
        row = idx.loc[(qid, k)]
        f1s.append(row["f1"])
        ems.append(row["em"])
        costs.append(row["cost_norm"])
        costs_t.append(row["cost_norm_time"])
        harms.append(row["harm_f1"])
        ks.append(k)
    return {
        "n": len(f1s),
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
        "em": float(np.mean(ems)) if ems else float("nan"),
        "realized_budget": float(np.mean(costs)) if costs else float("nan"),
        "realized_budget_time": float(np.mean(costs_t)) if costs_t else float("nan"),
        "harm": float(np.mean(harms)) if harms else float("nan"),
        "harm_rate": float(np.mean([h > 0 for h in harms])) if harms else float("nan"),
        "mean_k": float(np.mean(ks)) if ks else float("nan"),
        "missing": missing,
    }


# --- policies --------------------------------------------------------------

def policy_fixed(frame: pd.DataFrame, k: int) -> Dict[str, int]:
    return {q: k for q in questions(frame)}


def policy_similarity_gate(frame: pd.DataFrame, tau: float, k_on: int, k_off: int = 0) -> Dict[str, int]:
    """Retrieve k_on when the best BM25 match is confident enough, else k_off."""
    sig = question_signals(frame)["max_score"]
    return {q: (k_on if s >= tau else k_off) for q, s in sig.items()}


def policy_uncertainty_gate(frame: pd.DataFrame, tau: float, k_on: int, k_off: int = 0) -> Dict[str, int]:
    """Retrieve k_on when the closed-book answer is uncertain (low logprob)."""
    sig = question_signals(frame)["unc0"].fillna(-np.inf)
    return {q: (k_on if u >= tau else k_off) for q, u in sig.items()}


def fit_delta_model(frame: pd.DataFrame, train_qids: List[str]):
    """Ridge predictor of delta_f1 from cheap features (CHAB-lite benefit/harm estimator)."""
    from sklearn.linear_model import Ridge

    feat_cols = ["k", "cost_norm", "max_score", "mean_score", "score_gap", "context_chars", "query_len"]
    tr = frame[frame["qid"].isin(train_qids)].dropna(subset=feat_cols + ["delta_f1"])
    model = Ridge(alpha=1.0, random_state=0)
    model.fit(tr[feat_cols].values, tr["delta_f1"].values)
    return model, feat_cols


def policy_chab_lite(
    frame: pd.DataFrame,
    model,
    feat_cols: List[str],
    lam: float,
    mu: float,
    ks: List[int],
) -> Dict[str, int]:
    """k* = argmax_k [ dF1_hat - lam*cost_norm - mu*max(0, -dF1_hat) ] (proposal U(q,k))."""
    sub = frame[frame["k"].isin(ks)].dropna(subset=feat_cols)
    choices: Dict[str, int] = {}
    for qid, grp in sub.groupby("qid"):
        x = grp[feat_cols].values
        d_hat = model.predict(x)
        utility = d_hat - lam * grp["cost_norm"].values - mu * np.maximum(0.0, -d_hat)
        best = int(np.argmax(utility))
        choices[qid] = int(grp["k"].values[best])
    return choices
