"""Budget calibration: map a target budget B to a threshold tau and measure the
gap between realized and target budget (proposal RQ2 / H2 / E2, E6).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import policies


def _cost_series(frame: pd.DataFrame, k: int, cost_col: str) -> pd.Series:
    return frame[frame["k"] == k].set_index("qid")[cost_col]


def realized_budget(
    frame: pd.DataFrame,
    choices: Dict[str, int],
    cost_col: str = "cost_norm",
    qids: Optional[List[str]] = None,
) -> float:
    idx = frame.set_index(["qid", "k"])
    qids = qids if qids is not None else list(choices.keys())
    vals = [idx.loc[(q, choices[q]), cost_col] for q in qids if (q, choices[q]) in idx.index]
    return float(np.mean(vals)) if vals else float("nan")


def calibrate_gate(
    frame: pd.DataFrame,
    signal_col: str,
    target_B: float,
    k_on: int,
    k_off: int,
    calib_qids: List[str],
    cost_col: str = "cost_norm",
) -> float:
    """Pick the threshold tau whose realized budget on the calibration split is
    closest to target_B. High signal => retrieve k_on."""
    sig = policies.question_signals(frame)[signal_col]
    c_on = _cost_series(frame, k_on, cost_col)
    c_off = _cost_series(frame, k_off, cost_col)

    order = sig.reindex(calib_qids).dropna().sort_values(ascending=False)
    on_cost = c_on.reindex(order.index)
    off_cost = c_off.reindex(order.index)

    best_tau, best_err = float("inf"), float("inf")
    # cumulative: turning on the first m questions (largest signals) first.
    n = len(order)
    on_vals = on_cost.values
    off_vals = off_cost.values
    for m in range(n + 1):
        cur = on_vals[:m].sum() + off_vals[m:].sum()
        realized = cur / n if n else 0.0
        err = abs(realized - target_B)
        if err < best_err:
            best_err = err
            # threshold just below the smallest signal still turned on
            best_tau = float(order.values[m - 1]) if m > 0 else float("inf")
    return best_tau


def _bootstrap_error(
    frame: pd.DataFrame,
    choices: Dict[str, int],
    target_B: float,
    cost_col: str,
    qids: List[str],
    n_boot: int = 500,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = frame.set_index(["qid", "k"])
    per_q = np.array(
        [idx.loc[(q, choices[q]), cost_col] for q in qids if (q, choices[q]) in idx.index],
        dtype=float,
    )
    if per_q.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    errs = []
    for _ in range(n_boot):
        samp = rng.choice(per_q, size=per_q.size, replace=True)
        errs.append(abs(samp.mean() - target_B))
    errs = np.array(errs)
    return {
        "mean": float(errs.mean()),
        "median": float(np.median(errs)),
        "p95": float(np.percentile(errs, 95)),
    }


def calibration_study(frame: pd.DataFrame, cfg: dict) -> List[dict]:
    """Calibrated vs uncalibrated adaptive gates over the budget grid, in-domain
    and under difficulty transfer."""
    budgets = cfg["budgets"]
    k_on = max(cfg["ks"])
    k_off = 0
    qids = policies.questions(frame)
    calib, evald = policies.split_questions(frame, cfg)

    rows: List[dict] = []
    for B in budgets:
        for signal_col in ("max_score", "unc0"):
            tau = calibrate_gate(frame, signal_col, B, k_on, k_off, calib)
            choices = (
                policies.policy_similarity_gate(frame, tau, k_on, k_off)
                if signal_col == "max_score"
                else policies.policy_uncertainty_gate(frame, tau, k_on, k_off)
            )
            res = policies.evaluate(frame, {q: choices[q] for q in evald})
            err = _bootstrap_error(frame, choices, B, "cost_norm", evald, seed=cfg["seed"])
            rows.append(
                {
                    "setting": "in-domain",
                    "signal": signal_col,
                    "target_B": B,
                    "tau": tau,
                    "realized_B": res["realized_budget"],
                    "abs_err": abs(res["realized_budget"] - B),
                    "err_mean": err["mean"],
                    "err_median": err["median"],
                    "err_p95": err["p95"],
                    "f1": res["f1"],
                    "harm_rate": res["harm_rate"],
                }
            )

        # Uncalibrated adaptive baseline: one threshold fixed at the median
        # budget, evaluated at every target B (it cannot track B).
        sig = policies.question_signals(frame)["unc0"].fillna(-np.inf)
        tau_naive = float(np.median(sig.reindex(calib).dropna().values))
        choices = policies.policy_uncertainty_gate(frame, tau_naive, k_on, k_off)
        res = policies.evaluate(frame, {q: choices[q] for q in evald})
        rows.append(
            {
                "setting": "in-domain",
                "signal": "uncertainty_uncalibrated",
                "target_B": B,
                "tau": tau_naive,
                "realized_B": res["realized_budget"],
                "abs_err": abs(res["realized_budget"] - B),
                "err_mean": float("nan"),
                "err_median": float("nan"),
                "err_p95": float("nan"),
                "f1": res["f1"],
                "harm_rate": res["harm_rate"],
            }
        )

    # Difficulty transfer: calibrate on 'easy', evaluate on 'hard' (and vice versa).
    for a, b in [(cfg["splits"]["transfer_a"], cfg["splits"]["transfer_b"]),
                 (cfg["splits"]["transfer_b"], cfg["splits"]["transfer_a"])]:
        src = [q for q in qids if frame.loc[frame["qid"] == q, "level"].iloc[0] == a]
        dst = [q for q in qids if frame.loc[frame["qid"] == q, "level"].iloc[0] == b]
        if not src or not dst:
            continue
        for B in budgets:
            for signal_col in ("max_score", "unc0"):
                tau = calibrate_gate(frame, signal_col, B, k_on, k_off, src)
                choices = (
                    policies.policy_similarity_gate(frame, tau, k_on, k_off)
                    if signal_col == "max_score"
                    else policies.policy_uncertainty_gate(frame, tau, k_on, k_off)
                )
                res = policies.evaluate(frame, {q: choices[q] for q in dst})
                rows.append(
                    {
                        "setting": f"transfer {a}->{b}",
                        "signal": signal_col,
                        "target_B": B,
                        "tau": tau,
                        "realized_B": res["realized_budget"],
                        "abs_err": abs(res["realized_budget"] - B),
                        "err_mean": float("nan"),
                        "err_median": float("nan"),
                        "err_p95": float("nan"),
                        "f1": res["f1"],
                        "harm_rate": res["harm_rate"],
                    }
                )
    return rows
