"""Phase B: build the tidy results frame and assemble the coordinated systems
(S1, S4) offline from the cached base runs.

Systems
-------
closed_book : no retrieval
S0_full     : standard RAG (full context)
S2_fixed    : retrieval + fixed-ratio compression
S2_adaptive : retrieval + query-adaptive compression
S1          : prediction gating over {closed_book, S0_full}          (assembled)
S4          : prediction gating over {closed_book, S2_adaptive}      (assembled, primary RQ1)

Because S1/S4 only *select* among cached runs they add no generation cost,
keeping the component comparison matched and cheap.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import predict


def _derive(row: dict) -> dict:
    prefill_ms = (row.get("prompt_eval_duration") or 0) / 1e6
    decode_ms = (row.get("eval_duration") or 0) / 1e6
    retrieval_ms = row.get("retrieval_ms", 0.0) or 0.0
    compression_ms = row.get("compression_ms", 0.0) or 0.0
    return {
        "ttft_ms": row.get("ttft_ms"),
        "e2e_ms": row.get("e2e_ms"),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "retrieval_ms": retrieval_ms,
        "compression_ms": compression_ms,
        "policy_overhead_ms": retrieval_ms + compression_ms,
        "prompt_tokens": float(row.get("prompt_eval_count") or 0),
        "output_tokens": float(row.get("eval_count") or 0),
        "em": row["em"],
        "f1": row["f1"],
        "compression_ratio": row.get("compression_ratio", 1.0),
        "words_before": row.get("words_before", 0),
        "words_after": row.get("words_after", 0),
        "retrieved": bool(row.get("retrieved", False)),
        "mean_logprob": row.get("mean_logprob"),
    }


def build_frame(records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "qid": r["qid"],
                "system": r["system"],
                "question": r["question"],
                "gold": r["gold"],
                "qtype": r.get("qtype"),
                "level": r.get("level"),
                "context_chars": sum(len(c) for c in r.get("retrieved_titles", [])),
                **_derive(r),
            }
        )
    df = pd.DataFrame(rows)

    # Per-question references.
    cb = df[df["system"] == "closed_book"].set_index("qid")
    full = df[df["system"] == "S0_full"].set_index("qid")
    df["f1_closed_book"] = df["qid"].map(cb["f1"])
    df["f1_full"] = df["qid"].map(full["f1"])
    df["delta_f1_vs_cb"] = df["f1"] - df["f1_closed_book"]
    # A retrieval is "wasted" when it did not improve on the closed-book answer.
    df["wasted_retrieval"] = df["retrieved"] & (df["f1"] <= df["f1_closed_book"])
    return df


def split_questions(frame: pd.DataFrame, cfg: dict):
    rng = np.random.default_rng(cfg["seed"])
    qids = list(dict.fromkeys(frame["qid"].tolist()))
    rng.shuffle(qids)
    n = int(len(qids) * cfg["splits"]["calib_frac"])
    return qids[:n], qids[n:]


def calibrate_tau(frame: pd.DataFrame, cfg: dict, retrieve_system: str) -> float:
    cb = frame[frame["system"] == "closed_book"].set_index("qid")
    rt = frame[frame["system"] == retrieve_system].set_index("qid")
    conf = {q: float(v) for q, v in cb["mean_logprob"].dropna().items()}
    q_skip = {q: float(v) for q, v in cb["f1"].items()}
    q_ret = {q: float(v) for q, v in rt["f1"].items()}
    calib, _ = split_questions(frame, cfg)
    tol = cfg["controller"].get("quality_tolerance", 0.02)
    return predict.calibrate_tau(conf, q_skip, q_ret, calib, quality_tolerance=tol)


def _compose(frame: pd.DataFrame, retrieve_system: str, tau: float, new_name: str) -> pd.DataFrame:
    """Assemble a gated system: pick the closed-book or retrieve row per query."""
    cb = frame[frame["system"] == "closed_book"].set_index("qid")
    rt = frame[frame["system"] == retrieve_system].set_index("qid")
    rows = []
    for qid in cb.index:
        conf = cb.loc[qid, "mean_logprob"]
        is_nan = conf is None or conf != conf
        need = True if is_nan else (float(conf) < tau)  # NaN confidence -> retrieve
        if need and qid in rt.index:
            row = rt.loc[qid].copy()
        else:
            row = cb.loc[qid].copy()
            need = False
        row["qid"] = qid
        row["system"] = new_name
        row["retrieved"] = need
        row["gated_skip"] = not need
        rows.append(row)
    return pd.DataFrame(rows)


def assemble(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return the frame with assembled S1 and S4 systems appended."""
    parts = [frame]
    for retrieve_system, name in [("S0_full", "S1"), ("S2_adaptive", "S4")]:
        tau = calibrate_tau(frame, cfg, retrieve_system)
        comp = _compose(frame, retrieve_system, tau, name)
        comp["gate_tau"] = tau
        parts.append(comp)
    out = pd.concat(parts, ignore_index=True)
    # recompute wasted-retrieval on the assembled rows too
    out["wasted_retrieval"] = out["retrieved"] & (out["f1"] <= out["f1_closed_book"])
    return out


def system_summary(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("system")
    retr = frame[frame["retrieved"]]
    wasted = retr.groupby("system")["wasted_retrieval"].mean()
    summary = pd.DataFrame(
        {
            "n": g.size(),
            "ttft_ms": g["ttft_ms"].median(),
            "e2e_ms": g["e2e_ms"].median(),
            "f1": g["f1"].mean(),
            "em": g["em"].mean(),
            "prompt_tokens": g["prompt_tokens"].mean(),
            "output_tokens": g["output_tokens"].mean(),
            "retrieval_rate": g["retrieved"].mean(),
            "wasted_retrieval_rate": wasted,
            "compression_ratio": g["compression_ratio"].mean(),
            "policy_overhead_ms": g["policy_overhead_ms"].median(),
            "retrieval_ms": g["retrieval_ms"].median(),
            "compression_ms": g["compression_ms"].median(),
        }
    )
    summary["wasted_retrieval_rate"] = summary["wasted_retrieval_rate"].fillna(0.0)
    order = ["closed_book", "S0_full", "S1", "S2_fixed", "S2_adaptive", "S4"]
    return summary.reindex([s for s in order if s in summary.index])
