"""Retrieval-need prediction / gating.

The predictor uses only a model-internal signal — the closed-book answer's mean
token log-probability (a self-knowledge confidence proxy) — to decide whether
retrieval is needed.

Calibration is **quality-constrained**, matching the proposal's objective
(minimise latency subject to non-inferior answer quality): we choose the
threshold that retrieves as little as possible while keeping mean quality within
``quality_tolerance`` of the always-retrieve reference (H2).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def calibrate_tau(
    conf: Dict[str, float],
    q_skip: Dict[str, float],
    q_retrieve: Dict[str, float],
    calib_qids: List[str],
    quality_tolerance: float = 0.02,
) -> float:
    """Choose confidence threshold tau: retrieve when confidence < tau.

    Objective: minimise retrieval rate subject to
    ``mean_quality >= mean_quality(always retrieve) - quality_tolerance``.
    """
    qids = [q for q in calib_qids if q in conf and q in q_skip and q in q_retrieve]
    if not qids:
        return float("inf")
    target = float(np.mean([q_retrieve[q] for q in qids]))
    candidates = sorted({conf[q] for q in qids})

    best_tau, best_rate = float("inf"), float("inf")
    for tau in [float("-inf")] + list(candidates) + [float("inf")]:
        retrieve = {q for q in qids if conf[q] < tau}
        rate = len(retrieve) / len(qids)
        qual = np.mean([q_retrieve[q] if q in retrieve else q_skip[q] for q in qids])
        if qual >= target - quality_tolerance and rate < best_rate:
            best_rate, best_tau = rate, tau
    return best_tau


def apply_gate(conf: Dict[str, float], tau: float, qid: str) -> str:
    """Return the chosen evidence mode for a query: 'skip' or 'retrieve'."""
    return "retrieve" if conf.get(qid, float("-inf")) < tau else "skip"
