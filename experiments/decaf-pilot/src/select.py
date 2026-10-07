"""Evidence scoring and selection.

Methods
-------
full        : keep all retrieved sentences.
fixed_r     : top round(r * n) sentences by relevance (fixed ratio).
rel_iso     : top-m sentences by relevance, m matched to DECAF's retained count
              (the iso-budget relevance baseline; the sharp ablation of §17).
decaf       : single-pass CPU-aware greedy selection —
              accept while MarginalQualityGain >= lambda * MarginalCPUCost.
decaf_nocov : DECAF with the coverage term disabled (ablation).

MarginalCPUCost is proportional to a sentence's token count divided by the full
evidence token count (decode cost per token on CPU is ~constant at a fixed
context length, so the marginal cost is token-proportional).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .contexts import content_terms, tokens


def relevance_scores(sentences: List[str], question: str) -> np.ndarray:
    if not sentences:
        return np.zeros(0)
    bm = BM25Okapi([tokens(s) for s in sentences])
    s = np.asarray(bm.get_scores(tokens(question)), dtype=float)
    mx = s.max()
    if mx > 0:
        s = s / mx
    return np.clip(s, 0.0, 1.0)


def _tokens(s: str) -> int:
    return max(1, len(tokens(s)))


def select(
    sentences: List[str],
    question: str,
    method: str,
    lam: float = 1.5,
    n_iso: Optional[int] = None,
) -> Tuple[List[int], Dict]:
    """Return (retained_indices_in_relevance_order, stats)."""
    t0 = time.perf_counter()
    n = len(sentences)
    rel = relevance_scores(sentences, question)
    order = list(np.argsort(-rel))
    tok = [_tokens(s) for s in sentences]
    total_tok = max(1, sum(tok))
    qterms = content_terms(question)
    use_cov = method != "decaf_nocov"
    selected: List[int] = []

    if method == "full":
        selected = order
    elif method.startswith("fixed_"):
        r = float(method.split("_", 1)[1])
        m = max(1, int(math.ceil(r * n)))
        selected = order[:m]
    elif method == "rel_iso":
        m = n_iso if n_iso is not None else max(1, n // 2)
        selected = order[:m]
    elif method.startswith("decaf"):
        use_cov = "nocov" not in method
        lam_eff = lam
        if "lam" in method:
            lam_eff = float(method.split("lam")[1])
        covered = set()
        for i in order:
            new = (content_terms(sentences[i]) & qterms) - covered
            cov = len(new) / max(1, len(qterms))
            gain = rel[i] * (0.5 + 0.5 * cov) if use_cov else rel[i]
            cost = tok[i] / total_tok
            if gain - lam_eff * cost >= 0.0:
                selected.append(i)
                covered |= new
        if not selected:
            selected = [order[0]]
    else:
        raise ValueError(method)

    selected = list(dict.fromkeys(selected))  # keep order, dedupe
    sel_tok = sum(tok[i] for i in selected)
    stats = {
        "n_total": n,
        "n_selected": len(selected),
        "tokens_total": total_tok,
        "tokens_selected": sel_tok,
        "retained_ratio": sel_tok / total_tok,
        "compress_ms": (time.perf_counter() - t0) * 1000.0,
    }
    return selected, stats


def build_context(sentences: List[str], selected: List[int], titles: Optional[List[str]] = None) -> str:
    """Reassembled context text (retained sentences in relevance order)."""
    parts = []
    for i in selected:
        prefix = f"[{titles[i]}] " if titles else ""
        parts.append(prefix + sentences[i])
    return "\n".join(parts)
