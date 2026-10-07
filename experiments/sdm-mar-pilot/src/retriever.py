"""Retriever with a controllable representation dimension (RC).

* ``bm25`` : sparse lexical retrieval (the default operating point).
* ``svd``  : TF-IDF followed by truncated SVD at dimension ``d``; used for the
  dimension-sweep intervention.

Near-duplicate records (same subject and relation, different value) score almost
identically, so ties are broken with a small seeded jitter: this is what makes
concept-separation load a real difficulty factor rather than a cosmetic one.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .synth import Doc

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self, docs: List[Doc], kind: str = "bm25", dim: int = 256):
        self.docs = docs
        self.kind = kind
        self.dim = dim
        texts = [d.index_text for d in docs]
        self.ids = [d.id for d in docs]
        self._index = {d.id: d for d in docs}

        if kind == "svd":
            self.vectorizer = TfidfVectorizer(tokenizer=_tok, token_pattern=None, lowercase=True)
            X = self.vectorizer.fit_transform(texts)
            n_comp = max(2, min(dim, X.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            E = self.svd.fit_transform(X)
            self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        else:
            self.bm25 = BM25Okapi([_tok(t) for t in texts])

    def scores(self, query: str) -> np.ndarray:
        if self.kind == "svd":
            q = self.vectorizer.transform([query])
            qv = self.svd.transform(q)
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            return self.E @ qv[0]
        return np.asarray(self.bm25.get_scores(_tok(query)), dtype=float)

    def search(self, query: str, k: int, rng: Optional[np.random.Generator] = None) -> List[Doc]:
        s = self.scores(query)
        if rng is not None:
            s = s + rng.uniform(0, 1e-6, size=s.shape)  # break near-ties randomly
        order = np.argsort(-s)[:k]
        return [self.docs[i] for i in order]

    def first_round_stats(self, query: str, gold_id: str, k: int) -> Dict[str, float]:
        s = self.scores(query)
        order = np.argsort(-s)
        top = s[order[0]]
        second = s[order[1]] if len(order) > 1 else 0.0
        rank = int(np.where(order == self.ids.index(gold_id))[0][0]) + 1
        return {"top_score": float(top), "margin": float(top - second),
                "gold_rank": float(rank)}
