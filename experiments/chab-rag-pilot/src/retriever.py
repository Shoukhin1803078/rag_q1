"""Ranking of the candidate paragraphs for a question.

Gold paragraphs are NOT forced to the front, so hard negatives and distractors
can appear in the retrieved set. This is required for a meaningful retrieval
harm signal.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .data import Example

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_DENSE_CACHE = {}


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def rank_bm25(example: Example) -> List[Tuple[int, float]]:
    """Return (paragraph_index, score) sorted by decreasing BM25 score."""
    corpus = [_tokenize(p.text) for p in example.paragraphs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(example.question))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(i, float(s)) for i, s in ranked]


def rank_dense(example: Example, model_name: str) -> List[Tuple[int, float]]:
    """Dense ranking with a small sentence-transformer (CPU friendly)."""
    import numpy as np

    if model_name not in _DENSE_CACHE:
        from sentence_transformers import SentenceTransformer

        _DENSE_CACHE[model_name] = SentenceTransformer(model_name)
    model = _DENSE_CACHE[model_name]

    texts = [p.text for p in example.paragraphs]
    emb = model.encode([example.question] + texts, normalize_embeddings=True)
    q = emb[0]
    sims = emb[1:] @ q
    order = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in order]


def rank_paragraphs(example: Example, retriever: str = "bm25", dense_model: str = "") -> List[Tuple[int, float]]:
    if retriever == "dense":
        return rank_dense(example, dense_model)
    return rank_bm25(example)


def top_k(example: Example, k: int, retriever: str = "bm25", dense_model: str = "") -> List[Tuple[int, float]]:
    """Top-k paragraphs as (index, score). k<=0 returns an empty list."""
    if k <= 0:
        return []
    return rank_paragraphs(example, retriever, dense_model)[:k]


def gold_ranks(example: Example, retriever: str = "bm25", dense_model: str = "") -> List[int]:
    """Rank (1-indexed position) of each gold paragraph under the retriever."""
    order = [i for i, _ in rank_paragraphs(example, retriever, dense_model)]
    pos = {idx: r + 1 for r, idx in enumerate(order)}
    return [pos[i] for i in example.gold_indices()]
