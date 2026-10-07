"""Evidence-level context compression.

Retrieved paragraphs are split into sentences, scored against the question, and
a subset is retained. Two budgets are offered to mirror the PACER ablations:

* ``fixed``    — retain a constant fraction of the sentences (fixed ratio).
* ``adaptive`` — retain every sentence whose relevance is within a relative
  threshold of the best sentence (evidence-sufficiency style), so the retained
  amount varies per query.

Compression is deliberately training-free and cheap so its own overhead can be
measured, matching the proposal's requirement that compression cost be counted.
"""
from __future__ import annotations

import math
import re
import time
from typing import List, Tuple

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _score_sentences(sentences: List[str], question: str) -> List[float]:
    corpus = [_tokens(s) for s in sentences]
    if not corpus:
        return []
    bm25 = BM25Okapi(corpus)
    return [float(x) for x in bm25.get_scores(_tokens(question))]


def compress(
    retrieved: List[Tuple[int, str, str]],
    question: str,
    mode: str = "none",
    keep_frac: float = 0.5,
    rel_threshold: float = 0.5,
    max_words: int | None = None,
) -> Tuple[List[Tuple[str, str]], dict]:
    """Compress retrieved evidence.

    Parameters
    ----------
    retrieved : list of (title, text) for the retrieved paragraphs.
    mode : "none" | "fixed" | "adaptive".

    Returns
    -------
    (compressed_contexts, stats) where compressed_contexts is a list of
    (title, text) blocks and stats records compression overhead and ratio.
    """
    t0 = time.perf_counter()

    # Flatten to sentences with provenance.
    sent_records = []  # (para_idx, sentence)
    for pi, (_title, text) in enumerate(retrieved):
        for s in split_sentences(text):
            sent_records.append((pi, s))

    n_before = len(sent_records)
    words_before = sum(len(_tokens(s)) for _, s in sent_records)

    if mode == "none" or n_before == 0:
        stats = _stats(t0, n_before, n_before, words_before, words_before, 0.0)
        return [(t, txt) for t, txt in retrieved], stats

    scores = _score_sentences([s for _, s in sent_records], question)

    if mode == "fixed":
        k = max(1, int(math.ceil(keep_frac * n_before)))
        order = sorted(range(n_before), key=lambda i: scores[i], reverse=True)[:k]
        keep = set(order)
    elif mode == "adaptive":
        smax = max(scores) if scores else 0.0
        thr = rel_threshold * smax
        keep = {i for i in range(n_before) if scores[i] >= thr}
        if not keep:  # always keep the single best evidence unit
            keep = {max(range(n_before), key=lambda i: scores[i])}
    else:
        raise ValueError(f"unknown compression mode: {mode}")

    # Cap by a word budget if the relative threshold was too permissive.
    if max_words is not None:
        ranked = sorted(keep, key=lambda i: scores[i], reverse=True)
        capped, used = [], 0
        for i in ranked:
            w = len(_tokens(sent_records[i][1]))
            if used + w > max_words and capped:
                break
            capped.append(i)
            used += w
        keep = set(capped)

    # Rebuild paragraphs in original order (stable, cites preserved).
    by_para: dict[int, List[str]] = {}
    for i in sorted(keep):
        pi, s = sent_records[i]
        by_para.setdefault(pi, []).append(s)

    compressed = []
    for pi, (title, _text) in enumerate(retrieved):
        if pi in by_para:
            compressed.append((title, " ".join(by_para[pi])))

    words_after = sum(len(_tokens(txt)) for _, txt in compressed)
    compression_ms = (time.perf_counter() - t0) * 1000.0
    stats = _stats(t0, n_before, len(keep), words_before, words_after, compression_ms)
    return compressed, stats


def _stats(t0, n_before, n_after, words_before, words_after, compression_ms) -> dict:
    return {
        "sentences_before": n_before,
        "sentences_after": n_after,
        "words_before": words_before,
        "words_after": words_after,
        "compression_ratio": (words_before / words_after) if words_after else 1.0,
        "compression_ms": compression_ms,
    }
