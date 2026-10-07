"""Context construction: sentence segmentation, token estimation, and a
filler corpus used to hit a target context length for the decode-scaling curve.
"""
from __future__ import annotations

import re
from typing import List

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "by", "with", "as", "that", "this", "it", "its",
    "from", "into", "which", "who", "what", "when", "where", "how", "why",
}

_TOPICS = ["corridor", "manifold", "console", "beacon", "lattice", "turbine",
           "archive", "junction", "reactor", "registry", "harbour", "compass"]
_VERBS = ["regulates", "monitors", "catalogs", "stabilises", "records",
          "calibrates", "distributes", "inspects", "reinforces", "sequences"]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> set:
    return {t for t in tokens(text) if t not in _STOP and len(t) > 1}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~0.78 word->token ratio observed for Qwen BPE)."""
    return max(1, int(round(len(tokens(text)) * 1.3)))


def filler_sentences(target_tokens: int, seed: int = 0) -> List[str]:
    """Deterministic synthetic filler reaching ~target_tokens."""
    out: List[str] = []
    i = 0
    total = 0
    while total < target_tokens:
        t = _TOPICS[(i + seed) % len(_TOPICS)]
        v = _VERBS[(i * 7 + seed) % len(_VERBS)]
        t2 = _TOPICS[(i * 3 + 1 + seed) % len(_TOPICS)]
        s = (f"Section {i:04d} of the reference manual {v} the {t} assembly, "
             f"which connects to the {t2} subsystem and is inspected during the "
             f"quarterly maintenance window number {i*13 % 997}.")
        out.append(s)
        total += estimate_tokens(s)
        i += 1
    return out
