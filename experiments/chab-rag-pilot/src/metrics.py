"""Answer-quality metrics and cost accounting.

Quality uses the standard SQuAD/HotpotQA normalization + exact match / token F1.
Cost is tracked in two units so conclusions do not depend on one budget
definition (proposal Sec. 12.5):
  * token budget  = prompt tokens + completion tokens
  * time  budget  = prompt-eval (prefill) time + eval (decode) time (ms)
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def clean_prediction(raw: str) -> str:
    """Trim a generated answer to a short span (first line / up to 'Answer:')."""
    text = raw.strip()
    text = text.split("\n")[0]
    for marker in ("Answer:", "answer:"):
        if marker in text:
            text = text.split(marker, 1)[1]
    text = text.strip().strip('"').strip()
    return text


def score_answer(raw: str, gold: str) -> Dict[str, float]:
    pred = clean_prediction(raw)
    return {"em": exact_match(pred, gold), "f1": f1_score(pred, gold), "pred": pred}


# --- cost accounting -------------------------------------------------------

def cost_from_timings(rec: dict) -> Dict[str, float]:
    """Extract cost quantities (token/time units) from a cached generation record.

    Ollama reports durations in nanoseconds.
    """
    prompt_tokens = float(rec.get("prompt_eval_count") or 0)
    completion_tokens = float(rec.get("eval_count") or 0)
    prefill_ns = float(rec.get("prompt_eval_duration") or 0)
    decode_ns = float(rec.get("eval_duration") or 0)

    total_tokens = prompt_tokens + completion_tokens
    prefill_ms = prefill_ns / 1e6
    decode_ms = decode_ns / 1e6
    total_ms = prefill_ms + decode_ms
    tpot_ms = (decode_ms / completion_tokens) if completion_tokens else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "tpot_ms": tpot_ms,
    }
