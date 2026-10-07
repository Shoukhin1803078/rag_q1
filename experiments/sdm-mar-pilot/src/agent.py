"""The retrieval agent.

A hop-based agentic loop: query the retriever with the current bridge, read the
returned records, and have the LLM extract the value needed for the next hop.
The LLM is invoked only when the retrieved notes are genuinely ambiguous
(more than one competing value), which is exactly the concept-separation case;
unambiguous reads are resolved deterministically to keep the pilot affordable.

A wrong bridge is propagated (not corrected), so the harness exhibits the
error-propagation behaviour the proposal's hop-level model describes.
"""
from __future__ import annotations

import re
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm import OllamaClient
from .retriever import Retriever
from .synth import Doc, Question

_VALUE_RE = re.compile(r"the\s+([a-z]+)\s+of\s+([A-Za-z0-9\-]+)\s+is\s+([A-Za-z0-9\-]+)")


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def readable_text(doc: Doc, condition: str, rng: np.random.Generator, p_image: float) -> str:
    if doc.modality == "T":
        return doc.full_text()
    # Caption intervention supplies a descriptive caption for the *evidence*
    # image nodes only (the proposal's text->caption->text substitution).
    if condition == "caption" and doc.is_gold:
        return doc.full_text()
    # Image node read by a text-only reader: content recovered only sometimes.
    if rng.random() < p_image:
        return doc.full_text()
    return f"Figure: a diagram about {doc.subject}, filed under topic {doc.topic}."


def candidates_from_notes(notes: List[str], subject: str, relation: str) -> List[str]:
    vals = []
    for text in notes:
        for rel, subj, val in _VALUE_RE.findall(text):
            if _norm(subj) == _norm(subject) and _norm(rel) == _norm(relation):
                vals.append(val)
    # de-duplicate, keep order
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _llm_pick(subject: str, relation: str, notes: List[str], llm: OllamaClient) -> Optional[str]:
    prompt = (
        f"You are reading research notes to look up one value.\n"
        f"Question: what is the {relation} of {subject}?\n\n"
        f"Notes:\n" + "\n".join(f"- {n}" for n in notes) +
        f"\n\nReply with exactly one value copied from the notes, nothing else.\nValue:"
    )
    out = llm.generate(prompt)["response"].strip().split("\n")[0].strip()
    out = out.strip('."\'')
    return out or None


def run_agent(
    question: Question,
    retriever: Retriever,
    llm: OllamaClient,
    cfg: dict,
    condition: str = "baseline",
    replicate: int = 0,
) -> Tuple[bool, List[dict], int]:
    rng = np.random.default_rng(
        zlib.crc32(f"{question.qid}|{condition}|{replicate}".encode()) % (2**32)
    )
    k = cfg["retriever"]["top_k"]
    p_image = cfg["questions"]["p_image"]

    subject = question.e0
    hops_log: List[dict] = []
    n_llm = 0

    for hop in question.hops:
        docs = retriever.search(f"{hop.subject} {hop.relation}", k, rng)
        notes = [readable_text(d, condition, rng, p_image) for d in docs]
        candidates = candidates_from_notes(notes, hop.subject, hop.relation)

        if condition == "oracle_bridge":
            value = hop.obj
        elif len(candidates) == 1:
            value = candidates[0]
        elif len(candidates) == 0:
            value = None
        else:
            value = _llm_pick(hop.subject, hop.relation, notes, llm)
            n_llm += 1

        ok = _norm(value) == _norm(hop.obj)
        hops_log.append({
            "hop": hop.index,
            "subject": hop.subject,
            "relation": hop.relation,
            "value": value,
            "gold": hop.obj,
            "success": bool(ok),
            "n_candidates": len(candidates),
            "used_llm": bool(len(candidates) > 1),
        })
        subject = value if value is not None else "<unresolved>"

    success = _norm(subject) == _norm(question.answer)
    return success, hops_log, n_llm
