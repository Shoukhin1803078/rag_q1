"""Synthetic multimodal evidence-graph corpus with planted bridge chains.

The harness lets H (depth), CSL (concept-separation load), MS (modality switches)
and edge types be set by construction, so a real LLM agent can be run over a
corpus whose structural difficulty is known.

Mechanism modelled
------------------
* Every hop i >= 2 is bridge-dependent: its subject is only revealed by hop i-1,
  so a query built from the question alone cannot retrieve it (bridge-blindness).
* CSL: each hop adds c near-duplicate records that share the subject and relation
  with the gold record but state a different value, so the reader must keep them
  apart (hard-negative neighbourhood of size c).
* Modality: an image node is indexed only by a short caption and its content is
  not directly readable by a text-only reader; the bridge is recovered only with
  probability ``p_image`` (imperfect visual extraction). This is simulated,
  because the pilot's reader is a text-only LLM.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Doc:
    id: str
    modality: str          # "T" or "I"
    subject: str
    relation: str
    obj: str
    topic: str
    next_relation: Optional[str]
    index_text: str
    is_gold: bool = False
    qid: Optional[str] = None
    hop: Optional[int] = None
    distractor: bool = False

    def full_text(self) -> str:
        nxt = f" Next relation: {self.next_relation}." if self.next_relation else ""
        return (f"Record: the {self.relation} of {self.subject} is {self.obj}. "
                f"Filed under topic {self.topic}.{nxt}")


@dataclass
class Hop:
    index: int             # 1-based hop
    subject: str           # subject to query at this hop (bridge from previous hop)
    relation: str
    obj: str               # gold value this hop should reveal
    modality: str
    topic: str
    next_relation: Optional[str]
    csl: int               # number of near-duplicate distractors at this hop


@dataclass
class Question:
    qid: str
    text: str
    e0: str
    r1: str
    answer: str
    hops: List[Hop]
    edge_types: Dict[str, int]
    MS: int
    CSL: float
    n_images: int
    node_ids: List[str]        # gold doc ids per hop
    distractor_ids: List[str]

    def to_record(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_record(cls, rec: dict) -> "Question":
        rec = dict(rec)
        rec["hops"] = [Hop(**h) for h in rec["hops"]]
        return cls(**rec)


def _entity(topic: str, n: int) -> str:
    return f"{topic.capitalize()}-{n:04d}"


def build_corpus(cfg: dict, seed: Optional[int] = None):
    seed = cfg["seed"] if seed is None else seed
    rng = random.Random(seed)
    topics = cfg["corpus"]["topics"]
    relations = cfg["corpus"]["relations"]
    qcfg = cfg["questions"]

    docs: List[Doc] = []
    questions: List[Question] = []

    # Background filler documents so retrieval is non-trivial.
    for i in range(cfg["corpus"]["n_filler_docs"]):
        topic = rng.choice(topics)
        subj = _entity(topic, rng.randint(0, 9999))
        rel = rng.choice(relations)
        obj = _entity(rng.choice(topics), rng.randint(0, 9999))
        mod = "I" if rng.random() < qcfg["modality_p"] else "T"
        d = Doc(id=f"f{i}", modality=mod, subject=subj, relation=rel, obj=obj,
                topic=topic, next_relation=None, index_text="")
        d.index_text = d.full_text() if mod == "T" else _caption(d)
        docs.append(d)

    for qi in range(qcfg["n_questions"]):
        qid = f"q{qi:03d}"
        H = rng.choice(qcfg["hops"])
        modalities = ["I" if rng.random() < qcfg["modality_p"] else "T" for _ in range(H)]
        e0 = _entity(rng.choice(topics), rng.randint(0, 9999))
        rel_seq = [rng.choice(relations) for _ in range(H)]

        hops: List[Hop] = []
        node_ids: List[str] = []
        distractor_ids: List[str] = []
        subject = e0
        for j in range(H):
            topic = rng.choice(topics)
            obj = _entity(rng.choice(topics), rng.randint(0, 9999))
            nxt = rel_seq[j + 1] if j + 1 < H else None
            csl = rng.choice(qcfg["csl_levels"])
            mod = modalities[j]
            hop = Hop(index=j + 1, subject=subject, relation=rel_seq[j], obj=obj,
                      modality=mod, topic=topic, next_relation=nxt, csl=csl)
            hops.append(hop)

            gid = f"{qid}_h{j+1}"
            node_ids.append(gid)
            g = Doc(id=gid, modality=mod, subject=subject, relation=rel_seq[j], obj=obj,
                    topic=topic, next_relation=nxt, index_text="", is_gold=True,
                    qid=qid, hop=j + 1)
            g.index_text = g.full_text() if mod == "T" else _caption(g)
            docs.append(g)

            # Near-duplicate distractors: same subject + relation, different value.
            for k in range(csl):
                wrong = _entity(rng.choice(topics), rng.randint(0, 9999))
                dmod = "I" if rng.random() < qcfg["modality_p"] else "T"
                dd = Doc(id=f"{qid}_h{j+1}_d{k}", modality=dmod, subject=subject,
                         relation=rel_seq[j], obj=wrong, topic=topic, next_relation=nxt,
                         index_text="", qid=qid, hop=j + 1, distractor=True)
                dd.index_text = dd.full_text() if dmod == "T" else _caption(dd)
                docs.append(dd)
                distractor_ids.append(dd.id)

            subject = obj

        edge = {"T->T": 0, "T->I": 0, "I->T": 0, "I->I": 0}
        ms = 0
        for j in range(H - 1):
            a, b = modalities[j], modalities[j + 1]
            edge[f"{a}->{b}"] += 1
            if a != b:
                ms += 1
        csl_total = sum(math.log2(1 + h.csl) for h in hops)

        text = (f"Begin at {e0} using relation '{rel_seq[0]}'. "
                f"Each record states the next relation to use. Return the final value.")
        questions.append(Question(
            qid=qid, text=text, e0=e0, r1=rel_seq[0], answer=hops[-1].obj,
            hops=hops, edge_types=edge, MS=ms, CSL=csl_total,
            n_images=sum(1 for m in modalities if m == "I"),
            node_ids=node_ids, distractor_ids=distractor_ids,
        ))

    return docs, questions


def _caption(doc: Doc) -> str:
    return f"Figure: a diagram about {doc.subject}, filed under topic {doc.topic}."


# --- persistence -----------------------------------------------------------

def save(docs: List[Doc], questions: List[Question], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "docs": [asdict(d) for d in docs],
        "questions": [q.to_record() for q in questions],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load(path: str):
    with open(path) as f:
        payload = json.load(f)
    docs = [Doc(**d) for d in payload["docs"]]
    questions = [Question.from_record(q) for q in payload["questions"]]
    return docs, questions
