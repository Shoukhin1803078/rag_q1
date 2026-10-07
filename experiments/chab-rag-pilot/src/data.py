"""Loading and sampling of the HotpotQA distractor split.

The distractor setting ships 10 candidate paragraphs per question (2 gold +
8 distractors), so we can study retrieval depth and retrieval harm without
building a separate corpus index.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Paragraph:
    title: str
    sentences: List[str]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class Example:
    qid: str
    question: str
    answer: str
    paragraphs: List[Paragraph]
    gold_titles: List[str]
    qtype: str
    level: str
    support_sentences: Dict[str, List[int]] = field(default_factory=dict)

    def gold_indices(self) -> List[int]:
        """Indices of the paragraphs that contain supporting facts."""
        gold = set(self.gold_titles)
        return [i for i, p in enumerate(self.paragraphs) if p.title in gold]

    def to_record(self) -> dict:
        return {
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "paragraphs": [{"title": p.title, "sentences": p.sentences} for p in self.paragraphs],
            "gold_titles": self.gold_titles,
            "qtype": self.qtype,
            "level": self.level,
            "support_sentences": self.support_sentences,
        }

    @classmethod
    def from_record(cls, rec: dict) -> "Example":
        return cls(
            qid=rec["qid"],
            question=rec["question"],
            answer=rec["answer"],
            paragraphs=[Paragraph(p["title"], p["sentences"]) for p in rec["paragraphs"]],
            gold_titles=rec["gold_titles"],
            qtype=rec["qtype"],
            level=rec["level"],
            support_sentences=rec.get("support_sentences", {}),
        )


def _sample_file(cache_dir: str, split: str, n: int) -> str:
    return os.path.join(cache_dir, f"hotpot_{split}_{n}.json")


def load_examples(cfg: dict, seed: int | None = None) -> List[Example]:
    """Load (or materialise from HF) the sampled subset of HotpotQA."""
    seed = cfg["seed"] if seed is None else seed
    split = cfg["data"]["split"]
    n = cfg["data"]["n_questions"]
    cache_dir = cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    path = _sample_file(cache_dir, split, n)

    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
        return [Example.from_record(r) for r in records]

    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["hf_name"], cfg["data"]["hf_config"], split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]

    examples: List[Example] = []
    for i in idx:
        row = ds[i]
        context = row["context"]
        # HF stores context as list-of-dicts with "title" and "sentences" (or parallel lists).
        if isinstance(context, dict):
            titles = context["title"]
            sentence_lists = context["sentences"]
        else:
            titles = [c["title"] for c in context]
            sentence_lists = [c["sentences"] for c in context]
        paragraphs = [Paragraph(t, list(s)) for t, s in zip(titles, sentence_lists)]

        sf = row["supporting_facts"]
        if isinstance(sf, dict):
            sf_titles, sf_idx = sf["title"], sf["sent_id"]
        else:
            sf_titles = [x[0] for x in sf]
            sf_idx = [x[1] for x in sf]
        gold_titles = sorted(set(sf_titles))
        support: Dict[str, List[int]] = {}
        for t, s in zip(sf_titles, sf_idx):
            support.setdefault(t, []).append(int(s))

        examples.append(
            Example(
                qid=str(row.get("id", i)),
                question=row["question"],
                answer=row["answer"],
                paragraphs=paragraphs,
                gold_titles=gold_titles,
                qtype=row.get("type", "unknown"),
                level=row.get("level", "unknown"),
                support_sentences=support,
            )
        )

    with open(path, "w") as f:
        json.dump([e.to_record() for e in examples], f)
    return examples
