"""Phase A: run the two experiment families.

* scaling     — TPOT(L) decode-scaling curve on controlled context lengths (E1/E8)
* compression — evidence-selection methods over HotpotQA with quality, grounding
                and per-stage timing (E2/E3/E4/E7)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

from . import contexts
from .data import load_examples
from .llm import OllamaClient, build_prompt
from .metrics import score_answer
from .retriever import rank_paragraphs
from .select import build_context, select

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NORM = re.compile(r"[^a-z0-9 ]")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class JsonlCache:
    def __init__(self, path: str, key_fields: List[str]):
        self.path = path
        self.key_fields = key_fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.entries: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        self.entries[self._k(r)] = r

    def _k(self, r: dict) -> str:
        return "|".join(str(r[f]) for f in self.key_fields)

    def has(self, **kw) -> bool:
        return self._k(kw) in self.entries

    def put(self, rec: dict) -> None:
        self.entries[self._k(rec)] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def all(self) -> List[dict]:
        return list(self.entries.values())


def _timing(resp: dict) -> dict:
    prefill_ms = (resp.get("prompt_eval_duration") or 0) / 1e6
    decode_ms = (resp.get("eval_duration") or 0) / 1e6
    n_out = float(resp.get("eval_count") or 0)
    return {
        "prompt_tokens": float(resp.get("prompt_eval_count") or 0),
        "output_tokens": n_out,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": prefill_ms + decode_ms,
        "tpot_ms": (decode_ms / n_out) if n_out else 0.0,
        "ttft_ms": resp["ttft_ms"],
        "e2e_ms": resp["e2e_ms"],
    }


def run_scaling(cfg: dict, client: OllamaClient) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["scaling_cache"]), ["mode", "L", "rep"])
    q = cfg["scaling"]["question"]
    # warmup so the first measured point is not a cold-start outlier
    client.generate(build_prompt(q, ["Warmup."], ["reference"]))
    jobs = [(L, r) for L in cfg["scaling"]["lengths"] for r in range(cfg["scaling"]["replicates"])]
    n_new = 0
    for L, r in tqdm(jobs, desc="scaling"):
        if cache.has(mode="scaling", L=L, rep=r):
            continue
        filler = contexts.filler_sentences(L, seed=cfg["seed"] + r)
        # unique leading sentence avoids Ollama's cross-run prompt-prefix cache
        # (otherwise a longer prompt sharing a prefix reuses prefill for free)
        nonce = f"Run identifier {L}-{r}-{L * 1000 + r}."
        ctx = nonce + " " + " ".join(filler)
        prompt = build_prompt(q, [ctx], ["reference"])
        resp = client.generate(prompt)
        rec = {"mode": "scaling", "L": L, "rep": r, "question": q, **_timing(resp),
               "raw_response": resp["response"]}
        cache.put(rec)
        n_new += 1
    return n_new


def _gold_sentences(ex) -> set:
    gold = set()
    by_title = {p.title: p.sentences for p in ex.paragraphs}
    for title, ids in ex.support_sentences.items():
        sents = by_title.get(title, [])
        for sid in ids:
            if 0 <= sid < len(sents):
                gold.add(sents[sid].strip())
    return gold


def run_compression(cfg: dict, client: OllamaClient) -> int:
    cache = JsonlCache(os.path.join(ROOT, cfg["paths"]["compress_cache"]), ["mode", "qid", "method"])
    examples = load_examples(cfg)[: cfg["data"]["n_questions"]]
    methods = cfg["compression"]["methods"]
    lam = cfg["compression"]["lambda"]
    n_new = 0
    for ex in tqdm(examples, desc="compression"):
        ranked = rank_paragraphs(ex, cfg["retriever"])[: cfg["top_k"]]
        sentences: List[str] = []
        titles: List[str] = []
        for pi, _score in ranked:
            for s in contexts.split_sentences(ex.paragraphs[pi].text):
                sentences.append(s)
                titles.append(ex.paragraphs[pi].title)
        if not sentences:
            continue
        # DECAF first: its retained count defines the iso-budget baseline.
        primary = cfg["compression"].get("decaf_primary", "decaf")
        decaf_idx, decaf_stats = select(sentences, ex.question, primary, lam=lam)
        n_iso = len(decaf_idx)
        gold = _gold_sentences(ex)

        for method in methods:
            if cache.has(mode="compression", qid=ex.qid, method=method):
                continue
            idx, stats = select(sentences, ex.question, method, lam=lam, n_iso=n_iso)
            ctx = build_context(sentences, idx, titles)
            prompt = build_prompt(ex.question, [ctx], ["evidence"])
            resp = client.generate(prompt)
            scored = score_answer(resp["response"], ex.answer)

            retained = {sentences[i].strip() for i in idx}
            inter = len(retained & gold)
            evidence_recall = inter / len(gold) if gold else float("nan")
            evidence_precision = inter / len(retained) if retained else float("nan")
            ctx_norm = _NORM.sub(" ", ctx.lower())
            pred_norm = _NORM.sub(" ", scored["pred"].lower()).strip()
            gold_norm = _NORM.sub(" ", ex.answer.lower()).strip()
            answer_support = float(bool(pred_norm) and pred_norm in ctx_norm)
            gold_in_evidence = float(bool(gold_norm) and gold_norm in ctx_norm)

            rec = {
                "mode": "compression", "qid": ex.qid, "method": method,
                "question": ex.question, "gold": ex.answer, "pred": scored["pred"],
                "em": scored["em"], "f1": scored["f1"],
                "evidence_recall": evidence_recall, "evidence_precision": evidence_precision,
                "answer_support": answer_support, "gold_in_evidence": gold_in_evidence,
                **stats, **_timing(resp), "raw_response": resp["response"],
            }
            cache.put(rec)
            n_new += 1
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--mode", default="all", choices=["all", "scaling", "compression"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lengths", default=None, help="comma-separated scaling lengths")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.limit:
        cfg["data"]["n_questions"] = args.limit
    if args.lengths:
        cfg["scaling"]["lengths"] = [int(x) for x in args.lengths.split(",")]

    client = OllamaClient(cfg)
    if not client.ping():
        sys.exit(f"Ollama not reachable at {cfg['ollama']['host']}.")
    print(f"Ollama OK | model={cfg['ollama']['model']}")

    n = 0
    if args.mode in ("all", "scaling"):
        n += run_scaling(cfg, client)
    if args.mode in ("all", "compression"):
        n += run_compression(cfg, client)
    print(f"Done. {n} new runs cached.")


if __name__ == "__main__":
    main()
