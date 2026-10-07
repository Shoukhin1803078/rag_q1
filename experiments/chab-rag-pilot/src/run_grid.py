"""Phase A: build the (question, k) generation grid and cache it.

Every downstream policy and calibration analysis reads this cache, so the
LLM work is paid exactly once.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml
from tqdm import tqdm

from .data import load_examples
from .llm import GridCache, OllamaClient, build_prompt
from .metrics import score_answer
from .retriever import rank_paragraphs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="cap number of questions")
    ap.add_argument("--ks", default=None, help="comma-separated k values, e.g. 0,3,10")
    ap.add_argument("--retriever", default=None, choices=["bm25", "dense"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.retriever:
        cfg["retriever"] = args.retriever
    ks = [int(x) for x in args.ks.split(",")] if args.ks else cfg["ks"]

    client = OllamaClient(cfg)
    if not client.ping():
        sys.exit(f"Ollama not reachable at {cfg['ollama']['host']}. Start it and retry.")
    print(f"Ollama OK | model={cfg['ollama']['model']} | ks={ks} | retriever={cfg['retriever']}")

    examples = load_examples(cfg)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} questions")

    cache_path = os.path.join(ROOT, cfg["paths"]["cache"])
    cache = GridCache(cache_path)

    todo = [(ex, k) for ex in examples for k in ks]
    pbar = tqdm(todo, desc="grid")
    n_new = 0
    for ex, k in pbar:
        if cache.has(ex.qid, k):
            continue
        retrieved = rank_paragraphs(ex, cfg["retriever"], cfg["dense_model"])[: max(k, 0)]
        ctx_titles = [ex.paragraphs[i].title for i, _ in retrieved]
        ctx_texts = [ex.paragraphs[i].text for i, _ in retrieved]
        prompt = build_prompt(ex.question, ctx_texts, ctx_titles)
        resp = client.generate(prompt)
        raw = resp.get("response", "")
        scored = score_answer(raw, ex.answer)

        lps = resp.get("logprobs") or []
        lp_vals = [x.get("logprob") for x in lps if x.get("logprob") is not None]
        mean_logprob = (sum(lp_vals) / len(lp_vals)) if lp_vals else None

        rec = {
            "qid": ex.qid,
            "k": k,
            "question": ex.question,
            "gold": ex.answer,
            "qtype": ex.qtype,
            "level": ex.level,
            "retrieved_idx": [i for i, _ in retrieved],
            "retrieved_scores": [s for _, s in retrieved],
            "retrieved_titles": ctx_titles,
            "gold_titles": ex.gold_titles,
            "prompt": prompt,
            "raw_response": raw,
            "pred": scored["pred"],
            "em": scored["em"],
            "f1": scored["f1"],
            "mean_logprob": mean_logprob,
            "context_chars": sum(len(t) for t in ctx_texts),
            # raw timing/cost fields from Ollama (durations in ns)
            "prompt_eval_count": resp.get("prompt_eval_count"),
            "eval_count": resp.get("eval_count"),
            "prompt_eval_duration": resp.get("prompt_eval_duration"),
            "eval_duration": resp.get("eval_duration"),
            "total_duration": resp.get("total_duration"),
            "load_duration": resp.get("load_duration"),
        }
        cache.put(rec)
        n_new += 1
        pbar.set_postfix(new=n_new)

    print(f"Done. {n_new} new generations cached at {cache_path}")


if __name__ == "__main__":
    main()
