"""Phase A: run the base systems over the query set and cache every generation
with stage-level timing (retrieval, compression, TTFT, prefill, decode, E2E).

Base systems
------------
closed_book   : no retrieval (provides the predictor's self-knowledge signal)
S0_full       : standard RAG, full retrieved context
S2_fixed      : retrieval + fixed-ratio compression
S2_adaptive   : retrieval + query-adaptive compression

S1 (prediction) and S4 (prediction + compression) are assembled offline from
these runs, so no generation is repeated.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import yaml
from tqdm import tqdm

from .compress import compress
from .data import load_examples
from .llm import GridCache, OllamaClient, build_prompt
from .metrics import score_answer
from .retriever import rank_paragraphs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _retrieve(example, cfg):
    t0 = time.perf_counter()
    ranked = rank_paragraphs(example, cfg["retriever"])[: cfg["top_k"]]
    retrieval_ms = (time.perf_counter() - t0) * 1000.0
    retrieved = [(example.paragraphs[i].title, example.paragraphs[i].text) for i, _ in ranked]
    return retrieved, retrieval_ms


def build_context(example, cfg, system):
    """Return (contexts, stats) for a base system."""
    if system == "closed_book":
        return [], {"retrieval_ms": 0.0, "compression_ms": 0.0, "compression_ratio": 1.0,
                    "words_before": 0, "words_after": 0, "retrieved": False}

    retrieved, retrieval_ms = _retrieve(example, cfg)

    if system == "S0_full":
        stats = {"retrieval_ms": retrieval_ms, "compression_ms": 0.0, "compression_ratio": 1.0,
                 "words_before": sum(len(t.split()) for _, t in retrieved), "retrieved": True}
        stats["words_after"] = stats["words_before"]
        return retrieved, stats

    if system == "S2_fixed":
        contexts, cstats = compress(retrieved, example.question, mode="fixed",
                                    keep_frac=cfg["compression"]["fixed_keep_frac"])
    elif system == "S2_adaptive":
        contexts, cstats = compress(retrieved, example.question, mode="adaptive",
                                    rel_threshold=cfg["compression"]["adaptive_rel_threshold"],
                                    max_words=cfg["compression"]["adaptive_max_words"])
    else:
        raise ValueError(system)

    stats = {"retrieval_ms": retrieval_ms, "retrieved": True, **cstats}
    return contexts, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--systems", default=None, help="comma-separated subset of base systems")
    args = ap.parse_args()

    cfg = load_config(args.config)
    systems = args.systems.split(",") if args.systems else cfg["base_systems"]

    client = OllamaClient(cfg)
    if not client.ping():
        sys.exit(f"Ollama not reachable at {cfg['ollama']['host']}. Start it and retry.")
    print(f"Ollama OK | model={cfg['ollama']['model']} | systems={systems} | top_k={cfg['top_k']}")

    examples = load_examples(cfg)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} questions")

    cache = GridCache(os.path.join(ROOT, cfg["paths"]["cache"]))
    todo = [(ex, s) for ex in examples for s in systems]
    pbar = tqdm(todo, desc="systems")
    n_new = 0
    for ex, system in pbar:
        if cache.has(ex.qid, system):
            continue
        contexts, stats = build_context(ex, cfg, system)
        titles = [t for t, _ in contexts]
        texts = [t for _, t in contexts]
        prompt = build_prompt(ex.question, texts, titles)
        resp = client.generate(prompt)
        scored = score_answer(resp["response"], ex.answer)

        rec = {
            "qid": ex.qid,
            "system": system,
            "question": ex.question,
            "gold": ex.answer,
            "qtype": ex.qtype,
            "level": ex.level,
            "retrieved_titles": titles,
            "gold_titles": ex.gold_titles,
            "prompt": prompt,
            "raw_response": resp["response"],
            "pred": scored["pred"],
            "em": scored["em"],
            "f1": scored["f1"],
            "mean_logprob": resp["mean_logprob"],
            # stage-level timing
            "ttft_ms": resp["ttft_ms"],
            "e2e_ms": resp["e2e_ms"],
            "prompt_eval_count": resp["prompt_eval_count"],
            "eval_count": resp["eval_count"],
            "prompt_eval_duration": resp["prompt_eval_duration"],
            "eval_duration": resp["eval_duration"],
            "total_duration": resp["total_duration"],
            "load_duration": resp["load_duration"],
            **stats,
        }
        cache.put(rec)
        n_new += 1
        pbar.set_postfix(new=n_new)

    print(f"Done. {n_new} new generations cached at {cache.path}")


if __name__ == "__main__":
    main()
