"""Phase A: execute agent runs over the synthetic corpus and cache them.

Runs recorded
-------------
* baseline, oracle_bridge, caption  at the default retriever, R replicates each
* dimension sweep (baseline condition) at d in {32,...,512} on a subsample

Each record stores per-hop outcomes, so hop-level analysis and error propagation
are possible without re-running.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import yaml
from tqdm import tqdm

from . import synth
from .agent import run_agent
from .llm import OllamaClient
from .retriever import Retriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class RunCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._entries: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    self._entries[self._key(r["qid"], r["condition"], r["replicate"], r["dim"])] = r

    @staticmethod
    def _key(qid, condition, replicate, dim):
        return f"{qid}|{condition}|{replicate}|{dim}"

    def has(self, qid, condition, replicate, dim) -> bool:
        return self._key(qid, condition, replicate, dim) in self._entries

    def put(self, rec: dict) -> None:
        self._entries[self._key(rec["qid"], rec["condition"], rec["replicate"], rec["dim"])] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def all(self) -> List[dict]:
        return list(self._entries.values())


def _job_list(cfg: dict, questions):
    jobs = []  # (retriever_spec, qid, condition, replicate, dim)
    reps = cfg["runs"]["replicates"]
    for cond in cfg["runs"]["conditions"]:
        for q in questions:
            for r in range(reps):
                jobs.append(("bm25", 0, q.qid, cond, r, None))

    dims = cfg["retriever"]["dims"]
    sub = questions[: cfg["runs"]["dim_subsample"]]
    for d in dims:
        for q in sub:
            for r in range(cfg["runs"]["dim_replicates"]):
                jobs.append(("svd", d, q.qid, "baseline", r, d))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="cap number of questions")
    ap.add_argument("--conditions", default=None, help="comma-separated subset of conditions")
    ap.add_argument("--skip-dims", action="store_true", help="skip the dimension sweep")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.conditions:
        cfg["runs"]["conditions"] = args.conditions.split(",")
    if args.skip_dims:
        cfg["retriever"]["dims"] = []

    corpus_path = os.path.join(ROOT, cfg["paths"]["corpus"])
    if os.path.exists(corpus_path):
        docs, questions = synth.load(corpus_path)
    else:
        docs, questions = synth.build_corpus(cfg)
        synth.save(docs, questions, corpus_path)
    if args.limit:
        questions = questions[: args.limit]
    print(f"corpus: {len(docs)} docs | questions: {len(questions)}")

    client = OllamaClient(cfg)
    if not client.ping():
        sys.exit(f"Ollama not reachable at {cfg['ollama']['host']}.")
    print(f"Ollama OK | model={cfg['ollama']['model']}")

    bm25 = Retriever(docs, "bm25")
    svd_cache: Dict[int, Retriever] = {}
    qmap = {q.qid: q for q in questions}

    cache = RunCache(os.path.join(ROOT, cfg["paths"]["cache"]))
    jobs = _job_list(cfg, questions)
    pbar = tqdm(jobs, desc="runs")
    n_new, n_llm = 0, 0
    for kind, dim, qid, cond, rep, dimkey in pbar:
        if cache.has(qid, cond, rep, dimkey):
            continue
        retr = bm25
        if kind == "svd":
            if dim not in svd_cache:
                svd_cache[dim] = Retriever(docs, "svd", dim=dim)
            retr = svd_cache[dim]
        success, hops_log, n_calls = run_agent(qmap[qid], retr, client, cfg, condition=cond, replicate=rep)
        n_llm += n_calls
        cache.put({
            "qid": qid, "condition": cond, "replicate": rep, "dim": dimkey,
            "retriever": kind, "success": bool(success), "hops": hops_log, "n_llm": n_calls,
        })
        n_new += 1
        pbar.set_postfix(new=n_new, llm=n_llm)

    print(f"Done. {n_new} new runs cached at {cache.path}")


if __name__ == "__main__":
    main()
