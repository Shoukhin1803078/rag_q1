"""Ollama client with streaming (for TTFT), RAG prompt construction, and the
on-disk generation cache.

TTFT is measured directly as the wall-clock time from request dispatch to the
first non-empty streamed chunk; prefill/decode durations come from Ollama's
final-chunk telemetry.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


def build_prompt(question: str, contexts: List[str], titles: Optional[List[str]] = None) -> str:
    """Context-grounded prompt; empty context => closed-book (no retrieval)."""
    if not contexts:
        return (
            "Answer the question using only your own knowledge. "
            "Give a short, direct answer.\n\n"
            f"Question: {question}\nAnswer:"
        )
    titles = titles or ["" for _ in contexts]
    blocks = []
    for i, (t, c) in enumerate(zip(titles, contexts), start=1):
        header = f"[{i}] {t}:" if t else f"[{i}]"
        blocks.append(f"{header} {c}")
    context_str = "\n".join(blocks)
    return (
        "Answer the question using the provided context. "
        "Give a short, direct answer.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\nAnswer:"
    )


class OllamaClient:
    def __init__(self, cfg: dict):
        self.host = cfg["ollama"]["host"].rstrip("/")
        self.model = cfg["ollama"]["model"]
        self.temperature = cfg["ollama"].get("temperature", 0.0)
        self.num_predict = cfg["ollama"].get("num_predict", 32)
        self.timeout = cfg["ollama"].get("timeout", 300)
        self.logprobs = cfg["ollama"].get("logprobs", True)

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def generate(self, prompt: str) -> dict:
        """Streamed generation; returns text, TTFT and Ollama timing telemetry."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "logprobs": self.logprobs,
            "top_logprobs": 1,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "seed": 0,
            },
        }
        t0 = time.perf_counter()
        ttft_ms = None
        text_parts: List[str] = []
        lp_vals: List[float] = []
        final: Dict = {}

        with requests.post(f"{self.host}/api/generate", json=payload, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response", "")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    text_parts.append(piece)
                for lp in chunk.get("logprobs") or []:
                    v = lp.get("logprob")
                    if v is not None:
                        lp_vals.append(v)
                if chunk.get("done"):
                    final = chunk
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "response": "".join(text_parts),
            "ttft_ms": ttft_ms if ttft_ms is not None else e2e_ms,
            "e2e_ms": e2e_ms,
            "mean_logprob": (sum(lp_vals) / len(lp_vals)) if lp_vals else None,
            "prompt_eval_count": final.get("prompt_eval_count"),
            "eval_count": final.get("eval_count"),
            "prompt_eval_duration": final.get("prompt_eval_duration"),
            "eval_duration": final.get("eval_duration"),
            "total_duration": final.get("total_duration"),
            "load_duration": final.get("load_duration"),
        }


# --- cache -----------------------------------------------------------------

class GridCache:
    """Append-only JSONL cache keyed by (qid, system). Resumable."""

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
                    rec = json.loads(line)
                    self._entries[self._key(rec["qid"], rec["system"])] = rec

    @staticmethod
    def _key(qid: str, system: str) -> str:
        return f"{qid}::{system}"

    def has(self, qid: str, system: str) -> bool:
        return self._key(qid, system) in self._entries

    def get(self, qid: str, system: str) -> Optional[dict]:
        return self._entries.get(self._key(qid, system))

    def put(self, rec: dict) -> None:
        self._entries[self._key(rec["qid"], rec["system"])] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def all(self) -> List[dict]:
        return list(self._entries.values())
