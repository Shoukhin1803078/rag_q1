"""Ollama client, RAG prompt construction, and the on-disk generation cache."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


def build_prompt(question: str, contexts: List[str], titles: Optional[List[str]] = None) -> str:
    """Context-grounded prompt; empty context => closed-book (no-RAG)."""
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
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "logprobs": self.logprobs,
            "top_logprobs": 1,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "seed": 0,
            },
        }
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# --- cache -----------------------------------------------------------------

class GridCache:
    """Append-only JSONL cache keyed by (qid, k). Resumable."""

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
                    self._entries[self._key(rec["qid"], rec["k"])] = rec

    @staticmethod
    def _key(qid: str, k: int) -> str:
        return f"{qid}::{k}"

    def has(self, qid: str, k: int) -> bool:
        return self._key(qid, k) in self._entries

    def get(self, qid: str, k: int) -> Optional[dict]:
        return self._entries.get(self._key(qid, k))

    def put(self, rec: dict) -> None:
        self._entries[self._key(rec["qid"], rec["k"])] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def all(self) -> List[dict]:
        return list(self._entries.values())
