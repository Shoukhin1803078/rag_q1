# CHAB-RAG — smoke-test / preliminary-results pilot

A small, CPU-feasible pilot that puts real numbers behind the CHAB-RAG proposal's
core claims, so the proposal can cite concrete feasibility evidence.

It is a **sanity check**, not the full experiment set. Every result maps to a
proposal hypothesis (H1–H6) and to the Sec. 12.8 evaluation table.

## What it produces

| Figure / table | Proposal link |
|---|---|
| `fig_inverted_u.png` — quality vs depth k | Sec. 9, RQ3 |
| `fig_deltaQ.png` — beneficial/neutral/harmful split | RQ3 / H3 / E3 |
| `fig_cost_vs_length.png` — length drives cost, not doc count | RQ4 / H4 / C3 |
| `fig_calibration.png` + `calibration.csv` — realized vs target budget | RQ2 / H2 / E2, E6 |
| `fig_pareto.png` + `matched_budget_table.csv` | RQ1, RQ5 / H1, H5 |
| `PRELIMINARY_RESULTS.md`, `summary.json` | drop-in report for the supervisor |

## Design

One `(question, k)` grid of generations is cached once; **all policies and
calibration are then evaluated offline from that cache**, so the expensive LLM
work is paid exactly once.

- Backbone: `qwen2.5:3b-instruct` via Ollama (CPU).
- Data: HotpotQA `distractor` split — 10 candidate paragraphs/question, so no
  separate corpus index is needed.
- Ranking: BM25 (`rank_bm25`); `--retriever dense` optional (needs
  `sentence-transformers`). Gold paragraphs are not forced first, so hard
  negatives can appear.
- Cost: tracked in tokens (prompt + completion) and time (prefill + decode),
  captured from Ollama's timing fields.

## Setup

```bash
ollama pull qwen2.5:3b-instruct        # once
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Quick check (~a minute):

```bash
.venv/bin/python -m src.run_grid --limit 5 --ks 0,3,10
.venv/bin/python -m src.analyze
```

Full pilot (~150 questions × k∈{0,1,3,5,10}; one overnight run on a laptop CPU):

```bash
bash scripts/run_pilot.sh
```

The grid runner is **resumable** — re-running skips cached `(qid, k)` pairs.

## Config

`config.yaml` controls seed, `n_questions`, `ks`, model, budget grid, split
fractions, and controller weights (`lambda`, `mu`). Tune `n_questions` first if
runtime is a concern.

## Pass criteria (sanity signals)

- F1 rises then plateaus/falls (inverted-U).
- Harm rate > 0 at larger k.
- `corr(context length, latency)` clearly stronger than `corr(k, latency)`.
- Calibrated `|B̂−B|` clearly below uncalibrated.
- CHAB-lite on/above the fixed-k frontier at matched realized budget.

Mixed confirm/refute results are fine — negative results are useful evidence too.

## Non-goals

Sentence-level **evidence allocation** (proposal Level-2 decision) is out of
scope here; CHAB-lite operates only at retrieval-depth level. The ΔQ estimator is
a deliberately simple ridge model for the pilot.
