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
- Stratification is adaptive: annotation `level` → `qtype` → retrieval-confidence
  median split (HotpotQA distractor validation is all `hard`, so `qtype` is used).
- Ranking: BM25 (`rank_bm25`); `--retriever dense` optional (needs
  `sentence-transformers`). Gold paragraphs are not forced first, so hard
  negatives can appear.
- Cost: tracked in tokens (prompt + completion) and time (prefill + decode),
  captured from Ollama's timing fields. Pathological CPU stalls are flagged and
  excluded from cost analysis only.

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

Full pilot (config default: 50 questions × k∈{0,1,3,5,10} ≈ 250 generations;
~1–2 min when cached, ~10 min cold on a laptop CPU):

```bash
bash scripts/run_pilot.sh
```

Raise `data.n_questions` in `config.yaml` to scale up (each extra question ≈ 5
generations). The grid runner is **resumable** — re-running skips cached
`(qid, k)` pairs, so increasing N only pays for the new questions.

## Headline results (50-question pilot)

- **Non-monotonic benefit:** F1 peaks at k=3 (0.254) and falls at k=5/10 (0.230/0.201) — inverted-U reproduced (RQ3/premise).
- **Retrieval harm exists:** 8–12% of queries are harmed by retrieval, rising at k=10.
- **Cost ∝ context length, not doc count:** corr(context tokens, prefill time) = 0.90, but at a fixed k the context size still varies (CV up to 0.32).
- **Calibration is feasible:** calibrated uncertainty gate gets mean |B̂−B| = 0.080 vs 0.275 uncalibrated; transfer across `qtype` shows miscalibration (0.24) — a real finding.
- **Controller is honest-mixed:** the simple ridge ΔQ predictor is anti-correlated with actual ΔF1 (r=−0.29), so CHAB-lite only wins at low budgets; fixed-k is a strong baseline here. See `results/PRELIMINARY_RESULTS.md`.

Numbers are indicative only (small N, one backbone, one dataset).

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
