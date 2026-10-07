# CHAB-RAG Pilot — Status & Progress

**Date:** 2026-10-07
**Scope:** Phase 0 — smoke test / preliminary-results pilot for the CHAB-RAG proposal.
**State:** ✅ Complete and runnable end-to-end. First results produced.

Location: `experiments/chab-rag-pilot/`

---

## 1. Summary

The proposal argues that RAG retrieval should be treated as a **calibrated, harm-aware,
compute-constrained resource-allocation problem**. Before committing to the full experimental
programme, this pilot runs a small, CPU-feasible sanity test to check that the proposal's
premises are real and that the key measurements are obtainable.

The pilot builds **one cached grid** of `(question, retrieval-depth k)` generations — 50
HotpotQA-distractor questions × `k ∈ {0,1,3,5,10}` using `qwen2.5:3b-instruct` (Ollama, CPU) —
then evaluates **every policy and all budget-calibration analysis offline from that cache**, so
the expensive LLM work is paid exactly once.

**Bottom line:** the three premises behind the proposal are confirmed at pilot scale
(benefit is non-uniform and non-monotonic; retrieval harm exists; context length — not document
count — drives prefill cost), and realized-budget calibration is demonstrably feasible. The
harm-aware controller itself is **not yet competitive** on this small sample — an honest,
useful negative result explained in §4.

---

## 2. What has been done

### Codebase (new, under `experiments/chab-rag-pilot/`)

| File | Purpose |
|---|---|
| `src/data.py` | Load & seed-sample HotpotQA `distractor` split (10 candidate paras/question) with `level`/`qtype` labels |
| `src/retriever.py` | BM25 ranking of candidate paragraphs (optional dense via `sentence-transformers`); gold NOT forced first |
| `src/llm.py` | Ollama client, RAG/no-RAG prompt builder, resumable JSONL generation cache, logprob capture |
| `src/metrics.py` | SQuAD-style EM/F1 normalization + token/prefill/decode cost accounting |
| `src/run_grid.py` | **Phase A** — build the `(qid, k)` generation grid into `cache/grid.jsonl` |
| `src/policies.py` | **Phase B** — build the tidy frame; No-RAG / fixed-k / similarity-gate / uncertainty-gate / CHAB-lite policies; offline evaluation |
| `src/calibrate.py` | Threshold calibration to a target budget; realized-vs-target error; in-domain + transfer study |
| `src/analyze.py` | Figures, tables, `summary.json`, and `PRELIMINARY_RESULTS.md` |
| `config.yaml` | Seed, N, ks, model, budget grid, split fractions, controller weights |
| `scripts/run_pilot.sh` | One-shot Phase A → Phase B |
| `README.md` | Setup, run instructions, headline results |

### Infrastructure / decisions

- **Backbone:** `qwen2.5:3b-instruct` via Ollama on CPU.
- **Data:** HotpotQA `distractor` (ships 10 candidate paragraphs → no corpus index needed).
- **Cost:** tracked in **both** token units (prompt+completion) and time units (prefill+decode),
  captured from Ollama's timing fields, so conclusions don't depend on one budget definition.
- **Robustness:** a stall guard flags/excludes pathological CPU tail-latency events from cost
  analysis only (quality metrics unaffected).
- **Adaptive stratification:** difficulty axis resolves as annotation `level` → `qtype` →
  retrieval-confidence median split (HotpotQA distractor validation is all `hard`, so `qtype`
  is used here).

### Runs completed

- Full grid: 250 generations (50 questions × 5 depths), complete.
- Smoke test (`--limit 5 --ks 0,3,10`): 15 generations, verified.
- Analysis: 5 figures, 4 CSV tables, `summary.json`, report.

---

## 3. Results

Outputs in `experiments/chab-rag-pilot/results/`.

### 3.1 Quality vs retrieval depth — inverted-U reproduced
`fig_inverted_u.png`

| k | 0 | 1 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| F1 | 0.117 | 0.199 | **0.254** | 0.230 | 0.201 |

Peak at **k=3**; declines at k=5 and k=10 → benefit is non-monotonic in retrieval depth
(supports Section 9 / RQ3).

### 3.2 Retrieval harm exists
`fig_deltaQ.png`

| k | beneficial (Δ>0.1) | neutral | harmful (Δ<−0.1) |
|---|---|---|---|
| 1 | 22% | 70% | 8% |
| 3 | 40% | 50% | 10% |
| 5 | 44% | 48% | 8% |
| 10 | 50% | 38% | 12% |

Harmful-retrieval rate is non-zero at every depth and rises at k=10 (supports RQ3 / H3 / E3).

### 3.3 Cost is driven by context length, not document count
`fig_cost_vs_length.png`

- corr(context tokens, prefill time) = **0.90**; corr(context chars, total time) = **0.86**
- k is a monotone but **coarse** proxy: corr(k, prefill) = 0.84, yet at a fixed k the context
  size still varies (max coefficient of variation **0.32**) → equal-document-count queries
  differ materially in cost (supports C3 / RQ4).
- Caveat: decode tail latency is noisy on a shared CPU; prefill is the reliable signal here.

### 3.4 Budget calibration is feasible
`fig_calibration.png`, `calibration.csv`

| policy | mean \|B̂−B\| | max |
|---|---|---|
| calibrated (uncertainty) | **0.080** | 0.200 |
| calibrated (similarity) | 0.167 | 0.347 |
| uncalibrated | 0.275 | 0.637 |

Calibration cuts realized-budget error by ~3× vs an uncalibrated adaptive baseline
(supports RQ2 / H2 / E2). **Transfer** across `qtype` reveals genuine miscalibration
(bridge→comparison 0.239, comparison→bridge 0.187) — the E6-style shift effect is real.

### 3.5 Quality–cost frontier & matched-budget table
`fig_pareto.png`, `matched_budget_table.csv` (Section 12.8 template)

No single policy dominates the frontier: **CHAB-lite is best at low budget**, **fixed-k** at
mid budget, **uncertainty-gate** at high budget. This mirrors the literature finding that
simple methods are often strong (cf. refs [12], [16]).

---

## 4. Honest interpretation & limitations

**Confirmed:** non-uniform / non-monotonic retrieval benefit; retrieval harm exists; context
length drives prefill cost; realized-budget calibration is achievable and its transfer error is
measurable.

**Negative result (important):** the deliberately simple ridge ΔQ predictor is **anti-correlated**
with actual ΔF1 on the held-out split (**r = −0.29**). Consequently CHAB-lite's low-budget win is
partly incidental and fixed-k remains a strong baseline. The proposal's controller claim (H1/H3/H5)
is therefore **not yet demonstrated** — this is expected at N=50 with a linear estimator, and it is
exactly the kind of evidence that should shape the full study (better estimator, larger N).

**Limitations to state to the supervisor:**
- N=50 questions, single dataset, single backbone (3B), single retriever (BM25), CPU only.
- ΔQ estimator is a linear ridge on cheap features (pilot-only).
- Evidence-level allocation (proposal Level-2 decision) not implemented.
- Decode-cost signal noisy on shared CPU; a proper TPOT(L) curve is needed.

---

## 5. What's next / remaining

Mapped to the proposal's experiment set (Sec. 12.5–12.7).

### 5.1 Immediate (strengthen the pilot)
- [ ] Scale **N** to 150–300 questions (`data.n_questions`) — cheap, grid is resumable.
- [ ] Add a second backbone (**Llama-3.2-3B**) for a cross-model signal.
- [ ] Add a **dense retriever** (`--retriever dense`) and compare BM25 vs dense rankings.
- [ ] Replace the failing ridge ΔQ estimator with a better one (gradient boosting / logistic
      classifier on richer query + retrieval features); re-check the CHAB-lite frontier.
- [ ] Measure a proper **TPOT(L) curve** (decode time vs context length) on a quiet machine.

### 5.2 Core experiments (proposal E1–E6)
- [ ] **E1** Budget-matched retrieval: fixed-k vs adaptive vs calibrated vs CHAB at equal realized budget.
- [ ] **E2** Calibration accuracy: mean/median/P95 of \|B̂−B\|, conditional on difficulty, cross-dataset/model.
- [ ] **E3** Retrieval-harm analysis at scale (beneficial/neutral/harmful partition, magnitudes).
- [ ] **E4** Decode-aware vs document-count vs token vs latency budgets at matched realized compute.
- [ ] **E5** Quality–cost–harm Pareto frontiers across budget levels.
- [ ] **E6** Transferability: cross-dataset, cross-difficulty, cross-corpus, cross-backbone.

### 5.3 Ablations (proposal A1–A7)
- [ ] A1 fixed-k vs adaptive · A2 uncertainty vs calibrated · A3 benefit vs benefit+harm
- [ ] A4 doc-count vs token vs decode-aware cost · A5 no vs in-domain vs cross-domain calibration
- [ ] A6 retrieval-only vs + evidence allocation · A7 token-cost vs measured TPOT(L)

### 5.4 Missing components (proposal Sections 12.3–12.4)
- [ ] **Evidence-level allocation (Level 2):** sentence/passage scoring by net marginal utility
      per unit compute (`Score(sᵢ) = [MarginalBenefit − μ·MarginalHarm] / EstimatedComputeCost`).
- [ ] Full **decode-aware cost model**: `C = C_retrieval + C_rerank + C_prefill + C_decode`, with
      `C_decode ≈ N_out × TPOT(L)`.
- [ ] Conformal / risk-controlled threshold calibration (refs [28], [29]) with an ε guarantee.
- [ ] More datasets: 2WikiMultihopQA, MuSiQue, NQ, TriviaQA (multi-hop + single-hop + transfer).

### 5.5 Deliverables still to produce
- [ ] `tables/` with the final E1–E6 result tables.
- [ ] An extended `SUMMARY` / paper-ready “Preliminary Results” section for the proposal.
- [ ] Released config + seeds + telemetry for reproducibility.

---

## 6. How to run

```bash
cd experiments/chab-rag-pilot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
ollama pull qwen2.5:3b-instruct                                       # once

# quick check
.venv/bin/python -m src.run_grid --limit 5 --ks 0,3,10
.venv/bin/python -m src.analyze

# full pilot
bash scripts/run_pilot.sh
```

Outputs land in `results/`: `PRELIMINARY_RESULTS.md`, 5 figures, `summary.json`,
`calibration.csv`, `matched_budget_table.csv`, `frontiers.csv`, `frame.csv`.
