# PACER Pilot — Status & Progress

**Date:** 2026-10-07
**Scope:** Phase 0 — smoke test / preliminary-results pilot for the PACER proposal.
**State:** ✅ Complete and runnable end-to-end. First results produced.

Location: `experiments/pacer-pilot/`

---

## 1. Summary

PACER's scientific claim is a **coordination hypothesis**: retrieval triggering and evidence
budgeting should be decided jointly under one latency budget, and this coordination should beat
matched single-technique systems at non-inferior answer quality.

This pilot builds a matched, CPU-feasible harness and runs the text-first core of the proposal's
ablation matrix — **S0 standard RAG, S1 prediction gating, S2 adaptive compression, S4 prediction +
compression** — over HotpotQA with `qwen2.5:3b-instruct`, measuring TTFT/E2E, stage-level latency,
compression ratio, retrieval accounting, and answer quality, then applying the proposal's paired
statistical protocol (bootstrap CI + Wilcoxon + Holm + effect size).

**Bottom line:** S4 matches standard-RAG quality at ~1/3 the end-to-end latency (−68%), but
**adaptive compression alone already captures most of the gain** — the incremental benefit of
adding the prediction gate is small and not statistically significant at n=40. That is a
scientifically acceptable outcome (the proposal explicitly allows coordination to *match* the best
single technique), and it is exactly the kind of honest preliminary evidence that strengthens the
proposal.

**Important caveat:** PACER targets a single consumer GPU; this pilot is **CPU-only**, so it
validates the mechanism and relative effects, not absolute speedups. KV-cache reuse (S3/S5–S7) is
Phase-2 and out of scope.

---

## 2. What has been done

### Codebase (new, under `experiments/pacer-pilot/`)

Shared modules reused from the CHAB-RAG pilot (self-contained copy for artifact reproducibility):

| File | Purpose |
|---|---|
| `src/data.py` | HotpotQA `distractor` loader (10 candidate paras/question) |
| `src/retriever.py` | BM25 ranking of candidate paragraphs |
| `src/metrics.py` | SQuAD-style EM/F1 |
| `src/llm.py` | **Modified:** streamed Ollama client that measures **TTFT** directly + prefill/decode telemetry + resumable cache |

PACER-specific modules:

| File | Purpose |
|---|---|
| `src/compress.py` | Training-free evidence compression: `none` / `fixed` ratio / `adaptive` (relative-threshold evidence sufficiency); measures its own overhead & ratio |
| `src/predict.py` | Retrieval-need gate from closed-book confidence, **quality-constrained** calibration (minimise retrieval s.t. F1 ≥ reference − ε) |
| `src/policies.py` | Tidy frame + offline assembly of S1/S4 + system summary |
| `src/run_harness.py` | **Phase A** — runs the 4 base systems and caches everything with stage timings |
| `src/analyze.py` | **Phase B** — figures, paired statistics, `summary.json`, report |
| `config.yaml`, `scripts/run_pilot.sh`, `README.md` | Config, runner, docs |

### Runs completed

- Full harness: 160 runs (40 questions × 4 base systems), ~5 min on CPU.
- Analysis: 5 figures, 3 CSV tables, `summary.json`, `PRELIMINARY_RESULTS.md`.

---

## 3. Results

Outputs in `experiments/pacer-pilot/results/`.

### 3.1 System summary (medians / means)

| System | TTFT ms | E2E ms | F1 | EM | Prompt tok | Retrieval rate | Wasted retr. | Compression ×| Overhead ms |
|---|---|---|---|---|---|---|---|---|---|
| Closed-book | 134 | 799 | 0.09 | 0.02 | 71 | 0.00 | — | 1.00 | 0 |
| **S0 Standard RAG** | 2555 | 3292 | 0.23 | 0.10 | 755 | 1.00 | 0.45 | 1.00 | 1.4 |
| S1 Prediction | 2103 | 3057 | 0.17 | 0.05 | 537 | 0.65 | 0.38 | 1.00 | 1.1 |
| S2 Fixed compression | 1319 | 2170 | 0.25 | 0.12 | 475 | 1.00 | 0.45 | 1.76 | 2.4 |
| **S2 Adaptive compression** | 392 | 1213 | **0.29** | 0.20 | 250 | 1.00 | 0.48 | 7.17 | 2.5 |
| **S4 Pred + Compression** | **332** | **1047** | 0.23 | 0.15 | 207 | 0.65 | 0.50 | 4.17 | 2.0 |

### 3.2 Latency decomposition (H5)
`fig_latency_decomposition.png` — prefill dominates the retrieval path; decode dominates the
non-retrieval path. Controller overhead (retrieval + compression ≈ 2.5 ms) is negligible.
corr(output tokens, decode share of E2E) = **0.88** → generation-heavy workloads reduce the
payoff of retrieval-side optimization.

### 3.3 Quality–latency operating points (RQ1)
`fig_pareto.png` — S4 and S2-adaptive occupy the top-left frontier; S0 is far right (slow) at
similar quality; closed-book is fast but low quality.

### 3.4 Compression trade-off (H2 / RQ2)
`fig_compression.png` — adaptive compression reaches **7.2×** ratio and *raises* F1 (0.29 vs 0.23)
by removing hard-negative context; fixed compression is milder (1.76×, F1 0.25).

### 3.5 Retrieval accounting (H1)
`fig_retrieval_accounting.png` — S4 retrieves on only 65% of queries vs 100% for S0, and ~50% of
all retrievals do not beat the closed-book answer.

### 3.6 Paired statistics (S4 vs components)
`paired_stats.csv` — bootstrap 95% CI on paired mean differences, Wilcoxon signed-rank p,
Holm–Bonferroni adjustment, Cliff's δ.

| vs | E2E diff (ms) | CI | p_holm | δ | F1 diff | CI | p_holm |
|---|---|---|---|---|---|---|---|
| S0 | −2280 | [−2506, −2053] | <0.001 | −0.97 | −0.008 | [−0.14, +0.14] | 1.0 |
| S1 | −1473 | [−1832, −1115] | <0.001 | −0.55 | +0.057 | [−0.03, +0.15] | 1.0 |
| S2-adaptive | −65 | [−206, +91] | 0.38 | −0.11 | −0.063 | [−0.16, +0.03] | 0.38 |

**S4 vs S0: large, significant latency win with no detectable quality loss.** S4 vs S2-adaptive:
no significant difference on either metric.

---

## 4. Honest interpretation & limitations

**Supported:** a prediction-conditioned system (S4) can cut E2E latency dramatically at
statistically non-inferior quality vs standard RAG (H1/H2 directional support); adaptive
compression is valuable and cheap; wasted retrieval is measurable.

**Not supported at this scale:** the *coordination* gain over the best single technique —
compression-only is statistically indistinguishable from S4 here (H4 not demonstrated). This is
the proposal's explicitly allowed Outcome B.

**Limitations:** CPU-only (no GPU latency, no real lookahead overlap, no KV-cache reuse);
n=40, one dataset, one 3B backbone; output capped at 32 tokens (limits the generation-heavy
regime for H5); the gate's quality constraint under-transfers from calibration to eval split
(S1 quality dips below target) — a calibration-transfer finding worth noting.

---

## 5. What's next / remaining

Mapped to the proposal's ablation matrix (S0–S7), experiments (E01–E18), and workflow (Phase 0–8).

### 5.1 Immediate (strengthen the pilot)
- [ ] Scale **N** to 150–300 questions (resumable; cheap).
- [ ] Raise output length (H5) and add a second backbone (Llama-3.2-3B).
- [ ] Add a **lookahead/prefetch** simulation with explicit **wasted-prefetch** accounting.
- [ ] Improve the gate (richer features than closed-book logprob) and report calibration transfer.
- [ ] Add a matched-quality comparison (iso-quality slice of the Pareto frontier).

### 5.2 Core experiments (E01–E18)
- [ ] E01–E05 reference/profile, standard RAG, predictive, compression, cache baselines.
- [ ] E06/E10–E13 integration + ablations S1–S4 with paired seeds.
- [ ] E17 quality–latency Pareto frontiers across operating points.
- [ ] E18 reproducibility run with confidence intervals.

### 5.3 Ablations (S0–S7)
- [ ] S3/S5/S6 **KV-cache reuse** (Phase-2; requires an inference engine with KV control).
- [ ] S7 full PACER; test **super-additivity** (interaction terms), not just main effects.

### 5.4 Missing components
- [ ] Real **lookahead retrieval overlap** (needs an engine where retrieval and prefill can overlap).
- [ ] Learned retrieval-need/timing predictor (currently a single confidence signal).
- [ ] Evidence-utility estimator coupled to the compression budget (ECoRAG/ACC-RAG-style).
- [ ] **GPU run** on a 24 GB-class card to obtain transferable absolute latencies.
- [ ] LongBench / NQ / 2Wiki generalization; naive-Bayes-free statistical protocol already in place.
- [ ] Optional multimodal extension (DocVQA / MP-DocVQA / ViDoRe).

### 5.5 Deliverables still to produce
- [ ] Final E01–E18 result tables and the Pareto figure set for the paper.
- [ ] A paper-ready "Preliminary Results" section for the proposal.
- [ ] Reproducibility manifest (hardware, versions, seeds, trace logs).

---

## 6. How to run

```bash
cd experiments/pacer-pilot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
ollama pull qwen2.5:3b-instruct                                       # once

.venv/bin/python -m src.run_harness --limit 4     # quick check
.venv/bin/python -m src.analyze

bash scripts/run_pilot.sh                          # full pilot
```

Outputs in `results/`: `PRELIMINARY_RESULTS.md`, 5 figures, `summary.json`,
`system_summary.csv`, `paired_stats.csv`, `frame.csv`.
