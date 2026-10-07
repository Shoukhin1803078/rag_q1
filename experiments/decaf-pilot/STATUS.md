# DECAF Pilot — Status & Progress

**Date:** 2026-10-07
**Scope:** Phase 0 — smoke test / preliminary-results pilot for the DECAF proposal
(*"When Does Context Compression Accelerate CPU RAG? A Decode-Centric Study"*).
**State:** ✅ Complete and runnable end-to-end. Decode-scaling, break-even and
compression results produced.

Location: `experiments/decaf-pilot/`

---

## 1. Summary

DECAF's thesis is that compression's benefit on CPU must be **measured, not assumed**: the
compressor costs time, decode cost depends on context length, and the question is whether
compression is a *net* win. Its backbone is the empirical `TPOT(L)` curve and the break-even
context length `L*`.

This pilot builds that harness on the project's own CPU/llama.cpp setup and runs the core
experiments E1/E8 (scaling), E2/E3/E4 (compression, overhead, break-even), E7 (grounding) and the
§17 component ablation.

**Bottom line:** on this machine the proposal's central CPU premise is **reversed** — prefill, not
decode, dominates end-to-end latency for any realistic context, because TPOT is nearly flat in `L`.
Compression is nevertheless a clear net latency win (≈4×) above a trivial threshold, but that win
is in **prefill/TTFT, not decode**, and it costs grounding. The CPU-aware selector does not beat a
plain fixed-ratio ranking on quality.

---

## 2. What has been done

### Codebase (new, under `experiments/decaf-pilot/`)

| File | Purpose |
|---|---|
| `src/contexts.py` | sentence segmentation, token estimation, filler corpus for controlled length |
| `src/select.py` | BM25 relevance, coverage-aware DECAF selector, baselines, `T_compress` |
| `src/run_experiments.py` | **Phase A** — scaling sweep (with warmup + prefix-cache nonce) and compression methods; resumable |
| `src/analyze.py` | **Phase B** — `TPOT(L)`/`prefill(L)` fits, `L*`, net speedup, Pareto, grounding, ablation |
| `src/llm.py` | streaming Ollama client (TTFT) with explicit `num_ctx` |
| `src/data.py`, `src/retriever.py`, `src/metrics.py` | HotpotQA loader (with gold supporting facts), BM25, EM/F1 |

### Runs completed

220 runs (10 scaling + 210 compression), ~6 min CPU. Two measurement bugs found and fixed (§4).

---

## 3. Results

Outputs in `experiments/decaf-pilot/results/`.

### 3.1 Decode scaling (RQ1 / E1, E8) — `fig_scaling.png`

| L (tokens) | prefill ms | TPOT ms/tok | decode share |
|---|---|---|---|
| 362 | 1091 | 36.8 | 0.52 |
| 645 | 2124 | 37.9 | 0.36 |
| 1179 | 4322 | 37.0 | 0.22 |
| 2241 | 8447 | 36.9 | 0.12 |
| 4356 | 17507 | 40.2 | 0.07 |

- `prefill ≈ 4.1 ms/token` (r²=1.00, mildly super-linear); `TPOT ≈ 36.5 + 0.7 ms per 1k tokens` (r²=0.68).
- **Decode/prefill crossover L ≈ 414 tokens** (32-token outputs); **prefill dominates above it.**
- **H1 refuted**: the decode share *falls* with L, and compression's gain is in TTFT, not TPOT.

### 3.2 Break-even L* (RQ2 / E3, E4) — `fig_break_even.png`
- `T_compress = 1.36 ms`, retained `L_c ≈ 205 tokens` → **`L* ≈ 205 tokens`**.
- **H2 supported but trivial on CPU**: because overhead ≪ per-token prefill cost, compression is a
  net latency win for *any* context above the retained budget.

### 3.3 Compression: quality, grounding, latency (RQ3 / E2, E7) — `fig_pareto_grounding.png`

| method | F1 | ev. recall | gold-in-ev | retained | T_compress ms | E2E ms | net speedup |
|---|---|---|---|---|---|---|---|
| Full context | 0.221 | 0.751 | 0.833 | 1.00 | 0.84 | 3713 | 1.00× |
| Fixed 50% | 0.285 | 0.668 | 0.767 | 0.572 | 1.09 | 949 | 3.91× |
| Fixed 20% | 0.265 | 0.562 | 0.600 | 0.249 | 1.03 | 912 | 4.07× |
| Relevance top-m (iso) | 0.240 | 0.573 | 0.567 | 0.261 | 1.12 | 1172 | 3.17× |
| DECAF λ=6 | 0.268 | 0.532 | 0.600 | 0.190 | 1.36 | 988 | 3.76× |
| DECAF λ=15 | 0.122 | 0.279 | 0.200 | 0.059 | 1.38 | 1147 | 3.24× |
| DECAF no-cov (λ=6) | 0.302 | 0.671 | 0.733 | 0.470 | 1.00 | 1504 | 2.47× |

- Compression cuts latency **~4×** and, at moderate ratios, **raises F1** (0.221 → 0.285): distractor
  context hurts the small model.
- **Grounding degrades monotonically** with retained evidence (0.833 → 0.60 → 0.20) — accuracy alone
  hides this.

### 3.4 Component analysis (§17) — `fig_ablation.png`
- Iso-budget relevance vs DECAF(λ6): F1 0.240 → 0.268, gold-in-evidence 0.567 → 0.600 (small gain),
  evidence recall 0.573 → 0.532 (small loss).
- **H3 not supported**: DECAF does not beat fixed-ratio selection (fixed-50% 0.285; fixed-20% 0.265).

---

## 4. Bugs found and honest interpretation

Two measurement bugs were caught and fixed — both matter for a latency study:

1. **Ollama context cap.** With no `num_ctx` set, Ollama truncated prompts (~2048 tokens): a
   4096-token target reported only 2050 tokens. Fixed by setting `num_ctx: 8192`; the token counts
   are now monotonic (345 → 4337).
2. **Cross-run prompt-prefix cache.** Longer prompts sharing a prefix with earlier runs re-used the
   KV prefix, making prefill appear near-free (an impossible 131 ms for 1160 tokens). Fixed with a
   unique per-run leading nonce; the prefill curve is now clean and monotonic.

**Interpretation.** The proposal's CPU hypothesis (decode-dominated; H1) is not supported on this
3B/Q4/CPU setup with ~32-token outputs — prefill dominates above ~400 tokens. The strongest, most
useful result is the honest reframing: **compression always pays off on CPU latency (overhead is
negligible vs prefill cost), and the real decision is how much grounding you are willing to trade**.
DECAF's CPU-aware term does not beat plain relevance ranking here.

**Limits:** one model/quantization, 30 questions, 32-token generations, BM25-based scorer (not the
proposal's cross-encoder), Reddit-free synthetic filler for the scaling sweep.

---

## 5. What's next / remaining

### 5.1 Immediate (strengthen the pilot)
- [ ] Add the **Q4_K_M vs Q8_0** and **3B vs 7B** pairs (H4, E5, E6) — one `ollama pull` each.
- [ ] Sweep **generation length** N_out (16/64/128) to locate the decode/prefill crossover properly (H1).
- [ ] Replace BM25 scoring with the proposal's **cross-encoder** (`bge-reranker-base`) and re-measure `T_compress`.
- [ ] Add a **break-even by quantization/model** table (a stated contribution).

### 5.2 Core experiments (E1–E9)
- [ ] E1 long-context points (8K, 16K) on a small query subset.
- [ ] E2/E3 full compression-ratio sweep with `O_c`, `S_net` curves.
- [ ] E7 grounding with citation correctness where gold citations exist.
- [ ] E8 achieved-bandwidth / CPU-utilization telemetry against the roofline model.
- [ ] E9 generalization: 2WikiMultihopQA, TriviaQA/NQ, LongBench subset.

### 5.3 Baselines & ablations
- [ ] LLMLingua-2 and Perception Compressor as external CPU baselines (or state the compute barrier).
- [ ] Full §17 ablation: scorer on/off, CPU-cost on/off, coverage on/off, reordering on/off, λ sweep.

### 5.4 Secondary systems analysis (H5)
- [ ] llama.cpp prefix/KV cache on repeated queries; async retrieval prefetch overlapped with decode.

### 5.5 Statistics & deliverables
- [ ] Paired bootstrap CIs and McNemar tests per the proposal's §21; Holm correction.
- [ ] Release configs, seeds, telemetry logs; paper-ready "Preliminary Results" section.

---

## 6. How to run

```bash
cd experiments/decaf-pilot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
ollama pull qwen2.5:3b-instruct                                       # once

.venv/bin/python -m src.run_experiments --mode scaling --lengths 256,512   # quick
.venv/bin/python -m src.analyze
bash scripts/run_pilot.sh                                                  # full pilot
```

Outputs in `results/`: `PRELIMINARY_RESULTS.md`, 4 figures, `summary.json`.
