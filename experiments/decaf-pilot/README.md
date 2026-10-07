# DECAF — smoke-test / preliminary-results pilot

A CPU-only pilot for the proposal *"When Does Context Compression Accelerate CPU
RAG? A Decode-Centric Study with Adaptive Evidence Selection"*.

It measures the proposal's empirical backbone directly on this machine
(Ollama/llama.cpp, Qwen2.5-3B Q4_K_M, CPU):

- **E1/E8** — the `TPOT(L)` and `prefill(L)` decode-scaling curves;
- **E2/E3/E4** — compression ratio vs quality, compressor overhead, and the
  **break-even context length `L*`**;
- **E7** — grounding (gold-evidence recall/precision, gold-in-context) under compression;
- **§17** — the component ablation (CPU-cost term and coverage term on/off).

## Design

- **Scaling sweep:** contexts of controlled length (256–4096 target tokens, verified
  by the reported prompt-token count) with a warmup and a unique per-run prefix nonce
  (to defeat Ollama's cross-run prompt-prefix cache).
- **Compression:** HotpotQA(distractor) top-5, sentence-level evidence selection.
  Methods: `full`, `fixed_0.5`, `fixed_0.2`, `rel_iso` (relevance top-m matched to
  DECAF's retained count — the iso-budget baseline), `decaf_lam6`, `decaf_lam15`,
  `decaf_nocov_lam6` (coverage ablation at matched λ).
- **DECAF scorer:** relevance (BM25) × coverage (marginal query-term coverage),
  accepted while `gain ≥ λ · cost`, where `cost ∝ sentence tokens` (the CPU decode
  cost is token-proportional). Single-pass, decoder-free; its own cost is measured.
- **Grounding:** evidence recall/precision against HotpotQA gold supporting facts,
  and whether the gold answer survives in the retained context.
- **`L*`:** solved from the fitted `prefill(L)` and `TPOT(L)` lines plus the measured
  compressor overhead.

## Headline results (10 scaling + 210 compression runs)

- **H1 is refuted here.** Decode's share of end-to-end latency **decreases** with
  context length (0.52 → 0.07): prefill grows ~4 ms/token while TPOT is nearly flat
  (≈37 ms, +0.7 ms per 1k tokens). Decode and prefill cross at **L ≈ 414 tokens**
  (for 32-token outputs); above it **prefill dominates** — the opposite of the
  proposal's CPU decode-dominated premise.
- **H2 holds but the threshold is trivial on CPU.** Compressor overhead is ~1.4 ms
  vs ~4 ms/token prefill, so `L* ≈ 205 tokens` ≈ the retained length: compression is
  a **net latency win for any context longer than the retained budget**. The real cost
  is quality/grounding, not latency.
- **H3 is not supported.** DECAF (F1 0.268) does not beat fixed-ratio selection
  (fixed-50% 0.285) and only edges the iso-budget relevance baseline (0.240).
- **Compression helps quality *and* latency, but hurts grounding:** F1 rises
  0.221 → 0.285 at ~4× lower latency (distractor context hurts the small model), while
  gold-in-evidence falls 0.833 → 0.60 → 0.20 as evidence is removed.

Absolute latencies are machine-specific; the **shapes and break-even behaviour** are
the point. One 3B quantized model, one dataset, 30 questions.

## Setup / run

```bash
ollama pull qwen2.5:3b-instruct      # once
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m src.run_experiments --mode scaling --lengths 256,512  # quick
.venv/bin/python -m src.analyze
bash scripts/run_pilot.sh                                                 # full pilot
```

Outputs in `results/`: `PRELIMINARY_RESULTS.md`, 4 figures, `summary.json`.

## Non-goals

Q4/Q8 and 3B/7B pairs, a real cross-encoder scorer, LongBench/2Wiki generalization,
energy/RAPL telemetry, and the KV-reuse/prefetch secondary study are the proposal's
next stage.
