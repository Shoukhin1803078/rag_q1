# PACER — smoke-test / preliminary-results pilot

A small, CPU-feasible pilot that produces real numbers and figures for the PACER
proposal's core coordination hypothesis: *does prediction-conditioned adaptive
evidence budgeting reduce RAG latency at non-inferior quality?*

It is a **sanity check**, not the full experiment set (S0–S7 on a GPU).

> **Scope note.** PACER targets a single consumer GPU. This pilot runs on CPU
> (Ollama, `qwen2.5:3b-instruct`), so it validates the **mechanism and relative
> effects**, not published absolute speedups. KV-cache reuse (S3/S5–S7) is
> Phase-2 in the proposal and is **out of scope** here.

## Systems compared

| ID | System | How produced |
|---|---|---|
| — | Closed-book (no retrieval) | generated |
| S0 | Standard RAG (full context) | generated |
| S1 | Prediction gating over {closed-book, S0} | assembled offline |
| S2 | Fixed-ratio compression | generated |
| S2 | Query-adaptive compression | generated |
| S4 | Prediction gating over {closed-book, adaptive compression} | assembled offline |

S1/S4 only *select* among cached runs, so they add no generation cost and the
component comparison stays matched.

## What it produces

| Figure / table | Proposal link |
|---|---|
| `fig_latency_decomposition.png` — retrieval/compression/prefill/decode split | H5, S0 reference |
| `fig_pareto.png` — quality vs E2E operating points | RQ1 / H1 / H4 |
| `fig_compression.png` — compression ratio vs quality | H2 / RQ2 |
| `fig_retrieval_accounting.png` — retrieval & wasted-retrieval rates | H1 |
| `fig_workload.png` — stage share vs output length | H5 |
| `system_summary.csv`, `paired_stats.csv` | §21 statistical analysis |
| `PRELIMINARY_RESULTS.md`, `summary.json` | drop-in report |

## Headline results (40-question pilot)

- **S4 ≈ S0 quality at ~1/3 the latency:** 1047 ms vs 3292 ms E2E (−68%), F1 0.23 vs 0.23
  (paired 95% CI on the F1 difference includes 0; E2E difference p_holm < 0.001, Cliff's δ = −0.97).
- **Adaptive compression is the strongest single component:** 1213 ms, F1 0.29 (it raises quality
  by removing hard-negative context) at 7.2× compression.
- **Coordination's incremental gain is small here:** S4 vs compression-only is −65 ms E2E
  (p_holm = 0.38) — a legitimate "coordination may match the best single technique" outcome.
- **Wasted retrieval is real:** ~45–50% of retrievals do not beat the closed-book answer.
- **Generation dominates:** corr(output length, decode share of E2E) = 0.88 → retrieval-side
  optimization matters less as generations lengthen (H5).

Numbers are indicative only (small N, one backbone, one dataset, CPU).

## Design

- Data: HotpotQA `distractor` (10 candidate paragraphs/question, BM25 top-5).
- **TTFT is measured directly** via streamed Ollama responses; prefill/decode come from
  Ollama's final-chunk telemetry.
- Compression is training-free: BM25 sentence scoring + either a fixed retained fraction or a
  relative-threshold (evidence-sufficiency) budget; its own time is measured.
- The retrieval gate uses closed-book mean token log-probability (self-knowledge) and is
  calibrated to **minimise retrieval subject to F1 ≥ always-retrieve − ε** (quality-constrained).

## Setup

```bash
ollama pull qwen2.5:3b-instruct      # once
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# quick check
.venv/bin/python -m src.run_harness --limit 4
.venv/bin/python -m src.analyze

# full pilot (config default: 40 questions × 4 base systems = 160 runs, ~5 min CPU)
bash scripts/run_pilot.sh
```

The harness is **resumable** — re-running skips cached `(qid, system)` pairs. Scale with
`data.n_questions` in `config.yaml`.

## Config knobs

`config.yaml`: seed, `n_questions`, `top_k`, model, compression (`fixed_keep_frac`,
`adaptive_rel_threshold`, `adaptive_max_words`), and `controller.quality_tolerance` (the ε of the
quality constraint).

## Non-goals

KV-cache reuse (S3/S5/S6/S7), GPU latency measurement, LongBench/NQ/2Wiki generalization, and
multimodal extension are all out of scope for this first pass.
