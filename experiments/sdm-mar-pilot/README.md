# SDM-MAR — smoke-test / preliminary-results pilot

A small pilot for the proposal *"Beyond Hop Count: A Structural, Mechanistic and
Predictive Account of Difficulty in Multimodal Agentic Retrieval"*.

It asks the proposal's central question in miniature: **do structural variables
(H, CSL, MS, edge types, RC) predict per-question success of agentic retrieval
better than hop count alone?** — and runs the three one-factor interventions with
their pre-stated directions.

> **Harness check, not a discovery.** Structure is *planted* by construction, so
> structure→success links are partly expected. H and CSL are driven by a **real
> LLM reader** (its disambiguation under hard negatives, H and CSL); the
> image-modality barrier is **simulated**, because the pilot's reader
> (`qwen2.5:3b-instruct`) is text-only and cannot perform OCR. This mirrors the
> honesty of the proposal's own Appendix C.

## Design

A synthetic **multimodal evidence-graph corpus** (1,709 docs) with planted bridge
chains, plus a hop-based agent that queries, reads, and extracts the next bridge:

- **Bridge-blindness by construction:** hop *j≥2*'s subject is only revealed by hop
  *j−1*, so a query built from the question alone cannot retrieve it (`A1`).
- **CSL:** each hop adds *c* near-duplicate records sharing the subject and
  relation with the gold record but stating a different value → hard-negative
  neighbourhood of size *c*.
- **Modality:** image nodes are indexed only by a short caption, and a text-only
  reader recovers their content only with probability `p_image` (simulated
  imperfect OCR).
- **Retriever:** BM25 by default; TF-IDF→SVD at dimension *d* for the RC sweep.
- **Agent:** reads each round; the **LLM is invoked exactly when the retrieved
  notes are ambiguous** (more than one competing value), i.e. under high CSL;
  otherwise the read is resolved deterministically. A wrong bridge is propagated
  (not corrected), so error propagation is measured.

## Runs

`baseline`, `oracle_bridge`, `caption` × 60 questions × 3 replicates, plus a
dimension sweep (d ∈ {32,64,128,256,512}) on a subsample. 790 runs total,
~660 LLM calls, ~6 min on CPU. Resumable.

## What it produces

| Figure / table | Proposal link |
|---|---|
| `fig_success_structure.png` — success vs H, MS, CSL | RQ2 separability |
| `fig_models.png` — Models A/B/C AUC + ablations | RQ1, §12.5 |
| `fig_interventions.png` — oracle bridge / caption / dimension | RQ3 |
| `fig_product_law.png` — product law vs measured | RQ4 |
| `features.csv`, `ablations.csv`, `dimension_sweep.csv`, `product_law.csv` | tables |
| `PRELIMINARY_RESULTS.md`, `summary.json` | drop-in report |

## Headline results

- **Hop count barely predicts success:** Model A AUC **0.514** (≈ chance).
- **Structure predicts:** Model C AUC **0.800**; ΔAUC(C−A) = **0.286**, clustered
  95% CI **[0.142, 0.432]** — clears the proposal's pre-stated ΔAUC ≥ 0.03.
- **CSL carries the signal, depth does not:** `hop+csl` = 0.797; removing CSL drops
  to 0.769; removing H *raises* AUC to 0.820.
- **Images are the weak link:** hop success 0.34 (text) vs 0.20 (image).
- **Interventions:** (A) oracle-bridge gain grows with H → **confirmed**;
  (B) caption helps image-bearing chains → **confirmed (weak)**;
  (C) dimension gain grows with CSL → **not confirmed** (helps low-CSL more).
- **Hops are not independent:** product of marginals MAE 0.35, corr 0.10.

## Setup / run

```bash
ollama pull qwen2.5:3b-instruct      # once
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m src.run_runs --limit 4 --skip-dims   # quick check
.venv/bin/python -m src.analyze
bash scripts/run_pilot.sh                                 # full pilot
```

Config knobs in `config.yaml`: `n_questions`, `hops`, `csl_levels`, `p_image`,
`modality_p`, `top_k`, `replicates`, `dims`.

## Non-goals

Real images / a real VLM, real multimodal datasets (CrossModalQA, VisDocAgentBench),
per-question hop-level mixed-effects modelling, and cross-dataset frozen transfer
are the proposal's next stage, not this pilot.
