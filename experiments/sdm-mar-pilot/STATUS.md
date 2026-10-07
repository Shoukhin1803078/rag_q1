# SDM-MAR Pilot — Status & Progress

**Date:** 2026-10-07
**Scope:** Phase 0 — smoke test / preliminary-results pilot for the SDM-MAR proposal
("Beyond Hop Count", multimodal agentic retrieval difficulty).
**State:** ✅ Complete and runnable end-to-end. First real-agent results produced.

Location: `experiments/sdm-mar-pilot/`

---

## 1. Summary

The proposal's central claim is predictive and falsifiable: *the difficulty of a multimodal
agentic retrieval question has separable structural causes (depth H, concept-separation load CSL,
modality switches MS, edge type, retriever capacity RC) that predict agent success out of sample
beyond hop count.* The proposal states that only an idealised simulation exists and that **no real
agent results exist yet**.

This pilot builds the missing harness and runs a **real LLM reader** over a controllable
multimodal evidence-graph corpus whose structure is known, then fits the proposal's three models
(A hop only, B retrieval-score, C structural) with grouped cross-validation and clustered
statistics, runs the three one-factor interventions, and checks the hop-level product law.

**Bottom line:** hop count is nearly useless as a predictor (AUC 0.514), while the structural
model reaches AUC 0.800 with ΔAUC(C−A) = 0.286 (clustered 95% CI [0.142, 0.432]) — the core
hypothesis is supported in the harness. Of the three interventions, the oracle-bridge prediction
is confirmed and the caption prediction is weakly confirmed; the dimension prediction is **not**
confirmed, for reasons discussed below.

**Caveat that must travel with these numbers:** the structure is *planted* by construction, so
structure→success links are expected; H and CSL are driven by the real LLM, but the image-modality
barrier is **simulated** (the pilot's reader is text-only).

---

## 2. What has been done

### Codebase (new, under `experiments/sdm-mar-pilot/`)

| File | Purpose |
|---|---|
| `src/synth.py` | Synthetic multimodal corpus + planted bridge chains; controls H, CSL, MS, edge types |
| `src/retriever.py` | BM25 and TF-IDF→SVD retriever with controllable dimension (RC); seeded tie-breaking for hard negatives |
| `src/agent.py` | Hop-based agentic loop; LLM invoked only on ambiguous (high-CSL) reads; wrong bridges propagate |
| `src/run_runs.py` | **Phase A** — conditions × questions × replicates + dimension sweep, resumable cache |
| `src/models.py` | Models A/B/C, grouped CV (GroupKFold by question), AUC/Brier/log-loss/calibration, clustered bootstrap ΔAUC |
| `src/analyze.py` | **Phase B** — ablations, interventions, product law, figures, report |
| `src/llm.py`, `src/metrics.py` | shared streaming Ollama client and metrics (from earlier pilots) |
| `config.yaml`, `scripts/run_pilot.sh`, `README.md` | config, runner, docs |

### Runs completed

790 agent runs (60 questions × 3 conditions × 3 replicates + dimension sweep), ~660 LLM calls,
~6 min CPU. Two iterations: initial run, then a corrected **caption** intervention (see §4).

---

## 3. Results

Outputs in `experiments/sdm-mar-pilot/results/`.

### 3.1 Success vs structure (baseline)
`fig_success_structure.png`
- Overall baseline success **0.467**.
- By depth H: 1→0.53, 2→0.44, 3→0.42 (weak trend).
- By CSL tercile: **0.71 → 0.30 → 0.17** (strong).
- Hop success by node modality: **text 0.34, image 0.20** → images are the weak link.

### 3.2 Prediction: Models A/B/C (grouped CV)
`fig_models.png`, `ablations.csv`

| Model | AUC | Brier | log-loss | cal. slope |
|---|---|---|---|---|
| A — hop count | **0.514** | 0.252 | 0.698 | −0.19 |
| B — retrieval score | 0.701 | 0.216 | 0.628 | 0.78 |
| C — structure | **0.800** | 0.177 | 0.557 | 0.76 |

**ΔAUC(C−A) = 0.286**, clustered 95% CI **[0.142, 0.432]** (clears the pre-stated ≥ 0.03).

Ablations: `hop+csl` 0.797 · `minus_CSL` **0.769** · `minus_H` **0.820** · `minus_MS` 0.800.
→ **CSL carries the signal; depth H adds nothing** in this harness.

### 3.3 Interventions (RQ3)
`fig_interventions.png`
- **A (oracle bridge):** gain grows with H — +0.47, +0.56, +0.58 → **confirmed**.
- **B (image→caption):** gain +0.09 on image-bearing chains vs −0.07 without → **confirmed (weak)**.
- **C (dimension sweep):** gain d=32→512 is +0.42 (low CSL) vs +0.12 (high CSL) → **not confirmed**
  (prediction said gain grows with CSL).

### 3.4 Hop-level product law (RQ4)
`fig_product_law.png` — MAE 0.350, corr 0.10. The product of marginal per-hop success is mostly a
function of H and under-estimates measured success: hops are dependent, not independent.

---

## 4. Honest interpretation, bugs found, and limits

**Supported:** structure predicts success far better than hop count; CSL is the dominant factor;
image hops are the least reliable; the oracle-bridge remedy is confirmed and grows with depth.

**Not supported (and useful):** the dimension intervention's pre-stated direction is reversed. This
is **partly a harness degeneracy**: high-CSL distractors differ from gold mainly by a random
identifier token, so no embedding dimension can separate them; a real corpus would not have this
degeneracy. The caption intervention initially *hurt* because my first implementation enriched
distractor images as well as evidence images — a modelling bug I found and fixed so captions apply
to the evidence image nodes only (as the proposal intends).

**Limits:** structure is planted; the modality barrier is simulated (text-only reader); n=60
questions; the product-law comparison uses marginal per-hop rates rather than a per-question
hop-level model; there is no cross-dataset transfer yet.

---

## 5. What's next / remaining

### 5.1 Immediate
- [ ] Replace the simulated modality barrier with a **real VLM** (e.g. qwen2.5vl) on a small real image set.
- [ ] Add the **hop-level mixed-effects model with an error-propagation term** (proposal §12.5), nesting the product law.
- [ ] Make CSL distractors semantically (not lexically) similar so the dimension sweep is identified.
- [ ] Scale questions/replicates and add bootstrapped CIs to the intervention gains.

### 5.2 Real data (proposal §12.6)
- [ ] Obtain **CrossModalQA** (contact authors) or fall back to MultimodalQA / Double-Bench.
- [ ] Annotate H, MS, edge types, CSL; build the **100-question human-validation** set (κ / ICC).
- [ ] Run **three independent agent families** with 5 replicates per question.

### 5.3 Statistics & transfer
- [ ] Mixed-effects logistic regression with question random effects; cluster-robust errors.
- [ ] **Frozen-coefficient transfer** to VisDocAgentBench / Double-Bench / BrowseComp-V3.
- [ ] Holm correction across the pre-registered comparison family; report effect sizes + CIs.

### 5.4 Failure analysis
- [ ] Mechanism-labelled failure taxonomy (wrong bridge, wrong query, wrong image interpretation,
      retrieval miss, chain break, recombination failure) cross-tabulated by edge type, H and CSL.

### 5.5 Deliverables
- [ ] Annotation protocol + code release (Appendix B), run logs, prompts, seeds.
- [ ] Paper-ready "Preliminary Results" section for the proposal.

---

## 6. How to run

```bash
cd experiments/sdm-mar-pilot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
ollama pull qwen2.5:3b-instruct                                       # once

.venv/bin/python -m src.run_runs --limit 4 --skip-dims   # quick check
.venv/bin/python -m src.analyze
bash scripts/run_pilot.sh                                 # full pilot
```

Outputs in `results/`: `PRELIMINARY_RESULTS.md`, 4 figures, `summary.json`, `features.csv`,
`ablations.csv`, `dimension_sweep.csv`, `product_law.csv`.
