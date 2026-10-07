**RESEARCH PROJECT PROPOSAL**

**When Does Context Compression Accelerate CPU RAG?**

**A Decode-Centric Study with Adaptive Evidence Selection**

*Framework name: **DECAF** — Decode-Centric Adaptive evidence selection For CPU-only RAG.*

*A controlled characterization of how retrieved-context length affects TTFT, per-token decode latency (TPOT) and end-to-end latency in CPU-only, quantized small-LLM RAG — and a training-free, CPU-aware compressor that operates on that characterization.*

Prepared by

**Al Amin Tokder**

Research Project Proposal — Efficient Retrieval-Augmented Generation

AI / ML & Systems-for-ML Research



---

# 1. Executive Summary

Context compression is one of the most widely proposed remedies for Retrieval-Augmented Generation (RAG) latency: fewer retrieved tokens should mean cheaper inference. Almost all of this evidence comes from GPU serving, where **prefill dominates** (e.g. one paper reports prefill at 95.53% of inference time). On **CPU**, the cost profile is different: generation is slow and largely memory-bound, and per-token decode latency (TPOT) scales with the KV-cache size, which scales with context length. Whether compression therefore *actually* pays off on CPU — once the compressor's own cost is counted — has not been characterized under controlled small-model conditions.

Prior work covers the two ingredients separately: CPU-only RAG deployment and CPU-only quantized-LLM benchmarking exist (HS-RAG, IEEE 2026; quantized CPU benchmarks, IEEE 2026), and training-free and adaptive context/prompt compression exist (SARA, ACL 2026; Perception Compressor, NAACL 2025 Findings; BRIEF-Pro, ACL 2026 Findings; LLMLingua-2; ECoRAG; ACC-RAG). What is missing is the **interaction**: how retrieved-context length, compression, quantization and model size jointly shape TTFT, TPOT, end-to-end latency, memory and answer grounding for a small quantized model running exclusively on a CPU.

This proposal makes that interaction its subject. It (1) defines a **measurement methodology** that separates prefill from decode and formalizes compression efficiency, overhead and net speedup; (2) introduces **DECAF**, a **single-pass, CPU-aware adaptive evidence-selection** mechanism that decides what to keep *before* invoking the expensive decoder, scoring each evidence unit by relevance/coverage gain per unit of estimated decode cost; (3) performs a **break-even analysis** identifying the context length `L*` above which compression becomes a net latency win; and (4) provides a **reproducible efficiency benchmark** on identical CPU hardware with quality, latency, memory, grounding and (where available) energy metrics and accuracy–latency Pareto frontiers.

The contribution is therefore not "a new training-free compressor" — that space is now crowded — but a **decode-centric characterization of when compression helps on CPU**, plus a compressor built on that characterization.

**Primary Research Question:** When does reducing retrieved context actually translate into lower CPU decoding cost, and what compression strategy maximizes answer quality per unit of CPU time?

---

# 2. Problem Statement

Latency is the practical blocker for local RAG. On CPU-only hardware — the setting most deployments without a GPU can afford — inference is slow enough that the difference between an interactive tool and an unusable one is often a matter of seconds per query. Compression is the obvious lever, but its benefit is *not* automatic:

- The compressor itself consumes CPU time (`T_compress`), and a compressor that is not much cheaper than the decoding it saves can be a net loss.
- Compression can reduce answer accuracy and, as recent evidence shows, can damage **citation grounding far more than it damages correctness** — one 2026 study reports only a 2–4% correctness drop but a 40–50% grounding drop under aggressive compression.
- The benefit depends on context length, model size and quantization in ways that have been characterized on GPU but not, in a controlled way, on CPU.

Three 2025–2026 survey-level sources agree that efficiency under resource constraints is unresolved (Agentic RAG Survey §12.4 "Computational Cost, Efficiency, and Sustainability"; RAG Survey §4.3/§5.4, where ablating caching "increases inference time up to 4×"). A method and a methodology are therefore needed that: (1) compress to the minimum evidence needed for a *grounded* answer; (2) are **training-free**, so they work with any off-the-shelf small model; (3) **adapt** the compression rate per query without an expensive controller; (4) make the decision **before** invoking the decoder; and (5) are evaluated under realistic CPU constraints with the metrics that matter there (TTFT, TPOT, end-to-end latency, peak memory, grounding).

---

# 3. Related Work and Positioning

Four RAG-latency streams are relevant: KV-cache reuse; retrieval prefetching/pipelining; context compression; and CPU-only / on-device efficiency work.

***Table 1. Positioning of DECAF against related 2025–2026 work.*** (✗ = absent; ✓ = present; ~ = partial. HW = hardware used.)

| Work (venue) | Axis | CPU-only? | Training-free? | Adaptive rate? | Measures decode (TPOT)? | Measures grounding? | Scale / HW |
|---|---|---|---|---|---|---|---|
| CacheBlend (EuroSys 2025) | KV reuse | ✗ | ✓ | ✗ | ✗ (prefill-only) | ✗ | 7B–70B / A40 |
| TurboRAG (EMNLP 2025) | KV reuse | ✗ | ✗ (888 A100-hrs) | ✗ | ✗ ("orthogonal to decode") | ✗ | 1.5B–72B / A100 |
| RAGCache (ACM 2025) | KV reuse | ✗ | ✓ | ✗ | ✗ (TPOT unaddressed) | ✗ | 7B–70B / A10G |
| CacheTune (2026), FusionRAG (SIGMOD 2026), SpecCache (ACL 2026), CoinRAG (2026), PCR (2026) | KV reuse | ✗ | ~/✗ | ✗ | ✗ (TTFT) | ✗ | 7B–32B / GPU |
| Predictive Prefetching (ICML 2026) | Prefetch | ✗ | ✗ (trained) | ✗ | ✗ | ✗ | 8B–70B / 8×A100 |
| TeleRAG (MLSys 2026), HedraRAG (SOSP 2025), RAGO (ISCA 2025) | Prefetch/scheduling | ✗ | ✓ / sim | ✗ | ✗ | ✗ | 3B–405B / GPU-cluster |
| VoiceAgentRAG (2026) | Prefetch/cache | partial | ✓ | ✗ | ✗ | ✗ | API LLM |
| REFRAG (2025) | Compression | ✗ | ✗ (64×H100 CPT) | RL policy | ✓ (TTIT, GPU) | ✗ | 3B–13B / H100 |
| CORE-RAG (ICML 2026) | Compression | ✗ | ✗ (GRPO) | ✗ (emergent) | ✗ (tokens only) | ✗ | 1.5B / 8×H20 |
| ACC-RAG (EMNLP 2025 F) | Compression | ✗ | ✗ (~71 h GPU) | ✓ (RL selector) | ✗ (FTIT only) | ✗ | 3B–7B / A100 |
| ECoRAG (ACL 2025 F) | Compression | ✗ | ~ (110M+770M) | ✓ (evaluator loop) | ✗ (total) | ✗ | reader ≥8B / 8×3090 |
| **SARA (ACL 2026)** | Compression + selective RAG | ✗ | ~ | ✓ | ✗ | ✗ | 5 LLMs / GPU |
| **Perception Compressor (NAACL 2025 F)** | Training-free prompt compression | ✗ | ✓ | ✓ (dynamic ratio) | ✗ | ✗ | LLM API |
| **BRIEF-Pro (ACL 2026 F)** | Universal context compression | ✗ | ✗ | ~ | ✗ | ✗ | multi-hop / GPU |
| **HS-RAG (IEEE 2026)** | Lightweight RAG on CPU | ✓ | ✓ (heuristic) | ~ | partial (decode noted) | ✗ | CPU |
| **Quantized LLM CPU benchmark (IEEE 2026)** | Systems characterization | ✓ | n/a | n/a | ✓ (latency/memory) | ✗ | 3B–7B / CPU |
| **Evidence-grounding study (CustomNLP4U 2026)** | Compression evaluation | ✗ | n/a | n/a | ✗ | ✓ (grounding drop) | LLM / GPU |
| **DECAF — proposed** | **Compression + characterization** | **✓** | **✓** | **✓ (CPU-aware)** | **✓ (central)** | **✓** | **3B/7B CPU** |

No listed work combines a CPU-only, quantized small-model setting with decode-level attribution, an adaptive CPU-aware compression decision, and grounding measurement. Each ingredient exists; the **interaction** is what is under-characterized — and that is DECAF's subject.

---

# 4. Research Gap (reframed)

1. **Compression's CPU benefit is not decomposed.** Compression papers typically report token reduction, first-token time, or aggregate latency. Whether the savings come from prefill or from decode — the phase that dominates CPU — is generally not isolated. (REFRAG isolates decode but requires 64×H100 continual pretraining.)
2. **The compressor's own cost is under-reported.** Adaptive methods add a controller (a selector, an evaluator loop, extra models). Whether that overhead is smaller than the decoding it saves, on CPU, is rarely quantified.
3. **No break-even characterization on CPU.** The context length above which compression becomes net-positive may depend on model size and quantization; this threshold is, to our knowledge, not established for CPU inference of small quantized models.
4. **Grounding is under-measured.** Compression can preserve correctness while degrading citation grounding, so accuracy alone (EM/F1/ROUGE) is insufficient.
5. **GPU-era KV-reuse/prefetch transfer is untested on CPU.** Theory (Amdahl) predicts little end-to-end benefit when decode dominates; this is reported here as a secondary, honest systems analysis rather than a headline claim.

---

# 5. Theoretical Framework and Measurement Methodology

**Latency decomposition.**

```
T_total = T_retrieve + T_compress + T_prefill(L) + N_out · TPOT(L)
GPU (published):  T_prefill ≫ rest
CPU (hypothesis): N_out · TPOT(L) ≫ T_prefill     → decode-dominated end-to-end
```

**The roofline relation is a hypothesis to be tested, not a law.** A first-order model is `TPOT(L) ≈ (W_model + KV(L)) / BW_mem` with `KV(L) ∝ L·d·layers·bytes`. Real CPU inference is also shaped by the cache hierarchy, SIMD/vectorization, quantization kernels, GQA/MQA, attention implementation, memory locality, NUMA effects, thread scheduling, BLAS/kernel choice, weight reuse and KV-cache layout. DECAF therefore treats memory-bandwidth domination as an **empirical question** (E8), and measures CPU utilization, achieved memory bandwidth, cache/LLC behavior where the platform exposes it, tokens/s, and `TPOT` as functions of context length, quantization and model size.

***Table 2. Measurement framework***

| Quantity | Definition | Meaning |
|---|---|---|
| Compression efficiency | `CE = ΔT_decode / ΔL` | Decode time saved per removed context token |
| Compression overhead | `O_c = T_compress / T_baseline` | Controller/compressor cost as a fraction of the baseline |
| Net speedup | `S_net = T_baseline / (T_compress + T_prefill^c + T_decode^c)` | True end-to-end gain including overhead |
| Quality-adjusted efficiency | `QE = Quality / Latency` | Reported as a Pareto frontier, not a lone scalar |
| Break-even length | `L* = min L : T_compress + T_baseline_decode > T_decode^full` | Context length above which compression is a net win |

A central design principle follows: **the compressor must be cheaper than the decoding it saves.** DECAF is single-pass and decoder-free at decision time, so `T_compress` is bounded by a small cross-encoder forward pass rather than by repeated generation.

---

# 6. Research Questions

1. **RQ1:** How does retrieved-context length affect TTFT, TPOT and end-to-end latency in CPU-only quantized RAG?
2. **RQ2:** When does context compression produce **net** latency savings after accounting for compression overhead (i.e. what is the break-even context length `L*`)?
3. **RQ3:** Can a training-free, CPU-aware compressor achieve a better quality–latency Pareto frontier than fixed-ratio and existing prompt-compression methods at equal grounding?
4. **RQ4:** How do model size (3B vs 7B) and quantization (Q4_K_M vs Q8_0) alter the compression–quality–latency relationship?

KV-cache reuse and prefetch are treated as a **secondary systems analysis**, not an RQ.

---

# 7. Hypotheses

- **H1 (falsifiable systems hypothesis):** As retrieved-context length increases, the fraction of end-to-end CPU latency attributable to decoding increases, and context compression produces larger **relative** gains in TPOT than in TTFT.
- **H2:** For each (model, quantization) pair there exists a break-even context length `L*`; below it compression is a net loss (overhead dominates), above it compression is a net win (decode savings dominate).
- **H3:** A single-pass CPU-aware gate matches or beats fixed-ratio and existing training-free prompt compression on the quality–latency–grounding Pareto frontier.
- **H4:** Larger models and higher-precision quantization shift `L*` and the Pareto frontier, because they change the decode-cost/overhead ratio.
- **H5 (secondary):** KV-cache reuse and retrieval prefetching provide limited net end-to-end benefit when CPU decode dominates — reuse moving mainly TTFT, prefetch hiding less time than it costs.

---

# 8. Aim and Objectives

**Aim:** to characterize, empirically, when context compression accelerates CPU RAG, and to introduce a training-free CPU-aware compressor that operates on that characterization.

- **O1** — Build a CPU inference harness with per-stage telemetry (T_retrieve, T_compress, TTFT, TPOT, E2E, peak RSS, throughput / bandwidth / energy where exposed).
- **O2** — Define and validate the measurement methodology (CE, O_c, S_net, QE, L*) of §5.
- **O3** — Implement DECAF's single-pass CPU-aware evidence selection (§11–12) using off-the-shelf components.
- **O4** — Run the context-length scaling, compression-ratio, break-even, quantization and model-size experiments (E1–E9).
- **O5** — Evaluate grounding alongside accuracy, and produce accuracy–latency Pareto frontiers.
- **O6** — Run the secondary KV-reuse/prefetch transferability analysis (H5).

---

# 9. Scope and Target Setting

**In scope:** single-turn and multi-hop QA with a CPU-only, 16 GB / 4–8 core machine; Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct in GGUF form (Q4_K_M and Q8_0 core; Q5 optional); fully training-free methods.

**Out of scope:** GPU/vLLM serving and data-center batching; visual/multimodal retrieval; voice/multi-turn; and any fine-tuning, RL training or continual pretraining.

---

# 10. Evaluation Datasets

Datasets are chosen to stress the central phenomenon (context length → CPU decode cost → compression benefit) and to maximize comparability with the compression literature, while trading breadth for depth.

***Table 3. Evaluation datasets (depth over breadth)***

| Role | Dataset | QA type | Context | Baselines reporting it |
|---|---|---|---|---|
| Main benchmark | HotpotQA (distractor) | Multi-hop | Provided gold paragraphs | CacheBlend, ECoRAG, ACC-RAG |
| Main benchmark | 2WikiMultihopQA | Multi-hop | Provided contexts | CacheBlend, ECoRAG |
| Generalization | **Long-context benchmark** (LongBench multi-document QA; MuSiQue / NarrativeQA as alternates) | Long-context / multi-hop | Long retrieved sets (up to ~32K tokens) | Long-context RAG literature |
| Generalization | TriviaQA (or NQ) | Single-hop | Provided / small index | ECoRAG, CORE-RAG, ACC-RAG |

- **Deliberate addition of a long-context benchmark (feedback-driven):** short contexts cannot stress the decode-vs-compression phenomenon; at least one long-context set is required to measure `L*`.
- **Scale (depth > breadth):** main benchmark ~**500 queries × 2 datasets**; generalization ~**200 queries × 2 datasets**; 50 calibration queries per model/quantization for threshold selection only (no training anywhere). 3B across all datasets; 7B on the main benchmark and a generalization subset.
- **Retrieval modes:** provided-context (isolates compression; the core experiments) and a small `rank_bm25` / dense index for end-to-end retrieval baselines. The full 21M-passage DPR corpus is excluded — it would make retrieval, not prefill/decode, dominant and confound H1.
- **Protocol:** fixed dataset-native prompt, standard EM/F1 scoring, held identical across configurations; grounding scored against retained gold evidence (§15).

---

# 11. Proposed System Architecture

DECAF makes the compression decision **before** invoking the decoder — there is no iterative generation, which is what keeps `T_compress` small.
<img width="1536" height="582" alt="Untitled - Visual 1" src="https://github.com/user-attachments/assets/6efbdb78-0e7c-4e78-a733-27d85128fc68" />

```
Query
  │
  ▼
Retriever (top-N passages)
  │
  ▼
Sentence segmentation
  │
  ▼
Fine-grained evidence scoring (relevance, coverage, answer-uncertainty reduction)
  │
  ▼
CPU-aware budget controller   ← estimated marginal CPU decode cost per evidence unit
  │   select while MarginalQualityGain ≥ λ · MarginalLatencyCost
  ▼
Adaptive evidence selection (single pass)
  │
  ▼
Relevance reordering
  │
  ▼
Reader: Qwen2.5-3B / 7B (GGUF via llama.cpp, n_threads = cores)
  │
  ▼
Answer + telemetry (TTFT, TPOT, E2E, RSS, bandwidth/energy, grounding)
```

A separate branch measures llama.cpp prefix/KV cache and an async retrieval thread, for the secondary analysis (H5).

**Validity boundary:** the compressor may only change *which* retrieved sentences reach the decoder and *when* they are fetched; it never alters sentence content. Every decision is logged (retained count, ratio, per-stage latency, grounding of retained evidence) so quality and grounding can be attributed to a specific operating point.

---

# 12. Core Technical Modules

**12.1 Fine-grained evidence scoring.** Each candidate sentence `s_i` is scored with an off-the-shelf cross-encoder (`BAAI/bge-reranker-base`, ~110M; `bge-reranker-v2-m3` as a quality variant), run on CPU.

**12.2 CPU-aware budget controller (the core novel mechanism).** Rather than "add sentences until enough evidence," DECAF optimizes quality gain per unit of CPU cost:

```
Score(s_i) = [ Relevance(s_i,Q) × Coverage(s_i,Q) × UncertaintyReduction(s_i) ] / EstimatedCPUCost(s_i)

select evidence while:   MarginalQualityGain(s_i)  ≥  λ · MarginalLatencyCost(s_i)
```

- `EstimatedCPUCost(s_i)` is derived from the **measured decode-scaling curve** of §5 (`≈ tokens(s_i) × per-token TPOT estimate at the current context length`) — so the controller is explicitly grounded in the hardware/model/quantization, which is what makes it "CPU-aware."
- `Coverage` rewards evidence covering query aspects not yet covered; `UncertaintyReduction` is a decoder-free proxy (aspect coverage + lightweight answer-type signal), so no generation is required.
- The selection is **single-pass** and `λ` is a single interpretable budget knob, calibrated on the held-out split.

This yields a clearly defined algorithmic mechanism — *hardware-aware adaptive evidence selection for CPU decode efficiency* — rather than a composition of generic components.

**12.3 Relevance reordering.** Retained evidence is ordered by relevance to reduce lost-in-the-middle effects on small models.

**12.4 Telemetry and measurement.** Logs T_retrieve, T_compress, TTFT, TPOT, E2E, tokens/s, peak RSS (`resource`/psutil), achieved memory bandwidth, CPU utilization, cache/LLC counters where available, and energy (RAPL/`powerstat` where exposed); computes CE, O_c, S_net, QE and L*.

**12.5 Grounding evaluator.** Scores whether the generated answer is supported by the retained gold evidence (evidence recall/precision, answer-support rate, citation correctness where available) — decoder-free, so it does not corrupt the latency measurement.

---

# 13. Baselines

**Retrieval:** BM25 top-k; dense top-k; hybrid.

**Compression / selection:** Full context; fixed top-k sentences; LLMLingua-2; **Perception Compressor** (training-free, adaptive, NAACL 2025 Findings); **DECAF**. Where compute permits, ACC-RAG and ECoRAG are included as (trained/GPU) reference points. Methods whose training is infeasible on CPU (TurboRAG, REFRAG, CORE-RAG, CoinRAG, and the trained compressors of ACC-RAG/SARA) are cited qualitatively with the compute barrier stated, not approximated at reduced fidelity.

All baselines are re-run on identical CPU hardware, datasets and metrics.

---

# 14. Experimental Methodology

***Table 4. Core experiment matrix (trimmed to the experiments that test the central hypothesis)***

| ID | Experiment | Tests |
|---|---|---|
| E1 | **Context-length scaling**: TTFT/TPOT/E2E/memory at L ∈ {1K, 2K, 4K, 8K, 16K, 32K}, full vs compressed | RQ1, H1 |
| E2 | Compression ratio vs quality (0.8/0.6/0.4/0.2 retained) | RQ3 |
| E3 | Compression overhead vs decode savings (`O_c`, `S_net`) | RQ2 |
| E4 | **Break-even context length `L*`** per model×quantization | RQ2, H2 |
| E5 | Quantization interaction (Q4_K_M vs Q8_0) | RQ4 |
| E6 | Model-size interaction (3B vs 7B) | RQ4 |
| E7 | **Grounding preservation** under compression | RQ3 |
| E8 | CPU memory-bandwidth / roofline analysis | §5 |
| E9 | Generalization dataset | RQ3 |

Protocol: fixed hardware/threads/models/quants/prompts/seeds; warmup then ≥3 repeats per reported point; medians with CIs. The experiment count is deliberately trimmed (~25–30% smaller than the earlier draft) so that depth on the central hypothesis is not sacrificed to breadth.

---

# 15. Evaluation Metrics

- **Latency:** TTFT; **TPOT (ms/token)**; end-to-end latency; P50/P95; tokens/s.
- **System:** peak RSS; compression ratio; compressor overhead (% of E2E); achieved memory bandwidth; CPU utilization; energy per query / per token where the platform exposes it.
- **Quality:** EM, F1; ROUGE-L where applicable.
- **Grounding / evidence:** evidence recall, evidence precision, answer-support rate, citation correctness where gold citations exist.
- **Derived:** CE, O_c, S_net, QE, L* (Table 2).
- **Composite:** accuracy–latency Pareto frontier (not a single scalar).

---

# 16. Break-Even and Decode-Scaling Analysis

The empirical backbone of the paper is the `TPOT = f(L)` relationship for each (model, quantization), measured for full and compressed contexts, from which `L*` is read off:

```
TPOT
 ^                          Full context
 |                        /
 |                     /
 |                  /        Compressed
 |               /        /
 |            /        /
 |         /        /
 |______/________/__________________>  context length
                 L*
```

A positive, monotone `TPOT(L)` for full context together with a shallower/near-flat curve for compressed context yields a measurable `L*`; the corresponding table of `L*` per model×quantization is a practical contribution.

---

# 17. Component Analysis (Ablation)

A trimmed ablation isolates the necessity of each DECAF component rather than assuming the full system is best: scorer off/on; CPU-aware cost term on/off (i.e. relevance-only selection); coverage term on/off; reordering on/off; and `λ` sweep. This also quantifies how much the CPU-awareness itself contributes versus a plain relevance-ranked top-k, which is the sharpest test of the central mechanism.

---

# 18. Secondary Analysis: Transferability of KV-Reuse and Prefetching

A focused micro-study measures (i) llama.cpp prefix/KV cache on repeated queries and (ii) async retrieval prefetch overlapped with decoding, reporting net E2E and TTFT. Framed as a **secondary systems analysis**, it tests the Amdahl-based expectation that reuse mainly moves TTFT and prefetch hides little when decode dominates — an honest negative/positive result, not a headline contribution.

---

# 19. Risk Register

***Table 5. Risks and mitigations***

| Risk | Mitigation |
|---|---|
| Compressor cost could erase the compression benefit | Single-pass, decoder-free, ~110M cross-encoder; measure and report `O_c` explicitly; `O_c` is a first-class result |
| Adaptive decisions may be unstable / expensive | Single gate (not the earlier three-gate design); one budget knob `λ`; no iterative generation |
| 7B too slow on 4–8 cores | 3B primary; 7B on main benchmark + subset |
| Long-context (32K) points are slow on CPU | Limit high-`L` points to a small query subset sufficient to fit `TPOT(L)` |
| Grounding metrics may not exist natively | Evaluate support against retained gold evidence (dataset-provided supporting facts where available) |
| Roofline model over-simplified | Treat as hypothesis; report measured bandwidth/cache/utilization against it (E8) |
| Novelty overlap with SARA / Perception Compressor / BRIEF-Pro | Position the contribution as the CPU-specific decode-centric characterization + break-even analysis, not as "first training-free adaptive compression" |

---

# 20. Success Criteria

1. A measured, reproducible `TPOT(L)` relationship and TTFT/decode split on CPU for 3B/7B × Q4/Q8 (RQ1).
2. A measured break-even context length `L*` per model×quantization, with the overhead-vs-savings crossover (RQ2).
3. A quality–latency–grounding Pareto frontier on which DECAF is competitive with or ahead of fixed-ratio and Perception Compressor (RQ3).
4. A quantified model-size/quantization effect on `L*` and the frontier (RQ4).
5. The secondary transferability analysis answered in either direction with evidence.
6. Code, configs, seeds, telemetry logs and dataset splits released.

---

# 21. Statistical Analysis Plan

- **Latency:** paired bootstrap confidence intervals (matched per query), medians, P95, and effect sizes — not significance alone.
- **Accuracy (exact-answer):** McNemar's test where a per-query correct/incorrect pairing is available.
- **Accuracy/grounding (continuous):** permutation/randomization tests and paired bootstrap.
- **Multiplicity:** Holm–Bonferroni across the core comparison family.
- Report **absolute and relative** improvements together; the Pareto frontier, not a p-value, is the primary story.

---

# 22. Reproducibility Plan

Public release of the compressor, budget controller, scoring, grounding evaluator and telemetry code (llama.cpp / llama-cpp-python, sentence-transformers/ONNX, spaCy/NLTK); exact CPU model, physical cores, RAM and thread count; OS, llama.cpp build and library versions; model checkpoints by hash and quantization (GGUF Q4_K_M / Q8_0); fixed seeds and prompt templates; released dataset subsets and held-out splits; per-run telemetry logs (TTFT, TPOT, E2E, RSS, bandwidth, grounding) so results can be re-derived.

---

# 23. Timeline and Collaboration Roles

***Table 6. Indicative phased timeline***

| Phase | Activities | Indicative duration |
|---|---|---|
| 1. Harness | CPU harness + telemetry + measurement methodology validated | 1 weeks |
| 2. Characterization | E1 context-length scaling; E8 bandwidth analysis | 2 weeks |
| 3. Baselines | Full/top-k/LLMLingua-2/Perception Compressor on identical hardware | 2 weeks |
| 4. DECAF | Scorer + CPU-aware controller + reordering + grounding evaluator | 2 weeks |
| 5. Core results | E2–E7, E9; `L*` and Pareto frontiers; component analysis | 1 weeks |
| 6. Secondary + writing | Transferability study; manuscript; artifacts | 2 weeks |

***Table 7. Collaboration roles***

| Area | Primary responsibility |
|---|---|
| Research direction, hypothesis framing, manuscript review | Supervisor |
| Harness, baselines, module implementation, experiments | Student (Tasnia Haque) |
| Measurement-methodology and `L*` design review | Joint |
| Statistical analysis review | Joint |
| Venue selection and submission strategy | Joint |
| Reproducibility artifact release | Student (Tasnia Haque) |

---

# 24. Publication Strategy and Conclusion

**Venue decision (made explicitly).** The first version targets the **efficient-NLP** community — ACL/EMNLP (efficiency/retrieval tracks, main or Findings) and NAACL — because the contribution is framed as a RAG/evidence-selection and evaluation study with a strong measurement methodology. The methodology (decode attribution, break-even, bandwidth analysis, energy) is designed to be strong enough that **systems** reviewers also respect it, with MLSys / EuroSys / ASPLOS / USENIX ATC as an alternative home if the characterization results warrant a systems framing. The venue is chosen *now* rather than deferred, because the two framings imply different lead contributions.

**Conclusion.** DECAF's contribution is not a new training-free compressor — that space now includes SARA, Perception Compressor and BRIEF-Pro — but a **decode-centric characterization of when context compression actually accelerates CPU RAG**, a **training-free CPU-aware compressor built on that characterization**, a **break-even analysis** that yields practically useful `L*` thresholds, and a **reproducible efficiency benchmark** with grounding and energy metrics. The strongest expected result is not "DECAF is X% faster" but: *compression is a net loss below a hardware-, model- and quantization-dependent context-length threshold, and a net win above it* — a result that helps practitioners decide **when** to compress rather than assuming they always should.

**Immediate next steps:** (1) confirm CPU/cores/RAM and llama.cpp build; (2) fix dataset subsets/splits and model checkpoint hashes; (3) build the harness and reproduce the measurement methodology (E1, E8); (4) implement the single-pass CPU-aware controller; (5) run E2–E7, E9 and the secondary analysis.
