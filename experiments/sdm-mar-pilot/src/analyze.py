"""Phase B: predictive models, ablations, interventions, hop-level analysis,
figures and the preliminary-results report."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import models, synth
from .retriever import Retriever
from .run_runs import RunCache, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _runs_frame(records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({"qid": r["qid"], "condition": r["condition"], "replicate": r["replicate"],
                     "dim": r["dim"], "success": int(r["success"]), "n_llm": r.get("n_llm", 0)})
    return pd.DataFrame(rows)


def _hops_frame(records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for h in r.get("hops", []):
            rows.append({"qid": r["qid"], "condition": r["condition"], "replicate": r["replicate"],
                         "dim": r["dim"], "hop": h["hop"], "success": int(h["success"]),
                         "n_candidates": h["n_candidates"], "used_llm": int(h["used_llm"])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = os.path.join(ROOT, cfg["paths"]["results"])
    os.makedirs(out, exist_ok=True)

    docs, questions = synth.load(os.path.join(ROOT, cfg["paths"]["corpus"]))
    cache = RunCache(os.path.join(ROOT, cfg["paths"]["cache"]))
    records = cache.all()
    if not records:
        raise SystemExit("no runs cached — run `python -m src.run_runs` first")

    runs = _runs_frame(records)
    hops = _hops_frame(records)
    questions = [q for q in questions if q.qid in set(runs["qid"])]
    main = runs[runs["dim"].isna()].copy()          # bm25 conditions
    dim_runs = runs[runs["dim"].notna()].copy()      # dimension sweep

    # ----- feature table with first-hop retrieval statistics (Model B inputs) -----
    retr = Retriever(docs, "bm25")
    k = cfg["retriever"]["top_k"]
    retr_stats = {}
    for q in questions:
        retr_stats[q.qid] = retr.first_round_stats(
            f"{q.hops[0].subject} {q.hops[0].relation}", q.node_ids[0], k)
    feat = models.build_feature_table(questions, retr_stats)
    feat.to_csv(os.path.join(out, "features.csv"), index=False)

    summary: Dict = {"n_questions": len(questions), "n_runs": len(records),
                     "conditions": sorted(main["condition"].unique().tolist())}

    # ----- success by structure (baseline condition) -----
    base = main[main["condition"] == "baseline"].merge(feat, on="qid", how="left")
    by_h = base.groupby("H")["success"].mean()
    by_ms = base.groupby("MS")["success"].mean()
    by_csl_bin = base.assign(csl_bin=pd.qcut(base["CSL"], 3, duplicates="drop")).groupby(
        "csl_bin", observed=True)["success"].mean()
    summary["baseline_success_overall"] = float(base["success"].mean())
    summary["baseline_success_by_H"] = {int(k_): float(v) for k_, v in by_h.items()}

    # ----- predictive models -----
    fit = models.fit_all(feat, main[main["condition"] == "baseline"], summary, seed=cfg["seed"])
    abl = models.ablation_table(feat, main[main["condition"] == "baseline"], seed=cfg["seed"])
    abl.to_csv(os.path.join(out, "ablations.csv"), index=False)
    summary["ablations"] = abl.to_dict(orient="records")

    # ----- interventions -----
    cond_mean = main.groupby("condition")["success"].mean()
    summary["condition_success"] = {c: float(v) for c, v in cond_mean.items()}

    # oracle bridge: gain by depth H
    comp = main.merge(feat, on="qid", how="left")
    pivot = comp.pivot_table(index=["qid", "H", "CSL", "MS"], columns="condition",
                             values="success", aggfunc="mean").reset_index()
    gains = {"oracle_gain_by_H": {}, "caption_gain_by_MS": {}}
    if "oracle_bridge" in pivot:
        g = pivot.assign(gain=pivot["oracle_bridge"] - pivot["baseline"]).groupby("H")["gain"].mean()
        gains["oracle_gain_by_H"] = {int(k_): float(v) for k_, v in g.items()}
    if "caption" in pivot:
        g = pivot.assign(gain=pivot["caption"] - pivot["baseline"]).groupby("MS")["gain"].mean()
        gains["caption_gain_by_MS"] = {int(k_): float(v) for k_, v in g.items()}
        # caption gain on chains with vs without image nodes
        pv = pivot.assign(has_image=(pivot["MS"] > 0) | (pivot["qid"].map(
            {q.qid: q.n_images for q in questions}) > 0))
        gains["caption_gain_by_has_image"] = {
            str(k_): float(v) for k_, v in pv.assign(gain=pv["caption"] - pv["baseline"]).groupby("has_image")["gain"].mean().items()}
    summary["interventions"] = gains

    # dimension sweep: success vs d, split by CSL tercile
    if not dim_runs.empty:
        dr = dim_runs.merge(feat[["qid", "CSL", "H"]], on="qid", how="left")
        dr["csl_bin"] = pd.qcut(dr["CSL"], 2, labels=["low CSL", "high CSL"], duplicates="drop")
        dim_curve = dr.groupby(["dim", "csl_bin"], observed=True)["success"].mean().reset_index()
        dim_curve.to_csv(os.path.join(out, "dimension_sweep.csv"), index=False)
        summary["dimension_sweep"] = dim_curve.to_dict(orient="records")
        # gain from the smallest to the largest dimension, per CSL bin
        g = {}
        for lbl, gg in dim_curve.groupby("csl_bin", observed=True):
            gg = gg.sort_values("dim")
            g[str(lbl)] = float(gg["success"].iloc[-1] - gg["success"].iloc[0])
        summary["dimension_gain_by_csl"] = g

    # ----- hop-level analysis and the product law -----
    hop_base = hops[(hops["condition"] == "baseline")].copy()
    mod_map = {}
    for q in questions:
        for h in q.hops:
            mod_map[(q.qid, h.index)] = h.modality
    hop_base["node_modality"] = [mod_map.get((q, j)) for q, j in zip(hop_base["qid"], hop_base["hop"])]
    hop_by_mod = hop_base.groupby("node_modality")["success"].mean()
    summary["hop_success_by_modality"] = {str(k_): float(v) for k_, v in hop_by_mod.items()}

    # product law vs measured
    per_q = hop_base.groupby("qid")["success"].agg(["mean", "size"]).rename(columns={"mean": "hop_mean"})
    hprob = hop_base.groupby("hop")["success"].mean()  # marginal per-hop success
    prod_rows = []
    for q in questions:
        qh = hop_base[hop_base["qid"] == q.qid]
        if qh.empty:
            continue
        prod = float(np.prod([hprob[j] for j in range(1, len(q.hops) + 1)]))
        measured = float(runs[(runs["qid"] == q.qid) & (runs["condition"] == "baseline")]["success"].mean())
        ps = [hprob[j] for j in range(1, len(q.hops) + 1)]
        lower = max(0.0, 1 - sum(1 - p for p in ps))
        upper = min(ps)
        prod_rows.append({"qid": q.qid, "H": len(q.hops), "product": prod,
                          "measured": measured, "lower": lower, "upper": upper})
    prod_df = pd.DataFrame(prod_rows)
    prod_df.to_csv(os.path.join(out, "product_law.csv"), index=False)
    summary["product_law_mae"] = float((prod_df["product"] - prod_df["measured"]).abs().mean())
    summary["product_law_corr"] = float(np.corrcoef(prod_df["product"], prod_df["measured"])[0, 1]) \
        if len(prod_df) > 2 else float("nan")

    # ----- figures -----
    _figs(out, by_h, by_ms, by_csl_bin, abl, gains, summary, prod_df,
          dim_runs, feat, cfg)

    # ----- report -----
    _report(out, cfg, summary, abl, feat, prod_df, gains)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"Wrote figures + tables + summary.json + PRELIMINARY_RESULTS.md to {out}")


def _figs(out, by_h, by_ms, by_csl_bin, abl, gains, summary, prod_df, dim_runs, feat, cfg):
    # Fig 1: success vs structure
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax[0].bar([str(i) for i in by_h.index], by_h.values, color="#4c72b0")
    ax[0].set_xlabel("depth H"); ax[0].set_ylabel("success rate"); ax[0].set_title("Success vs depth")
    ax[1].bar([str(i) for i in by_ms.index], by_ms.values, color="#dd8452")
    ax[1].set_xlabel("modality switches MS"); ax[1].set_title("Success vs modality switches")
    ax[2].bar([str(i) for i in by_csl_bin.index], by_csl_bin.values, color="#55a868")
    ax[2].set_xlabel("CSL tercile"); ax[2].set_title("Success vs concept-separation load")
    for a in ax:
        a.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_success_structure.png")); plt.close(fig)

    # Fig 2: model AUC + ablations
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    mm = summary["model_metrics"]
    ax[0].bar(list(mm.keys()), [mm[m]["auc"] for m in mm], color="#4c72b0")
    ax[0].set_ylabel("AUC (grouped CV)"); ax[0].set_title("Models A (hop) / B (score) / C (structure)")
    ax[0].set_ylim(0, 1)
    ax[1].barh(abl["variant"], abl["auc"], color="#55a868")
    ax[1].invert_yaxis(); ax[1].set_xlabel("AUC"); ax[1].set_title("Model C ablations"); ax[1].set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_models.png")); plt.close(fig)

    # Fig 3: interventions
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    oh = gains.get("oracle_gain_by_H", {})
    if oh:
        ax[0].bar([str(k) for k in oh], list(oh.values()), color="#8172b3")
    ax[0].set_xlabel("depth H"); ax[0].set_ylabel("oracle-bridge gain"); ax[0].set_title("Intervention A: oracle bridge")
    cm = gains.get("caption_gain_by_MS", {})
    if cm:
        ax[1].bar([str(k) for k in cm], list(cm.values()), color="#c44e52")
    ax[1].set_xlabel("modality switches MS"); ax[1].set_ylabel("caption gain"); ax[1].set_title("Intervention B: image->caption")
    if not dim_runs.empty:
        dr = dim_runs.merge(feat[["qid", "CSL"]], on="qid", how="left")
        dr["csl_bin"] = pd.qcut(dr["CSL"], 2, labels=["low CSL", "high CSL"], duplicates="drop")
        for lbl, g in dr.groupby("csl_bin", observed=True):
            c = g.groupby("dim")["success"].mean()
            ax[2].plot(c.index, c.values, "o-", label=str(lbl))
        ax[2].set_xscale("log", base=2)
        ax[2].legend(fontsize=7)
    ax[2].set_xlabel("retriever dimension d"); ax[2].set_ylabel("success rate"); ax[2].set_title("Intervention C: dimension sweep")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_interventions.png")); plt.close(fig)

    # Fig 4: product law (colored by depth)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sc = ax.scatter(prod_df["product"], prod_df["measured"], c=prod_df["H"], cmap="viridis", s=22)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    fig.colorbar(sc, ax=ax, label="depth H")
    ax.set_xlabel("product of per-hop success"); ax.set_ylabel("measured question success")
    ax.set_title(f"Product law (MAE={summary['product_law_mae']:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_product_law.png")); plt.close(fig)


def _report(out, cfg, summary, abl, feat, prod_df, gains):
    L = []
    L.append("# SDM-MAR — Preliminary Results (smoke test)\n")
    L.append(f"Synthetic multimodal evidence-graph corpus; real LLM reader `{cfg['ollama']['model']}` "
             f"(CPU, Ollama) · {summary['n_questions']} questions · {summary['n_runs']} agent runs\n")
    L.append("> **Harness check, not a discovery.** Structure (H, CSL, MS, edge types) is planted by "
             "construction, so structure-to-success links are partly expected. H and CSL are driven by the "
             "real LLM reader (its disambiguation under hard negatives); the image-modality barrier is "
             "**simulated** (the pilot's reader is text-only and cannot perform OCR). See Section 3.\n")

    d = summary["delta_auc_C_minus_A"]
    L.append(f"**Bottom line.** Grouped-CV AUC — Model A (hop count) **{summary['model_metrics']['A']['auc']:.3f}**, "
             f"Model B (retrieval score) **{summary['model_metrics']['B']['auc']:.3f}**, Model C (structure) "
             f"**{summary['model_metrics']['C']['auc']:.3f}**; ΔAUC(C−A) = **{d['delta_auc']:.3f}** "
             f"(clustered 95% CI [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]).\n")

    L.append("## 1. Success vs structure (baseline)\n")
    L.append("![](" + "fig_success_structure.png" + ")\n")
    L.append(f"- overall baseline success = **{summary['baseline_success_overall']:.3f}**")
    L.append(f"- by depth H: " + ", ".join(f"H={k}: {v:.2f}" for k, v in summary["baseline_success_by_H"].items()))
    L.append(f"- hop success by node modality: "
             + ", ".join(f"{k}: {v:.2f}" for k, v in summary["hop_success_by_modality"].items()) + "\n")

    L.append("## 2. Prediction: Models A/B/C and ablations\n")
    L.append("![](" + "fig_models.png" + ")\n")
    mm = pd.DataFrame(summary["model_metrics"]).T
    L.append(mm.round(3).to_markdown() + "\n")
    L.append("Ablations (grouped CV AUC):\n")
    L.append(abl.round(3).to_markdown(index=False) + "\n")

    L.append("## 3. Interventions\n")
    L.append("![](" + "fig_interventions.png" + ")\n")
    oh = gains.get("oracle_gain_by_H", {})
    oh_verdict = "confirmed" if (len(oh) >= 2 and list(oh.values())[-1] > list(oh.values())[0]) else "not confirmed"
    L.append(f"- **Intervention A (oracle bridge)** — gain by depth H: {_fmt(oh)}; prediction 'grows with H' → "
             f"**{oh_verdict}**.")
    cm = gains.get("caption_gain_by_MS", {})
    ci = gains.get("caption_gain_by_has_image", {})
    cap_ok = ci.get("True", 0) > ci.get("False", 0)
    L.append(f"- **Intervention B (image→caption)** — gain by MS: {_fmt(cm)}; on chains with/without image nodes: "
             f"{_fmt(ci)}; prediction 'helps image chains' → **{'confirmed' if cap_ok else 'not confirmed'}**.")
    dg = summary.get("dimension_gain_by_csl", {})
    if dg:
        hi = dg.get("high CSL", 0.0); lo = dg.get("low CSL", 0.0)
        L.append(f"- **Intervention C (dimension sweep)** — success gain from d=32→512: low CSL {lo:+.2f}, "
                 f"high CSL {hi:+.2f}; prediction 'gain grows with CSL' → "
                 f"**{'confirmed' if hi > lo else 'not confirmed'}**.")
    L.append("- Note: the dimension sweep is weakly identified in this synthetic corpus, because high-CSL "
             "distractors differ from gold mainly by a random identifier token, so no embedding dimension "
             "separates them; a real corpus should not have this degeneracy.\n")

    L.append("## 4. Hop-level analysis and the product law\n")
    L.append("![](" + "fig_product_law.png" + ")\n")
    L.append(f"- MAE(product of per-hop success, measured question success) = "
             f"**{summary['product_law_mae']:.3f}**; corr = **{summary['product_law_corr']:.2f}**")
    L.append("- The product of marginal per-hop success is determined almost entirely by H, so it tracks the "
             "measured rate poorly and generally under-estimates it: hop outcomes are strongly dependent "
             "(a wrong bridge degrades the next hop), not independent. The Fréchet bounds are in "
             "`product_law.csv`. A genuine per-question hop model with an error-propagation term "
             "(proposal Section 12.5) is the correct next step.\n")

    L.append("## 5. Reading for the proposal\n")
    L.append("- Section 2 is the **head-to-head predictive test** (Models A/B/C, ΔAUC with clustered CI).")
    L.append("- Section 3 gives the **three one-factor interventions** with their pre-stated directions.")
    L.append("- Section 1 & 4 give the **structure-to-success** and **hop-level product-law** evidence.")
    L.append("- Next: replace the simulated modality barrier with a real VLM, use real multimodal corpora "
             "(CrossModalQA / VisDocAgentBench), and scale questions and replicates.\n")

    with open(os.path.join(out, "PRELIMINARY_RESULTS.md"), "w") as f:
        f.write("\n".join(L))


def _fmt(d) -> str:
    return ", ".join(f"{k}: {v:+.2f}" for k, v in d.items()) if d else "(none)"


if __name__ == "__main__":
    main()
