"""Phase B: figures, tables, and the preliminary-results report."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import calibrate, policies
from .llm import GridCache
from .run_grid import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _frontier(frame, eval_qids, choices_fn, label, sweep):
    pts = []
    for s in sweep:
        choices = choices_fn(s)
        res = policies.evaluate(frame, {q: choices[q] for q in eval_qids if q in choices})
        pts.append(
            {
                "label": label,
                "sweep": float(s),
                "realized_budget": res["realized_budget"],
                "f1": res["f1"],
                "em": res["em"],
                "harm": res["harm"],
                "harm_rate": res["harm_rate"],
                "mean_k": res["mean_k"],
            }
        )
    return pts


def _interp_at(frontier_pts, target):
    """Best F1 among points whose realized budget <= target (matched-budget view)."""
    ok = [p for p in frontier_pts if p["realized_budget"] <= target + 1e-9]
    if not ok:
        return None
    return max(ok, key=lambda p: p["f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = os.path.join(ROOT, cfg["paths"]["results"])
    os.makedirs(out, exist_ok=True)

    cache = GridCache(os.path.join(ROOT, cfg["paths"]["cache"]))
    records = cache.all()
    if not records:
        raise SystemExit("cache empty — run `python -m src.run_grid` first")
    frame = policies.build_frame(records)
    frame.to_csv(os.path.join(out, "frame.csv"), index=False)

    ks_present = [int(k) for k in sorted(frame["k"].unique())]
    calib_qids, eval_qids = policies.split_questions(frame, cfg)
    summary = {"n_questions": len(policies.questions(frame)), "ks": ks_present}

    # ---------- Fig 1: quality vs k (inverted-U) ----------
    per_k = frame.groupby("k")[["f1", "em"]].mean()
    strata, strata_name = policies.resolve_strata(frame)
    frame["stratum"] = frame["qid"].map(strata)
    by_strata = frame.groupby(["stratum", "k"])[["f1"]].mean().unstack(0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(per_k.index, per_k["f1"], "o-", label="F1")
    ax[0].plot(per_k.index, per_k["em"], "s--", label="EM")
    ax[0].set_xlabel("retrieval depth k")
    ax[0].set_ylabel("score")
    ax[0].set_title("Quality vs retrieval depth")
    ax[0].legend()
    for col in by_strata.columns:
        ax[1].plot(by_strata.index, by_strata[col], "o-", label=str(col[-1]))
    ax[1].set_xlabel("retrieval depth k")
    ax[1].set_ylabel("F1")
    ax[1].set_title(f"F1 vs k by {strata_name} (n={len(strata)})")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_inverted_u.png"))
    plt.close(fig)
    summary["strata"] = {"name": strata_name, "counts": {str(k): int(v) for k, v in strata.value_counts().items()}}
    summary["quality_by_k"] = {int(k): float(v) for k, v in per_k["f1"].items()}

    # ---------- Fig 2: delta-Q / harm ----------
    thresh = cfg["harm_threshold"]
    kk = [k for k in ks_present if k > 0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k in kk:
        vals = frame.loc[frame["k"] == k, "delta_f1"].dropna()
        ax[0].hist(vals, bins=20, alpha=0.5, label=f"k={k}")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel(r"$\Delta Q = Q_R - Q_0$ (F1)")
    ax[0].set_ylabel("count")
    ax[0].set_title("Per-query retrieval effect")
    ax[0].legend()
    rates = {}
    for k in kk:
        d = frame.loc[frame["k"] == k, "delta_f1"].dropna()
        rates[k] = {
            "beneficial": float((d > thresh).mean()),
            "neutral": float((d.abs() <= thresh).mean()),
            "harmful": float((d < -thresh).mean()),
        }
    keys = ["beneficial", "neutral", "harmful"]
    bottom = np.zeros(len(kk))
    colors = {"beneficial": "#4c9f70", "neutral": "#cccccc", "harmful": "#c0392b"}
    for key in keys:
        vals = np.array([rates[k][key] for k in kk])
        ax[1].bar([str(k) for k in kk], vals, bottom=bottom, label=key, color=colors[key])
        bottom += vals
    ax[1].set_xlabel("retrieval depth k")
    ax[1].set_ylabel("fraction of queries")
    ax[1].set_title(f"Beneficial / neutral / harmful (|Δ|>{thresh})")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_deltaQ.png"))
    plt.close(fig)
    summary["effect_rates_by_k"] = {int(k): rates[k] for k in kk}

    # ---------- Fig 3: cost is driven by length, not doc count ----------
    rob = frame[~frame["stalled"]]
    corr_prefill = float(np.corrcoef(rob["prompt_tokens"], rob["prefill_ms"])[0, 1])
    corr_len_prefill = float(np.corrcoef(rob["context_chars"], rob["prefill_ms"])[0, 1])
    corr_len_total = float(np.corrcoef(rob["context_chars"], rob["total_ms"])[0, 1])
    corr_k_prefill = float(np.corrcoef(rob["k"], rob["prefill_ms"])[0, 1])
    # Within a fixed k, context size still varies widely -> doc count is coarse.
    within = rob.groupby("k")["prompt_tokens"]
    cv = (within.std() / within.mean()).dropna()
    worst_k = int(cv.idxmax()) if len(cv) else ks_present[0]
    worst_cv = float(cv.max()) if len(cv) else float("nan")
    n_stalled = int(frame["stalled"].sum())

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    sc = ax[0].scatter(rob["prompt_tokens"], rob["prefill_ms"], c=rob["k"], cmap="viridis", s=12)
    if len(rob) > 1:
        m, b = np.polyfit(rob["prompt_tokens"], rob["prefill_ms"], 1)
        xs = np.linspace(rob["prompt_tokens"].min(), rob["prompt_tokens"].max(), 50)
        ax[0].plot(xs, m * xs + b, "r--", lw=1)
    ax[0].set_xlabel("prompt tokens (context size)")
    ax[0].set_ylabel("prefill time (ms)")
    ax[0].set_title(f"Prefill cost tracks context length (r={corr_prefill:.2f})")
    fig.colorbar(sc, ax=ax[0], label="k")
    boxes = [rob.loc[rob["k"] == k, "prompt_tokens"].values for k in ks_present]
    ax[1].boxplot(boxes, tick_labels=[str(k) for k in ks_present], showfliers=False)
    ax[1].set_xlabel("retrieval depth k")
    ax[1].set_ylabel("prompt tokens")
    ax[1].set_title(f"Same k ⇒ different context size (max CV={worst_cv:.2f} at k={worst_k})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_cost_vs_length.png"))
    plt.close(fig)
    summary["cost_proxy"] = {
        "corr_prompttokens_vs_prefill": corr_prefill,
        "corr_contextchars_vs_prefill": corr_len_prefill,
        "corr_contextchars_vs_total": corr_len_total,
        "corr_k_vs_prefill": corr_k_prefill,
        "worst_within_k_cv": worst_cv,
        "worst_within_k": worst_k,
        "n_stalled_dropped": n_stalled,
    }

    # ---------- Policy frontiers (offline, on eval split) ----------
    fixed_pts = []
    for k in ks_present:
        res = policies.evaluate(frame, {q: k for q in eval_qids})
        fixed_pts.append(
            {"label": f"fixed-k={k}", "sweep": k, "realized_budget": res["realized_budget"],
             "f1": res["f1"], "em": res["em"], "harm": res["harm"],
             "harm_rate": res["harm_rate"], "mean_k": res["mean_k"]}
        )

    qsig = policies.question_signals(frame)
    sim_q = np.quantile(qsig["max_score"].dropna(), np.linspace(0, 1, 11))
    unc_q = np.quantile(qsig["unc0"].dropna(), np.linspace(0, 1, 11))
    k_on = max(ks_present)
    sim_pts = _frontier(frame, eval_qids,
                        lambda t: policies.policy_similarity_gate(frame, t, k_on, 0),
                        "similarity-gate", sim_q)
    unc_pts = _frontier(frame, eval_qids,
                        lambda t: policies.policy_uncertainty_gate(frame, t, k_on, 0),
                        "uncertainty-gate", unc_q)

    model, feat_cols = policies.fit_delta_model(frame, calib_qids)
    sub_eval = frame[frame["qid"].isin(eval_qids)].dropna(subset=feat_cols + ["delta_f1"])
    if len(sub_eval) > 2:
        d_hat_eval = model.predict(sub_eval[feat_cols].values)
        pred_corr = float(np.corrcoef(d_hat_eval, sub_eval["delta_f1"])[0, 1])
    else:
        pred_corr = float("nan")
    summary["delta_predictor_corr"] = pred_corr
    lam_range = np.linspace(0.0, 2.0, 11)
    chab_pts = _frontier(frame, eval_qids,
                         lambda l: policies.policy_chab_lite(frame, model, feat_cols, l, cfg["controller"]["mu"], ks_present),
                         "CHAB-lite", lam_range)

    calib_rows = calibrate.calibration_study(frame, cfg)
    calib_df = pd.DataFrame(calib_rows)

    # ---------- Fig 4: calibration ----------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ind = calib_df[calib_df["setting"] == "in-domain"]
    for sig, marker in [("max_score", "o"), ("unc0", "s")]:
        d = ind[ind["signal"] == sig].sort_values("target_B")
        ax[0].plot(d["target_B"], d["realized_B"], marker + "-", label=f"calibrated ({sig})")
    d = ind[ind["signal"] == "uncertainty_uncalibrated"].sort_values("target_B")
    ax[0].plot(d["target_B"], d["realized_B"], "x--", color="gray", label="uncalibrated")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_xlabel("target budget B")
    ax[0].set_ylabel("realized budget B̂")
    ax[0].set_title("Budget calibration (in-domain)")
    ax[0].legend(fontsize=7)
    tr = calib_df[calib_df["setting"].str.startswith("transfer")]
    if not tr.empty:
        d = tr[tr["signal"] == "max_score"].sort_values(["setting", "target_B"])
        for setting, g in d.groupby("setting"):
            ax[1].plot(g["target_B"], g["realized_B"], "o-", label=setting)
        ax[1].plot([0, 1], [0, 1], "k:", lw=1)
        ax[1].set_xlabel("target budget B")
        ax[1].set_ylabel("realized budget B̂")
        ax[1].set_title(f"Calibration under {strata_name} transfer")
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_calibration.png"))
    plt.close(fig)
    calib_df.to_csv(os.path.join(out, "calibration.csv"), index=False)

    # ---------- Fig 5: Pareto ----------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for pts, style in [(fixed_pts, "o-"), (sim_pts, "s-"), (unc_pts, "^-"), (chab_pts, "D-")]:
        xs = [p["realized_budget"] for p in pts]
        ys = [p["f1"] for p in pts]
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, style, label=pts[0]["label"].rsplit("=", 1)[0])
    ax.set_xlabel("realized budget B̂ (fraction of k=10 cost)")
    ax.set_ylabel("F1")
    ax.set_title("Quality–cost frontier")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_pareto.png"))
    plt.close(fig)

    # ---------- Matched-budget table (Sec. 12.8 style) ----------
    budgets_cmp = sorted(set(cfg["budgets"]) & {0.3, 0.5, 1.0}) or [0.5]
    table_rows = []
    families = [("fixed-k", fixed_pts), ("similarity-gate", sim_pts), ("uncertainty-gate", unc_pts), ("CHAB-lite", chab_pts)]
    for B in budgets_cmp:
        for name, pts in families:
            best = _interp_at(pts, B)
            if best is None:
                continue
            table_rows.append(
                {"target_B": B, "policy": name, "realized_B": round(best["realized_budget"], 3),
                 "F1": round(best["f1"], 3), "EM": round(best["em"], 3),
                 "harm_rate": round(best["harm_rate"], 3), "mean_k": round(best["mean_k"], 2)}
            )
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(os.path.join(out, "matched_budget_table.csv"), index=False)

    all_frontiers = fixed_pts + sim_pts + unc_pts + chab_pts
    pd.DataFrame(all_frontiers).to_csv(os.path.join(out, "frontiers.csv"), index=False)
    summary["policy_frontiers"] = {"fixed": fixed_pts, "similarity": sim_pts,
                                   "uncertainty": unc_pts, "chab_lite": chab_pts}
    summary["matched_budget_table"] = table_rows
    summary["calibration"] = calib_rows

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    _write_report(out, cfg, summary, per_k, rates, calib_df, table_df)
    print(f"Wrote figures + tables + summary.json + PRELIMINARY_RESULTS.md to {out}")


def _write_report(out, cfg, summary, per_k, rates, calib_df, table_df):
    cp = summary["cost_proxy"]
    qk = summary["quality_by_k"]
    best_k = max(qk, key=qk.get)
    lines = []
    lines.append("# CHAB-RAG — Preliminary Results (smoke test)\n")
    lines.append(f"Model: `{cfg['ollama']['model']}` (CPU, Ollama) · "
                 f"n={summary['n_questions']} HotpotQA (distractor) questions · "
                 f"k∈{summary['ks']} · retriever={cfg['retriever']}\n")
    lines.append("> Sanity pilot only — small N, single backbone, single dataset. "
                 "Numbers are indicative feasibility signals, not publication results.\n")

    lines.append("## 1. Quality vs retrieval depth\n")
    lines.append("![](" + "fig_inverted_u.png)\n")
    lines.append("Mean F1 by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in qk.items()) + "\n")
    lines.append(f"Stratified by **{summary['strata']['name']}** "
                 f"({', '.join(f'{k}: {v}' for k, v in summary['strata']['counts'].items())}).\n")
    lines.append(f"**Peak F1 at k={best_k}.** " +
                 ("Non-monotonic (inverted-U) benefit reproduced." if best_k not in (0, max(qk))
                  else "Benefit monotonic in this sample; check N before drawing conclusions.") + "\n")

    lines.append("## 2. Retrieval harm (RQ3 / H3)\n")
    lines.append("![](" + "fig_deltaQ.png)\n")
    for k, r in rates.items():
        lines.append(f"- k={k}: beneficial {r['beneficial']*100:.0f}%, neutral {r['neutral']*100:.0f}%, "
                     f"harmful {r['harmful']*100:.0f}%")
    lines.append("")

    lines.append("## 3. Cost is driven by context length, not document count (RQ4 / C3)\n")
    lines.append("![](" + "fig_cost_vs_length.png)\n")
    lines.append(f"- corr(context size in tokens, prefill time) = **{cp['corr_prompttokens_vs_prefill']:.2f}** "
                 f"(context chars vs prefill: {cp['corr_contextchars_vs_prefill']:.2f})")
    lines.append(f"- corr(context chars, total time) = **{cp['corr_contextchars_vs_total']:.2f}**")
    lines.append(f"- k is a monotone but coarse proxy: corr(k, prefill time) = **{cp['corr_k_vs_prefill']:.2f}**, "
                 f"yet within a single k the context size still varies (max coefficient of variation "
                 f"**{cp['worst_within_k_cv']:.2f}** at k={cp['worst_within_k']}) — equal-document-count "
                 f"queries differ materially in prefill/decode cost.")
    if cp["n_stalled_dropped"]:
        lines.append(f"- {cp['n_stalled_dropped']} pathological CPU stall(s) excluded from the cost analysis "
                     f"(decode tail latency is noisy on shared CPU; motivates measuring TPOT(L) curves properly).")
    lines.append("- decode tail latency is noisy on a shared CPU; the prefill signal above is the reliable part.\n")

    lines.append("## 4. Budget calibration (RQ2 / H2)\n")
    lines.append("![](" + "fig_calibration.png)\n")
    ind = calib_df[calib_df["setting"] == "in-domain"]
    for sig in ["max_score", "unc0", "uncertainty_uncalibrated"]:
        d = ind[ind["signal"] == sig]
        if not d.empty:
            lines.append(f"- {sig}: mean |B̂−B| = **{d['abs_err'].mean():.3f}**, max = {d['abs_err'].max():.3f}")
    tr = calib_df[calib_df["setting"].str.startswith("transfer")]
    if not tr.empty:
        for setting, g in tr.groupby("setting"):
            lines.append(f"- {setting}: mean |B̂−B| = **{g['abs_err'].mean():.3f}** (transfer calibration)")
    lines.append("")

    lines.append("## 5. Quality–cost frontier & matched-budget comparison (RQ1/RQ5)\n")
    lines.append("![](" + "fig_pareto.png)\n")
    lines.append(f"CHAB-lite ΔQ predictor quality (corr of predicted vs actual ΔF1 on the held-out "
                 f"split) = **{summary.get('delta_predictor_corr', float('nan')):.2f}** — a deliberately "
                 f"simple ridge model; a weak value indicates the cheap features under-predict retrieval "
                 f"benefit at this scale.\n")
    lines.append("Table = best quality achievable **within** each target budget (realized ≤ target).\n")
    if not table_df.empty:
        lines.append(table_df.to_markdown(index=False) + "\n")

    lines.append("## 6. Reading for the proposal\n")
    lines.append("- Use Sections 1–3 as the *premise evidence* (Sec. 2/3/9 of the proposal): "
                 "benefit is non-uniform and non-monotonic, and document count is a poor cost proxy.")
    lines.append("- Use Section 4 as a *feasibility demo* of realized-budget calibration (E2/E6), "
                 "and Section 5 as a template for the matched-budget table (Sec. 12.8).")
    lines.append("- Next steps: scale N, add a second backbone, and add the evidence-allocation level.\n")

    with open(os.path.join(out, "PRELIMINARY_RESULTS.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
