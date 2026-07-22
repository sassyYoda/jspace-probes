"""Figures for the A+G extension.

  rsa_matrix.png  : (a) 7x7 heatmap — lower triangle = mean mid-fraction J-space
                    RSA, upper triangle = unembedding-baseline RSA; (b) size
                    ladder — mean cross-family mid RSA vs params.
  corpus_grid.png : per refit lens, RSA-vs-fraction rows (refit solid, wikitext
                    dashed) + same-model cross-corpus panel.

Reads results/metrics_extend_a.json (+ results/metrics_corpus_grid.json if
present). CPU-only, run anywhere.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ORDER = ["gpt2-small", "qwen3-1.7b", "qwen3-8b", "qwen3.6-27b",
         "gemma-2-2b", "gemma-2-9b", "llama3.1-8b"]
# fixed categorical palette, one color per model
COLOR = {"gpt2-small": "#2a78d6", "qwen3-1.7b": "#d0509b", "qwen3-8b": "#d08700",
         "qwen3.6-27b": "#0e9aa7", "gemma-2-2b": "#6a49c8",
         "gemma-2-9b": "#1a9e6e", "llama3.1-8b": "#e4573d"}
FAM_SPANS = [(0, 1), (1, 4), (4, 6), (6, 7)]  # gpt2 | qwen | gemma | llama


def pair_get(pairs, a, b):
    return pairs.get(f"{a}|{b}") or pairs[f"{b}|{a}"]


def fig_matrix(m):
    pairs = m["pairs"]
    mid = [str(f) for f in m["mid_fractions"]]
    n = len(ORDER)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if i == j:
                continue
            p = pair_get(pairs, a, b)
            if i > j:   # lower: J-space mean mid RSA
                M[i, j] = np.mean([p["rsa_j"][f] for f in mid])
            else:       # upper: unembedding baseline
                M[i, j] = p["rsa_unembed"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 6), dpi=150,
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else "#1a2733")
    for lo, hi in FAM_SPANS[:-1]:
        ax.axhline(hi - 0.5, color="#fcfcfb", lw=2.2)
        ax.axvline(hi - 0.5, color="#fcfcfb", lw=2.2)
    ax.set_xticks(range(n), ORDER, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(n), ORDER, fontsize=8.5)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_color("#1a2733")
    ax.set_title("lower △: J-space RSA (mean over fractions 0.4–0.7)\n"
                 "upper △: unembedding-baseline RSA", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("RSA (Spearman)", fontsize=9)
    cb.outline.set_visible(False)

    pm = m["analysis"]["per_model_mean_cross_family_alignment"]
    for name in ORDER:
        d = pm[name]
        ax2.scatter(d["params_b"], d["mean_cross_family_mid_rsa"], s=64,
                    color=COLOR[name], zorder=3)
        ax2.annotate(name, (d["params_b"], d["mean_cross_family_mid_rsa"]),
                     xytext=(7, -3), textcoords="offset points", fontsize=8.5,
                     color="#1a2733")
    fams = {"qwen": ["qwen3-1.7b", "qwen3-8b", "qwen3.6-27b"],
            "gemma": ["gemma-2-2b", "gemma-2-9b"]}
    for fam, names in fams.items():
        xs = [pm[nm]["params_b"] for nm in names]
        ys = [pm[nm]["mean_cross_family_mid_rsa"] for nm in names]
        ax2.plot(xs, ys, color="#9aa4ad", lw=1.4, zorder=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("parameters (B, log scale)", fontsize=9)
    ax2.set_ylabel("mean cross-family J-space RSA (mid fractions)", fontsize=9)
    ax2.set_title("size ladder: cross-family alignment vs scale", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="0.92", lw=0.8)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / "results/rsa_matrix.png")
    print("wrote results/rsa_matrix.png")


def fig_corpus(m1, m2):
    fracs = [float(f) for f in m1["fractions"]]
    grid = m2["corpus_grid"]
    tags = ["qwen3-1.7b|chat", "qwen3-1.7b|code", "gemma-2-2b|code"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), dpi=150)
    axes = axes.ravel()
    for ax, tag in zip(axes[:3], tags):
        mname, corpus = tag.split("|")
        rows = grid[tag]["rows"]
        for other, r in rows.items():
            if not r["cross_family"]:
                continue
            c = COLOR[other]
            wiki = pair_get(m1["pairs"], mname, other)
            y_ref = [r["rsa_j_all_fracs"][str(f)] for f in fracs]
            y_wik = [wiki["rsa_j"][str(f)] for f in fracs]
            ax.plot(fracs, y_ref, color=c, lw=2, marker="o", ms=3.5, label=other)
            ax.plot(fracs, y_wik, color=c, lw=1.3, ls="--", alpha=0.55)
            ax.axhline(wiki["rsa_unembed"], color=c, lw=0.8, ls=(0, (1, 3)),
                       alpha=0.7)
        v = "PASS" if grid[tag]["verdict_cross_family"] else "FAIL"
        ax.set_title(f"{mname} refit on {corpus} vs others' wikitext lenses "
                     f"[{v}]", fontsize=9.5)
        ax.set_ylim(-0.05, 1.0)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", title=None)
    ax = axes[3]
    sm = m2["same_model_cross_corpus"]
    styles = {("qwen3-1.7b", "wikitext|chat"): ("-", "o"),
              ("qwen3-1.7b", "wikitext|code"): ("--", "s"),
              ("qwen3-1.7b", "chat|code"): (":", "^"),
              ("gemma-2-2b", "wikitext|code"): ("--", "s"),
              ("gemma-2-2b", "wikitext|wiki100"): ("-", "o"),
              ("gemma-2-2b", "wiki100|code"): (":", "^")}
    for mname, combos in sm.items():
        for combo, vals in combos.items():
            ls, mk = styles[(mname, combo)]
            ax.plot(fracs, [vals[str(f)] for f in fracs], color=COLOR[mname],
                    lw=1.8, ls=ls, marker=mk, ms=3.5,
                    label=f"{mname}: {combo.replace('|', ' vs ')}")
    ax.set_title("same-model cross-corpus lens RSA", fontsize=9.5)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for ax in axes:
        ax.set_xlabel("layer fraction", fontsize=9)
        ax.set_ylabel("RSA (Spearman)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Corpus grid: solid = refit lens, dashed = wikitext lens, "
                 "dotted = unembedding baseline", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(ROOT / "results/corpus_grid.png")
    print("wrote results/corpus_grid.png")


if __name__ == "__main__":
    m1 = json.load(open(ROOT / "results/metrics_extend_a.json"))
    fig_matrix(m1)
    p2 = ROOT / "results/metrics_corpus_grid.json"
    if p2.exists():
        fig_corpus(m1, json.load(open(p2)))
    else:
        print("corpus grid metrics not present yet; skipped corpus_grid.png")
