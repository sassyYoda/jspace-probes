"""Re-run Idea A's universality comparison with the code-corpus Qwen lens.

Imports ../universality/run.py machinery (no reimplementation):
same 507 concepts, same fractions, same baselines (unembed, shuffled, random-orth),
same norm_scaled/raw variants. Only qwen3-1.7b's lens is swapped: wikitext ->
codeparrot refit (results/qwen3-1.7b_codeparrot_lens.pt).

Also computes the SAME-model cross-corpus RSA (qwen wikitext lens vs qwen code
lens) per fraction — Idea G's first lens-stability datapoint.
"""

import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
A_ROOT = (ROOT / "../universality").resolve()
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))

spec = importlib.util.spec_from_file_location("run_a", A_ROOT / "run.py")
run_a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_a)  # also puts the jlens repo + universality dir on sys.path
import jlens  # noqa: E402  (importable after run_a's sys.path inserts)
from words import candidate_pool  # noqa: E402


def main(lens_path=None, outdir=None):
    """lens_path/outdir: test overrides (default: config lens, results/)."""
    t0 = time.time()
    outdir = Path(outdir) if outdir else ROOT / "results"
    cfg_g = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg = yaml.safe_load(open(A_ROOT / "config.yaml"))  # A's comparison config
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    cap = json.load(open(VENDOR / "data/experiments/capacity.json"))
    countries = next(p["pool"] for p in cap["candidate_pools"] if p["name"] == "countries")
    pool = candidate_pool(countries)

    # A's models with their wikitext lenses (patch A_ROOT so nothing resolves
    # relative to this experiment by accident)
    models = {name: run_a.load_model_assets(name, mcfg, cfg["lens_repo"], cfg["lens_revision"])
              for name, mcfg in cfg["models"].items()}

    # the refit lens replaces qwen's wikitext lens
    code_lens = jlens.JacobianLens.load(
        str(lens_path or ROOT / cfg_g["comparison"]["refit_lens_file"]))
    qwen_wiki_lens = models["qwen3-1.7b"]["lens"]
    models["qwen3-1.7b"] = {**models["qwen3-1.7b"], "lens": code_lens}

    concepts = [w for w in pool
                if all(run_a.single_token_id(m["tok"], w) is not None for m in models.values())]
    n = len(concepts)
    print(f"pool {len(pool)} -> {n} concepts single-token in all models")
    assert n >= cfg["min_concepts"], f"only {n} shared concepts"

    fracs = cfg["fractions"]

    # the code lens was fitted at exactly the layers A's frac_to_layer picked for
    # the wikitext lens; assert the mapping is unchanged
    a_metrics = json.load(open(A_ROOT / "results/metrics.json"))
    qwen_map = {f: run_a.frac_to_layer(models["qwen3-1.7b"], f) for f in fracs}
    assert {str(k): v for k, v in qwen_map.items()} == a_metrics["layer_map"]["qwen3-1.7b"], \
        f"fraction->layer mapping drifted: {qwen_map}"

    variants = {}
    for variant, apply_norm in (("norm_scaled", True), ("raw", False)):
        per_model = {}
        for idx, (name, m) in enumerate(models.items()):
            u = run_a.concept_rows(m, concepts, apply_norm)
            layer_map = {f: run_a.frac_to_layer(m, f) for f in fracs}
            jvecs = {f: u @ m["lens"].jacobians[layer_map[f]] for f in fracs}
            per_model[name] = {
                "u": u, "S_u": run_a.cosine_sim(u), "layer_map": layer_map,
                "jvecs": jvecs,
                "S_j": {f: run_a.cosine_sim(jv) for f, jv in jvecs.items()},
                "orth": {s: u @ run_a.random_orthogonal(u.shape[1], seed=s + 1000 * idx)
                         for s in cfg["baseline_seeds"]},
            }
        variants[variant] = {"per_model": per_model,
                             "pairs": run_a.compute_pairs(per_model, fracs, cfg, n)}

    # ---- same-model cross-corpus stability (bonus / Idea G datapoint) ----
    same_model = {}
    for variant, apply_norm in (("norm_scaled", True), ("raw", False)):
        m = models["qwen3-1.7b"]
        u = run_a.concept_rows(m, concepts, apply_norm)
        res = {}
        for f in fracs:
            l = qwen_map[f]
            S_code = run_a.cosine_sim(u @ code_lens.jacobians[l])
            S_wiki = run_a.cosine_sim(u @ qwen_wiki_lens.jacobians[l])
            shuf = []
            for s in cfg["baseline_seeds"]:
                perm = np.random.RandomState(s).permutation(n)
                shuf.append(run_a.rsa(S_wiki, S_code[perm][:, perm]))
            res[f] = {"rsa": run_a.rsa(S_wiki, S_code),
                      "rsa_vs_unembed_code": run_a.rsa(run_a.cosine_sim(u), S_code),
                      "rsa_shuffled_mean": float(np.mean(shuf))}
        same_model[variant] = res

    # ---- cross-corpus vs A's same-corpus values, pre-registered read ----
    tol = cfg_g["comparison"]["rsa_tolerance"]
    a_pairs = a_metrics["variants"]["norm_scaled"]["pairs"]
    g_pairs = variants["norm_scaled"]["pairs"]
    confound = {}
    for pair in ("qwen3-1.7b|gemma-2-2b", "qwen3-1.7b|gpt2-small"):
        rows = {}
        for f in cfg["mid_fractions"]:
            wiki = a_pairs[pair]["rsa_j"][str(f)]
            code = g_pairs[pair]["rsa_j"][f]
            base = g_pairs[pair]["rsa_unembed"]
            rows[f] = {"rsa_wiki_lens": wiki, "rsa_code_lens": code,
                       "delta": code - wiki, "within_tolerance": abs(code - wiki) <= tol,
                       "rsa_unembed": base, "margin_over_unembed": code - base,
                       "above_baseline": code - base > (
                           max(g_pairs[pair]["rsa_shuffled"][f]["values"])
                           - min(g_pairs[pair]["rsa_shuffled"][f]["values"]))}
        rows_ok = all(r["within_tolerance"] and r["above_baseline"] for r in rows.values())
        confound[pair] = {"fractions": rows, "survives": rows_ok}
    confound_dead = all(p["survives"] for p in confound.values())

    # sanity: the untouched gemma|gpt2 pair must reproduce A's numbers exactly
    sanity = max(abs(g_pairs["gemma-2-2b|gpt2-small"]["rsa_j"][f]
                     - a_pairs["gemma-2-2b|gpt2-small"]["rsa_j"][str(f)]) for f in fracs)

    metrics = {
        "manifest": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_clock_s": None,
            "refit_lens": {"file": cfg_g["comparison"]["refit_lens_file"],
                           "corpus": cfg_g["corpus"]["hf_dataset"],
                           "n_prompts": code_lens.n_prompts,
                           "source_layers": code_lens.source_layers},
            "seed": cfg["seed"], "baseline_seeds": cfg["baseline_seeds"],
            "n_concepts": n, "primary_variant": "norm_scaled",
            "rsa_tolerance": tol,
            "note": "identical to idea-A run.py except qwen3-1.7b's wikitext lens "
                    "is replaced by the codeparrot refit; gemma-2-2b|gpt2-small "
                    "pair is an unchanged-reproduction sanity check",
            "sanity_max_abs_diff_gemma_gpt2_vs_A": sanity,
        },
        "fractions": fracs,
        "layer_map": {name: pm["layer_map"]
                      for name, pm in variants["norm_scaled"]["per_model"].items()},
        "variants": {v: {"pairs": variants[v]["pairs"]} for v in variants},
        "same_model_cross_corpus": same_model,
        "confound_check": {
            "note": f"pre-registered: cross-corpus RSA within {tol} of same-corpus "
                    "values at mid fractions AND margin over unembed exceeds the "
                    "shuffled spread => shared-corpus confound is dead",
            "pairs": confound, "confound_dead": confound_dead,
        },
    }
    metrics["manifest"]["wall_clock_s"] = round(time.time() - t0, 1)
    outdir.mkdir(exist_ok=True, parents=True)
    json.dump(metrics, open(outdir / "metrics.json", "w"), indent=1)

    plot(g_pairs, a_pairs, same_model["norm_scaled"], fracs,
         outdir / "rsa_cross_corpus.png")
    print(f"confound_dead={confound_dead} sanity_diff={sanity:.2e} "
          f"wall={metrics['manifest']['wall_clock_s']}s")
    for pair, res in confound.items():
        for f, r in res["fractions"].items():
            print(f"  {pair} @{f}: wiki={r['rsa_wiki_lens']:.3f} "
                  f"code={r['rsa_code_lens']:.3f} d={r['delta']:+.3f} "
                  f"tol_ok={r['within_tolerance']} above_base={r['above_baseline']}")


def plot(g_pairs, a_pairs, same_model, fracs, path):
    """Hue = model pair (entity); linestyle = lens corpus. One axis; legend +
    end labels (palette validated: CVD pass, contrast WARN -> direct labels)."""
    colors = {"qwen3-1.7b|gemma-2-2b": "#2a78d6", "qwen3-1.7b|gpt2-small": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    for pair, c in colors.items():
        y_code = [g_pairs[pair]["rsa_j"][f] for f in fracs]
        y_wiki = [a_pairs[pair]["rsa_j"][str(f)] for f in fracs]
        ax.plot(fracs, y_code, color=c, lw=2, marker="o", ms=4,
                label=pair.replace("|", " vs ") + " (code lens)")
        ax.plot(fracs, y_wiki, color=c, lw=1.6, ls="--", alpha=0.65,
                label=pair.replace("|", " vs ") + " (wikitext lens, A)")
        ax.annotate(f"{y_code[-1]:.2f}", (fracs[-1], y_code[-1]), xytext=(5, 0),
                    textcoords="offset points", color=c, fontsize=8, va="center")
        ax.axhline(g_pairs[pair]["rsa_unembed"], color=c, lw=1, ls=(0, (1, 3)), alpha=0.8)
        ax.plot(fracs, [g_pairs[pair]["rsa_shuffled"][f]["mean"] for f in fracs],
                color=c, lw=0.8, ls=":", alpha=0.4)
    y_same = [same_model[f]["rsa"] for f in fracs]
    ax.plot(fracs, y_same, color="#6b6b6b", lw=1.6, marker="s", ms=3,
            label="qwen wikitext vs code lens (same model)")
    ax.annotate(f"{y_same[-1]:.2f}", (fracs[-1], y_same[-1]), xytext=(5, 0),
                textcoords="offset points", color="#6b6b6b", fontsize=8, va="center")
    ax.plot([], [], color="0.4", ls=(0, (1, 3)), label="unembedding baseline")
    ax.set_xlabel("layer fraction")
    ax.set_ylabel("RSA (Spearman)")
    ax.set_title("Cross-model J-space RSA survives a corpus swap? (code-refit Qwen lens)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=min(0, min(y_same) - 0.05))
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default=None, help="override refit lens path (testing)")
    ap.add_argument("--outdir", default=None, help="override output dir (testing)")
    args = ap.parse_args()
    main(lens_path=args.lens, outdir=args.outdir)
