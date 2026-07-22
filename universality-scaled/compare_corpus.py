"""Part 2 analysis: corpus-grid RSA rows for the refit lenses.

For each refit lens (qwen3-1.7b chat, qwen3-1.7b code [from `../corpus-refit/`],
gemma-2-2b code):
  - cross-model RSA rows: refit-lens model vs the OTHER 6 models' wikitext
    lenses, on the 7-way shared concept set (A-style comparison, norm-scaled);
  - delta vs the wikitext-lens rows from extend_a's metrics;
  - verdict per the pre-registered corpus-refit tolerance: at mid fractions
    |RSA_refit - RSA_wiki| <= 0.1 AND margin over unembed > max(0.1, shuffled
    spread);
  - same-model cross-corpus lens RSA (wikitext vs chat vs code where available).

Requires results/metrics_extend_a.json (Part 1) and the fitted lens files.
CPU-only; run after fits complete.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import extend_a  # noqa: E402  (also sets up jlens/words sys.path)
import jlens  # noqa: E402


def refit_jv(base, lens_path, fracs):
    lens = jlens.JacobianLens.load(str(lens_path))
    lmap = base["layer_map"]
    missing = [lmap[f] for f in fracs if lmap[f] not in lens.source_layers]
    assert not missing, f"refit lens lacks layers {missing}"
    jv = {f: base["u_norm"] @ lens.jacobians[lmap[f]].float() for f in fracs}
    meta = {"n_prompts": getattr(lens, "n_prompts", None),
            "source_layers": list(lens.source_layers)}
    del lens
    return jv, meta


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    fracs = cfg["fractions"]
    mid = cfg["mid_fractions"]
    tol = cfg["comparison"]["rsa_tolerance"]
    seeds = cfg["baseline_seeds"]

    part1 = json.load(open(ROOT / "results/metrics_extend_a.json"))
    shared = part1["concepts"]

    cap = json.load(open(extend_a.VENDOR / "data/experiments/capacity.json"))
    countries = next(p["pool"] for p in cap["candidate_pools"] if p["name"] == "countries")
    from words import candidate_pool  # noqa: E402
    pool = candidate_pool(countries)

    models = {name: extend_a.load_model(name, mcfg, cfg, pool, fracs)
              for name, mcfg in cfg["models"].items()}

    refits = {
        "qwen3-1.7b|chat": ("qwen3-1.7b", ROOT / cfg["fits"]["qwen3-1.7b-chat"]["lens_out"]),
        "qwen3-1.7b|code": ("qwen3-1.7b", ROOT / cfg["comparison"]["qwen_code_lens"]),
        "gemma-2-2b|code": ("gemma-2-2b", ROOT / cfg["fits"]["gemma-2-2b-code"]["lens_out"]),
        # n=100 wikitext refit CONTROL: its deltas vs the neuronpedia wikitext
        # lens measure pure fit noise (same corpus, fewer prompts)
        "gemma-2-2b|wiki100": ("gemma-2-2b", ROOT / cfg["fits"]["gemma-2-2b-wiki"]["lens_out"]),
    }

    fam = part1["families"]
    out_rows, jv_cache, meta_cache = {}, {}, {}
    for tag, (mname, path) in refits.items():
        base = models[mname]
        jv, meta = refit_jv(base, path, fracs)
        jv_cache[tag] = jv
        meta_cache[tag] = {**meta, "lens_file": str(path.name)}
        variant = {**base, "jv": jv}
        rows = {}
        for other in models:
            if other == mname:
                continue
            key = f"{mname}|{other}" if f"{mname}|{other}" in part1["pairs"] \
                else f"{other}|{mname}"
            wiki = part1["pairs"][key]
            a, b = (variant, models[other]) if key.startswith(mname) \
                else (models[other], variant)
            res = extend_a.pair_metrics(a, b, shared, fracs, mid, seeds)
            per_frac = {}
            for f in mid:
                r_ref = res["rsa_j"][f]
                r_wiki = wiki["rsa_j"][str(f)]
                spread = (max(res["rsa_shuffled"][f]["values"])
                          - min(res["rsa_shuffled"][f]["values"]))
                margin = r_ref - res["rsa_unembed"]
                per_frac[f] = {
                    "rsa_wiki_lens": r_wiki, "rsa_refit_lens": r_ref,
                    "delta": r_ref - r_wiki,
                    "within_tolerance": abs(r_ref - r_wiki) <= tol,
                    "rsa_unembed": res["rsa_unembed"],
                    "margin_over_unembed": margin,
                    "above_baseline": margin > max(0.1, spread),
                }
            rows[other] = {
                "cross_family": fam[other] != fam[mname],
                "rsa_j_all_fracs": res["rsa_j"],
                "fractions": per_frac,
                "survives": all(r["within_tolerance"] and r["above_baseline"]
                                for r in per_frac.values()),
            }
        cross_ok = all(r["survives"] for r in rows.values() if r["cross_family"])
        all_ok = all(r["survives"] for r in rows.values())
        out_rows[tag] = {"rows": rows, "verdict_cross_family": cross_ok,
                         "verdict_all": all_ok}
        print(f"{tag}: cross-family verdict={'PASS' if cross_ok else 'FAIL'} "
              f"(all-pairs {'PASS' if all_ok else 'FAIL'})", flush=True)

    # ---- same-model cross-corpus lens RSA ----
    def lens_rsa(mname, jv_a, jv_b):
        base = models[mname]
        idx = torch.tensor([base["words"].index(w) for w in shared])
        return {f: extend_a.rsa(extend_a.cosine_sim(jv_a[f][idx]),
                                extend_a.cosine_sim(jv_b[f][idx])) for f in fracs}

    same_model = {}
    q = models["qwen3-1.7b"]
    same_model["qwen3-1.7b"] = {
        "wikitext|chat": lens_rsa("qwen3-1.7b", q["jv"], jv_cache["qwen3-1.7b|chat"]),
        "wikitext|code": lens_rsa("qwen3-1.7b", q["jv"], jv_cache["qwen3-1.7b|code"]),
        "chat|code": lens_rsa("qwen3-1.7b", jv_cache["qwen3-1.7b|chat"],
                              jv_cache["qwen3-1.7b|code"]),
    }
    g = models["gemma-2-2b"]
    same_model["gemma-2-2b"] = {
        "wikitext|code": lens_rsa("gemma-2-2b", g["jv"], jv_cache["gemma-2-2b|code"]),
        "wikitext|wiki100": lens_rsa("gemma-2-2b", g["jv"],
                                     jv_cache["gemma-2-2b|wiki100"]),
        "wiki100|code": lens_rsa("gemma-2-2b", jv_cache["gemma-2-2b|wiki100"],
                                 jv_cache["gemma-2-2b|code"]),
    }

    metrics = {
        "manifest": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_clock_s": None,
            "n_shared_concepts": len(shared),
            "rsa_tolerance": tol,
            "verdict_rule": "at mid fractions: |RSA_refit - RSA_wiki| <= 0.1 AND "
                            "margin over unembed > max(0.1, shuffled spread)",
            "refit_lenses": meta_cache,
            "chat_corpus_note": "lmsys/lmsys-chat-1m gated (403); used "
                                "HuggingFaceH4/ultrachat_200k train_sft user turns",
        },
        "fractions": fracs,
        "mid_fractions": mid,
        "corpus_grid": out_rows,
        "same_model_cross_corpus": same_model,
    }
    metrics["manifest"]["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(metrics, open(ROOT / "results/metrics_corpus_grid.json", "w"), indent=1)
    print(f"DONE wall={metrics['manifest']['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
