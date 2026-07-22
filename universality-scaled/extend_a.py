"""Universality at scale: 7-model cross-family J-space universality grid.

Generalizes `../universality/run.py` to 7 models
(gpt2-small, qwen3-1.7b, qwen3-8b, gemma-2-2b, gemma-2-9b, llama3.1-8b,
qwen3.6-27b). CPU + network only: tokenizers, unembedding/final-norm tensors
(shard-wise via the safetensors index, no full model download) and pre-fitted
lenses from neuronpedia/jacobian-lens. Norm-scaled readout convention
(primary, as validated in `../universality/`); baselines = unembedding RSA
(both norm conventions) + shuffled-labels x3.

Run from the repo root:
  python universality-scaled/extend_a.py
"""

import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from scipy.stats import spearmanr
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
for p in (VENDOR, ROOT / "../universality"):
    if p.exists():
        sys.path.insert(0, str(p))
import jlens  # noqa: E402
from words import candidate_pool  # noqa: E402

# priority-ordered: for untied models lm_head must win over embed_tokens
UNEMBED_KEYS = ("lm_head.weight", "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight", "embed_tokens.weight",
                "transformer.wte.weight", "wte.weight")
NORM_KEYS = ("model.norm.weight", "model.language_model.norm.weight",
             "norm.weight", "transformer.ln_f.weight", "ln_f.weight")


def fetch_tensor(repo, revision, keys):
    """Download only the shard holding the FIRST key (priority order) present."""
    try:
        wmap = json.load(open(hf_hub_download(repo, "model.safetensors.index.json",
                                              revision=revision)))["weight_map"]
    except EntryNotFoundError:
        wmap = None
    if wmap is None:
        with safe_open(hf_hub_download(repo, "model.safetensors", revision=revision),
                       framework="pt") as f:
            for k in keys:
                if k in f.keys():
                    return k, f.get_tensor(k)
        raise KeyError(f"{keys} not found in {repo}")
    for k in keys:
        if k in wmap:
            with safe_open(hf_hub_download(repo, wmap[k], revision=revision),
                           framework="pt") as f:
                return k, f.get_tensor(k)
    raise KeyError(f"{keys} not found in {repo} index")


def single_token_id(tok, word):
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def frac_to_layer(source_layers, n_layers, frac):
    return min(source_layers, key=lambda l: abs(l / n_layers - frac))


def cosine_sim(X):
    Xn = X / X.norm(dim=1, keepdim=True)
    return (Xn @ Xn.T).numpy()


def rsa(Sa, Sb):
    iu = np.triu_indices_from(Sa, k=1)
    return float(spearmanr(Sa[iu], Sb[iu]).statistic)


def load_model(name, mcfg, cfg, pool, fracs):
    """Return per-model dict: single-token words, u rows (both conventions),
    j-vectors per fraction (norm-scaled), layer map. Lens freed on return."""
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mcfg["hf_repo"], revision=mcfg["hf_revision"])
    st = {w: single_token_id(tok, w) for w in pool}
    words = [w for w in pool if st[w] is not None]
    ids = torch.tensor([st[w] for w in words])

    ukey, U = fetch_tensor(mcfg["hf_repo"], mcfg["hf_revision"], UNEMBED_KEYS)
    u_raw = U[ids].float()
    del U
    nkey, g = fetch_tensor(mcfg["hf_repo"], mcfg["hf_revision"], NORM_KEYS)
    g = g.float() + (1.0 if mcfg["norm_plus_one"] else 0.0)
    u_norm = u_raw * g

    lens = jlens.JacobianLens.from_pretrained(
        cfg["lens_repo"], filename=mcfg["lens_file"], revision=mcfg["lens_revision"])
    layer_map = {f: frac_to_layer(lens.source_layers, mcfg["n_layers"], f)
                 for f in fracs}
    jv = {f: (u_norm @ lens.jacobians[layer_map[f]].float()) for f in fracs}
    src = list(lens.source_layers)
    del lens
    print(f"[{name}] {len(words)} single-token | unembed={ukey} norm={nkey} "
          f"| lens layers {src[0]}..{src[-1]} (n={len(src)}) | map={layer_map} "
          f"| {time.time()-t0:.0f}s", flush=True)
    return {"words": words, "u_raw": u_raw, "u_norm": u_norm, "jv": jv,
            "layer_map": layer_map, "ukey": ukey, "nkey": nkey,
            "source_layers": src, "family": mcfg["family"],
            "params_b": mcfg["params_b"]}


def pair_metrics(A, B, concepts, fracs, mid_fracs, seeds):
    ia = torch.tensor([A["words"].index(w) for w in concepts])
    ib = torch.tensor([B["words"].index(w) for w in concepts])
    n = len(concepts)
    S_u = {m: {"norm": cosine_sim(m_d["u_norm"][i]), "raw": cosine_sim(m_d["u_raw"][i])}
           for m, (m_d, i) in {"a": (A, ia), "b": (B, ib)}.items()}
    res = {"n_concepts": n,
           "rsa_unembed": rsa(S_u["a"]["norm"], S_u["b"]["norm"]),
           "rsa_unembed_raw": rsa(S_u["a"]["raw"], S_u["b"]["raw"]),
           "rsa_j": {}, "rsa_shuffled": {}}
    for f in fracs:
        Sa = cosine_sim(A["jv"][f][ia])
        Sb = cosine_sim(B["jv"][f][ib])
        res["rsa_j"][f] = rsa(Sa, Sb)
        vals = []
        for s in seeds:
            perm = np.random.RandomState(s).permutation(n)
            vals.append(rsa(Sa, Sb[perm][:, perm]))
        res["rsa_shuffled"][f] = {"values": vals, "mean": float(np.mean(vals)),
                                  "std": float(np.std(vals))}
    details = {}
    for f in mid_fracs:
        margin = res["rsa_j"][f] - res["rsa_unembed"]
        spread = (max(res["rsa_shuffled"][f]["values"])
                  - min(res["rsa_shuffled"][f]["values"]))
        details[f] = {"margin_over_unembed": margin, "shuffled_spread": spread,
                      "pass": margin > spread and margin > 0}
    res["criterion"] = {"passed": all(d["pass"] for d in details.values()),
                        "details": details}
    base_best = max(res["rsa_unembed"], res["rsa_unembed_raw"])
    strict = {f: res["rsa_j"][f] - base_best > max(
        res["rsa_shuffled"][f]["values"]) - min(res["rsa_shuffled"][f]["values"])
        for f in mid_fracs}
    res["strict_criterion"] = {"best_unembed_baseline": base_best,
                               "per_fraction": strict,
                               "passed": all(strict.values())}
    res["mid_mean_rsa_j"] = float(np.mean([res["rsa_j"][f] for f in mid_fracs]))
    return res


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    print(f"disk free at start: {free_gb:.1f} GB", flush=True)

    cap = json.load(open(VENDOR / "data/experiments/capacity.json"))
    countries = next(p["pool"] for p in cap["candidate_pools"] if p["name"] == "countries")
    pool = candidate_pool(countries)

    fracs = cfg["fractions"]
    mid = cfg["mid_fractions"]
    models = {name: load_model(name, mcfg, cfg, pool, fracs)
              for name, mcfg in cfg["models"].items()}

    shared = [w for w in pool if all(w in m["words"] for m in models.values())]
    n_shared = len(shared)
    print(f"pool {len(pool)} -> {n_shared} concepts single-token in ALL 7 models",
          flush=True)
    assert n_shared >= cfg["min_concepts"], f"only {n_shared} shared concepts"
    use_pairwise_max = n_shared < cfg["pairwise_max_threshold"]

    pairs, pairs_max = {}, {}
    for a, b in itertools.combinations(models, 2):
        key = f"{a}|{b}"
        pairs[key] = pair_metrics(models[a], models[b], shared, fracs, mid,
                                  cfg["baseline_seeds"])
        pw = [w for w in pool if w in models[a]["words"] and w in models[b]["words"]]
        if use_pairwise_max:
            pairs_max[key] = pair_metrics(models[a], models[b], pw, fracs, mid,
                                          cfg["baseline_seeds"])
        else:
            pairs_max[key] = {"n_concepts": len(pw), "note": "not computed; "
                              "7-way shared set >= threshold"}
        print(f"{key}: n={pairs[key]['n_concepts']} "
              f"mid_mean_J={pairs[key]['mid_mean_rsa_j']:.3f} "
              f"unembed={pairs[key]['rsa_unembed']:.3f} "
              f"pass={pairs[key]['criterion']['passed']}", flush=True)

    # ---- family structure ----
    fam = {m: models[m]["family"] for m in models}
    within = [k for k in pairs if fam[k.split("|")[0]] == fam[k.split("|")[1]]]
    cross = [k for k in pairs if k not in within]
    mid_means = {k: pairs[k]["mid_mean_rsa_j"] for k in pairs}
    per_model_cross = {}
    for m in models:
        vals = [mid_means[k] for k in cross if m in k.split("|")]
        per_model_cross[m] = {"mean_cross_family_mid_rsa": float(np.mean(vals)),
                              "n_pairs": len(vals),
                              "params_b": models[m]["params_b"]}
    anchor = "qwen3.6-27b"
    anchor_vals = [mid_means[k] for k in pairs if anchor in k.split("|")]
    others_vals = [mid_means[k] for k in pairs if anchor not in k.split("|")]
    ranking = sorted(per_model_cross, key=lambda m: -per_model_cross[m]
                     ["mean_cross_family_mid_rsa"])
    analysis = {
        "within_family_pairs": {k: mid_means[k] for k in within},
        "cross_family_pairs": {k: mid_means[k] for k in cross},
        "cross_family_range": [float(min(mid_means[k] for k in cross)),
                               float(max(mid_means[k] for k in cross))],
        "within_family_range": [float(min(mid_means[k] for k in within)),
                                float(max(mid_means[k] for k in within))],
        "cross_family_mean": float(np.mean([mid_means[k] for k in cross])),
        "within_family_mean": float(np.mean([mid_means[k] for k in within])),
        "size_ladders": {
            "qwen": {k: mid_means[k] for k in within if fam[k.split("|")[0]] == "qwen"},
            "gemma": {k: mid_means[k] for k in within if fam[k.split("|")[0]] == "gemma"},
            "note": "qwen ladder top (27b) is the qwen3.6 generation, 1.7b/8b are qwen3",
        },
        "per_model_mean_cross_family_alignment": per_model_cross,
        "alignment_ranking_most_to_least": ranking,
        "anchor_27b": {
            "model": anchor,
            "mean_mid_rsa_vs_all_others": float(np.mean(anchor_vals)),
            "mean_mid_rsa_among_non_anchor_pairs": float(np.mean(others_vals)),
            "rank_by_cross_family_alignment":
                ranking.index(anchor) + 1,
        },
    }

    metrics = {
        "manifest": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_clock_s": None,
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "numpy": np.__version__,
                         "transformers": __import__("transformers").__version__,
                         "scipy": __import__("scipy").__version__},
            "seed": cfg["seed"], "baseline_seeds": cfg["baseline_seeds"],
            "lens_repo": cfg["lens_repo"],
            "n_shared_concepts": n_shared,
            "pairwise_max_computed": use_pairwise_max,
            "convention": "norm_scaled (primary readout convention); rsa_unembed_raw "
                          "kept as extra baseline; J-space raw variant and SVCCA "
                          "omitted at this scale (A showed raw collapses for qwen "
                          "and SVCCA shows no separation)",
            "tensor_keys": {m: {"unembed": models[m]["ukey"], "norm": models[m]["nkey"]}
                            for m in models},
            "lens_source_layers": {m: models[m]["source_layers"] for m in models},
        },
        "concepts": shared,
        "fractions": fracs,
        "mid_fractions": mid,
        "layer_map": {m: models[m]["layer_map"] for m in models},
        "families": fam,
        "pairs": pairs,
        "pairs_pairwise_max": pairs_max,
        "analysis": analysis,
        "criterion_passed_all_pairs": all(p["criterion"]["passed"]
                                          for p in pairs.values()),
        "strict_passed_all_pairs": all(p["strict_criterion"]["passed"]
                                       for p in pairs.values()),
    }
    metrics["manifest"]["wall_clock_s"] = round(time.time() - t0, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(metrics, open(ROOT / "results/metrics_extend_a.json", "w"), indent=1)
    print(f"DONE criterion={metrics['criterion_passed_all_pairs']} "
          f"strict={metrics['strict_passed_all_pairs']} "
          f"wall={metrics['manifest']['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
