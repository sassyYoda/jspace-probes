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
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from scipy.stats import spearmanr
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))
import jlens
from words import candidate_pool

UNEMBED_KEYS = ("lm_head.weight", "model.embed_tokens.weight", "embed_tokens.weight",
                "transformer.wte.weight", "wte.weight")
NORM_KEYS = ("model.norm.weight", "norm.weight", "transformer.ln_f.weight", "ln_f.weight")


def fetch_tensor(repo, revision, keys):
    try:
        wmap = json.load(open(hf_hub_download(repo, "model.safetensors.index.json",
                                              revision=revision)))["weight_map"]
        files = {wmap[k] for k in keys if k in wmap}
    except EntryNotFoundError:
        files = {"model.safetensors"}
    for fname in sorted(files):
        with safe_open(hf_hub_download(repo, fname, revision=revision), framework="pt") as f:
            for k in keys:
                if k in f.keys():
                    return k, f.get_tensor(k)
    raise KeyError(f"{keys} not found in {repo}")


def load_model_assets(name, mcfg, lens_repo, lens_revision):
    tok = AutoTokenizer.from_pretrained(mcfg["hf_repo"], revision=mcfg["hf_revision"])
    ukey, U = fetch_tensor(mcfg["hf_repo"], mcfg["hf_revision"], UNEMBED_KEYS)
    nkey, g = fetch_tensor(mcfg["hf_repo"], mcfg["hf_revision"], NORM_KEYS)
    g = g.float()
    if mcfg["norm_plus_one"]:
        g = g + 1.0
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=mcfg["lens_file"],
                                              revision=lens_revision)
    return {"name": name, "tok": tok, "U": U, "ukey": ukey, "g": g, "nkey": nkey,
            "lens": lens, "n_layers": mcfg["n_layers"]}


def single_token_id(tok, word):
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def concept_rows(m, concepts, apply_norm):
    ids = [single_token_id(m["tok"], w) for w in concepts]
    u = m["U"][torch.tensor(ids)].float()
    return u * m["g"] if apply_norm else u


def frac_to_layer(m, frac):
    layers = m["lens"].source_layers
    return min(layers, key=lambda l: abs(l / m["n_layers"] - frac))


def cosine_sim(X):
    Xn = X / X.norm(dim=1, keepdim=True)
    return (Xn @ Xn.T).numpy()


def rsa(Sa, Sb):
    iu = np.triu_indices_from(Sa, k=1)
    return float(spearmanr(Sa[iu], Sb[iu]).statistic)


def svcca(Xa, Xb, k):
    def pcs(X):
        Xc = X - X.mean(0, keepdim=True)
        Uf, _, _ = torch.linalg.svd(Xc, full_matrices=False)
        return Uf[:, :k]
    s = torch.linalg.svdvals(pcs(Xa).T @ pcs(Xb))
    return float(s.clamp(max=1.0).mean())


def random_orthogonal(d, seed):
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=gen))
    return q * torch.sign(torch.diagonal(r))


def compute_pairs(per_model, fracs, cfg, n):
    k = cfg["svcca_components"]
    pairs = {}
    for a, b in itertools.combinations(per_model, 2):
        A, B = per_model[a], per_model[b]
        rsa_u = rsa(A["S_u"], B["S_u"])
        res = {
            "rsa_j": {f: rsa(A["S_j"][f], B["S_j"][f]) for f in fracs},
            "svcca_j": {f: svcca(A["jvecs"][f], B["jvecs"][f], k) for f in fracs},
            "rsa_unembed": rsa_u,
            "svcca_unembed": svcca(A["u"], B["u"], k),
        }
        shuf = {}
        for f in fracs:
            vals = []
            for s in cfg["baseline_seeds"]:
                perm = np.random.RandomState(s).permutation(n)
                vals.append(rsa(A["S_j"][f], B["S_j"][f][perm][:, perm]))
            shuf[f] = {"values": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals))}
        res["rsa_shuffled"] = shuf
        orth_vals = [rsa(cosine_sim(A["orth"][s]), cosine_sim(B["orth"][s]))
                     for s in cfg["baseline_seeds"]]
        res["rsa_random_orth"] = {"values": orth_vals, "mean": float(np.mean(orth_vals)),
                                  "std": float(np.std(orth_vals))}

        details = {}
        for f in cfg["mid_fractions"]:
            margin = res["rsa_j"][f] - rsa_u
            spread = max(shuf[f]["values"]) - min(shuf[f]["values"])
            details[f] = {"margin_over_unembed": margin, "shuffled_spread": spread,
                          "pass": margin > spread and margin > 0}
        res["criterion"] = {"passed": all(d["pass"] for d in details.values()),
                            "details": details}
        pairs[f"{a}|{b}"] = res
    return pairs


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    cap = json.load(open(VENDOR / "data/experiments/capacity.json"))
    countries = next(p["pool"] for p in cap["candidate_pools"] if p["name"] == "countries")
    pool = candidate_pool(countries)

    models = {name: load_model_assets(name, mcfg, cfg["lens_repo"], cfg["lens_revision"])
              for name, mcfg in cfg["models"].items()}

    concepts = [w for w in pool
                if all(single_token_id(m["tok"], w) is not None for m in models.values())]
    n = len(concepts)
    print(f"pool {len(pool)} -> {n} concepts single-token in all models")
    assert n >= cfg["min_concepts"], f"only {n} shared concepts"

    fracs = cfg["fractions"]
    variants = {}
    for variant, apply_norm in (("norm_scaled", True), ("raw", False)):
        per_model = {}
        for idx, (name, m) in enumerate(models.items()):
            u = concept_rows(m, concepts, apply_norm)
            layer_map = {f: frac_to_layer(m, f) for f in fracs}
            jvecs = {f: u @ m["lens"].jacobians[layer_map[f]] for f in fracs}
            per_model[name] = {
                "u": u, "S_u": cosine_sim(u), "layer_map": layer_map,
                "jvecs": jvecs, "S_j": {f: cosine_sim(jv) for f, jv in jvecs.items()},
                "orth": {s: u @ random_orthogonal(u.shape[1], seed=s + 1000 * idx)
                         for s in cfg["baseline_seeds"]},
            }
        variants[variant] = {"per_model": per_model,
                             "pairs": compute_pairs(per_model, fracs, cfg, n)}
    for name, m in models.items():
        print(f"{name}: unembed={m['ukey']} norm={m['nkey']} "
              f"layers={variants['norm_scaled']['per_model'][name]['layer_map']}")

    passed = {v: all(p["criterion"]["passed"] for p in variants[v]["pairs"].values())
              for v in variants}

    strict = {}
    for p, res in variants["norm_scaled"]["pairs"].items():
        base = max(variants[v]["pairs"][p]["rsa_unembed"] for v in variants)
        details = {}
        for f in cfg["mid_fractions"]:
            margin = res["rsa_j"][f] - base
            spread = (max(res["rsa_shuffled"][f]["values"])
                      - min(res["rsa_shuffled"][f]["values"]))
            details[f] = {"margin_over_best_unembed": margin, "shuffled_spread": spread,
                          "pass": margin > spread and margin > 0}
        strict[p] = {"best_unembed_baseline": base, "details": details,
                     "passed": all(d["pass"] for d in details.values())}
    strict_passed = all(s["passed"] for s in strict.values())
    metrics = {
        "manifest": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_clock_s": None,
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "numpy": np.__version__,
                         "transformers": __import__("transformers").__version__,
                         "scipy": __import__("scipy").__version__},
            "seed": cfg["seed"], "baseline_seeds": cfg["baseline_seeds"],
            "lens_repo": cfg["lens_repo"], "lens_revision": cfg["lens_revision"],
            "n_concepts": n,
            "primary_variant": "norm_scaled",
            "note_variants": "norm_scaled applies the final-norm diagonal weight to "
                             "u_v ((1+w) for gemma-2); raw uses bare unembedding rows. "
                             "Both run the full pipeline; raw is the robustness check.",
            "note_random_orth": "row-normalized cosine geometry is invariant to "
                                "orthogonal maps, so this baseline equals rsa_unembed "
                                "by construction; kept as a numerical sanity check",
        },
        "concepts": concepts,
        "fractions": fracs,
        "layer_map": {name: pm["layer_map"]
                      for name, pm in variants["norm_scaled"]["per_model"].items()},
        "variants": {v: {"pairs": variants[v]["pairs"],
                         "criterion_passed_all_pairs": passed[v]} for v in variants},
        "criterion_passed_all_pairs": passed["norm_scaled"],
        "strict_criterion": {"note": "norm_scaled J-space RSA vs the best unembedding "
                                     "baseline across both variants",
                             "pairs": strict, "passed_all_pairs": strict_passed},
    }
    metrics["manifest"]["wall_clock_s"] = round(time.time() - t0, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(metrics, open(ROOT / "results/metrics.json", "w"), indent=1)

    plot(variants["norm_scaled"]["pairs"], fracs, ROOT / "results/rsa_by_layer.png")
    print(f"criterion passed: norm_scaled={passed['norm_scaled']} raw={passed['raw']} "
          f"strict={strict_passed} wall={metrics['manifest']['wall_clock_s']}s")


def plot(pairs, fracs, path):
    colors = {p: c for p, c in zip(pairs, ["#2a78d6", "#1baf7a", "#eda100"])}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for p, res in pairs.items():
        c = colors[p]
        y = [res["rsa_j"][f] for f in fracs]
        ax.plot(fracs, y, color=c, lw=2, marker="o", ms=4, label=p.replace("|", " vs "))
        ax.annotate(f"{y[-1]:.2f}", (fracs[-1], y[-1]), xytext=(5, 0),
                    textcoords="offset points", color=c, fontsize=8, va="center")
        ax.axhline(res["rsa_unembed"], color=c, lw=1.5, ls="--", alpha=0.7)
        ax.plot(fracs, [res["rsa_shuffled"][f]["mean"] for f in fracs],
                color=c, lw=1.2, ls=":", alpha=0.7)
    ax.plot([], [], color="0.4", ls="--", label="unembedding baseline")
    ax.plot([], [], color="0.4", ls=":", label="shuffled floor")
    ax.set_xlabel("layer fraction")
    ax.set_ylabel("cross-model RSA (Spearman)")
    ax.set_title("J-space relational geometry across models")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
