"""Idea B signal check: does J-space presence vs stimulus strength ignite (sigmoid)
or scale (linear)? Arm 1: 8-step prompt evidence ladders, 40 concepts. Arm 2: graded
j-vector injection at layer 17, presence read downstream (layers 18-22).

Presence = max over mid-band layers x all positions of J-lens softmax probability of
the concept's leading-space token (norm-scaled readout, as validated in idea-D).
"""

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
import yaml
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
sys.path.insert(0, str(ROOT / "../harness"))
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

import jlens
from jspace_interventions import InterventionHooks, JVecs, find_blocks, layer_band, noop_check
from ladders import PROBES, concepts, ladder


class Recorder:
    def __init__(self, blocks, layers):
        self.acts = {}
        self._handles = [blocks[l].register_forward_hook(self._mk(l)) for l in layers]

    def _mk(self, l):
        def fn(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            self.acts[l] = h.detach()
        return fn

    def remove(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def record_acts(model, blocks, ids, layers):
    rec = Recorder(blocks, layers)
    model(ids)
    rec.remove()
    return rec.acts


@torch.no_grad()
def presence(model, acts, J_dev, layers, tids):
    """Per target token: max softmax prob over layers x positions, its (layer, pos),
    rank at that cell, and the max over layers at the last position."""
    norm, head = model.model.norm, model.lm_head
    best = {t: (0.0, -1, -1) for t in tids}
    rows, p_last = {}, {t: 0.0 for t in tids}
    for l in layers:
        z = acts[l][0].float() @ J_dev[l].T
        probs = head(norm(z.to(model.dtype))).float().softmax(-1)
        for t in tids:
            col = probs[:, t]
            pos = int(col.argmax())
            v = float(col[pos])
            if v > best[t][0]:
                best[t] = (v, l, pos)
                rows[t] = probs[pos].cpu()
            p_last[t] = max(p_last[t], float(probs[-1, t]))
    out = {}
    for t in tids:
        v, l, pos = best[t]
        rank = int((rows[t] > rows[t][t]).sum()) + 1 if t in rows else -1
        out[t] = {"p": v, "layer": l, "pos": pos, "rank": rank, "p_last": p_last[t]}
    return out


def add_op(vec, scale, start):
    def op(h):
        out = h.clone()
        hs = out[:, start:].float()
        out[:, start:] = (hs + scale * vec.to(h.device)).to(h.dtype)
        return out
    return op


def sig_f(x, b, t, x0, w):
    return b + (t - b) / (1 + np.exp(np.clip(-(x - x0) / w, -500, 500)))


def aicc(rss, n, k):
    return n * math.log(max(rss, 1e-300) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_curves(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, r = len(x), x.max() - x.min()
    coef = np.polyfit(x, y, 1)
    rss_l = float(((np.polyval(coef, x) - y) ** 2).sum())
    lin = {"slope": coef[0], "intercept": coef[1], "rss": rss_l,
           "aicc": aicc(rss_l, n, 3), "aic": n * math.log(max(rss_l, 1e-300) / n) + 6}
    lo, hi = [0, 0, x.min() - r, 1e-4], [1, 1.5, x.max() + r, 3 * r]
    best = None
    steep = x[max(1, int(np.argmax(np.diff(y))))] if n > 2 else x.mean()
    for x0 in list(np.quantile(x, [0.25, 0.5, 0.75])) + [steep]:
        for w in [r / 20, r / 8, r / 3, r]:
            try:
                p, _ = curve_fit(sig_f, x, y, p0=[y.min(), y.max(), x0, w],
                                 bounds=(lo, hi), maxfev=20000)
                rss = float(((sig_f(x, *p) - y) ** 2).sum())
                if best is None or rss < best[1]:
                    best = (p, rss)
            except (RuntimeError, ValueError):
                continue
    if best is None:
        return {"lin": lin, "sig": None, "daicc": None, "sig_preferred": False,
                "width_frac": None}
    p, rss_s = best
    sig = {"b": p[0], "t": p[1], "x0": p[2], "w": p[3], "rss": rss_s,
           "aicc": aicc(rss_s, n, 5), "aic": n * math.log(max(rss_s, 1e-300) / n) + 10,
           "width_10_90": math.log(81) * p[3]}
    daicc = lin["aicc"] - sig["aicc"]
    return {"lin": lin, "sig": sig, "daicc": daicc,
            "sig_preferred": bool(daicc > 2.0),
            "width_frac": sig["width_10_90"] / r}


def summarize(fits, sig_frac_min, width_frac_max):
    names = sorted(fits)
    pref = [n for n in names if fits[n]["sig_preferred"]]
    widths = [fits[n]["width_frac"] for n in pref if fits[n]["width_frac"] is not None]
    med_w = float(np.median(widths)) if widths else None
    return {"n": len(names), "n_sig_preferred": len(pref),
            "sig_frac": len(pref) / len(names),
            "median_width_frac_sig_preferred": med_w,
            "median_daicc": float(np.median([fits[n]["daicc"] for n in names
                                             if fits[n]["daicc"] is not None])),
            "passes_sig_frac": len(pref) / len(names) > sig_frac_min,
            "passes_width": med_w is not None and med_w < width_frac_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="concepts (pilot)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    dev = cfg["device"]
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    model = (transformers.AutoModelForCausalLM
             .from_pretrained(cfg["model"], dtype=torch.bfloat16).to(dev).eval())
    lens = jlens.JacobianLens.from_pretrained(cfg["lens_repo"], filename=cfg["lens_file"])
    blocks = find_blocks(model)
    band = layer_band(lens, *cfg["band"])
    L = cfg["inject_layer"]
    down = [l for l in band if l > L]
    J_dev = {l: lens.jacobians[l].to(dev) for l in band}
    jv = JVecs(lens, model, tok)
    print(f"loaded {time.time()-t0:.0f}s; band {band[0]}-{band[-1]}, inject L{L}, "
          f"downstream {down}", flush=True)

    specs = concepts()
    names = sorted(specs)
    if args.limit:
        names = names[: args.limit]
    tid = {}
    for n in names + PROBES:
        ids = tok.encode(" " + n, add_special_tokens=False)
        assert len(ids) == 1, (n, ids)
        tid[n] = ids[0]
    for n in names:
        steps = ladder(specs[n])
        for s in steps[:7]:
            assert tid[n] not in tok.encode(s, add_special_tokens=False), (n, s)

    carrier_ids = tok(cfg["carrier"].rstrip(), return_tensors="pt").input_ids.to(dev)
    assert noop_check(model, carrier_ids, band), "noop_check FAILED"
    print("noop_check PASS", flush=True)

    def enc(prompt):
        return tok(prompt.rstrip(), return_tensors="pt").input_ids.to(dev)

    # Arm 1
    arm1 = {}
    for i, n in enumerate(names):
        t1 = time.time()
        curve = []
        for s, prompt in enumerate(ladder(specs[n]), start=1):
            ids = enc(prompt)
            r = presence(model, record_acts(model, blocks, ids, band), J_dev, band,
                         [tid[n]])[tid[n]]
            curve.append({"step": s, **r})
        arm1[n] = curve
        print(f"[arm1 {i+1}/{len(names)}] {n} p={['%.3g' % c['p'] for c in curve]} "
              f"({time.time()-t1:.1f}s)", flush=True)

    # Arm 2
    alphas = cfg["alphas"]
    start = carrier_ids.shape[1] - 1
    base_acts = record_acts(model, blocks, carrier_ids, down + [L])
    mean_norm = float(base_acts[L][0].float().norm(dim=-1).mean())
    probe_tids = [tid[p] for p in PROBES]
    base_read = presence(model, base_acts, J_dev, down,
                         [tid[n] for n in names] + probe_tids)

    v0 = jv.vec(L, names[0])
    with InterventionHooks(blocks) as iv:
        iv.add(L, add_op(v0, 0.0, start))
        a0 = record_acts(model, blocks, carrier_ids, down)
    assert all(torch.equal(a0[l], base_acts[l]) for l in down), "alpha=0 not baseline"
    g = torch.Generator().manual_seed(1000)
    r0 = torch.randn(lens.d_model, generator=g)
    with InterventionHooks(blocks) as iv:
        iv.add(L, add_op(r0 / r0.norm(), 0.0, start))
        a0 = record_acts(model, blocks, carrier_ids, down)
    assert all(torch.equal(a0[l], base_acts[l]) for l in down), "alpha=0 rand not baseline"
    print(f"alpha=0 bit-exact PASS; mean resid norm L{L} = {mean_norm:.1f}", flush=True)

    arm2, arm2_rand, probe_curves = {}, {}, {}
    for i, n in enumerate(names):
        t1 = time.time()
        vec = jv.vec(L, n)
        g = torch.Generator().manual_seed(1000 + i)
        rvec = torch.randn(lens.d_model, generator=g)
        rvec = rvec / rvec.norm()
        real, rand, probes = [], [], []
        for a in alphas:
            if a == 0.0:
                b = base_read[tid[n]]
                real.append({"alpha": a, **b})
                rand.append({"alpha": a, **b})
                probes.append({"alpha": a,
                               **{p: base_read[tid[p]]["p"] for p in PROBES}})
                continue
            for v, dst in [(vec, real), (rvec, rand)]:
                with InterventionHooks(blocks) as iv:
                    iv.add(L, add_op(v, a * mean_norm, start))
                    acts = record_acts(model, blocks, carrier_ids, down)
                targets = [tid[n]] + (probe_tids if dst is real else [])
                r = presence(model, acts, J_dev, down, targets)
                dst.append({"alpha": a, **r[tid[n]]})
                if dst is real:
                    probes.append({"alpha": a, **{p: r[tid[p]]["p"] for p in PROBES}})
        arm2[n], arm2_rand[n], probe_curves[n] = real, rand, probes
        print(f"[arm2 {i+1}/{len(names)}] {n} p={['%.3g' % c['p'] for c in real]} "
              f"rand_max={max(c['p'] for c in rand):.3g} ({time.time()-t1:.1f}s)",
              flush=True)

    # Competition probe. X is injected at ALL positions (deviation from arm 2's
    # last-position convention): a prompt-evoked Y registers in the mid-band J-space
    # only at its hint-token positions, never at the final position (measured: France
    # carrier, model next-token p(France)=0.42 yet mid-band p_last<1e-4), so a
    # last-position injection could never contact Y.
    comp = []
    for seed in cfg["pair_seeds"]:
        rng = random.Random(seed)
        for k in range(cfg["n_pairs_per_seed"]):
            x, y = rng.sample(names, 2)
            ids = enc(ladder(specs[y])[6])
            before = presence(model, record_acts(model, blocks, ids, down), J_dev,
                              down, [tid[x], tid[y]])
            with InterventionHooks(blocks) as iv:
                iv.add(L, add_op(jv.vec(L, x), cfg["competition_alpha"] * mean_norm, 0))
                acts = record_acts(model, blocks, ids, down)
            after = presence(model, acts, J_dev, down, [tid[x], tid[y]])
            g = torch.Generator().manual_seed(2000 + seed * 100 + k)
            rv = torch.randn(lens.d_model, generator=g)
            with InterventionHooks(blocks) as iv:
                iv.add(L, add_op(rv / rv.norm(),
                                 cfg["competition_alpha"] * mean_norm, 0))
                acts = record_acts(model, blocks, ids, down)
            rnd = presence(model, acts, J_dev, down, [tid[x], tid[y]])
            comp.append({"seed": seed, "x": x, "y": y,
                         "x_before": before[tid[x]]["p"],
                         "x_after": after[tid[x]]["p"],
                         "y_before": before[tid[y]]["p"],
                         "y_after": after[tid[y]]["p"],
                         "x_after_rand": rnd[tid[x]]["p"],
                         "y_after_rand": rnd[tid[y]]["p"]})
    print(f"competition probe done ({len(comp)} pairs)", flush=True)

    # Fits
    fits1 = {n: fit_curves([c["step"] for c in arm1[n]], [c["p"] for c in arm1[n]])
             for n in names}
    fits2 = {n: fit_curves([c["alpha"] for c in arm2[n]], [c["p"] for c in arm2[n]])
             for n in names}
    fits2_last = {n: fit_curves([c["alpha"] for c in arm2[n]],
                                [c["p_last"] for c in arm2[n]]) for n in names}
    crit = cfg["criterion"]
    sum1 = summarize(fits1, crit["sig_frac_min"], crit["width_frac_max"])
    sum2 = summarize(fits2, crit["sig_frac_min"], crit["width_frac_max"])
    sum2_last = summarize(fits2_last, crit["sig_frac_min"], crit["width_frac_max"])
    verdict = (sum1["passes_sig_frac"] and sum2["passes_sig_frac"]
               and sum1["passes_width"] and sum2["passes_width"])

    thr = 0.05
    valid = [c for c in comp if c["y_before"] > thr]
    ev = [c for c in valid if c["x_after"] - c["x_before"] > thr
          and c["y_after"] - c["y_before"] < -thr]
    co = [c for c in valid if c["x_after"] - c["x_before"] > thr
          and c["y_after"] - c["y_before"] >= -thr]
    dx = [c["x_after"] - c["x_before"] for c in valid]
    dy = [c["y_after"] - c["y_before"] for c in valid]
    comp_sum = {
        "n_pairs": len(comp), "n_valid_y_evoked": len(valid), "threshold": thr,
        "n_x_ignited": sum(1 for c in valid if c["x_after"] - c["x_before"] > thr),
        "n_one_in_one_out": len(ev), "n_coexist": len(co),
        "median_dY_given_X_ignited": float(np.median(
            [c["y_after"] - c["y_before"] for c in ev + co])) if ev + co else None,
        "corr_dX_dY": float(np.corrcoef(dx, dy)[0, 1]) if len(valid) > 2 else None,
        "median_dY_random_inject": float(np.median(
            [c["y_after_rand"] - c["y_before"] for c in valid])) if valid else None,
        "median_dX_random_inject": float(np.median(
            [c["x_after_rand"] - c["x_before"] for c in valid])) if valid else None,
    }

    rand_inflate = {n: max(c["p"] for c in arm2_rand[n]) - arm2_rand[n][0]["p"]
                    for n in names}
    probe_inflate = {p: float(np.median(
        [max(pc[p] for pc in probe_curves[n]) - probe_curves[n][0][p] for n in names]))
        for p in PROBES}
    controls = {
        "random_dir_max_presence_increase": {
            "median": float(np.median(list(rand_inflate.values()))),
            "max": float(np.max(list(rand_inflate.values()))),
            "per_concept": rand_inflate},
        "probe_flatness_median_increase": probe_inflate,
        "probe_max_increase": float(np.max(
            [max(pc[p] for pc in probe_curves[n]) - probe_curves[n][0][p]
             for n in names for p in PROBES])),
    }

    manifest = {
        "model": cfg["model"], "lens": cfg["lens_file"], "device": dev,
        "band_layers": [band[0], band[-1]], "inject_layer": L,
        "downstream_layers": down, "alphas": alphas,
        "carrier": cfg["carrier"], "mean_resid_norm_L17": mean_norm,
        "n_concepts": len(names), "probes": PROBES,
        "presence_def": "max over layers x positions of J-lens softmax prob of the "
                        "leading-space concept token, norm-scaled readout",
        "aicc_k": {"linear": 3, "sigmoid": 5},
        "hysteresis": "SKIPPED: decreasing context accumulation within one prompt is "
                      "not implementable in a causal LM (context only accumulates; "
                      "prompt-side retraction leaves hint tokens attended), so an "
                      "honest matched-evidence descending branch does not exist.",
        "random_dir_seeds": "deterministic per concept (1000+idx); pair sampling uses "
                            "cfg pair_seeds",
        "competition_note": "X injected at all positions (not last-position-onward): "
                            "prompt-evoked Y lives in mid-band J-space only at its "
                            "hint-token positions, so last-position injection cannot "
                            "contact it. Carrier = Y's step-7 ladder; presence = max "
                            "measure; validity filter y_before > 0.05.",
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "platform": platform.platform(),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    out = {
        "manifest": manifest,
        "criterion": {**crit, "arm1": sum1, "arm2": sum2,
                      "arm2_lastpos_secondary": sum2_last, "signal": bool(verdict)},
        "competition": {"summary": comp_sum, "pairs": comp},
        "controls": controls,
        "fits": {"arm1": fits1, "arm2": fits2, "arm2_lastpos": fits2_last},
        "curves": {"arm1": arm1, "arm2": arm2, "arm2_random_dir": arm2_rand,
                   "arm2_probes": probe_curves},
    }
    json.dump(out, open(ROOT / "results" / "metrics.json", "w"), indent=1,
              default=float)
    plot(names, arm1, arm2, fits1, fits2, ROOT / "results" / "ignition_curves.png")
    print(json.dumps({"arm1": sum1, "arm2": sum2, "competition": comp_sum,
                      "signal": bool(verdict),
                      "wall_clock_s": manifest["wall_clock_seconds"]}, indent=2),
          flush=True)


def plot(names, arm1, arm2, fits1, fits2, path):
    def pick(fits):
        ranked = sorted(names, key=lambda n: -(fits[n]["daicc"] or -1e9))
        return [ranked[0], ranked[len(ranked) // 4], ranked[len(ranked) // 2],
                ranked[-1]]

    fig, axes = plt.subplots(3, 4, figsize=(13, 9), facecolor="#fcfcfb")
    for row, (arm, fits, xkey, xlab) in enumerate(
            [(arm1, fits1, "step", "ladder step"), (arm2, fits2, "alpha", "alpha")]):
        for ax, n in zip(axes[row], pick(fits)):
            x = np.array([c[xkey] for c in arm[n]], float)
            y = [c["p"] for c in arm[n]]
            ax.plot(x, y, "o", color="#2a78d6", markersize=5)
            xs = np.linspace(x.min(), x.max(), 200)
            f = fits[n]
            ax.plot(xs, np.polyval([f["lin"]["slope"], f["lin"]["intercept"]], xs),
                    "--", color="#c98500", linewidth=1.5, label="linear")
            if f["sig"]:
                s = f["sig"]
                ax.plot(xs, sig_f(xs, s["b"], s["t"], s["x0"], s["w"]), "-",
                        color="#2a78d6", linewidth=1.8, label="sigmoid")
            d = f["daicc"]
            ax.set_title(f"{n}  dAICc={d:.1f}" if d is not None else n, fontsize=9,
                         color="#0b0b0b")
            ax.set_xlabel(xlab, fontsize=8, color="#52514e")
            ax.grid(True, color="#e8e8e5", linewidth=0.7)
            ax.set_facecolor("#fcfcfb")
            for sp in ax.spines.values():
                sp.set_color("#c3c2b7")
            ax.tick_params(colors="#52514e", labelsize=7)
        axes[row][0].set_ylabel(f"arm {row+1} presence", fontsize=9, color="#52514e")
        axes[row][0].legend(fontsize=7, frameon=False)
    for col, (fits, lab) in enumerate([(fits1, "arm 1 (ladders)"),
                                       (fits2, "arm 2 (injection)")]):
        ax = axes[2][col]
        d = [fits[n]["daicc"] for n in names if fits[n]["daicc"] is not None]
        ax.hist(d, bins=20, color="#2a78d6", edgecolor="#fcfcfb")
        ax.axvline(2, color="#c98500", linestyle="--", linewidth=1.5)
        ax.set_title(f"dAICc (lin - sig), {lab}", fontsize=9, color="#0b0b0b")
        ax.set_xlabel("dAICc (>2 = sigmoid preferred)", fontsize=8, color="#52514e")
    for col, (fits, lab) in enumerate([(fits1, "arm 1"), (fits2, "arm 2")]):
        ax = axes[2][col + 2]
        pts = [(fits[n]["daicc"], fits[n]["width_frac"]) for n in names
               if fits[n]["width_frac"] is not None]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14,
                       color="#4a3aa7")
            ax.axhline(1 / 3, color="#c98500", linestyle="--", linewidth=1.5)
            ax.axvline(2, color="#c98500", linestyle=":", linewidth=1)
        ax.set_title(f"width_frac vs dAICc, {lab}", fontsize=9, color="#0b0b0b")
        ax.set_xlabel("dAICc", fontsize=8, color="#52514e")
        ax.set_ylabel("10-90% width / range", fontsize=8, color="#52514e")
    for ax in axes[2]:
        ax.grid(True, color="#e8e8e5", linewidth=0.7)
        ax.set_facecolor("#fcfcfb")
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=7)
    fig.suptitle("J-space ignition: presence vs stimulus strength (Qwen3.5-4B)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
