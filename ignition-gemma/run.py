"""Cross-family replication of the ignition/competition result on
google/gemma-2-2b. Same design as ../ignition (READ its run.py): arm 1 =
8-step prompt evidence ladders, arm 2 = graded j-vector injection at the mid-band
midpoint with downstream readout, plus the competition probe and all controls.
Shared machinery (fits, summaries, hooks, ladders) is imported from the Qwen run;
only the model-family-specific pieces are adapted here:

- Gemma-2 final RMSNorm uses the (1+w) convention. Readout goes through the model's
  own `model.model.norm` module (which applies (1+w) internally), so the norm-scaled
  readout of idea-A is automatic. J-vectors keep the flagship harness convention
  j = normalize(J^T u) with raw unembedding rows — measured head-to-head, the
  (1+w)-weighted variant is a much weaker steering direction on gemma (France at
  alpha=1: downstream presence 0.05 vs 0.62 raw), and raw-u is what the Qwen run
  used, so raw-u maximizes construct equivalence.
- Gemma-2 applies a final logit softcap (30.0 * tanh(logits/30)). It is monotone, so
  readout ranks are unaffected; we apply it before the softmax anyway so presence
  probabilities match the model's own output convention.
- Gemma's tokenizer prepends <bos>, and the BOS residual is a high-norm attention
  sink. Position 0 is excluded from the presence scan and from the mean-residual-norm
  used to scale injections; competition injections start at position 1 (the Qwen
  run's "all positions" = all content positions; there was no BOS there).
- 26 layers; lens covers 0-24. Mid band = fractions 0.40-0.75 -> layers 10-18;
  injection at the band midpoint, layer 14; downstream readout layers 15-18.
"""

import argparse
import importlib.util
import json
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

ROOT = Path(__file__).resolve().parent
QWEN_DIR = (ROOT / "../ignition").resolve()

# Load the flagship run as a module: its module-level code inserts the harness,
# the Jacobian-lens repo (JLENS_REPO or ../vendor/jacobian-lens) and its own dir
# into sys.path, so the imports below resolve.
_spec = importlib.util.spec_from_file_location("qwen_ignition_run", QWEN_DIR / "run.py")
qi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qi)

import jlens
from jspace_interventions import InterventionHooks, JVecs, find_blocks, layer_band, noop_check
from ladders import PROBES, concepts, ladder

add_op, fit_curves, summarize, sig_f = qi.add_op, qi.fit_curves, qi.summarize, qi.sig_f
record_acts = qi.record_acts


@torch.no_grad()
def presence(model, acts, J_dev, layers, tids, softcap, pos_start):
    """Same presence definition as the Qwen run (max over layers x positions of the
    J-lens softmax prob of the leading-space concept token, norm-scaled readout via
    the model's own final norm), with two gemma adaptations: positions < pos_start
    (the BOS sink) are excluded, and the final logit softcap is applied (monotone,
    ranks unchanged)."""
    norm, head = model.model.norm, model.lm_head
    best = {t: (0.0, -1, -1) for t in tids}
    rows, p_last = {}, {t: 0.0 for t in tids}
    for l in layers:
        z = acts[l][0, pos_start:].float() @ J_dev[l].T
        logits = head(norm(z.to(model.dtype))).float()
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        probs = logits.softmax(-1)
        for t in tids:
            col = probs[:, t]
            pos = int(col.argmax())
            v = float(col[pos])
            if v > best[t][0]:
                best[t] = (v, l, pos + pos_start)
                rows[t] = probs[pos].cpu()
            p_last[t] = max(p_last[t], float(probs[-1, t]))
    out = {}
    for t in tids:
        v, l, pos = best[t]
        rank = int((rows[t] > rows[t][t]).sum()) + 1 if t in rows else -1
        out[t] = {"p": v, "layer": l, "pos": pos, "rank": rank, "p_last": p_last[t]}
    return out


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
             .from_pretrained(cfg["model"], dtype=torch.bfloat16,
                              attn_implementation="eager").to(dev).eval())
    softcap = float(model.config.final_logit_softcapping or 0.0)
    lens = jlens.JacobianLens.from_pretrained(cfg["lens_repo"], filename=cfg["lens_file"])
    blocks = find_blocks(model)
    band = layer_band(lens, *cfg["band"])
    L = band[len(band) // 2] if cfg["inject_layer"] == "auto" else cfg["inject_layer"]
    down = [l for l in band if l > L]
    J_dev = {l: lens.jacobians[l].to(dev) for l in band}
    jv = JVecs(lens, model, tok)
    print(f"loaded {time.time()-t0:.0f}s; band {band[0]}-{band[-1]}, inject L{L}, "
          f"downstream {down}, softcap {softcap}", flush=True)

    specs = concepts()
    names = sorted(specs)
    # Tokenizer filter: keep concepts that are single tokens with leading space.
    tid, dropped = {}, []
    for n in list(names) + PROBES:
        ids = tok.encode(" " + n, add_special_tokens=False)
        if len(ids) == 1:
            tid[n] = ids[0]
        else:
            dropped.append(n)
    names = [n for n in names if n in tid]
    probes = [p for p in PROBES if p in tid]
    assert len(names) >= 30, f"only {len(names)} concepts survive gemma tokenizer"
    assert not dropped, f"unexpected drops (all 40+5 verified single-token): {dropped}"
    if args.limit:
        names = names[: args.limit]
    for n in names:
        steps = ladder(specs[n])
        for s in steps[:7]:
            assert tid[n] not in tok.encode(s, add_special_tokens=False), (n, s)

    carrier_ids = tok(cfg["carrier"].rstrip(), return_tensors="pt").input_ids.to(dev)
    pos_start = 1 if carrier_ids[0, 0].item() == tok.bos_token_id else 0
    assert noop_check(model, carrier_ids, band), "noop_check FAILED"
    print(f"noop_check PASS (pos_start={pos_start})", flush=True)

    def enc(prompt):
        return tok(prompt.rstrip(), return_tensors="pt").input_ids.to(dev)

    def read(acts, layers, tids):
        return presence(model, acts, J_dev, layers, tids, softcap, pos_start)

    # Arm 1
    arm1 = {}
    for i, n in enumerate(names):
        t1 = time.time()
        curve = []
        for s, prompt in enumerate(ladder(specs[n]), start=1):
            ids = enc(prompt)
            r = read(record_acts(model, blocks, ids, band), band, [tid[n]])[tid[n]]
            curve.append({"step": s, **r})
        arm1[n] = curve
        print(f"[arm1 {i+1}/{len(names)}] {n} p={['%.3g' % c['p'] for c in curve]} "
              f"({time.time()-t1:.1f}s)", flush=True)

    # Arm 2
    alphas = cfg["alphas"]
    start = carrier_ids.shape[1] - 1
    base_acts = record_acts(model, blocks, carrier_ids, down + [L])
    # BOS is a high-norm sink: scale injections by the mean CONTENT-token norm.
    mean_norm = float(base_acts[L][0, pos_start:].float().norm(dim=-1).mean())
    mean_norm_with_bos = float(base_acts[L][0].float().norm(dim=-1).mean())
    probe_tids = [tid[p] for p in probes]
    base_read = read(base_acts, down, [tid[n] for n in names] + probe_tids)

    # NB: always pass token_id explicitly. JVecs.token_id() encodes WITH special
    # tokens; gemma prepends <bos>, whose decode is non-empty, so it would silently
    # build every j-vector from the BOS unembedding row.
    v0 = jv.vec(L, token_id=tid[names[0]])
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
    print(f"alpha=0 bit-exact PASS; mean resid norm L{L} = {mean_norm:.1f} "
          f"(with BOS: {mean_norm_with_bos:.1f})", flush=True)

    arm2, arm2_rand, probe_curves = {}, {}, {}
    for i, n in enumerate(names):
        t1 = time.time()
        vec = jv.vec(L, token_id=tid[n])
        g = torch.Generator().manual_seed(1000 + i)
        rvec = torch.randn(lens.d_model, generator=g)
        rvec = rvec / rvec.norm()
        real, rand, pcurve = [], [], []
        for a in alphas:
            if a == 0.0:
                b = base_read[tid[n]]
                real.append({"alpha": a, **b})
                rand.append({"alpha": a, **b})
                pcurve.append({"alpha": a,
                               **{p: base_read[tid[p]]["p"] for p in probes}})
                continue
            for v, dst in [(vec, real), (rvec, rand)]:
                with InterventionHooks(blocks) as iv:
                    iv.add(L, add_op(v, a * mean_norm, start))
                    acts = record_acts(model, blocks, carrier_ids, down)
                targets = [tid[n]] + (probe_tids if dst is real else [])
                r = read(acts, down, targets)
                dst.append({"alpha": a, **r[tid[n]]})
                if dst is real:
                    pcurve.append({"alpha": a, **{p: r[tid[p]]["p"] for p in probes}})
        arm2[n], arm2_rand[n], probe_curves[n] = real, rand, pcurve
        print(f"[arm2 {i+1}/{len(names)}] {n} p={['%.3g' % c['p'] for c in real]} "
              f"rand_max={max(c['p'] for c in rand):.3g} ({time.time()-t1:.1f}s)",
              flush=True)

    # Competition probe. As in the Qwen run, X is injected at ALL content positions
    # (prompt-evoked Y lives in mid-band J-space only at its hint-token positions);
    # here "all" starts at pos_start so the BOS sink is never edited.
    comp = []
    for seed in cfg["pair_seeds"]:
        rng = random.Random(seed)
        for k in range(cfg["n_pairs_per_seed"]):
            x, y = rng.sample(names, 2)
            ids = enc(ladder(specs[y])[6])
            before = read(record_acts(model, blocks, ids, down), down,
                          [tid[x], tid[y]])
            with InterventionHooks(blocks) as iv:
                iv.add(L, add_op(jv.vec(L, token_id=tid[x]),
                                 cfg["competition_alpha"] * mean_norm, pos_start))
                acts = record_acts(model, blocks, ids, down)
            after = read(acts, down, [tid[x], tid[y]])
            g = torch.Generator().manual_seed(2000 + seed * 100 + k)
            rv = torch.randn(lens.d_model, generator=g)
            with InterventionHooks(blocks) as iv:
                iv.add(L, add_op(rv / rv.norm(),
                                 cfg["competition_alpha"] * mean_norm, pos_start))
                acts = record_acts(model, blocks, ids, down)
            rnd = read(acts, down, [tid[x], tid[y]])
            comp.append({"seed": seed, "x": x, "y": y,
                         "x_before": before[tid[x]]["p"],
                         "x_after": after[tid[x]]["p"],
                         "y_before": before[tid[y]]["p"],
                         "y_after": after[tid[y]]["p"],
                         "x_after_rand": rnd[tid[x]]["p"],
                         "y_after_rand": rnd[tid[y]]["p"]})
    print(f"competition probe done ({len(comp)} pairs)", flush=True)

    # Fits (identical machinery to the Qwen run)
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

    # Plain-AIC secondary count (reported alongside AICc, as in the Qwen README)
    def aic_pref(fits):
        return sum(1 for n in names if fits[n]["sig"] is not None
                   and fits[n]["lin"]["aic"] - fits[n]["sig"]["aic"] > crit["daicc_min"])
    aic_counts = {"arm1": aic_pref(fits1), "arm2": aic_pref(fits2)}

    thr = 0.05
    valid = [c for c in comp if c["y_before"] > thr]
    ev = [c for c in valid if c["x_after"] - c["x_before"] > thr
          and c["y_after"] - c["y_before"] < -thr]
    co = [c for c in valid if c["x_after"] - c["x_before"] > thr
          and c["y_after"] - c["y_before"] >= -thr]
    dx = [c["x_after"] - c["x_before"] for c in valid]
    dy = [c["y_after"] - c["y_before"] for c in valid]
    excess = [c["y_after"] - c["y_after_rand"] for c in valid]
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
        "n_y_below_random_control": sum(1 for e in excess if e < 0),
        "median_specific_excess": float(np.median(excess)) if excess else None,
    }

    rand_inflate = {n: max(c["p"] for c in arm2_rand[n]) - arm2_rand[n][0]["p"]
                    for n in names}
    probe_inflate = {p: float(np.median(
        [max(pc[p] for pc in probe_curves[n]) - probe_curves[n][0][p] for n in names]))
        for p in probes}
    controls = {
        "random_dir_max_presence_increase": {
            "median": float(np.median(list(rand_inflate.values()))),
            "max": float(np.max(list(rand_inflate.values()))),
            "per_concept": rand_inflate},
        "probe_flatness_median_increase": probe_inflate,
        "probe_max_increase": float(np.max(
            [max(pc[p] for pc in probe_curves[n]) - probe_curves[n][0][p]
             for n in names for p in probes])),
    }

    manifest = {
        "model": cfg["model"], "lens": cfg["lens_file"], "device": dev,
        "band_layers": [band[0], band[-1]], "inject_layer": L,
        "downstream_layers": down, "alphas": alphas,
        "carrier": cfg["carrier"], "mean_resid_norm_inject_layer": mean_norm,
        "mean_resid_norm_incl_bos": mean_norm_with_bos,
        "n_concepts": len(names), "probes": probes,
        "concepts_dropped_by_tokenizer": dropped,
        "presence_def": "max over layers x positions of J-lens softmax prob of the "
                        "leading-space concept token, norm-scaled readout via the "
                        "model's own final norm ((1+w) convention); final logit "
                        "softcap applied (monotone, ranks unaffected); BOS position "
                        "excluded from the scan",
        "jvec_def": "normalize(J^T u) with raw unembedding rows (flagship harness "
                    "convention); the (1+w)-weighted variant was tested and is a "
                    "much weaker steering direction (France alpha=1: 0.05 vs 0.62)",
        "final_logit_softcap": softcap,
        "aicc_k": {"linear": 3, "sigmoid": 5},
        "hysteresis": "SKIPPED (same rationale as Qwen run: decreasing context "
                      "accumulation is not implementable in a causal LM)",
        "random_dir_seeds": "deterministic per concept (1000+idx); pair sampling uses "
                            "cfg pair_seeds",
        "competition_note": "X injected at all content positions (pos_start onward, "
                            "BOS excluded); carrier = Y's step-7 ladder; presence = "
                            "max measure; validity filter y_before > 0.05.",
        "gemma_adaptations": [
            "attn_implementation=eager (gemma-2 attention softcap)",
            "BOS position excluded from presence scan, injection positions and "
            "mean-norm (high-norm attention sink)",
            "final logit softcap applied before softmax in the readout",
            "j-vectors from raw unembedding rows as in the flagship run; the "
            "(1+w)-scaled alternative measured and rejected (weak steering)",
            "band recomputed as fractions 0.40-0.75 of lens layers 0-24 -> 10-18; "
            "injection at band midpoint L14",
        ],
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "platform": platform.platform(),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    out = {
        "manifest": manifest,
        "criterion": {**crit, "arm1": sum1, "arm2": sum2,
                      "arm2_lastpos_secondary": sum2_last,
                      "plain_aic_n_sig_preferred": aic_counts,
                      "signal": bool(verdict)},
        "competition": {"summary": comp_sum, "pairs": comp},
        "controls": controls,
        "fits": {"arm1": fits1, "arm2": fits2, "arm2_lastpos": fits2_last},
        "curves": {"arm1": arm1, "arm2": arm2, "arm2_random_dir": arm2_rand,
                   "arm2_probes": probe_curves},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "metrics.json", "w"), indent=1,
              default=float)
    plot(names, arm1, arm2, fits1, fits2, ROOT / "results" / "ignition_curves.png")
    print(json.dumps({"arm1": sum1, "arm2": sum2,
                      "plain_aic": aic_counts, "competition": comp_sum,
                      "signal": bool(verdict),
                      "wall_clock_s": manifest["wall_clock_seconds"]}, indent=2),
          flush=True)


def plot(names, arm1, arm2, fits1, fits2, path):
    """Same figure as the Qwen run (local copy: only the suptitle differs)."""
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
    fig.suptitle("J-space ignition: presence vs stimulus strength (gemma-2-2b)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
