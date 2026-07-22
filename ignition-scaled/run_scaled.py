"""Scaled ignition + competition — config-parametrized across three model
families (Qwen3-8B / Gemma-2-9B / Llama-3.1-8B), run as separate processes.
Measured runs used a single MI300X; device auto-detects.

Design deltas vs the validated 4B signal check (`../ignition/`, whose machinery
this imports together with `../harness/`):

- 80 concepts for arm 2 / competition (original 40 + 40 same-family extras from
  concepts_extended.py; per-tokenizer single-token filter, survivors recorded).
- Arm 2: finer 15-point alpha grid; 3 carrier prompts (seed = carrier); fits report
  AIC and AICc AND the pre-registered scaled PRIMARY criterion = median 10-90%
  transition width / alpha range < 1/3 over all arm-2 sigmoid fits.
- Arm 1: the original 40-concept prompt ladders unchanged (comparability anchor).
- Competition matrix: 20 high-evocability residents x 20 injected concepts at
  alpha=1.0, all content positions (positional-asymmetry finding), per-resident
  norm-matched random-direction controls (2 seeds): eviction rate, specific-excess
  sign test, dY vs cos(j_X, j_Y) structure.
- Controls kept: noop_check, alpha=0 bit-exactness (real + random), random-direction
  non-ignition curves, unrelated-probe flatness.
- Family adaptations are config-driven, following the validated gemma-2-2b run:
  expect_bos_prepend (BOS excluded from presence scan / mean-norm / injection
  positions), expect_softcap (final logit softcap applied in readout, monotone),
  attn_implementation (eager for gemma-2). J-vectors stay raw-u (harness
  convention); token ids always passed explicitly to JVecs (BOS token_id gotcha).
- Checkpointed (resumable) at per-unit granularity; run under nohup.

Presence = max over mid-band layers x content positions of the J-lens softmax
probability of the concept's leading-space token (norm-scaled readout via the
model's final norm), identical to the validated runs.
"""

import argparse
import json
import math
import os
import platform
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
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default="config.yaml")
_known, _ = _pre.parse_known_args()
cfg = yaml.safe_load(open(ROOT / _known.config))
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
for p in (ROOT / "../harness", ROOT / "../ignition", VENDOR, ROOT):
    sys.path.insert(0, str(p))

import jlens  # noqa: E402
from jspace_interventions import (  # noqa: E402
    InterventionHooks, JVecs, find_blocks, layer_band, noop_check)
from ladders import PROBES, concepts, ladder  # noqa: E402
from concepts_extended import extra_concepts, resident_carrier  # noqa: E402


# ---------------------------------------------------------------- shared machinery

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
def presence(model, acts, J_dev, layers, tids, softcap, pos_start):
    """Per target token: max softmax prob over layers x content positions, its
    (layer, pos), rank at that cell, and the max over layers at the last position.
    Positions < pos_start (BOS sink, if any) are excluded; the final logit softcap
    (if the family has one) is applied — monotone, ranks unchanged."""
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
    """Linear (k=3) vs 4-param logistic (k=5), reporting BOTH AICc and plain AIC,
    plus the 10-90% width fraction (the scaled primary criterion)."""
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
        return {"lin": lin, "sig": None, "daicc": None, "daic": None,
                "sig_preferred": False, "sig_preferred_aic": False, "width_frac": None}
    p, rss_s = best
    sig = {"b": p[0], "t": p[1], "x0": p[2], "w": p[3], "rss": rss_s,
           "aicc": aicc(rss_s, n, 5), "aic": n * math.log(max(rss_s, 1e-300) / n) + 10,
           "width_10_90": math.log(81) * p[3]}
    daicc = lin["aicc"] - sig["aicc"]
    daic = lin["aic"] - sig["aic"]
    return {"lin": lin, "sig": sig, "daicc": daicc, "daic": daic,
            "sig_preferred": bool(daicc > 2.0), "sig_preferred_aic": bool(daic > 2.0),
            "width_frac": sig["width_10_90"] / r}


def summarize(fits, sig_frac_min, width_frac_max):
    """AICc + plain-AIC preference fractions, widths over all sigmoid fits (scaled
    primary) and over the AICc-preferred subset (4B-comparable)."""
    keys = sorted(fits)
    if not keys:
        return None
    pref = [k for k in keys if fits[k]["sig_preferred"]]
    pref_aic = [k for k in keys if fits[k]["sig_preferred_aic"]]
    w_all = [fits[k]["width_frac"] for k in keys if fits[k]["width_frac"] is not None]
    w_pref = [fits[k]["width_frac"] for k in pref if fits[k]["width_frac"] is not None]
    med_all = float(np.median(w_all)) if w_all else None
    med_pref = float(np.median(w_pref)) if w_pref else None
    return {"n": len(keys),
            "n_sig_preferred_aicc": len(pref), "sig_frac_aicc": len(pref) / len(keys),
            "n_sig_preferred_aic": len(pref_aic), "sig_frac_aic": len(pref_aic) / len(keys),
            "median_daicc": float(np.median([fits[k]["daicc"] for k in keys
                                             if fits[k]["daicc"] is not None])),
            "median_daic": float(np.median([fits[k]["daic"] for k in keys
                                            if fits[k]["daic"] is not None])),
            "median_width_frac_all_fits": med_all,
            "median_width_frac_aicc_preferred": med_pref,
            "passes_sig_frac_aicc": len(pref) / len(keys) > sig_frac_min,
            "passes_sig_frac_aic": len(pref_aic) / len(keys) > sig_frac_min,
            "passes_width_primary": med_all is not None and med_all < width_frac_max}


def sign_test_p(n_neg, n_valid):
    """Exact two-sided sign test (no scipy.stats dependency)."""
    if n_valid == 0:
        return None
    cdf = sum(math.comb(n_valid, i) for i in range(0, min(n_neg, n_valid - n_neg) + 1))
    return min(1.0, 2.0 * cdf * 0.5 ** n_valid)


# ---------------------------------------------------------------- checkpointing

def load_ckpt(path):
    if path.exists():
        return json.load(open(path))
    return {"arm1": {}, "arm2": {}, "evoc": {}, "comp": {}}


def save_ckpt(path, ck):
    tmp = path.with_suffix(".tmp")
    json.dump(ck, open(tmp, "w"), default=float)
    os.replace(tmp, path)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pilot", type=int, default=0,
                    help="pilot with N concepts (writes to <tag>/pilot, 1 random "
                         "seed, NxN competition); 0 = full run")
    args = ap.parse_args()
    pilot = args.pilot
    tag = cfg["tag"]

    outdir = ROOT / "results" / tag / ("pilot" if pilot else "")
    outdir.mkdir(parents=True, exist_ok=True)
    ck_path = outdir / "checkpoint.json"
    ck = load_ckpt(ck_path)

    t0 = time.time()
    dev = cfg["device"]
    if dev in (None, "auto"):
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    kw = {}
    if cfg.get("attn_implementation"):
        kw["attn_implementation"] = cfg["attn_implementation"]
    model = (transformers.AutoModelForCausalLM
             .from_pretrained(cfg["model"], dtype=torch.bfloat16, **kw)
             .to(dev).eval())
    softcap = float(getattr(model.config, "final_logit_softcapping", None) or 0.0)
    assert bool(softcap) == bool(cfg.get("expect_softcap", False)), \
        f"softcap surprise: model has {softcap}, config expects " \
        f"{cfg.get('expect_softcap', False)}"
    lens = jlens.JacobianLens.from_pretrained(cfg["lens_repo"], filename=cfg["lens_file"])
    blocks = find_blocks(model)
    band = layer_band(lens, *cfg["band"])
    L = band[len(band) // 2] if cfg["inject_layer"] == "auto" else cfg["inject_layer"]
    down = [l for l in band if l > L]
    J_dev = {l: lens.jacobians[l].to(dev) for l in band}
    jv = JVecs(lens, model, tok)
    print(f"[{tag}] loaded {time.time()-t0:.0f}s; lens layers "
          f"{lens.source_layers[0]}-{lens.source_layers[-1]}, band "
          f"{band[0]}-{band[-1]}, inject L{L}, downstream {down}, "
          f"d_model {lens.d_model}, softcap {softcap}", flush=True)

    # BOS / special-token behavior must match the config expectation exactly.
    n_prepend = len(tok.encode("hi")) - len(tok.encode("hi", add_special_tokens=False))
    assert n_prepend == cfg.get("expect_bos_prepend", 0), \
        f"tokenizer prepends {n_prepend} special tokens, config expects " \
        f"{cfg.get('expect_bos_prepend', 0)}"
    pos_start = n_prepend
    if pos_start:
        probe_ids = tok("The weather is fine.", return_tensors="pt").input_ids
        assert probe_ids[0, 0].item() == tok.bos_token_id

    # Concept assembly: original 40 (with ladders) + extras; single-token filter.
    orig_specs = concepts()
    orig_names = sorted(orig_specs)
    extras = extra_concepts()
    extra_names = sorted(extras)
    kind = {n: orig_specs[n]["kind"] for n in orig_names}
    kind.update(extras)

    tid, dropped = {}, []
    for n in orig_names + extra_names + PROBES:
        ids = tok.encode(" " + n, add_special_tokens=False)
        if len(ids) == 1:
            tid[n] = ids[0]
        else:
            dropped.append(n)
    orig_surv = [n for n in orig_names if n in tid]
    extra_surv = [n for n in extra_names if n in tid]
    probes = [p for p in PROBES if p in tid]
    names = orig_surv + extra_surv          # arm-2 / competition concept set
    assert len(names) >= 60 and len(orig_surv) >= 30 and len(probes) >= 3, \
        f"too few survivors: {len(orig_surv)} orig, {len(extra_surv)} extra, " \
        f"{len(probes)} probes (dropped: {dropped})"
    print(f"[{tag}] concepts: {len(orig_surv)}/40 original + "
          f"{len(extra_surv)}/{len(extra_names)} extras survive single-token "
          f"filter; probes {len(probes)}/5 (dropped: {dropped})", flush=True)

    for n in orig_surv:                     # arm-1 ladder integrity
        for s in ladder(orig_specs[n])[:7]:
            assert tid[n] not in tok.encode(s, add_special_tokens=False), (n, s)

    def enc(prompt):
        return tok(prompt.rstrip(), return_tensors="pt").input_ids.to(dev)

    def read(acts, layers, tids):
        return presence(model, acts, J_dev, layers, tids, softcap, pos_start)

    carriers = cfg["carriers"]
    carrier_ids = [enc(c) for c in carriers]
    for c_i, cids in enumerate(carrier_ids):
        toks = set(cids[0].tolist())
        clash = [n for n in names + probes if tid[n] in toks]
        assert not clash, f"carrier {c_i} contains concept tokens {clash}"

    assert noop_check(model, carrier_ids[0], band), "noop_check FAILED"
    print(f"[{tag}] noop_check PASS; pos_start={pos_start}", flush=True)

    if pilot:
        names = names[:pilot]

    alphas = cfg["alphas"]
    probe_tids = [tid[p] for p in probes]

    # Per-carrier baselines + alpha=0 bit-exactness (real and random direction).
    base_acts_c, mean_norm, base_read = {}, {}, {}
    for c_i, cids in enumerate(carrier_ids):
        start = cids.shape[1] - 1
        acts = record_acts(model, blocks, cids, down + [L])
        base_acts_c[c_i] = acts
        mean_norm[c_i] = float(acts[L][0, pos_start:].float().norm(dim=-1).mean())
        base_read[c_i] = read(acts, down, [tid[n] for n in names] + probe_tids)
        v0 = jv.vec(L, token_id=tid[names[0]])
        g = torch.Generator().manual_seed(1000)
        r0 = torch.randn(lens.d_model, generator=g)
        for v in (v0, r0 / r0.norm()):
            with InterventionHooks(blocks) as iv:
                iv.add(L, add_op(v, 0.0, start))
                a0 = record_acts(model, blocks, cids, down)
            assert all(torch.equal(a0[l], acts[l]) for l in down), \
                f"alpha=0 not bit-exact on carrier {c_i}"
        print(f"[{tag}] carrier {c_i}: alpha=0 bit-exact PASS; mean content resid "
              f"norm L{L} = {mean_norm[c_i]:.1f}", flush=True)

    # ---------------- Arm 1: original 40-concept prompt ladders, single pass ----
    arm1_names = orig_surv[:pilot] if pilot else orig_surv
    t_arm1 = time.time()
    for i, n in enumerate(arm1_names):
        if n in ck["arm1"]:
            continue
        t1 = time.time()
        curve = []
        for s, prompt in enumerate(ladder(orig_specs[n]), start=1):
            r = read(record_acts(model, blocks, enc(prompt), band), band,
                     [tid[n]])[tid[n]]
            curve.append({"step": s, **r})
        ck["arm1"][n] = curve
        save_ckpt(ck_path, ck)
        print(f"[{tag} arm1 {i+1}/{len(arm1_names)}] {n} "
              f"p={['%.3g' % c['p'] for c in curve]} ({time.time()-t1:.1f}s)",
              flush=True)
    arm1_unit_s = (time.time() - t_arm1) / max(1, len(arm1_names))

    # ---------------- Arm 2: injection dose-response, 3 carriers x concepts -----
    t_arm2 = time.time()
    n_units = 0
    for c_i, cids in enumerate(carrier_ids):
        start = cids.shape[1] - 1
        for i, n in enumerate(names):
            key = f"c{c_i}::{n}"
            if key in ck["arm2"]:
                continue
            t1 = time.time()
            vec = jv.vec(L, token_id=tid[n])
            g = torch.Generator().manual_seed(1000 + i)
            rvec = torch.randn(lens.d_model, generator=g)
            rvec = rvec / rvec.norm()
            real, rand, pcurve = [], [], []
            for a in alphas:
                if a == 0.0:
                    b = base_read[c_i][tid[n]]
                    real.append({"alpha": a, **b})
                    rand.append({"alpha": a, **b})
                    pcurve.append({"alpha": a,
                                   **{p: base_read[c_i][tid[p]]["p"] for p in probes}})
                    continue
                for v, dst in [(vec, real), (rvec, rand)]:
                    with InterventionHooks(blocks) as iv:
                        iv.add(L, add_op(v, a * mean_norm[c_i], start))
                        acts = record_acts(model, blocks, cids, down)
                    targets = [tid[n]] + (probe_tids if dst is real else [])
                    r = read(acts, down, targets)
                    dst.append({"alpha": a, **r[tid[n]]})
                    if dst is real:
                        pcurve.append({"alpha": a,
                                       **{p: r[tid[p]]["p"] for p in probes}})
            ck["arm2"][key] = {"real": real, "rand": rand, "probes": pcurve}
            save_ckpt(ck_path, ck)
            n_units += 1
            print(f"[{tag} arm2 c{c_i} {i+1}/{len(names)}] {n} "
                  f"p={['%.3g' % c['p'] for c in real]} "
                  f"rand_max={max(c['p'] for c in rand):.3g} "
                  f"({time.time()-t1:.1f}s)", flush=True)
    arm2_unit_s = (time.time() - t_arm2) / max(1, n_units)

    # ---------------- Competition matrix ----------------------------------------
    evoc_names = names if not pilot else names[: 2 * pilot]
    t_ev = time.time()
    n_ev = 0
    for n in evoc_names:
        if n in ck["evoc"]:
            continue
        cy = resident_carrier(n, kind[n])
        ids = enc(cy)
        assert tid[n] in ids[0].tolist(), (n, cy)
        r = read(record_acts(model, blocks, ids, down), down, [tid[n]])[tid[n]]
        ck["evoc"][n] = {"carrier": cy, **r}
        save_ckpt(ck_path, ck)
        n_ev += 1
    evoc_unit_s = (time.time() - t_ev) / max(1, n_ev)
    thr = cfg["competition"]["evocability_threshold"]
    ranked = sorted(evoc_names, key=lambda n: -ck["evoc"][n]["p"])
    n_res = min(cfg["competition"]["n_residents"] if not pilot else pilot, len(ranked))
    residents = [n for n in ranked if ck["evoc"][n]["p"] > thr][:n_res]
    injected = residents                      # same fixed subset on both axes
    print(f"[{tag}] evocability: "
          f"{sum(1 for n in evoc_names if ck['evoc'][n]['p'] > thr)}"
          f"/{len(evoc_names)} above {thr}; residents = {residents}", flush=True)

    # Primary matrix at the pre-registered alpha=1.0; secondary alphas added
    # because at alpha=1.0 the norm-matched random control can already floor Y
    # (seen in the 8B pilot: dY_random ~ -0.87), making dY_specific
    # floor-dominated; a lower alpha resolves the specific component.
    comp_alphas = ([cfg["competition"]["alpha"]]
                   + list(cfg["competition"].get("secondary_alphas", [])))
    n_rseeds = 1 if pilot else cfg["competition"]["n_random_seeds"]
    inj_tids = [tid[x] for x in injected]
    t_comp = time.time()
    n_cells = 0
    for yi, y in enumerate(residents):
        ids = enc(ck["evoc"][y]["carrier"])
        for comp_alpha in comp_alphas:
            key = f"{y}@a{comp_alpha}"
            if key in ck["comp"]:
                continue
            t1 = time.time()
            m_norm = float(record_acts(model, blocks, ids, [L])[L][0, pos_start:]
                           .float().norm(dim=-1).mean())
            before = read(record_acts(model, blocks, ids, down), down,
                          inj_tids + [tid[y]])
            rand_after = []
            for s in range(n_rseeds):
                g = torch.Generator().manual_seed(3000 + yi * 10 + s)
                rv = torch.randn(lens.d_model, generator=g)
                with InterventionHooks(blocks) as iv:
                    iv.add(L, add_op(rv / rv.norm(), comp_alpha * m_norm, pos_start))
                    acts = record_acts(model, blocks, ids, down)
                rand_after.append(read(acts, down, [tid[y]])[tid[y]]["p"])
            cells = {}
            for x in injected:
                with InterventionHooks(blocks) as iv:
                    iv.add(L, add_op(jv.vec(L, token_id=tid[x]), comp_alpha * m_norm,
                                     pos_start))
                    acts = record_acts(model, blocks, ids, down)
                r = read(acts, down, [tid[x], tid[y]])
                cells[x] = {"x_before": before[tid[x]]["p"],
                            "x_after": r[tid[x]]["p"],
                            "y_after": r[tid[y]]["p"]}
                n_cells += 1
            ck["comp"][key] = {"carrier": ck["evoc"][y]["carrier"],
                               "alpha": comp_alpha, "mean_norm": m_norm,
                               "y_before": before[tid[y]]["p"],
                               "y_after_rand": rand_after, "cells": cells}
            save_ckpt(ck_path, ck)
            print(f"[{tag} comp {yi+1}/{len(residents)} a={comp_alpha}] resident {y} "
                  f"y_before={before[tid[y]]['p']:.3f} "
                  f"rand={['%.3f' % v for v in rand_after]} ({time.time()-t1:.1f}s)",
                  flush=True)
    comp_unit_s = (time.time() - t_comp) / max(1, n_cells)

    if pilot:
        n_full = len(orig_surv) + len(extra_surv)
        est = (len(orig_surv) * arm1_unit_s + 3 * n_full * arm2_unit_s
               + n_full * evoc_unit_s + (20 * 20 + 20 * 2) * comp_unit_s)
        print(json.dumps({
            "tag": tag, "pilot": pilot,
            "unit_seconds": {"arm1_per_concept": round(arm1_unit_s, 2),
                             "arm2_per_concept_carrier": round(arm2_unit_s, 2),
                             "evoc_per_concept": round(evoc_unit_s, 2),
                             "comp_per_cell": round(comp_unit_s, 2)},
            "estimated_full_run_minutes": round(est / 60, 1),
            "wall_clock_s": round(time.time() - t0, 1)}, indent=2), flush=True)

    # ---------------- Fits + summaries ------------------------------------------
    crit = cfg["criterion"]
    fits1 = {n: fit_curves([c["step"] for c in ck["arm1"][n]],
                           [c["p"] for c in ck["arm1"][n]]) for n in ck["arm1"]}
    fits2 = {k: fit_curves([c["alpha"] for c in v["real"]],
                           [c["p"] for c in v["real"]])
             for k, v in ck["arm2"].items()}
    sum1 = summarize(fits1, crit["sig_frac_min"], crit["width_frac_max"])
    sum2 = summarize(fits2, crit["sig_frac_min"], crit["width_frac_max"])
    sum2_by_carrier = {
        f"c{c_i}": summarize({k: f for k, f in fits2.items()
                              if k.startswith(f"c{c_i}::")},
                             crit["sig_frac_min"], crit["width_frac_max"])
        for c_i in range(len(carriers))}

    # Over-driving: peak presence exceeds the value at alpha_max by > margin.
    margin = cfg["overdrive_margin"]
    overdrive = {}
    for k, v in ck["arm2"].items():
        ps = [c["p"] for c in v["real"]]
        pk = int(np.argmax(ps))
        if pk < len(ps) - 1 and ps[pk] - ps[-1] > margin and ps[pk] >= 0.15:
            overdrive[k] = {"peak_alpha": v["real"][pk]["alpha"], "peak_p": ps[pk],
                            "p_at_alpha_max": ps[-1], "drop": ps[pk] - ps[-1]}
    od_concepts = sorted({k.split("::")[1] for k in overdrive})

    # Controls: random-direction non-ignition + unrelated-probe flatness.
    rand_inflate = {k: max(c["p"] for c in v["rand"]) - v["rand"][0]["p"]
                    for k, v in ck["arm2"].items()}
    probe_inflate_max = max(
        max(pc[p] for pc in v["probes"]) - v["probes"][0][p]
        for v in ck["arm2"].values() for p in probes)
    controls = {
        "random_dir_max_presence_increase": {
            "median": float(np.median(list(rand_inflate.values()))),
            "max": float(np.max(list(rand_inflate.values())))},
        "probe_max_increase": float(probe_inflate_max),
        "noop_check": "PASS", "alpha0_bit_exact": "PASS (all carriers, real+random)",
        "bos_prepend": pos_start,
    }

    # Competition analysis (per matrix alpha; the first is the pre-registered 1.0).
    thr_e = 0.05

    def corr(a, b):
        if len(a) < 3:
            return None
        pear = float(np.corrcoef(a, b)[0, 1])
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        spear = float(np.corrcoef(ra, rb)[0, 1])
        return {"pearson": pear, "spearman": spear}

    def analyze_comp(comp_alpha):
        cells_flat, mat = [], {}
        for y in residents:
            rec = ck["comp"].get(f"{y}@a{comp_alpha}")
            if rec is None:
                continue
            dY_rand = float(np.mean(rec["y_after_rand"])) - rec["y_before"]
            for x in injected:
                c = rec["cells"][x]
                dY_real = c["y_after"] - rec["y_before"]
                cell = {"y": y, "x": x, "diag": x == y,
                        "y_before": rec["y_before"], "y_after": c["y_after"],
                        "x_before": c["x_before"], "x_after": c["x_after"],
                        "dY_real": dY_real, "dY_random": dY_rand,
                        "dY_specific": dY_real - dY_rand,
                        "cos_jx_jy": float(jv.vec(L, token_id=tid[x])
                                           @ jv.vec(L, token_id=tid[y]))}
                cells_flat.append(cell)
                mat[(y, x)] = cell
        valid = [c for c in cells_flat if not c["diag"] and c["y_before"] > thr_e]
        evict = [c for c in valid if c["x_after"] - c["x_before"] > thr_e
                 and c["dY_real"] < -thr_e]
        coexist = [c for c in valid if c["x_after"] - c["x_before"] > thr_e
                   and c["dY_real"] >= -thr_e]
        n_neg = sum(1 for c in valid if c["dY_specific"] < 0)
        cos_v = [c["cos_jx_jy"] for c in valid]
        dspec = [c["dY_specific"] for c in valid]
        dreal = [c["dY_real"] for c in valid]
        summary = {
            "matrix_alpha": comp_alpha,
            "n_residents": len(residents), "n_injected": len(injected),
            "n_cells": len(cells_flat), "n_valid_offdiag": len(valid),
            "threshold": thr_e,
            "n_x_ignited": sum(1 for c in valid
                               if c["x_after"] - c["x_before"] > thr_e),
            "n_evictions": len(evict),
            "eviction_rate": len(evict) / len(valid) if valid else None,
            "n_coexist": len(coexist),
            "median_dY_real": float(np.median(dreal)) if valid else None,
            "median_dY_random": float(np.median([c["dY_random"] for c in valid]))
            if valid else None,
            "median_dY_specific": float(np.median(dspec)) if valid else None,
            "n_specific_excess_negative": n_neg,
            "sign_test_p_two_sided": sign_test_p(n_neg, len(valid)),
            "corr_dY_specific_vs_cos": corr(dspec, cos_v),
            "corr_dY_real_vs_cos": corr(dreal, cos_v),
            "corr_x_after_vs_cos": corr([c["x_after"] for c in valid], cos_v),
        }
        return summary, cells_flat, mat

    comp_results = {ca: analyze_comp(ca) for ca in comp_alphas}
    comp_summary, cells_flat, mat = comp_results[comp_alphas[0]]

    manifest = {
        "tag": tag, "model": cfg["model"], "lens": cfg["lens_file"], "device": dev,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "band_layers": [band[0], band[-1]], "inject_layer": L,
        "downstream_layers": down, "alphas": alphas, "carriers": carriers,
        "mean_resid_norm_per_carrier": mean_norm,
        "n_concepts": len(names), "orig_surviving": len(orig_surv),
        "extra_surviving": extra_surv, "dropped_by_tokenizer": dropped,
        "probes": probes, "pos_start": pos_start,
        "final_logit_softcap": softcap,
        "attn_implementation": cfg.get("attn_implementation"),
        "presence_def": "max over layers x content positions of J-lens softmax prob "
                        "of the leading-space concept token, norm-scaled readout via "
                        "the model's final norm; softcap applied if the family has "
                        "one (monotone); BOS position excluded when present",
        "arm2_injection": "alpha * mean_content_resid_norm(L) * unit j-vector "
                          "(raw-u harness convention) at inject layer, "
                          "last-position-onward (validated convention)",
        "competition_injection": "all content positions (positional asymmetry: "
                                 "prompt-evoked residents live at their hint-token "
                                 "positions), alpha=1.0, per-resident mean-norm "
                                 "scaling, BOS never edited",
        "resident_carrier_template": "The story was about (the) <Y>.",
        "random_dir_seeds": "arm2: 1000+concept_idx; competition: "
                            "3000+resident_idx*10+s",
        "aicc_k": {"linear": 3, "sigmoid": 5},
        "pilot": pilot,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "platform": platform.platform(),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }

    verdict_width = sum2["passes_width_primary"] if sum2 else None
    out = {
        "manifest": manifest,
        "criterion": {**crit,
                      "primary": "arm2 median 10-90% width / alpha range < 1/3 over "
                                 "all concept x carrier sigmoid fits",
                      "arm1": sum1, "arm2": sum2, "arm2_by_carrier": sum2_by_carrier,
                      "width_primary_pass": bool(verdict_width)},
        "overdrive": {"margin": margin, "n_curves": len(overdrive),
                      "n_concepts": len(od_concepts), "concepts": od_concepts,
                      "curves": overdrive},
        "competition": {"summary": comp_summary,
                        "secondary_alpha_summaries": {
                            str(ca): comp_results[ca][0] for ca in comp_alphas[1:]},
                        "residents": residents, "injected": injected,
                        "evocability": {n: ck["evoc"][n] for n in ck["evoc"]},
                        "cells": cells_flat},
        "controls": controls,
        "fits": {"arm1": fits1, "arm2": fits2},
        "curves": {"arm1": ck["arm1"], "arm2": ck["arm2"]},
    }
    json.dump(out, open(outdir / "metrics.json", "w"), indent=1, default=float)
    json.dump({"residents": residents, "injected": injected,
               "matrices": {str(ca): {"summary": comp_results[ca][0],
                                      "cells": comp_results[ca][1]}
                            for ca in comp_alphas}},
              open(outdir / "competition_matrix.json", "w"), indent=1, default=float)

    plot_curves(tag, ck, fits1, fits2, overdrive, outdir / "ignition_curves.png")
    for ca in comp_alphas:
        s_ca, _, mat_ca = comp_results[ca]
        if residents and mat_ca:
            suffix = "" if ca == comp_alphas[0] else f"_a{ca}"
            plot_matrix(tag, residents, injected, mat_ca, s_ca,
                        outdir / f"competition_matrix{suffix}.png")

    print(json.dumps({"tag": tag, "arm1": sum1, "arm2": sum2,
                      "width_primary_pass": bool(verdict_width),
                      "overdrive_curves": len(overdrive),
                      "overdrive_concepts": len(od_concepts),
                      "competition": comp_summary,
                      "competition_secondary": {
                          str(ca): comp_results[ca][0] for ca in comp_alphas[1:]},
                      "controls": controls,
                      "wall_clock_s": manifest["wall_clock_seconds"]}, indent=2),
          flush=True)
    (outdir / "DONE").write_text(f"{time.time()-t0:.0f}s\n")


# ---------------------------------------------------------------- figures

STYLE = {"bg": "#fcfcfb", "blue": "#2a78d6", "amber": "#c98500", "violet": "#4a3aa7",
         "ink": "#0b0b0b", "sub": "#52514e", "grid": "#e8e8e5", "spine": "#c3c2b7"}


def _ax_style(ax):
    ax.grid(True, color=STYLE["grid"], linewidth=0.7)
    ax.set_facecolor(STYLE["bg"])
    for sp in ax.spines.values():
        sp.set_color(STYLE["spine"])
    ax.tick_params(colors=STYLE["sub"], labelsize=7)


def plot_curves(tag, ck, fits1, fits2, overdrive, path):
    fig, axes = plt.subplots(4, 4, figsize=(13, 12), facecolor=STYLE["bg"])

    def draw(ax, xs_pts, ys_pts, f, title, xlab):
        ax.plot(xs_pts, ys_pts, "o", color=STYLE["blue"], markersize=4)
        xs = np.linspace(min(xs_pts), max(xs_pts), 200)
        ax.plot(xs, np.polyval([f["lin"]["slope"], f["lin"]["intercept"]], xs),
                "--", color=STYLE["amber"], linewidth=1.5, label="linear")
        if f["sig"]:
            s = f["sig"]
            ax.plot(xs, sig_f(xs, s["b"], s["t"], s["x0"], s["w"]), "-",
                    color=STYLE["blue"], linewidth=1.8, label="sigmoid")
        d = f["daicc"]
        ax.set_title(f"{title}  dAICc={d:.1f}" if d is not None else title,
                     fontsize=8, color=STYLE["ink"])
        ax.set_xlabel(xlab, fontsize=8, color=STYLE["sub"])
        _ax_style(ax)

    def pick(fits):
        ranked = sorted(fits, key=lambda k: -(fits[k]["daicc"] or -1e9))
        return [ranked[0], ranked[len(ranked) // 4], ranked[len(ranked) // 2],
                ranked[-1]]

    for ax, k in zip(axes[0], pick(fits1)):
        draw(ax, [c["step"] for c in ck["arm1"][k]], [c["p"] for c in ck["arm1"][k]],
             fits1[k], k, "ladder step")
    axes[0][0].set_ylabel("arm 1 presence", fontsize=9, color=STYLE["sub"])
    axes[0][0].legend(fontsize=7, frameon=False)
    for ax, k in zip(axes[1], pick(fits2)):
        draw(ax, [c["alpha"] for c in ck["arm2"][k]["real"]],
             [c["p"] for c in ck["arm2"][k]["real"]], fits2[k], k, "alpha")
    axes[1][0].set_ylabel("arm 2 presence", fontsize=9, color=STYLE["sub"])

    ax = axes[2][0]
    d1 = [f["daicc"] for f in fits1.values() if f["daicc"] is not None]
    ax.hist(d1, bins=20, color=STYLE["blue"], edgecolor=STYLE["bg"])
    ax.axvline(2, color=STYLE["amber"], linestyle="--", linewidth=1.5)
    ax.set_title("dAICc, arm 1 (ladders)", fontsize=9, color=STYLE["ink"])
    ax = axes[2][1]
    d2 = [f["daicc"] for f in fits2.values() if f["daicc"] is not None]
    ax.hist(d2, bins=25, color=STYLE["blue"], edgecolor=STYLE["bg"])
    ax.axvline(2, color=STYLE["amber"], linestyle="--", linewidth=1.5)
    ax.set_title("dAICc, arm 2 (all carriers)", fontsize=9, color=STYLE["ink"])
    ax = axes[2][2]
    pts = [(f["daicc"], f["width_frac"]) for f in fits2.values()
           if f["width_frac"] is not None]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=10,
                   color=STYLE["violet"], alpha=0.7)
        ax.axhline(1 / 3, color=STYLE["amber"], linestyle="--", linewidth=1.5)
        ax.axvline(2, color=STYLE["amber"], linestyle=":", linewidth=1)
    ax.set_title("arm 2: width_frac vs dAICc", fontsize=9, color=STYLE["ink"])
    ax.set_ylabel("10-90% width / range", fontsize=8, color=STYLE["sub"])
    ax = axes[2][3]
    w = [f["width_frac"] for f in fits2.values() if f["width_frac"] is not None]
    if w:
        ax.hist(w, bins=25, color=STYLE["violet"], edgecolor=STYLE["bg"])
        ax.axvline(1 / 3, color=STYLE["amber"], linestyle="--", linewidth=1.5,
                   label="pre-reg 1/3")
        ax.axvline(float(np.median(w)), color=STYLE["blue"], linewidth=1.5,
                   label=f"median {np.median(w):.2f}")
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("arm 2 width_frac (PRIMARY)", fontsize=9, color=STYLE["ink"])
    for ax in axes[2]:
        _ax_style(ax)

    od = sorted(overdrive, key=lambda k: -overdrive[k]["drop"])[:4]
    for ax, k in zip(axes[3], od):
        v = ck["arm2"][k]["real"]
        ax.plot([c["alpha"] for c in v], [c["p"] for c in v], "o-",
                color=STYLE["amber"], markersize=4)
        ax.set_title(f"over-driven: {k} (drop {overdrive[k]['drop']:.2f})",
                     fontsize=8, color=STYLE["ink"])
        ax.set_xlabel("alpha", fontsize=8, color=STYLE["sub"])
        _ax_style(ax)
    for ax in axes[3][len(od):]:
        ax.axis("off")
    if od:
        axes[3][0].set_ylabel("presence", fontsize=9, color=STYLE["sub"])

    fig.suptitle(f"Scaled J-space ignition ({tag}, MI300X): "
                 "3 carriers x 15 alphas", fontsize=12, color=STYLE["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"figure -> {path}", flush=True)


def plot_matrix(tag, residents, injected, mat, summary, path):
    ny, nx = len(residents), len(injected)
    Ya = np.array([[mat[(y, x)]["y_after"] for x in injected] for y in residents])
    Ds = np.array([[mat[(y, x)]["dY_specific"] for x in injected] for y in residents])
    Cs = np.array([[mat[(y, x)]["cos_jx_jy"] for x in injected] for y in residents])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor=STYLE["bg"])
    for ax, M, title, cmap, sym in [
            (axes[0][0], Ya, "resident Y presence AFTER X injection", "viridis", False),
            (axes[0][1], Ds, "dY specific (real - random control)", "RdBu", True),
            (axes[1][0], Cs, "cos(j_X, j_Y) at inject layer", "RdBu", True)]:
        vmax = np.abs(M).max() if sym else None
        im = ax.imshow(M, cmap=cmap, aspect="auto",
                       vmin=-vmax if sym else None, vmax=vmax)
        ax.set_xticks(range(nx), injected, rotation=90, fontsize=6,
                      color=STYLE["sub"])
        ax.set_yticks(range(ny), residents, fontsize=6, color=STYLE["sub"])
        ax.set_title(title, fontsize=10, color=STYLE["ink"])
        ax.set_xlabel("injected X", fontsize=8, color=STYLE["sub"])
        ax.set_ylabel("resident Y", fontsize=8, color=STYLE["sub"])
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1][1]
    cells = [mat[(y, x)] for y in residents for x in injected
             if x != y and mat[(y, x)]["y_before"] > 0.05]
    if cells:
        ax.scatter([c["cos_jx_jy"] for c in cells],
                   [c["dY_specific"] for c in cells], s=12, alpha=0.7,
                   color=STYLE["violet"])
        cc = summary.get("corr_dY_specific_vs_cos") or {}
        ax.set_title(f"dY_specific vs cos(j_X, j_Y)  "
                     f"r={cc.get('pearson', float('nan')):.2f} "
                     f"rho={cc.get('spearman', float('nan')):.2f}",
                     fontsize=10, color=STYLE["ink"])
    ax.set_xlabel("cos(j_X, j_Y)", fontsize=8, color=STYLE["sub"])
    ax.set_ylabel("dY_specific", fontsize=8, color=STYLE["sub"])
    ax.axhline(0, color=STYLE["spine"], linewidth=1)
    _ax_style(ax)
    p = summary["sign_test_p_two_sided"]
    fig.suptitle(f"Competition matrix, {tag} "
                 f"(alpha={summary['matrix_alpha']}, all content positions): "
                 f"eviction {summary['n_evictions']}/{summary['n_valid_offdiag']}, "
                 f"sign-test p={p:.2e}" if p is not None else
                 f"Competition matrix, {tag}", fontsize=12, color=STYLE["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
