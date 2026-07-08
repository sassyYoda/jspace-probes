"""Idea J signal check: workspace capacity psychophysics on Qwen3.5-4B.

Load-N lists from the paper's capacity.json families; per-word J-lens presence
in the retention region (mid band, literal-occurrence masked); yes/no recall by
teacher-forced logit comparison; matched non-listed-word null; no-hold control;
plateau-vs-linear occupancy fit and the pre-registered behavioral-bind test.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from scipy.stats import norm as norm_dist

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.environ.get("JLENS_REPO",
                        os.path.join(HERE, "..", "vendor", "jacobian-lens"))
sys.path.insert(0, VENDOR)
RESULTS = os.path.join(HERE, "results")


def load_cfg():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    if cfg["device"] == "auto":
        cfg["device"] = ("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available() else "cpu")
    return cfg


def build_pools(cfg, tok):
    data = json.load(open(os.path.join(VENDOR, cfg["capacity_json"])))
    pools = {}
    for p in data["candidate_pools"]:
        if p["name"] not in cfg["families"]:
            continue
        surv = []
        for w in p["pool"]:
            ids = tok.encode(" " + w, add_special_tokens=False)
            if len(ids) == 1:
                surv.append((w, ids[0]))
        pools[p["name"]] = surv
    return pools


def build_items(cfg, pools):
    items = []
    fams = cfg["families"]
    n_ctl = cfg["n_controls_per_prompt"]
    for N in cfg["Ns"]:
        for seed in cfg["seeds"]:
            for i in range(cfg["prompts_per_cell"]):
                fam = fams[i % len(fams)]
                present = ((i // len(fams)) + (i % len(fams)) + seed) % 2 == 0
                rng = random.Random(f"ideaJ-{N}-{seed}-{i}")
                sample = rng.sample(pools[fam], N + 1 + n_ctl)
                words = sample[:N]
                if present:
                    pos = rng.randrange(N)
                    probe = words[pos]
                else:
                    pos = None
                    probe = sample[N]
                items.append({
                    "N": N, "seed": seed, "idx": i, "family": fam,
                    "present": present, "probe_serial": pos,
                    "words": words, "probe": probe,
                    "controls": sample[N + 1:],
                })
    return items


def prompts_for(cfg, item):
    lst = ", ".join(w for w, _ in item["words"])
    hold_prefix = f"Remember this list: {lst}."
    hold = cfg["prompt_hold"].format(LIST=lst, PROBE=item["probe"][0])
    assert hold.startswith(hold_prefix)
    nohold = cfg["prompt_nohold"].format(LIST=lst)
    nohold_prefix = nohold.split(". ")[0] + "."
    assert nohold.startswith(nohold_prefix)
    return hold, hold_prefix, nohold, nohold_prefix


@torch.no_grad()
def measure(lens, lm, tok, prompt, prefix, layers, word_tids, ctl_tids,
            probe_tid, mask_radius):
    """Per-token presence (max softmax prob over the layer band) in three
    regions: retention (after the list; primary, literal occurrences masked
    +-mask_radius), carry (list region strictly after a word's own position;
    exploratory), self (the word's own list position; positive control).
    Also returns final-position model logits."""
    ids = tok(prompt).input_ids
    pids = tok(prefix).input_ids
    assert ids[: len(pids)] == pids, "prefix tokenization boundary broke"
    T, q0 = len(ids), len(pids)
    lens_logits, model_logits, _ = lens.apply(lm, prompt, layers=layers)
    tids = word_tids + ctl_tids + [probe_tid]
    tids_t = torch.tensor(tids)
    probs = torch.stack(
        [lens_logits[l].float().softmax(-1)[:, tids_t] for l in layers]
    ).amax(0)  # [T, K] max over band
    ids_arr = np.asarray(ids)
    ret_pos = np.arange(q0, T)

    def band_max(pos_idx, k):
        return float(probs[pos_idx, k].max()) if len(pos_idx) else float("nan")

    out = {"self": [], "carry": [], "retention": []}
    for k, tid in enumerate(tids):
        occ = np.where(ids_arr == tid)[0]
        masked = set()
        for o in occ:
            masked.update(range(o - mask_radius, o + mask_radius + 1))
        keep_ret = np.array([a for a in ret_pos if a not in masked], int)
        out["retention"].append(band_max(keep_ret, k))
        if k < len(word_tids):  # listed word: has an own position in the list
            own = int(occ[0])
            out["self"].append(float(probs[own, k]))
            carry = np.arange(own + mask_radius + 1, q0)
            out["carry"].append(band_max(carry, k))
        elif k < len(word_tids) + len(ctl_tids):  # control: whole list region
            carry = np.array([a for a in range(q0) if a not in masked], int)
            out["carry"].append(band_max(carry, k))
    return out, model_logits[-1].float()


def run_measurements(cfg, items, out_path):
    import jlens
    import transformers

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=getattr(torch, cfg["dtype"])
    ).to(cfg["device"]).eval()
    lm = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        cfg["lens_repo"], filename=cfg["lens_file"]
    )
    lo, hi = cfg["mid_band_layers"]
    layers = [l for l in lens.source_layers if lo <= l <= hi]
    print(f"model+lens loaded {time.time()-t0:.0f}s; layers {layers}", flush=True)

    yes_ids = [tok.encode(v, add_special_tokens=False)[0] for v in (" yes", " Yes")]
    no_ids = [tok.encode(v, add_special_tokens=False)[0] for v in (" no", " No")]
    rad = cfg["literal_mask_radius"]

    records = []
    for n, item in enumerate(items):
        t1 = time.time()
        hold, hold_pre, nohold, nohold_pre = prompts_for(cfg, item)
        word_tids = [t for _, t in item["words"]]
        ctl_tids = [t for _, t in item["controls"]]
        pres, final = measure(lens, lm, tok, hold, hold_pre, layers,
                              word_tids, ctl_tids, item["probe"][1], rad)
        yes = torch.logsumexp(final[yes_ids], 0).item()
        no = torch.logsumexp(final[no_ids], 0).item()
        Nw = len(word_tids)
        pres_nh, _ = measure(lens, lm, tok, nohold, nohold_pre, layers,
                             word_tids, ctl_tids, item["probe"][1], rad)
        rec = {k: item[k] for k in
               ("N", "seed", "idx", "family", "present", "probe_serial")}
        rec.update({
            "probe": item["probe"][0],
            "word_presence": pres["retention"][:Nw],
            "control_presence": pres["retention"][Nw:Nw + len(ctl_tids)],
            "probe_presence": pres["retention"][-1],
            "word_carry": pres["carry"][:Nw],
            "control_carry": pres["carry"][Nw:],
            "word_self": pres["self"],
            "yes_lse": yes, "no_lse": no,
            "said_yes": yes > no,
            "correct": (yes > no) == item["present"],
            "nohold_word_presence": pres_nh["retention"][:Nw],
            "nohold_control_presence": pres_nh["retention"][Nw:Nw + len(ctl_tids)],
            "nohold_word_carry": pres_nh["carry"][:Nw],
            "nohold_control_carry": pres_nh["carry"][Nw:],
            "wall_s": round(time.time() - t1, 2),
        })
        records.append(rec)
        if (n + 1) % 12 == 0 or n + 1 == len(items):
            json.dump(records, open(out_path, "w"))
            done = rec
            print(f"[{n+1}/{len(items)}] N={done['N']} seed={done['seed']} "
                  f"({done['wall_s']}s/prompt, total {time.time()-t0:.0f}s)",
                  flush=True)
    return records, time.time() - t0


def dprime(n_hit, n_present, n_fa, n_absent):
    hr = (n_hit + 0.5) / (n_present + 1)
    far = (n_fa + 0.5) / (n_absent + 1)
    return float(norm_dist.ppf(hr) - norm_dist.ppf(far))


def dprime_of(recs):
    pres = [r for r in recs if r["present"]]
    abs_ = [r for r in recs if not r["present"]]
    return dprime(sum(r["said_yes"] for r in pres), len(pres),
                  sum(r["said_yes"] for r in abs_), len(abs_))


def fit_models(Ns_arr, counts):
    """LS fits of count=min(N,C) vs count=a*N; AICc (both k=2: 1 param + var)."""
    Ns_arr, counts = np.asarray(Ns_arr, float), np.asarray(counts, float)
    Cs = np.arange(0.5, 64.25, 0.25)
    rss_p = ((counts[None, :] - np.minimum(Ns_arr[None, :], Cs[:, None])) ** 2
             ).sum(1)
    i = int(rss_p.argmin())
    a = float((Ns_arr * counts).sum() / (Ns_arr ** 2).sum())
    rss_l = float(((counts - a * Ns_arr) ** 2).sum())
    n, k = len(counts), 2

    def aicc(rss):
        return float(n * np.log(max(rss, 1e-9) / n) + 2 * k
                     + 2 * k * (k + 1) / (n - k - 1))

    return {"C": float(Cs[i]), "rss_plateau": float(rss_p[i]),
            "a_linear": a, "rss_linear": rss_l,
            "aicc_plateau": aicc(rss_p[i]), "aicc_linear": aicc(rss_l),
            "plateau_wins": bool(aicc(rss_p[i]) < aicc(rss_l))}


def boot_ci(vals, iters, rng, fn=np.mean):
    vals = np.asarray(vals, float)
    stats = [fn(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(iters)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def analyze(cfg, records):
    rng = np.random.default_rng(0)
    B = cfg["bootstrap_iters"]
    Ns = cfg["Ns"]
    seeds = cfg["seeds"]

    pct = cfg["threshold_percentile"]
    ctl_all = np.concatenate([r["control_presence"] for r in records])
    theta = float(np.percentile(ctl_all, pct))
    theta_per_N = {N: float(np.percentile(
        np.concatenate([r["control_presence"] for r in records if r["N"] == N]),
        pct)) for N in Ns}
    ctl_nh = np.concatenate([r["nohold_control_presence"] for r in records])
    theta_nh = float(np.percentile(ctl_nh, pct))
    ctl_carry = np.concatenate([r["control_carry"] for r in records])
    theta_c = float(np.percentile(ctl_carry, pct))
    ctl_carry_nh = np.concatenate([r["nohold_control_carry"] for r in records])
    theta_c_nh = float(np.percentile(ctl_carry_nh, pct))

    def auc(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        return float(((a[:, None] > b[None, :]).mean()
                      + 0.5 * (a[:, None] == b[None, :]).mean()))

    listed_all = np.concatenate([r["word_presence"] for r in records])
    listed_carry = np.concatenate([r["word_carry"] for r in records])
    aucs = {"retention": auc(listed_all, ctl_all),
            "carry": auc(listed_carry, ctl_carry)}

    for r in records:
        wp = np.asarray(r["word_presence"])
        r["count"] = int((wp > theta).sum())
        r["count_perN"] = int((wp > theta_per_N[r["N"]]).sum())
        r["mean_presence"] = float(np.nanmean(wp))
        r["count_nohold"] = int(
            (np.asarray(r["nohold_word_presence"]) > theta_nh).sum())
        r["mean_presence_nohold"] = float(np.nanmean(r["nohold_word_presence"]))
        wc = np.asarray(r["word_carry"], float)
        r["carry_count"] = int(np.nansum(wc > theta_c))
        r["mean_carry"] = float(np.nanmean(wc))
        r["mean_self"] = float(np.mean(r["word_self"]))
        nc = np.asarray(r["nohold_word_carry"], float)
        r["carry_count_nohold"] = int(np.nansum(nc > theta_c_nh))
        r["mean_carry_nohold"] = float(np.nanmean(nc))

    per_N = {}
    for N in Ns:
        rs = [r for r in records if r["N"] == N]
        pres = [r for r in rs if r["present"]]
        abs_ = [r for r in rs if not r["present"]]
        per_N[N] = {
            "mean_presence": float(np.mean([r["mean_presence"] for r in rs])),
            "mean_presence_ci": boot_ci([r["mean_presence"] for r in rs], B, rng),
            "mean_presence_nohold": float(
                np.mean([r["mean_presence_nohold"] for r in rs])),
            "mean_presence_nohold_ci": boot_ci(
                [r["mean_presence_nohold"] for r in rs], B, rng),
            "count": float(np.mean([r["count"] for r in rs])),
            "count_ci": boot_ci([r["count"] for r in rs], B, rng),
            "count_perN_threshold": float(np.mean([r["count_perN"] for r in rs])),
            "count_nohold": float(np.mean([r["count_nohold"] for r in rs])),
            "carry_count": float(np.mean([r["carry_count"] for r in rs])),
            "carry_count_ci": boot_ci([r["carry_count"] for r in rs], B, rng),
            "carry_count_nohold": float(
                np.mean([r["carry_count_nohold"] for r in rs])),
            "mean_carry": float(np.mean([r["mean_carry"] for r in rs])),
            "mean_carry_ci": boot_ci([r["mean_carry"] for r in rs], B, rng),
            "mean_carry_nohold": float(
                np.mean([r["mean_carry_nohold"] for r in rs])),
            "mean_self": float(np.mean([r["mean_self"] for r in rs])),
            "accuracy": float(np.mean([r["correct"] for r in rs])),
            "hit_rate": float(np.mean([r["said_yes"] for r in pres])),
            "fa_rate": float(np.mean([r["said_yes"] for r in abs_])),
            "dprime": dprime_of(rs),
            "n_prompts": len(rs),
        }

    fits = {f"seed{s}": fit_models(
        [r["N"] for r in records if r["seed"] == s],
        [r["count"] for r in records if r["seed"] == s]) for s in seeds}
    fits["pooled"] = fit_models([r["N"] for r in records],
                                [r["count"] for r in records])
    fits_carry = {f"seed{s}": fit_models(
        [r["N"] for r in records if r["seed"] == s],
        [r["carry_count"] for r in records if r["seed"] == s]) for s in seeds}
    fits_carry["pooled"] = fit_models([r["N"] for r in records],
                                      [r["carry_count"] for r in records])

    lo_C, hi_C = cfg["plateau_C_range"]
    wins = [fits[f"seed{s}"]["plateau_wins"]
            and lo_C <= fits[f"seed{s}"]["C"] <= hi_C for s in seeds]
    crit_i = sum(wins) >= 2

    C_pool = fits["pooled"]["C"]
    N_star = next((N for N in Ns if N >= C_pool), Ns[-1])
    low = [r for r in records if r["N"] in cfg["low_N_band"]]
    high = [r for r in records if r["N"] >= N_star]

    def boot_dp_diff():
        diffs = []
        for _ in range(B):
            lo_s = [low[i] for i in rng.integers(0, len(low), len(low))]
            hi_s = [high[i] for i in rng.integers(0, len(high), len(high))]
            diffs.append(dprime_of(lo_s) - dprime_of(hi_s))
        return diffs

    dp_diffs = boot_dp_diff()
    dp_low, dp_high = dprime_of(low), dprime_of(high)
    diff_ci = [float(np.percentile(dp_diffs, 2.5)),
               float(np.percentile(dp_diffs, 97.5))]
    crit_ii = diff_ci[0] > 0

    serial = {}
    for N in Ns:
        rs = [r for r in records if r["N"] == N]
        M = np.stack([r["word_presence"] for r in rs])
        C = np.stack([r["word_carry"] for r in rs]).astype(float)
        S = np.stack([r["word_self"] for r in rs])
        serial[N] = {"mean": M.mean(0).tolist(),
                     "sem": (M.std(0) / np.sqrt(len(rs))).tolist(),
                     "carry_mean": np.nanmean(C, 0).tolist(),
                     "self_mean": S.mean(0).tolist()}

    pp = [r for r in records if r["present"]]

    def pb(x, y, nperm=5000):
        ok = ~np.isnan(x)
        x, y = x[ok], y[ok]
        if y.std() == 0 or x.std() == 0:
            return float("nan"), float("nan")
        r = float(np.corrcoef(x, y)[0, 1])
        perm = [abs(np.corrcoef(x, rng.permutation(y))[0, 1])
                for _ in range(nperm)]
        return r, float(np.mean(np.asarray(perm) >= abs(r)))

    x = np.asarray([r["probe_presence"] for r in pp])
    xc = np.asarray([r["word_carry"][r["probe_serial"]] for r in pp], float)
    y = np.asarray([r["correct"] for r in pp], float)
    r_pb, p_pb = pb(x, y)
    r_pb_c, p_pb_c = pb(xc, y)
    xz = x.copy()
    for N in Ns:
        m = np.asarray([r["N"] == N for r in pp])
        if m.sum() > 1 and x[m].std() > 0:
            xz[m] = (x[m] - x[m].mean()) / x[m].std()
    r_pb_z = (float(np.corrcoef(xz, y)[0, 1])
              if y.std() > 0 and xz.std() > 0 else float("nan"))

    return {
        "threshold": theta, "threshold_per_N": theta_per_N,
        "threshold_nohold": theta_nh,
        "threshold_carry": theta_c, "threshold_carry_nohold": theta_c_nh,
        "auc_listed_vs_control": aucs,
        "n_control_presences": int(len(ctl_all)),
        "per_N": per_N, "plateau_fits": fits,
        "plateau_fits_carry_exploratory": fits_carry,
        "criterion": {
            "i_plateau_wins_per_seed": {f"seed{s}": bool(w)
                                        for s, w in zip(seeds, wins)},
            "i_pass": bool(crit_i),
            "C_pooled": C_pool, "N_star_grid": int(N_star),
            "dprime_lowN": dp_low, "dprime_highN": dp_high,
            "dprime_diff_ci95": diff_ci, "ii_pass": bool(crit_ii),
            "verdict": "SIGNAL" if (crit_i and crit_ii) else "NO-SIGNAL",
        },
        "serial_position": serial,
        "presence_recall": {"r_pointbiserial": r_pb, "p_perm": p_pb,
                            "r_pointbiserial_zwithinN": r_pb_z,
                            "r_pointbiserial_carry": r_pb_c,
                            "p_perm_carry": p_pb_c,
                            "n_present_items": len(pp)},
    }


def plot(cfg, records, metrics, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    CAT = ["#2a78d6", "#1baf7a", "#eda100"]  # seeds 0/1/2
    BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#6da7ec", "#5598e7",
                 "#3987e5", "#256abf", "#184f95", "#0d366b"]
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9,
    })
    Ns = cfg["Ns"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    fig.suptitle("Idea J: workspace capacity psychophysics (Qwen3.5-4B, "
                 "J-lens layers 12-22)", fontsize=11)

    ax = axes[0, 0]
    xs = np.linspace(0, max(Ns), 200)
    ax.plot(xs, xs, color=INK2, lw=1, ls=":", label="count = N")
    for si, s in enumerate(cfg["seeds"]):
        rs = [r for r in records if r["seed"] == s]
        means = [np.mean([r["count"] for r in rs if r["N"] == N]) for N in Ns]
        ax.plot(Ns, means, "o-", color=CAT[si], lw=2, ms=4,
                label=f"seed {s} (C={metrics['plateau_fits'][f'seed{s}']['C']:.0f})")
    Cp = metrics["plateau_fits"]["pooled"]["C"]
    ax.plot(xs, np.minimum(xs, Cp), color=INK, lw=1.4, ls="--",
            label=f"min(N, C={Cp:.1f}) pooled")
    cc = [metrics["per_N"][N]["carry_count"] for N in Ns]
    Cc = metrics["plateau_fits_carry_exploratory"]["pooled"]["C"]
    ax.plot(Ns, cc, "d-", color="#4a3aa7", lw=1.6, ms=4,
            label=f"carry (list-region, expl.; C={Cc:.1f})")
    ax.set_xlabel("list length N"); ax.set_ylabel("words above null threshold")
    ax.set_title("Occupancy count vs N (retention region)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    dp = [metrics["per_N"][N]["dprime"] for N in Ns]
    acc = [metrics["per_N"][N]["accuracy"] for N in Ns]
    ax.plot(Ns, dp, "o-", color=CAT[0], lw=2, ms=4, label="d'")
    ax.plot(Ns, [a * max(dp) for a in acc], "s--", color=CAT[2], lw=1.4, ms=3,
            label=f"accuracy (x{max(dp):.1f})")
    ax.axvline(metrics["criterion"]["N_star_grid"], color=INK2, lw=1, ls=":")
    ax.set_xlabel("list length N"); ax.set_ylabel("recall d'")
    ax.set_title("Recall vs N (pooled seeds)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for ni, N in enumerate(Ns):
        m = metrics["serial_position"][N]["carry_mean"]
        ax.plot(np.arange(1, N + 1), m, color=BLUE_RAMP[ni % len(BLUE_RAMP)],
                lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("serial position in list")
    ax.set_ylabel("mean carry presence")
    ax.set_title("Serial-position curves, list-region carry "
                 "(light=N2 ... dark=N64)", fontsize=10)

    ax = axes[1, 1]
    mh = [metrics["per_N"][N]["mean_presence"] for N in Ns]
    mn = [metrics["per_N"][N]["mean_presence_nohold"] for N in Ns]
    ci_h = np.array([metrics["per_N"][N]["mean_presence_ci"] for N in Ns]).T
    ci_n = np.array([metrics["per_N"][N]["mean_presence_nohold_ci"] for N in Ns]).T
    ax.fill_between(Ns, ci_h[0], ci_h[1], color=CAT[0], alpha=0.18, lw=0)
    ax.plot(Ns, mh, "o-", color=CAT[0], lw=2, ms=4, label="hold, retention")
    ax.fill_between(Ns, ci_n[0], ci_n[1], color=CAT[2], alpha=0.18, lw=0)
    ax.plot(Ns, mn, "s-", color=CAT[2], lw=2, ms=4, label="no-hold, retention")
    ax.plot(Ns, [metrics["per_N"][N]["mean_carry"] for N in Ns], "o--",
            color=CAT[0], lw=1.2, ms=3, alpha=0.7, label="hold, carry")
    ax.plot(Ns, [metrics["per_N"][N]["mean_carry_nohold"] for N in Ns], "s--",
            color=CAT[2], lw=1.2, ms=3, alpha=0.7, label="no-hold, carry")
    ax.set_yscale("log")
    ax.set_xlabel("list length N"); ax.set_ylabel("mean per-word presence")
    ax.set_title("Instruction dependence of loading", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=170)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.smoke:
        cfg.update(Ns=[2, 8], seeds=[0], prompts_per_cell=4, bootstrap_iters=100)

    os.makedirs(RESULTS, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    rec_path = os.path.join(RESULTS, f"records{tag}.json")

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    pools = build_pools(cfg, tok)
    survivors = {f: len(p) for f, p in pools.items()}
    print("single-token survivors:", survivors, flush=True)
    items = build_items(cfg, pools)
    print(f"{len(items)} prompts", flush=True)

    if args.analyze_only:
        records = json.load(open(rec_path))
        wall = None
    else:
        records, wall = run_measurements(cfg, items, rec_path)

    metrics = analyze(cfg, records)
    metrics["survivors_per_family"] = survivors
    metrics["n_prompts"] = len(records)
    metrics["wall_clock_s"] = round(wall, 1) if wall else None
    metrics["config"] = cfg
    json.dump(metrics, open(os.path.join(RESULTS, f"metrics{tag}.json"), "w"),
              indent=1)
    plot(cfg, records, metrics,
         os.path.join(RESULTS, f"capacity_curves{tag}.png"))
    print(json.dumps(metrics["criterion"], indent=2))


if __name__ == "__main__":
    main()
