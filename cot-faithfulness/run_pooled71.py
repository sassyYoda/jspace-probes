"""Idea I POOLED n=71 definitive analysis.

Pools all four unfaithful cohorts (seed 0 orig n=16, seed 1 ext n=15,
seed 2 gpu n=19, seed 3 gpu n=21 = 71) and re-runs detection (both probe
sets, J-lens + logit lens), reversion (k=10, random s0/s1 + content
controls), and the no-CoT supplementary probe with the IDENTICAL pipeline
(phase2/phase3 imported verbatim from run.py) on ONE machine (local MPS),
so no cross-hardware numeric mixing.

Adds, per the pooled-analysis pre-registration:
- pooled AUCs with 2000-resample paired-bootstrap 95% CIs;
- per-seed-cohort AUC breakdown (16/15/19/21);
- formal tests of the pooled covert (set-b J-lens) AUC vs 0.5 (paired
  within-item label-swap permutation test, 20000 perms, + bootstrap) and
  vs the 0.7 pre-reg bar (bootstrap one-sided);
- pre-registered verdict logic: null_stands = AUC in [0.4, 0.6] and CI
  excludes 0.7; signal_flag = CI lower bound > 0.5. Both may hold.
- if signal_flag: per-concept AUC diagnostic, per-cue-cohort covert AUCs,
  and an explicit set-b token surface-variant audit vs the cue texts.

Never overwrites existing results files: all outputs carry _pooled71.
Phases: detect / revert / nocot / finalize (each cached / resumable).
"""

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "../harness"))
sys.path.insert(0, str(VENDOR))

from run import (LETTERS, auc, cached, chat_ids, mcq_text, phase2, phase3,
                 read_answer, roc, style)
from run_ext import boot_ci_auc, boot_ci_gap, load_model, rev_frac

RES = ROOT / "results"
N_BOOT = 2000
N_PERM = 20000

COHORT_FILES = {0: "phase1.json", 1: "phase1_ext.json",
                2: "phase1_s2.json", 3: "phase1_s3.json"}
EXPECTED_N = {0: 16, 1: 15, 2: 19, 3: 21}


def cohort_of(idx):
    return min(idx // 10000, 3)


def load_pooled_p1():
    items = []
    for s, f in COHORT_FILES.items():
        d = json.load(open(RES / f))
        for it in d["items"]:
            it.setdefault("phase", "orig" if s == 0 else f"seed{s}")
        items += d["items"]
        n_unf = sum(1 for it in d["items"] if it["unfaithful_cue"]
                    and cohort_of(it["idx"]) == s)
        assert n_unf == EXPECTED_N[s], (s, n_unf)
    n = sum(1 for it in items if it["unfaithful_cue"])
    assert n == 71, n
    return {"items": items}


# ---------------------------------------------------------------- phases

def phase_detect(cfg):
    p1 = load_pooled_p1()
    model, tok, dev, blocks, band, J_dev, jv, lens = load_model(cfg)
    p2 = cached(RES / "phase2_pooled71.json",
                lambda: phase2(cfg, model, tok, dev, blocks, band, J_dev, p1))
    print(json.dumps({k: round(v["auc"], 4) for k, v in p2["aucs"].items()}),
          flush=True)


def phase_revert(cfg):
    p1 = load_pooled_p1()
    model, tok, dev, blocks, band, J_dev, jv, lens = load_model(cfg)
    p2 = json.load(open(RES / "phase2_pooled71.json"))
    p3 = cached(RES / "phase3_pooled71.json",
                lambda: phase3(cfg, model, tok, dev, blocks, band, J_dev, jv,
                               p1, p2))
    print(json.dumps(p3["criterion"]), flush=True)


def phase_nocot(cfg):
    """Identical to run_ext.phase_nocot but on the 71-pool with _pooled71
    paths (real/content tids come from phase3_pooled71.json)."""
    import torch
    p1 = load_pooled_p1()
    model, tok, dev, blocks, band, J_dev, jv, lens = load_model(cfg)
    p3 = json.load(open(RES / "phase3_pooled71.json"))
    p3rows = {r["idx"]: r for r in p3["items"]}
    unf = [it for it in p1["items"]
           if it["unfaithful_cue"] and it["idx"] in p3rows]
    prefix_ids = tok.encode("Answer: (", add_special_tokens=False)
    lset = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    d_model = lens.d_model
    t0 = time.time()
    out_items = []
    for i, it in enumerate(unf):
        cue = it["unfaithful_cue"]
        h = it["hinted"][cue]
        r3 = p3rows[it["idx"]]

        def ctx(cue_text):
            pids = chat_ids(tok, mcq_text(it, cfg["instruction"], cue_text), dev)
            return torch.cat([pids, torch.tensor([prefix_ids], device=dev)],
                             dim=1)

        ids_h = ctx(cfg["cues"][cue].format(L=h["hint_letter"]))
        ids_u = ctx(None)
        base_h, _ = read_answer(model, ids_h, lset)
        base_u, _ = read_answer(model, ids_u, lset)
        real_tids = [t for t, _, _ in r3["real_tids"]]
        content_tids = [t for t, _, _ in r3["content_tids"]]

        def dirs_for(tids):
            return {l: torch.stack([jv.vec(l, token_id=t) for t in tids], dim=1)
                    for l in band}

        conds = {"real": dirs_for(real_tids), "content": dirs_for(content_tids)}
        for s in cfg["random_seeds"]:
            rd = {}
            for l in band:
                g = torch.Generator().manual_seed(s * 1000003 + l * 4099)
                rd[l] = torch.randn(d_model, len(real_tids), generator=g)
            conds[f"random{s}"] = rd
        res = {"baseline_hinted": base_h, "baseline_unhinted": base_u}
        for name, dirs in conds.items():
            letter, isletter = read_answer(model, ids_h, lset, dirs, blocks)
            res[name] = {"letter": letter, "top1_is_letter": isletter}
        out_items.append({"idx": it["idx"], "phase": it["phase"], "cue": cue,
                          "correct": LETTERS[it["label"]],
                          "hint": h["hint_letter"], "outcomes": res})
        print(f"[nocot71 {i+1}/{len(unf)}] idx={it['idx']} baseH={base_h} "
              f"baseU={base_u} real={res['real']['letter']}", flush=True)
    json.dump({"items": out_items, "wall_clock_s": round(time.time() - t0, 1)},
              open(RES / "supp_nocot_pooled71.json", "w"), default=float)
    print("nocot71 done", flush=True)


# ---------------------------------------------------------------- stats

def stat(row, tag, lens, key):
    d = row[tag]
    return max(d[lens][str(t)]["p"] for t in row[key]) if d else None


def perm_p_vs_half(pos, neg, seed=0, n_perm=N_PERM):
    """Paired within-item label-swap permutation test of AUC = 0.5.
    Returns one-sided (greater) and two-sided p with the null distribution
    summary. Add-one correction."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n = len(pos)
    obs = auc(pos, neg)
    null = np.empty(n_perm)
    for b in range(n_perm):
        sw = rng.integers(0, 2, n).astype(bool)
        p = np.where(sw, neg, pos)
        q = np.where(sw, pos, neg)
        null[b] = auc(p, q)
    p_greater = (1 + np.sum(null >= obs)) / (n_perm + 1)
    p_two = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (n_perm + 1)
    return {"auc_obs": obs, "p_one_sided_greater": float(p_greater),
            "p_two_sided": float(p_two), "n_perm": n_perm,
            "null_mean": float(null.mean()), "null_sd": float(null.std())}


def boot_dist(pos, neg, seed=0, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n = len(pos)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(0, n, n)
        vals[b] = auc(pos[k], neg[k])
    return vals


# ---------------------------------------------------------------- finalize

INK, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#e8e8e5", "#fcfcfb"
BLUE, AMBER = "#2a78d6", "#c98500"


def plot_detection_71(aucs, cohort_covert, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), facecolor=SURF)
    for ax, key, title in [(axes[0], "a_surface", "(a) hint-surface tokens"),
                           (axes[1], "b_concepts",
                            "(b) social-influence concepts")]:
        for lens, color in (("jlens", BLUE), ("logitlens", AMBER)):
            d = aucs[f"{key}_{lens}"]
            fpr, tpr = roc(d["pos_scores"], d["neg_scores"])
            lo, hi = d["ci95"]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{'J-lens' if lens == 'jlens' else 'logit lens'} "
                          f"AUC={d['auc']:.2f} [{lo:.2f},{hi:.2f}]")
        ax.plot([0, 1], [0, 1], "--", color=MUT, linewidth=1)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("FPR (unhinted controls)", fontsize=9, color=MUT)
        ax.set_ylabel("TPR (unfaithful-biased)", fontsize=9, color=MUT)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        style(ax)
    ax = axes[2]
    names = list(cohort_covert.keys())
    x = np.arange(len(names))
    vals = [cohort_covert[c]["auc"] for c in names]
    los = [cohort_covert[c]["ci95"][0] for c in names]
    his = [cohort_covert[c]["ci95"][1] for c in names]
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los),
                               np.array(his) - np.array(vals)],
                fmt="o", color=BLUE, capsize=4, markersize=6)
    ax.axhline(0.5, ls="--", color=MUT, lw=1)
    ax.axhline(0.7, ls=":", color=AMBER, lw=1.2)
    ax.text(len(names) - 0.4, 0.705, "pre-reg bar 0.7", fontsize=7,
            color=AMBER, ha="right")
    ax.set_xticks(x, [f"{c}\n(n={cohort_covert[c]['n']})" for c in names],
                  fontsize=8, color=INK)
    ax.set_ylim(0.2, 1.0)
    ax.set_title("(c) set-(b) covert J-lens AUC by cohort", fontsize=10,
                 color=INK)
    ax.set_ylabel("AUC [95% CI]", fontsize=9, color=MUT)
    style(ax)
    fig.suptitle("Detection, POOLED unfaithful set n=71 (seeds 0/1/2/3), "
                 "95% paired-bootstrap CI (Qwen2.5-7B-Instruct)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180)


def surface_variant_audit(cfg, tok, set_b_words):
    """List the exact set-b probe tokens and check none is a surface variant
    of the cue text or the set-a surface vocab."""
    cue_texts = " ".join(cfg["cues"][c].format(L="X").lower()
                         for c in cfg["cues"])
    surface = {w.lower() for c in cfg["surface_words"].values() for w in c}
    rows = []
    for w, t in set_b_words.items():
        s = tok.decode([t])
        norm = s.strip().lower()
        in_cue = norm in re.findall(r"[a-z]+", cue_texts)
        sub_cue = norm in cue_texts
        overlap_surface = any(norm == sw or norm in sw or sw in norm
                              for sw in surface)
        rows.append({"word": w, "token_id": int(t), "token_str": s,
                     "is_word_of_cue_text": bool(in_cue),
                     "is_substring_of_cue_text": bool(sub_cue),
                     "overlaps_set_a_surface_vocab": bool(overlap_surface)})
    clean = all(not (r["is_word_of_cue_text"]
                     or r["overlaps_set_a_surface_vocab"]) for r in rows)
    return {"tokens": rows, "no_surface_variant_leak": bool(clean)}


def phase_finalize(cfg, t0):
    import torch
    import transformers
    p1 = load_pooled_p1()
    p2 = json.load(open(RES / "phase2_pooled71.json"))
    p3 = json.load(open(RES / "phase3_pooled71.json"))
    supp = json.load(open(RES / "supp_nocot_pooled71.json"))
    unf = {it["idx"]: it for it in p1["items"] if it["unfaithful_cue"]}
    rows = p2["per_item"]
    assert len(rows) == 71

    # ---- detection AUCs: pooled + per cohort, with CIs
    def auc_block(sub_rows, key, lens):
        pairs = [(stat(r, "hinted", lens, key), stat(r, "control", lens, key))
                 for r in sub_rows]
        pairs = [(p, n) for p, n in pairs if p is not None and n is not None]
        pos = [p for p, _ in pairs]
        neg = [n for _, n in pairs]
        return pos, neg, {"auc": auc(pos, neg), "n": len(pairs),
                          "ci95": boot_ci_auc(pos, neg)}

    aucs_pooled, cohort_aucs = {}, {f"seed{s}": {} for s in range(4)}
    pooled_scores = {}
    for key, name in (("set_a_tids", "a_surface"), ("set_b_tids", "b_concepts")):
        for lens in ("jlens", "logitlens"):
            pos, neg, blk = auc_block(rows, key, lens)
            blk["pos_scores"], blk["neg_scores"] = pos, neg
            aucs_pooled[f"{name}_{lens}"] = blk
            pooled_scores[f"{name}_{lens}"] = (pos, neg)
            for s in range(4):
                sub = [r for r in rows if cohort_of(r["idx"]) == s]
                _, _, b = auc_block(sub, key, lens)
                cohort_aucs[f"seed{s}"][f"{name}_{lens}"] = b

    # ---- formal tests on the covert statistic (set-b J-lens), pooled
    pos, neg = pooled_scores["b_concepts_jlens"]
    perm = perm_p_vs_half(pos, neg)
    bd = boot_dist(pos, neg)
    tests = {
        "vs_0.5_permutation_paired": perm,
        "vs_0.5_bootstrap": {
            "p_le_0.5_one_sided": float((1 + np.sum(bd <= 0.5))
                                        / (len(bd) + 1)),
            "note": "fraction of paired-bootstrap AUCs <= 0.5 (small -> "
                    "AUC reliably above 0.5)"},
        "vs_0.7_bar_bootstrap": {
            "p_ge_0.7_one_sided": float((1 + np.sum(bd >= 0.7))
                                        / (len(bd) + 1)),
            "note": "fraction of paired-bootstrap AUCs >= 0.7 (small -> "
                    "AUC reliably below the 0.7 pre-reg bar)"}}
    # same formal tests for the logit-lens covert baseline, for symmetry
    posL, negL = pooled_scores["b_concepts_logitlens"]
    tests["logitlens_vs_0.5_permutation_paired"] = perm_p_vs_half(posL, negL)

    cov = aucs_pooled["b_concepts_jlens"]
    verdict = {
        "covert_auc_pooled": cov["auc"], "covert_auc_ci95": cov["ci95"],
        "null_stands": bool(0.4 <= cov["auc"] <= 0.6
                            and cov["ci95"][1] < 0.7),
        "signal_flag_ci_excludes_0.5": bool(cov["ci95"][0] > 0.5),
        "below_bar_ci_excludes_0.7": bool(cov["ci95"][1] < 0.7)}

    # ---- reversion: pooled + per cohort
    conds = ["real", "content"] + [f"random{s}" for s in cfg["random_seeds"]]
    p3_items = p3["items"]
    rev_pooled = {c: rev_frac(p3_items, c) for c in ["baseline"] + conds}
    rev_cohort = {f"seed{s}": {c: rev_frac(
        [o for o in p3_items if cohort_of(o["idx"]) == s], c)
        for c in ["baseline"] + conds} for s in range(4)}
    rand_rev = float(np.mean([rev_pooled[f"random{s}"]["reversion"]
                              for s in cfg["random_seeds"]]))
    gap = rev_pooled["real"]["reversion"] - rand_rev
    gap_ci = boot_ci_gap(p3_items, cfg["random_seeds"])

    # ---- no-CoT probe (restricted to hint-taking items, as before)
    nc_all = supp["items"]
    nc_biased = [o for o in nc_all
                 if o["outcomes"]["baseline_hinted"] == o["hint"]]
    nocot = {
        "n_items": len(nc_all),
        "n_nocot_baseline_takes_hint": len(nc_biased),
        "n_nocot_unhinted_correct": sum(
            1 for o in nc_all
            if o["outcomes"]["baseline_unhinted"] == o["correct"]),
        "summary_on_hint_taking_items":
            {c: rev_frac(nc_biased, c) for c in conds}}
    nc_rand = float(np.mean(
        [nocot["summary_on_hint_taking_items"][f"random{s}"]["reversion"]
         for s in cfg["random_seeds"]]))
    nocot["gap_real_minus_random"] = (
        nocot["summary_on_hint_taking_items"]["real"]["reversion"] - nc_rand)
    nocot["gap_ci95"] = boot_ci_gap(nc_biased, cfg["random_seeds"])

    # ---- driving-concept diagnostic (run if CI excludes 0.5)
    import transformers as _tr
    tok = _tr.AutoTokenizer.from_pretrained(cfg["model"])
    set_b_words = {w: int(t) for w, t in p2["set_b_words"].items()}
    diagnostic = {"triggered": verdict["signal_flag_ci_excludes_0.5"],
                  "surface_variant_audit":
                      surface_variant_audit(cfg, tok, set_b_words)}
    if diagnostic["triggered"]:
        per_concept = {}
        for w, t in set_b_words.items():
            pairs = [(r["hinted"]["jlens"][str(t)]["p"],
                      r["control"]["jlens"][str(t)]["p"])
                     for r in rows if r["hinted"] and r["control"]]
            po = [p for p, _ in pairs]
            ne = [n for _, n in pairs]
            per_concept[w] = {"token_id": t, "auc": auc(po, ne),
                              "n": len(pairs), "ci95": boot_ci_auc(po, ne)}
        # covert (max-over-set-b) AUC split by cue type of the item
        by_cue = {}
        for cue in ("stanford", "experience"):
            sub = [r for r in rows if r["cue"] == cue]
            _, _, b = auc_block(sub, "set_b_tids", "jlens")
            by_cue[cue] = b
        # per-concept x cue grid (heterogeneity of the driver)
        per_concept_by_cue = {}
        for w, t in set_b_words.items():
            per_concept_by_cue[w] = {}
            for cue in ("stanford", "experience"):
                pairs = [(r["hinted"]["jlens"][str(t)]["p"],
                          r["control"]["jlens"][str(t)]["p"])
                         for r in rows
                         if r["cue"] == cue and r["hinted"] and r["control"]]
                per_concept_by_cue[w][cue] = {
                    "auc": auc([p for p, _ in pairs], [n for _, n in pairs]),
                    "n": len(pairs)}
        diagnostic.update({"per_concept_auc_jlens": per_concept,
                           "covert_auc_by_cue_cohort": by_cue,
                           "per_concept_auc_by_cue": per_concept_by_cue})

    cue_counts = {c: sum(1 for it in unf.values() if it["unfaithful_cue"] == c)
                  for c in ("stanford", "experience")}
    strip = lambda d: {k: {kk: vv for kk, vv in v.items()
                           if kk in ("auc", "n", "ci95")} for k, v in d.items()}
    metrics = {
        "manifest": {
            "analysis": "pooled n=71 definitive re-analysis; all 71 items + "
                        "71 matched unhinted controls re-run locally through "
                        "the identical phase2/phase3/no-CoT pipeline "
                        "(functions imported verbatim from run.py) on one "
                        "machine (MPS); no cross-hardware score mixing",
            "cohorts": {f"seed{s}": EXPECTED_N[s] for s in range(4)},
            "unfaithful_cue_counts": cue_counts,
            "model": cfg["model"], "lens": cfg["lens_file"],
            "device": cfg["device"], "n_boot": N_BOOT, "n_perm": N_PERM,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "finalize_wall_clock_s": round(time.time() - t0, 1)},
        "detection_pooled": strip(aucs_pooled),
        "detection_per_cohort": {k: strip(v) for k, v in cohort_aucs.items()},
        "covert_formal_tests": tests,
        "preregistered_verdict": verdict,
        "reversion_pooled": {
            "summary": rev_pooled,
            "criterion": {
                "reversion_real": rev_pooled["real"]["reversion"],
                "reversion_random_mean": rand_rev,
                "gap": gap, "gap_ci95": gap_ci,
                "gap_pass": gap >= cfg["criterion"]["reversion_gap_min"],
                "reversion_gt_disruption":
                    rev_pooled["real"]["reversion"]
                    > rev_pooled["real"]["disruption"],
                "signal": bool(gap >= cfg["criterion"]["reversion_gap_min"]
                               and rev_pooled["real"]["reversion"]
                               > rev_pooled["real"]["disruption"])}},
        "reversion_per_cohort": rev_cohort,
        "supp_nocot_pooled": nocot,
        "driving_concept_diagnostic": diagnostic,
    }
    json.dump(metrics, open(RES / "metrics_pooled71.json", "w"), indent=1,
              default=float)
    cohort_covert = {k: v["b_concepts_jlens"] for k, v in cohort_aucs.items()}
    plot_detection_71(aucs_pooled, cohort_covert,
                      RES / "detection_auc_pooled71.png")
    show = {k: v for k, v in metrics.items()
            if k not in ("reversion_per_cohort",)}
    print(json.dumps(show, indent=1, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["detect", "revert", "nocot", "finalize"])
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    {"detect": phase_detect, "revert": phase_revert,
     "nocot": phase_nocot,
     "finalize": lambda c: phase_finalize(c, t0)}[args.phase](cfg)
    print(f"phase {args.phase} wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
