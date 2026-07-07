"""Idea D signal check: J-space ablation selectivity dose-response on Qwen3.5-4B.

Per prompt: read the J-lens at the mid band (all positions), rank vocab tokens
by max readout probability, exclude prompt-literal/function-word/non-alphabetic
tokens, then ablate the top-k concept j-directions (QR-orthogonalized, harness
ablate op) and re-score. Controls: dimension/count-matched random subspaces
(2 seeds) and bottom-k of the active pool.
"""

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "../harness"))
import os
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

import datasets as datasets_lib
import jlens
from jspace_interventions import InterventionHooks, JVecs, find_blocks, layer_band, noop_check
from tasks import build_battery

STOP = set("""a an the and or but if then else of in on at by for with to from as is are was
were be been being am do does did have has had will would can could may might shall should
must not no nor so than that this these those it its he she his her him they them their we
our you your i me my mine who whom whose which what when where why how there here all any
both each few more most other some such only own same too very just also once about above
below between into through during before after again further out off over under up down""".split())

EXCLUSION_RULE = (
    "Token excluded from active-concept ranking if: (1) decoded string, stripped+"
    "lowercased, is empty or has no alphabetic character (punctuation/digits/"
    "whitespace); (2) single ASCII letter; (3) in fixed English function-word list "
    "(~110 words, see STOP in run.py); (4) token id occurs in prompt token ids; "
    "(5) normalized string matches lowercased prompt at word boundaries; (6) tokenizer "
    "special token (chat markup). Candidates "
    "deduplicated by normalized string, highest score kept. Score = max over "
    "mid-band layers x all positions of J-lens softmax probability."
)

DEP_TASKS = ["multihop", "arithmetic", "rhyme"]
INDEP_TASKS = ["mcqa", "sentiment", "singlehop"]


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
def active_scores(model, blocks, ids, J_dev, band):
    rec = Recorder(blocks, band)
    model(ids)
    rec.remove()
    norm, head = model.model.norm, model.lm_head
    best = None
    for l in band:
        z = rec.acts[l][0].float() @ J_dev[l].T
        p = head(norm(z.to(model.dtype))).float().softmax(-1).amax(0)
        best = p if best is None else torch.maximum(best, p)
    return best.cpu()


def select_concepts(score, prompt, prompt_ids, tok, scan_top, pool):
    prompt_l = prompt.lower()
    pset = set(prompt_ids) | set(tok.all_special_ids)
    vals, idxs = score.topk(scan_top)
    seen, accepted = set(), []
    for v, t in zip(vals.tolist(), idxs.tolist()):
        s = tok.decode([t])
        n = s.strip().lower()
        if not n or not any(c.isalpha() for c in n):
            continue
        if len(n) == 1 and n.isascii():
            continue
        if n in STOP or n in seen or t in pset:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", prompt_l):
            continue
        seen.add(n)
        accepted.append((t, round(v, 5)))
        if len(accepted) >= pool:
            break
    return accepted


@torch.no_grad()
def gen_correct(model, tok, ids, answers):
    cache = {}

    def logits_at(prefix):
        if prefix not in cache:
            full = ids
            if prefix:
                full = torch.cat(
                    [ids, torch.tensor([list(prefix)], device=ids.device)], dim=1)
            cache[prefix] = model(full).logits[0, -1].float()
        return cache[prefix]

    for ans in answers:
        cont = tok.encode(" " + ans.strip(), add_special_tokens=False)
        if all(logits_at(tuple(cont[:i])).argmax().item() == t
               for i, t in enumerate(cont)):
            return True
    return False


@torch.no_grad()
def logit_correct(model, ids, options, label):
    lg = model(ids).logits[0, -1].float()
    return int(torch.stack([lg[o] for o in options]).argmax().item()) == label


def score_item(model, tok, ids, item):
    if item["type"] == "gen":
        return gen_correct(model, tok, ids, item["answers"])
    return logit_correct(model, ids, item["options"], item["label"])


def jdirs(jv, band, tids):
    return {l: torch.stack([jv.vec(l, token_id=t) for t in tids], dim=1) for l in band}


def rand_dirs(band, k, d, seed):
    out = {}
    for l in band:
        g = torch.Generator().manual_seed(seed * 1000003 + l * 4099 + k)
        out[l] = torch.randn(d, k, generator=g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="items per task (pilot)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    dev = cfg["device"]
    if dev in (None, "auto"):
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")

    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    model = (transformers.AutoModelForCausalLM
             .from_pretrained(cfg["model"], dtype=torch.bfloat16).to(dev).eval())
    lens = jlens.JacobianLens.from_pretrained(cfg["lens_repo"], filename=cfg["lens_file"])
    blocks = find_blocks(model)
    band = layer_band(lens, *cfg["band"])
    J_dev = {l: lens.jacobians[l].to(dev) for l in band}
    jv = JVecs(lens, model, tok)
    print(f"loaded model+lens {time.time()-t0:.0f}s; band {band[0]}-{band[-1]}", flush=True)

    probe = tok("Fact: The capital of France is", return_tensors="pt").input_ids.to(dev)
    assert noop_check(model, probe, band), "noop_check FAILED"
    print("noop_check PASS", flush=True)

    items = build_battery(cfg, tok)
    if args.limit:
        by_task = {}
        items = [it for it in items
                 if by_task.setdefault(it["task"], []).append(it["id"]) or
                 len(by_task[it["task"]]) <= args.limit]
    ks, seeds, pool = cfg["ks"], cfg["random_seeds"], cfg["active_pool"]
    counts = {}
    for it in items:
        counts[it["task"]] = counts.get(it["task"], 0) + 1
    print(f"battery: {counts}", flush=True)

    jsonl = open(ROOT / "results" / "items.jsonl", "w")
    records = []
    for i, item in enumerate(items):
        t1 = time.time()
        ids = tok(item["prompt"], return_tensors="pt").input_ids.to(dev)
        base = score_item(model, tok, ids, item)
        score = active_scores(model, blocks, ids, J_dev, band)
        accepted = select_concepts(score, item["prompt"], ids[0].tolist(), tok,
                                   cfg["scan_top"], pool)
        assert len(accepted) >= 2 * max(ks), f"pool too small: {len(accepted)}"
        conds = {}
        for k in ks:
            sets = {"real": [t for t, _ in accepted[:k]],
                    "bottomk": [t for t, _ in accepted[-k:]]}
            for name, tids in sets.items():
                with InterventionHooks(blocks) as iv:
                    iv.ablate(jdirs(jv, band, tids), positions=None)
                    conds[f"{name}_k{k}"] = score_item(model, tok, ids, item)
            for s in seeds:
                with InterventionHooks(blocks) as iv:
                    iv.ablate(rand_dirs(band, k, lens.d_model, s), positions=None)
                    conds[f"rand_s{s}_k{k}"] = score_item(model, tok, ids, item)
        jv._cache.clear()
        rec = {
            "task": item["task"], "id": item["id"], "baseline_correct": base,
            "conditions": conds,
            "top_concepts": [(tok.decode([t]), v) for t, v in accepted[:10]],
            "bottom_concepts": [(tok.decode([t]), v) for t, v in accepted[-5:]],
            "n_accepted": len(accepted),
            "seconds": round(time.time() - t1, 2),
        }
        records.append(rec)
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        done, total = i + 1, len(items)
        eta = (time.time() - t0) / done * (total - done)
        print(f"[{done}/{total}] {item['task']}/{item['id']} base={base} "
              f"real_k20={conds.get('real_k20')} ({rec['seconds']}s, eta {eta/60:.0f}m)",
              flush=True)
    jsonl.close()

    cond_names = ([f"{c}_k{k}" for k in ks for c in ["real", "bottomk"]]
                  + [f"rand_s{s}_k{k}" for k in ks for s in seeds])
    tasks = sorted(counts)
    acc = {}
    for task in tasks:
        rs = [r for r in records if r["task"] == task]
        acc[task] = {"baseline": sum(r["baseline_correct"] for r in rs) / len(rs)}
        for c in cond_names:
            acc[task][c] = sum(r["conditions"][c] for r in rs) / len(rs)

    ck = cfg["criterion"]["k"]

    def drop(task, cond):
        return acc[task][cond] - acc[task][f"real_k{ck}"]

    rand_mean = {t: sum(acc[t][f"rand_s{s}_k{ck}"] for s in seeds) / len(seeds)
                 for t in tasks}
    dep_drop = sum(rand_mean[t] - acc[t][f"real_k{ck}"] for t in DEP_TASKS) / len(DEP_TASKS)
    indep_drop = sum(rand_mean[t] - acc[t][f"real_k{ck}"] for t in INDEP_TASKS) / len(INDEP_TASKS)
    verdict = (dep_drop > cfg["criterion"]["dep_min_drop"]
               and indep_drop < cfg["criterion"]["indep_max_drop"])
    criterion = {
        "k": ck,
        "dep_drop_real_vs_random": round(dep_drop, 4),
        "indep_drop_real_vs_random": round(indep_drop, 4),
        "thresholds": cfg["criterion"],
        "per_task_drop": {t: round(rand_mean[t] - acc[t][f"real_k{ck}"], 4) for t in tasks},
        "signal": bool(verdict),
    }

    manifest = {
        "model": cfg["model"], "lens": cfg["lens_file"], "device": dev,
        "band_layers": [band[0], band[-1]], "ks": ks, "random_seeds": seeds,
        "active_pool": pool, "scan_top": cfg["scan_top"],
        "n_items": counts, "exclusion_rule": EXCLUSION_RULE,
        "bottomk_rule": f"ranks {pool}-k+1..{pool} of the filtered active pool",
        "random_control": "k iid gaussian directions per layer, QR-orthonormalized "
                          "by the ablate op; subspace dimension exactly matches real",
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets_lib.__version__,
        "platform": platform.platform(),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    per_item = [{"task": r["task"], "id": r["id"], "base": r["baseline_correct"],
                 "conditions": r["conditions"], "top_concepts": r["top_concepts"]}
                for r in records]
    out = {"manifest": manifest, "accuracy": acc, "criterion": criterion,
           "items": per_item}
    json.dump(out, open(ROOT / "results" / "metrics.json", "w"), indent=1)

    plot(acc, tasks, ks, seeds, ROOT / "results" / "selectivity_dose_response.png")
    print(json.dumps({"criterion": criterion, "wall_clock_s":
                      manifest["wall_clock_seconds"]}, indent=2), flush=True)
    for t in tasks:
        row = {c: round(acc[t][c], 3) for c in ["baseline"] + cond_names}
        print(t, row, flush=True)


def plot(acc, tasks, ks, seeds, path):
    order = [t for t in DEP_TASKS + INDEP_TASKS if t in tasks]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True,
                             facecolor="#fcfcfb")
    xs = [0] + ks
    series = [("real", "real ablation", "#2a78d6", "-", "o"),
              ("rand_s0", "random s0", "#c98500", "--", "s"),
              ("rand_s1", "random s1", "#c98500", "--", "^"),
              ("bottomk", "bottom-k", "#4a3aa7", ":", "D")]
    for ax, task in zip(axes.flat, order):
        for key, label, color, ls, m in series:
            ys = [acc[task]["baseline"]] + [acc[task][f"{key}_k{k}"] for k in ks]
            ax.plot(xs, [y * 100 for y in ys], ls, color=color, marker=m,
                    markersize=4, linewidth=2, label=label)
        kind = "workspace-dep" if task in DEP_TASKS else "workspace-indep"
        ax.set_title(f"{task} ({kind})", fontsize=10, color="#0b0b0b")
        ax.set_xticks(xs)
        ax.set_ylim(-3, 103)
        ax.grid(True, color="#e8e8e5", linewidth=0.7)
        ax.set_facecolor("#fcfcfb")
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("k (ablated concept directions)", fontsize=9, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)", fontsize=9, color="#52514e")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9)
    fig.suptitle("J-space ablation selectivity dose-response (Qwen3.5-4B, mid band)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=180)
    print(f"figure -> {path}", flush=True)


if __name__ == "__main__":
    main()
