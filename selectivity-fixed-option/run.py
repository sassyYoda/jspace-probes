"""D redesign: fixed-option-scoring test of the answer-in-workspace reframe.

Same underlying questions (singlehop / multihop / rhyme, 60 each) scored two
ways: GEN (open completion, teacher-forced exact match) vs FIXED (4-option A-D
letter-logit comparison, answer word present only in the option list). Ablate
the top-k=20 active mid-band J-space concepts (same selection + exclusion rule
as the signal check, copied verbatim) vs matched random (2 seeds) and bottom-k
controls.

Pre-registered predictions (see README.md):
- OURS (answer-in-workspace): GEN real-vs-random drop >= 20 pts for all three
  task types; FIXED drop < 5 pts for all three, including multihop.
- PAPER'S (task-type): multihop drops >= 15 pts in BOTH formats; singlehop
  spared (< 5 pts) in both. The FIXED multihop cell decides.
"""

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parent
import os
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
HARNESS = Path(os.environ.get("JSPACE_HARNESS", ROOT / "../harness"))
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

import jlens
from jspace_interventions import InterventionHooks, JVecs, find_blocks, layer_band, noop_check
from tasks import LETTERS, build_battery

# ---- active-concept selection: copied VERBATIM from selectivity/run.py ----
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
# ---- end verbatim block ---------------------------------------------------


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
def active_scores(model, blocks, ids, J_dev, band, pos_start=0, softcap=0.0):
    """Max J-lens softmax prob per vocab token over band layers x positions.
    pos_start excludes the BOS attention sink on BOS-prepending models (gemma);
    softcap applies the final logit softcap (monotone; matches model readout)."""
    rec = Recorder(blocks, band)
    model(ids)
    rec.remove()
    norm, head = model.model.norm, model.lm_head
    best = None
    for l in band:
        z = rec.acts[l][0, pos_start:].float() @ J_dev[l].T
        logits = head(norm(z.to(model.dtype))).float()
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        p = logits.softmax(-1).amax(0)
        best = p if best is None else torch.maximum(best, p)
    return best.cpu()


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


def score_item(model, tok, ids, item, opt_ids):
    if item["type"] == "gen":
        return gen_correct(model, tok, ids, item["answers"])
    return logit_correct(model, ids, opt_ids, item["label"])


def jdirs(jv, band, tids):
    return {l: torch.stack([jv.vec(l, token_id=t) for t in tids], dim=1) for l in band}


def rand_dirs(band, k, d, seed):
    out = {}
    for l in band:
        g = torch.Generator().manual_seed(seed * 1000003 + l * 4099 + k)
        out[l] = torch.randn(d, k, generator=g)
    return out


def answer_leak(topk_strings, answer_words):
    """Which top-k concept strings match an answer content word (the answer
    direction 'sneaking in' despite the prompt-literal exclusion)."""
    hits = []
    for s in topk_strings:
        for w in answer_words:
            if len(w) < 3:
                continue
            if (s == w or s.rstrip("s") == w.rstrip("s")
                    or (len(s) >= 4 and len(w) >= 4
                        and (s.startswith(w) or w.startswith(s)))):
                hits.append((s, w))
    return hits


def key_of(item):
    return f"{item['task']}/{item['fmt']}/{item['id']}"


def analyze(records, cfg, manifest_extra=None):
    k = cfg["k"]
    seeds = cfg["random_seeds"]
    conds = [f"real_k{k}", f"bottomk_k{k}"] + [f"rand_s{s}_k{k}" for s in seeds]
    cells = sorted({(r["task"], r["fmt"]) for r in records})
    acc, drops, leak_frac = {}, {}, {}
    for cell in cells:
        task, fmt = cell
        rs = [r for r in records if r["task"] == task and r["fmt"] == fmt]
        variants = {"all": rs}
        if fmt == "fixed":
            variants["noleak"] = [r for r in rs if not r["leak"]]
            leak_frac[task] = round(sum(bool(r["leak"]) for r in rs) / len(rs), 4)
        for vn, rr in variants.items():
            name = f"{task}/{fmt}" + ("" if vn == "all" else "@noleak")
            if not rr:
                continue
            a = {"n": len(rr),
                 "baseline": sum(r["baseline_correct"] for r in rr) / len(rr)}
            for c in conds:
                a[c] = sum(r["conditions"][c] for r in rr) / len(rr)
            rand = sum(a[f"rand_s{s}_k{k}"] for s in seeds) / len(seeds)
            a["drop_real_vs_random"] = round(rand - a[f"real_k{k}"], 4)
            a["answer_in_topk_frac"] = round(
                sum(bool(r["leak"]) for r in rr) / len(rr), 4)
            acc[name] = {kk: (round(v, 4) if isinstance(v, float) else v)
                         for kk, v in a.items()}
            drops[name] = a["drop_real_vs_random"]

    # primary drops: GEN = all items; FIXED = leak-excluded
    def d(task, fmt):
        name = f"{task}/{fmt}@noleak" if fmt == "fixed" else f"{task}/{fmt}"
        if name not in drops:
            name = f"{task}/{fmt}"
        return drops.get(name)

    tasks3 = ["singlehop", "multihop", "rhyme"]
    grid = {t: {"gen": d(t, "gen"), "fixed": d(t, "fixed")} for t in tasks3}
    th = cfg["criterion"]
    ours = (all(grid[t]["gen"] is not None and grid[t]["gen"] >= th["gen_min_drop"]
                for t in tasks3)
            and all(grid[t]["fixed"] is not None
                    and grid[t]["fixed"] < th["fixed_max_drop"] for t in tasks3))
    paper = (grid["multihop"]["gen"] is not None
             and grid["multihop"]["gen"] >= th["paper_dep_min_drop"]
             and grid["multihop"]["fixed"] is not None
             and grid["multihop"]["fixed"] >= th["paper_dep_min_drop"]
             and grid["singlehop"]["gen"] < th["fixed_max_drop"]
             and grid["singlehop"]["fixed"] < th["fixed_max_drop"])
    verdict = {
        "grid_drop_real_vs_random": grid,
        "answer_leak_fraction_fixed": leak_frac,
        "ours_answer_in_workspace": bool(ours),
        "papers_task_type": bool(paper),
        "decisive_cell_multihop_fixed": grid["multihop"]["fixed"],
        "thresholds": th,
    }
    out = {"accuracy": acc, "verdict": verdict}
    if manifest_extra:
        out["manifest"] = manifest_extra
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="items per (task,fmt) cell")
    ap.add_argument("--analyze", action="store_true", help="recompute from jsonl only")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    outdir = ROOT / args.out
    outdir.mkdir(exist_ok=True)
    jsonl_path = outdir / "items.jsonl"

    if args.analyze:
        records = [json.loads(l) for l in open(jsonl_path)]
        res = analyze(records, cfg)
        json.dump(res, open(outdir / "metrics.json", "w"), indent=1)
        print(json.dumps(res["verdict"], indent=2))
        return

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

    # BOS handling (gemma): exclude the BOS attention sink from readout scans
    # and from ablation position ranges; apply final logit softcap if present.
    probe_ids = tok("Fact:", return_tensors="pt").input_ids
    bos_prepended = (tok.bos_token_id is not None
                     and probe_ids[0, 0].item() == tok.bos_token_id)
    pos_start = 1 if bos_prepended else 0
    softcap = float(getattr(model.config, "final_logit_softcapping", None) or 0.0)
    abl_positions = (pos_start, None) if pos_start else None
    print(f"loaded model+lens {time.time()-t0:.0f}s; band {band[0]}-{band[-1]} "
          f"({len(band)} layers); bos_prepended={bos_prepended} softcap={softcap}",
          flush=True)

    probe = tok("Fact: The capital of France is", return_tensors="pt").input_ids.to(dev)
    assert noop_check(model, probe, band), "noop_check FAILED"
    print("noop_check PASS", flush=True)

    items = build_battery(cfg)
    if args.limit:
        by_cell = {}
        items = [it for it in items
                 if by_cell.setdefault((it["task"], it["fmt"]), []).append(it["id"]) or
                 len(by_cell[(it["task"], it["fmt"])]) <= args.limit]
    k, seeds, pool = cfg["k"], cfg["random_seeds"], cfg["active_pool"]
    opt_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    assert len(set(opt_ids)) == 4, f"letter tokens collide: {opt_ids}"

    done = {}
    if jsonl_path.exists():
        for line in open(jsonl_path):
            r = json.loads(line)
            done[f"{r['task']}/{r['fmt']}/{r['id']}"] = r
    todo = [it for it in items if key_of(it) not in done]
    counts = {}
    for it in items:
        counts[f"{it['task']}/{it['fmt']}"] = counts.get(f"{it['task']}/{it['fmt']}", 0) + 1
    print(f"battery: {counts}; resuming past {len(done)} done, {len(todo)} to run",
          flush=True)

    jsonl = open(jsonl_path, "a")
    records = [done[key_of(it)] for it in items if key_of(it) in done]
    for i, item in enumerate(todo):
        t1 = time.time()
        ids = tok(item["prompt"], return_tensors="pt").input_ids.to(dev)
        base = score_item(model, tok, ids, item, opt_ids)
        score = active_scores(model, blocks, ids, J_dev, band, pos_start, softcap)
        accepted = select_concepts(score, item["prompt"], ids[0].tolist(), tok,
                                   cfg["scan_top"], pool)
        assert len(accepted) >= 2 * k, f"pool too small: {len(accepted)}"
        topk_strings = [tok.decode([t]).strip().lower() for t, _ in accepted[:k]]
        leak = answer_leak(topk_strings, item["answer_words"])
        conds = {}
        sets = {"real": [t for t, _ in accepted[:k]],
                "bottomk": [t for t, _ in accepted[-k:]]}
        for name, tids in sets.items():
            with InterventionHooks(blocks) as iv:
                iv.ablate(jdirs(jv, band, tids), positions=abl_positions)
                conds[f"{name}_k{k}"] = score_item(model, tok, ids, item, opt_ids)
        for s in seeds:
            with InterventionHooks(blocks) as iv:
                iv.ablate(rand_dirs(band, k, lens.d_model, s), positions=abl_positions)
                conds[f"rand_s{s}_k{k}"] = score_item(model, tok, ids, item, opt_ids)
        jv._cache.clear()
        rec = {
            "task": item["task"], "fmt": item["fmt"], "id": item["id"],
            "baseline_correct": base, "conditions": conds,
            "top_concepts": [(tok.decode([t]), v) for t, v in accepted[:10]],
            "n_accepted": len(accepted),
            "leak": leak,
            "label": item.get("label"),
            "seconds": round(time.time() - t1, 2),
        }
        records.append(rec)
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        nd, total = i + 1, len(todo)
        eta = (time.time() - t0) / nd * (total - nd)
        print(f"[{nd}/{total}] {key_of(item)} base={base} "
              f"real_k{k}={conds[f'real_k{k}']} leak={bool(leak)} "
              f"({rec['seconds']}s, eta {eta/60:.0f}m)", flush=True)
    jsonl.close()

    label_hist = {}
    for it in items:
        if it["fmt"] == "fixed":
            label_hist[LETTERS[it["label"]]] = label_hist.get(LETTERS[it["label"]], 0) + 1
    manifest = {
        "model": cfg["model"], "lens": cfg["lens_file"], "device": dev,
        "band_layers": [band[0], band[-1]], "n_band_layers": len(band),
        "k": k, "random_seeds": seeds,
        "active_pool": pool, "scan_top": cfg["scan_top"],
        "n_items": counts, "item_seed": cfg["item_seed"],
        "exclusion_rule": EXCLUSION_RULE,
        "bottomk_rule": f"ranks {pool}-k+1..{pool} of the filtered active pool",
        "random_control": "k iid gaussian directions per layer, QR-orthonormalized "
                          "by the ablate op; subspace dimension exactly matches real",
        "fixed_scoring": "argmax over letter-token logits ' A'..' D' at last position",
        "leak_rule": "top-k concept string equals an answer content word, or "
                     "plural-stripped equal, or (len>=4) prefix relation",
        "bos_prepended": bos_prepended, "pos_start": pos_start,
        "softcap": softcap, "fixed_label_histogram": label_hist,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    res = analyze(records, cfg, manifest)
    json.dump(res, open(outdir / "metrics.json", "w"), indent=1)
    print(json.dumps(res["verdict"], indent=2), flush=True)
    for name, row in res["accuracy"].items():
        print(name, row, flush=True)
    print(f"wall clock {manifest['wall_clock_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
