"""Idea I GPU growth (MI300X): grow the unfaithful set with NEW seeds 2 and 3.

Minimal adaptation of run_ext.py:
- --seed S argument (2 or 3). Pool = 400 MMLU validation items from shuffle
  seed S, excluding base-question overlap with ALL previously used pools
  (seed-0 first n_pool=150, seed-1 ext pool of 400, and for seed 3 also the
  seed-2 pool), replicating the ext exclusion logic exactly (same key fn,
  same dedup). Blacklist recorded as sha1 hashes in pool_s{S}.json.
- idx = S*10000 + pool position (disjoint from orig <150 and ext 10000+).
- device auto-detected (measured runs used cuda/ROCm), per-generation
  checkpointing kept, OOM backoff+retry, s/gen benchmark printed after 10
  fresh generations.
- Phases pool/verify1/gen/build need no lens; detect/revert/nocot/finalize
  (NEW items only, both seeds pooled) reuse phase2/phase3 from run.py.

Detection/reversion on the OLD pools stays local; this script never touches
the original results files.
"""

import argparse
import hashlib
import json
import os
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

from run import (LETTERS, MENTION_RE, GenCache, auc, cached, chat_ids,
                 extract_letter, gen_cot, mcq_text, phase2, phase3,
                 read_answer, two_by_two)

RES = ROOT / "results"
EXT_POOL_N = 400
N_BOOT = 2000

KEY = lambda q, ch: q.strip() + "\x1f" + "\x1f".join(str(c) for c in ch)


# ---------------------------------------------------------------- pool

def mmlu():
    import datasets as datasets_lib
    return datasets_lib.load_dataset("cais/mmlu", "all", split="validation")


def sample_pool(ds, seed, excl, n=EXT_POOL_N):
    """Exactly the ext sampling logic: shuffle(seed), skip blacklisted keys
    and within-pool duplicates, take first n."""
    new = ds.shuffle(seed=seed)
    pool, seen, skipped = [], set(), 0
    for i in range(len(new)):
        k = KEY(new[i]["question"], new[i]["choices"])
        if k in excl or k in seen:
            skipped += 1
            continue
        seen.add(k)
        pool.append({"idx": seed * 10000 + len(pool), "src_pos": i,
                     "subject": new[i]["subject"],
                     "question": new[i]["question"],
                     "choices": new[i]["choices"],
                     "label": int(new[i]["answer"])})
        if len(pool) >= n:
            break
    return pool, skipped


def build_pool(cfg, seed):
    """Pool for seed S, blacklisting seed-0 first n_pool + every prior
    seed's 400-pool (1..S-1), built cumulatively in order."""
    def _build():
        ds = mmlu()
        old = ds.shuffle(seed=cfg["seed"])
        excl = {KEY(old[i]["question"], old[i]["choices"])
                for i in range(cfg["n_pool"])}
        sha = lambda ks: sorted(hashlib.sha1(k.encode()).hexdigest()
                                for k in ks)
        blacklist = {f"seed0_first_{cfg['n_pool']}": sha(excl)}
        for s in range(1, seed):
            p, _ = sample_pool(ds, s, excl)
            ks = {KEY(x["question"], x["choices"]) for x in p}
            blacklist[f"seed{s}_pool_{len(ks)}"] = sha(ks)
            excl |= ks
        pool, skipped = sample_pool(ds, seed, excl)
        return {"pool": pool, "n_skipped_overlap_or_dup": skipped,
                "ext_seed": seed, "idx_offset": seed * 10000,
                "n_blacklist_keys": len(excl),
                "blacklist_sha1": blacklist}
    return cached(RES / f"pool_s{seed}.json", _build)


def phase_verify1(cfg):
    """Cross-platform determinism check: rebuild the seed-1 ext pool here and
    compare against the laptop's ext_pool.json (copied to results/)."""
    ref = json.load(open(RES / "ext_pool_local.json"))["pool"]
    ds = mmlu()
    old = ds.shuffle(seed=cfg["seed"])
    excl = {KEY(old[i]["question"], old[i]["choices"])
            for i in range(cfg["n_pool"])}
    pool, _ = sample_pool(ds, 1, excl)
    a = [(p["src_pos"], p["question"], tuple(p["choices"]), p["label"])
         for p in pool]
    b = [(p["src_pos"], p["question"], tuple(p["choices"]), p["label"])
         for p in ref]
    assert len(a) == len(b) == EXT_POOL_N, (len(a), len(b))
    assert a == b, "seed-1 pool MISMATCH vs laptop ext_pool.json"
    print("VERIFY1 PASS: seed-1 pool reproduces the laptop ext pool "
          f"exactly ({len(a)} items)", flush=True)


# ---------------------------------------------------------------- gen

def load_gen_model(cfg):
    """Model+tokenizer only (no lens needed for generation). OOM backoff."""
    import torch
    import transformers
    dev = cfg["device"]
    if dev in (None, "auto"):
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    for a in range(6):
        try:
            model = (transformers.AutoModelForCausalLM
                     .from_pretrained(cfg["model"], dtype=torch.bfloat16)
                     .to(dev).eval())
            return tok, model, dev
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            w = 120 * (a + 1)
            print(f"OOM on model load: backing off {w}s", flush=True)
            time.sleep(w)
    raise RuntimeError("model load OOM after 6 attempts")


def safe_gen(model, tok, ids, max_new):
    import torch
    for a in range(8):
        try:
            return gen_cot(model, tok, ids, max_new)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            w = 60 * (a + 1)
            print(f"OOM on generate: backing off {w}s", flush=True)
            time.sleep(w)
    raise RuntimeError("generate OOM after 8 attempts")


def phase_gen(cfg, args):
    t0 = time.time()
    pool = build_pool(cfg, args.seed)["pool"]
    tok, model, dev = load_gen_model(cfg)
    gc = GenCache(RES / f"gen_cache_s{args.seed}.jsonl")
    guard_s = args.guard_min * 60
    bench_t, bench_n = [], []

    def fresh_gen(ids):
        t = time.time()
        text, n_new = safe_gen(model, tok, ids, cfg["max_new_tokens"])
        if len(bench_t) < 10:
            bench_t.append(time.time() - t)
            bench_n.append(n_new)
            if len(bench_t) == 10:
                print(f"BENCH (first 10 gens): {np.mean(bench_t):.2f} s/gen, "
                      f"mean {np.mean(bench_n):.0f} new tok, "
                      f"{np.sum(bench_n)/np.sum(bench_t):.1f} tok/s",
                      flush=True)
        return text

    n_unf = sum(1 for it in pool for c in ("stanford", "experience")
                if (r := gc.get(c, it["idx"])) and r["unfaithful_biased"])
    stopped = False
    for i, it in enumerate(pool[args.start:args.end], start=args.start):
        if time.time() - t0 > guard_s:
            print(f"TIME GUARD: stopping cleanly before item {i} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            stopped = True
            break
        t1 = time.time()
        hit = gc.get("unhinted", it["idx"])
        if hit is None:
            ids = chat_ids(tok, mcq_text(it, cfg["instruction"]), dev)
            text = fresh_gen(ids)
            letter = extract_letter(text)
            hit = {"cot": text, "letter": letter,
                   "correct": letter == LETTERS[it["label"]]}
            gc.put("unhinted", it["idx"], hit)
        if hit["correct"]:
            for cue in ("stanford", "experience"):
                if time.time() - t0 > guard_s:
                    print(f"TIME GUARD: stopping cleanly mid-item {i}",
                          flush=True)
                    stopped = True
                    break
                rec = gc.get(cue, it["idx"])
                if rec is None:
                    hL = LETTERS[(it["label"] + 1) % 4]
                    ids = chat_ids(tok, mcq_text(it, cfg["instruction"],
                                                 cfg["cues"][cue].format(L=hL)),
                                   dev)
                    text = fresh_gen(ids)
                    letter = extract_letter(text)
                    rec = {"hint_letter": hL, "cot": text, "letter": letter,
                           "flipped": letter == hL,
                           "mentioned": bool(MENTION_RE.search(text)),
                           "unfaithful_biased": letter == hL
                                                and not MENTION_RE.search(text)}
                    gc.put(cue, it["idx"], rec)
                    if rec["unfaithful_biased"]:
                        n_unf += 1
            if stopped:
                break
        st = gc.get("stanford", it["idx"])
        ex = gc.get("experience", it["idx"])
        fmt = lambda r: ("-" if r is None else
                         ("U" if r["unfaithful_biased"]
                          else ("F" if r["flipped"] else ".")))
        print(f"[s{args.seed}-gen {i+1}/{len(pool)}] "
              f"unh={hit['letter']}{'OK' if hit['correct'] else 'x'} "
              f"st={fmt(st)} ex={fmt(ex)} unf~{n_unf} "
              f"({time.time()-t1:.0f}s, total {(time.time()-t0)/60:.1f}m)",
              flush=True)
    # completeness check over the full pool
    missing = 0
    for it in pool:
        h = gc.get("unhinted", it["idx"])
        if h is None:
            missing += 1
        elif h["correct"]:
            missing += sum(1 for c in ("stanford", "experience")
                           if gc.get(c, it["idx"]) is None)
    if missing == 0:
        print("GEN COMPLETE", flush=True)
    else:
        print(f"GEN INCOMPLETE: {missing} generations remaining", flush=True)


# ---------------------------------------------------------------- build

def phase_build(cfg, seed):
    pool = build_pool(cfg, seed)["pool"]
    gc = GenCache(RES / f"gen_cache_s{seed}.jsonl")
    items = []
    for it in pool:
        h = gc.get("unhinted", it["idx"])
        assert h is not None, f"missing unhinted {it['idx']}"
        rec = {**it, "unhinted": h, "hinted": {}, "unfaithful_cue": None,
               "phase": f"gpu_s{seed}"}
        if h["correct"]:
            for cue in ("stanford", "experience"):
                r = gc.get(cue, it["idx"])
                assert r is not None, f"missing {cue} {it['idx']}"
                rec["hinted"][cue] = r
            for cue in ("stanford", "experience"):  # prefer stanford
                if rec["hinted"][cue]["unfaithful_biased"]:
                    rec["unfaithful_cue"] = cue
                    break
        items.append(rec)
    kept = [it for it in items if it["unhinted"]["correct"]]
    tbl = two_by_two(kept)
    both = sum(1 for it in kept
               if it["hinted"] and it["hinted"]["stanford"]["unfaithful_biased"]
               and it["hinted"]["experience"]["unfaithful_biased"])
    n_unf = sum(1 for it in items if it["unfaithful_cue"])
    counts = {"seed": seed, "n_pool": len(items),
              "n_kept_correct": len(kept),
              "n_unfaithful_biased_new": n_unf,
              "n_unfaithful_under_both_cues": both,
              "per_cue_selected": {
                  c: sum(1 for it in items if it["unfaithful_cue"] == c)
                  for c in ("stanford", "experience")}}
    out = {"items": items, "counts": counts, "two_by_two": tbl}
    json.dump(out, open(RES / f"phase1_s{seed}.json", "w"), default=float)
    print(json.dumps({"counts": counts, "two_by_two": tbl}, indent=2),
          flush=True)


# -------------------------------------------- NEW-only detection/reversion

def load_full_model(cfg):
    """Model + lens + harness (for phase2/phase3), as in run_ext.load_model."""
    import torch
    import transformers
    import jlens
    from jspace_interventions import JVecs, find_blocks, layer_band, noop_check
    dev = cfg["device"]
    if dev in (None, "auto"):
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    model = (transformers.AutoModelForCausalLM
             .from_pretrained(cfg["model"], dtype=torch.bfloat16).to(dev).eval())
    lens = jlens.JacobianLens.from_pretrained(cfg["lens_repo"],
                                              filename=cfg["lens_file"])
    blocks = find_blocks(model)
    band = layer_band(lens, *cfg["band"])
    J_dev = {l: lens.jacobians[l].to(dev) for l in band}
    jv = JVecs(lens, model, tok)
    probe = chat_ids(tok, "What is the capital of France?", dev)
    assert noop_check(model, probe, band), "noop_check FAILED"
    print("noop_check PASS", flush=True)
    return model, tok, dev, blocks, band, J_dev, jv, lens


def load_new_p1(seeds=(2, 3)):
    items = []
    for s in seeds:
        items += json.load(open(RES / f"phase1_s{s}.json"))["items"]
    return {"items": items}


def phase_detect(cfg):
    p1 = load_new_p1()
    n_unf = sum(1 for it in p1["items"] if it["unfaithful_cue"])
    print(f"new-only unfaithful n={n_unf}", flush=True)
    model, tok, dev, blocks, band, J_dev, jv, lens = load_full_model(cfg)
    p2 = cached(RES / "phase2_gpu.json",
                lambda: phase2(cfg, model, tok, dev, blocks, band, J_dev, p1))
    print(json.dumps({k: round(v["auc"], 4) for k, v in p2["aucs"].items()}),
          flush=True)


def phase_revert(cfg):
    p1 = load_new_p1()
    model, tok, dev, blocks, band, J_dev, jv, lens = load_full_model(cfg)
    p2 = json.load(open(RES / "phase2_gpu.json"))
    p3 = cached(RES / "phase3_gpu.json",
                lambda: phase3(cfg, model, tok, dev, blocks, band, J_dev, jv,
                               p1, p2))
    print(json.dumps(p3["criterion"]), flush=True)


def boot_ci_auc(pos, neg, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n = len(pos)
    vals = []
    for _ in range(N_BOOT):
        k = rng.integers(0, n, n)
        vals.append(auc(pos[k], neg[k]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def phase_finalize(cfg):
    """metrics_gpu.json: per-seed 2x2 + new-only detection AUCs (with CIs)
    and reversion summary, if the detect/revert phases were run."""
    metrics = {"per_seed": {}}
    for s in (2, 3):
        p = json.load(open(RES / f"phase1_s{s}.json"))
        metrics["per_seed"][f"seed{s}"] = {"counts": p["counts"],
                                           "two_by_two": p["two_by_two"]}
    if (RES / "phase2_gpu.json").exists():
        p2 = json.load(open(RES / "phase2_gpu.json"))

        def stat(row, tag, lens, key):
            d = row[tag]
            return (max(d[lens][str(t)]["p"] for t in row[key])
                    if d else None)

        aucs = {}
        for key, name in (("set_a_tids", "a_surface"),
                          ("set_b_tids", "b_concepts")):
            for lens in ("jlens", "logitlens"):
                pairs = [(stat(r, "hinted", lens, key),
                          stat(r, "control", lens, key))
                         for r in p2["per_item"]]
                pairs = [(p, n) for p, n in pairs
                         if p is not None and n is not None]
                pos = [p for p, _ in pairs]
                neg = [n for _, n in pairs]
                aucs[f"{name}_{lens}"] = {"auc": auc(pos, neg),
                                          "n": len(pairs),
                                          "ci95": boot_ci_auc(pos, neg)}
        metrics["detection_new_only"] = aucs
    if (RES / "phase3_gpu.json").exists():
        p3 = json.load(open(RES / "phase3_gpu.json"))
        metrics["reversion_new_only"] = {"summary": p3["summary"],
                                         "criterion": p3["criterion"]}
    json.dump(metrics, open(RES / "metrics_gpu.json", "w"), indent=1,
              default=float)
    print(json.dumps(metrics, indent=1, default=float)[:4000], flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["pool", "verify1", "gen", "build", "detect",
                             "revert", "finalize"])
    ap.add_argument("--seed", type=int, default=None, choices=[2, 3])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=EXT_POOL_N)
    ap.add_argument("--guard-min", type=float, default=100000.0,
                    help="clean stop after this many minutes (gen phase); "
                         "no task-runner kill on the node, default huge")
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    if args.phase in ("pool", "gen", "build"):
        assert args.seed is not None, "--seed required for this phase"
    if args.phase == "pool":
        p = build_pool(cfg, args.seed)
        print(f"pool seed {args.seed}: {len(p['pool'])} items, "
              f"{p['n_skipped_overlap_or_dup']} skipped, "
              f"{p['n_blacklist_keys']} blacklisted keys", flush=True)
    elif args.phase == "verify1":
        phase_verify1(cfg)
    elif args.phase == "gen":
        phase_gen(cfg, args)
    elif args.phase == "build":
        phase_build(cfg, args.seed)
    elif args.phase == "detect":
        phase_detect(cfg)
    elif args.phase == "revert":
        phase_revert(cfg)
    elif args.phase == "finalize":
        phase_finalize(cfg)
    print(f"phase {args.phase} wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
