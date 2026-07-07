"""Causal validation of the intervention harness on the paper's probe-swap items.

Swap j_intermediate -> j_swap_to at all positions over a mid-layer band; flip =
greedy continuation becomes swap_answer. Controls: angle-matched random
directions (3 seeds) and the same swap at an early band.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import JacobianLens
from jspace_interventions import (
    InterventionHooks,
    JVecs,
    find_blocks,
    layer_band,
    noop_check,
    random_pairs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(
    os.environ.get("JLENS_REPO", os.path.join(HERE, "..", "vendor", "jacobian-lens")),
    "data/experiments/probe-swap.json",
)
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def cont_ids(tok, word):
    ids = tok.encode(" " + word.strip())
    for k in range(1, len(ids) + 1):
        if tok.decode(ids[:k]).strip():
            return ids[:k]
    return ids


@torch.no_grad()
def scores(model, tok, ids, conts):
    cache = {}

    def last_logprobs(prefix):
        if prefix not in cache:
            full = ids
            if prefix:
                ext = torch.tensor([list(prefix)], device=ids.device)
                full = torch.cat([ids, ext], dim=1)
            cache[prefix] = model(full).logits[0, -1].float().log_softmax(-1)
        return cache[prefix]

    out = {"greedy": tok.decode([last_logprobs(()).argmax().item()])}
    for name, cont in conts.items():
        lp, greedy = 0.0, True
        for i, t in enumerate(cont):
            d = last_logprobs(tuple(cont[:i]))
            lp += d[t].item()
            greedy = greedy and d.argmax().item() == t
        out[f"lp_{name}"] = lp
        out[f"greedy_{name}"] = greedy
    return out


def main():
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    model = (
        AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16)
        .to(DEVICE)
        .eval()
    )
    lens = JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens",
        filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    )
    blocks = find_blocks(model)
    assert len(blocks) == 32 and lens.source_layers == list(range(31))
    jv = JVecs(lens, model, tok)

    items = json.load(open(DATA))["items"]
    probe_ids = tok(items[0]["prompt"].rstrip(), return_tensors="pt").input_ids.to(
        DEVICE
    )
    mid = layer_band(lens, 0.4, 0.75)
    early = layer_band(lens, 0.05, 0.2)
    noop_ok = noop_check(model, probe_ids, mid)
    print(f"no-op check (band {mid[0]}-{mid[-1]}): {'PASS' if noop_ok else 'FAIL'}")
    assert noop_ok

    bands = {"mid": mid, "early": early}
    sweep_bands = {
        "band_0.30-0.65": layer_band(lens, 0.30, 0.65),
        "band_0.50-0.85": layer_band(lens, 0.50, 0.85),
        "band_0.25-0.90": layer_band(lens, 0.25, 0.90),
    }
    seeds = [0, 1, 2]

    def conditions_for(item, include_sweep):
        pairs = {
            name: jv.swap_pairs(item["intermediate"], item["swap_to"], band)
            for name, band in bands.items()
        }
        conds = {
            "real_mid": (pairs["mid"], 1.0),
            "real_mid_a2": (pairs["mid"], 2.0),
            "early": (pairs["early"], 1.0),
        }
        for s in seeds:
            conds[f"rand_mid_s{s}"] = (
                random_pairs(mid, pairs["mid"], lens.d_model, seed=s),
                1.0,
            )
        if include_sweep:
            for name, band in sweep_bands.items():
                p = jv.swap_pairs(item["intermediate"], item["swap_to"], band)
                conds[name] = (p, 1.0)
        return conds

    def run(include_sweep):
        results = []
        for item in items:
            t0 = time.time()
            prompt = item["prompt"].rstrip()
            ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
            conts = {
                "answer": cont_ids(tok, item["answer"]),
                "swap_answer": cont_ids(tok, item["swap_answer"]),
            }
            rec = {"name": item["name"], "conditions": {}}
            base = scores(model, tok, ids, conts)
            base_margin = base["lp_swap_answer"] - base["lp_answer"]
            rec["baseline"] = {
                "greedy": base["greedy"],
                "answer_correct": base["greedy_answer"],
                "lp_answer": base["lp_answer"],
                "lp_swap_answer": base["lp_swap_answer"],
            }
            for cname, (pairs, alpha) in conditions_for(item, include_sweep).items():
                with InterventionHooks(blocks) as iv:
                    iv.swap(pairs, positions=None, alpha=alpha)
                    s = scores(model, tok, ids, conts)
                rec["conditions"][cname] = {
                    "greedy": s["greedy"],
                    "flip": s["greedy_swap_answer"],
                    "still_answer": s["greedy_answer"],
                    "margin_shift": (s["lp_swap_answer"] - s["lp_answer"])
                    - base_margin,
                }
            rec["seconds"] = round(time.time() - t0, 2)
            results.append(rec)
            flips = [c for c, v in rec["conditions"].items() if v["flip"]]
            print(f"{item['name']}: base={base['greedy']!r} flips={flips} "
                  f"({rec['seconds']}s)")
        return results

    results = run(include_sweep=False)
    cond_names = list(results[0]["conditions"])

    def rates(results):
        out = {}
        correct = [r for r in results if r["baseline"]["answer_correct"]]
        for c in set().union(*(r["conditions"] for r in results)):
            rs = [r["conditions"][c] for r in results if c in r["conditions"]]
            cs = [r["conditions"][c] for r in correct if c in r["conditions"]]
            out[c] = {
                "flip_rate": round(sum(v["flip"] for v in rs) / len(rs), 3),
                "flip_rate_baseline_correct": round(
                    sum(v["flip"] for v in cs) / len(cs), 3
                ),
                "mean_margin_shift": round(
                    sum(v["margin_shift"] for v in rs) / len(rs), 3
                ),
                "n": len(rs),
                "n_baseline_correct": len(cs),
            }
        return out

    r = rates(results)
    rand_rate = max(r[f"rand_mid_s{s}"]["flip_rate"] for s in seeds)
    if r["real_mid"]["flip_rate"] < max(0.10, 2 * rand_rate):
        print("mid-band flips rare; sweeping bands")
        results = run(include_sweep=True)
        r = rates(results)

    summary = {
        "model": "Qwen/Qwen3.5-4B",
        "n_items": len(items),
        "noop_check": "pass",
        "bands": {k: [v[0], v[-1]] for k, v in {**bands, **sweep_bands}.items()},
        "positions": "all",
        "random_seeds": seeds,
        "flip_rates": r,
        "baseline_accuracy": round(
            sum(x["baseline"]["answer_correct"] for x in results) / len(results), 3
        ),
        "wall_clock_seconds": round(time.time() - t_start, 1),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "validation.json"), "w") as f:
        json.dump({"summary": summary, "items": results}, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
