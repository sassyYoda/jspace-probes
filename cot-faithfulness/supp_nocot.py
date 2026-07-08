"""POST-HOC supplementary arm (not pre-registered): no-CoT direct-answer probe.

Phase 3 was inert because the answer forward pass copies a conclusion the CoT
already states (8/16 truncated CoTs literally contain '(hint)'). Here the copy
shortcut is removed: context = hinted chat prompt + forced 'Answer: (' with no
CoT. Measure the letter under baseline and under the same per-item ablations
(real hint-linked top-k, random k x 2 seeds, content top-k), mid band, all
positions. Also records the unhinted no-CoT baseline letter.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
sys.path.insert(0, str(ROOT / "../harness"))
sys.path.insert(0, str(VENDOR))

import jlens
from jspace_interventions import InterventionHooks, JVecs, find_blocks, layer_band, noop_check
from run import LETTERS, chat_ids, mcq_text, read_answer

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
jv = JVecs(lens, model, tok)

p1 = json.load(open(ROOT / "results/phase1.json"))
p3 = json.load(open(ROOT / "results/phase3.json"))
p3rows = {r["idx"]: r for r in p3["items"]}
unf = [it for it in p1["items"] if it["unfaithful_cue"] and it["idx"] in p3rows]

probe = chat_ids(tok, "What is the capital of France?", dev)
assert noop_check(model, probe, band), "noop_check FAILED"

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
        return torch.cat([pids, torch.tensor([prefix_ids], device=dev)], dim=1)

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
    out_items.append({"idx": it["idx"], "cue": cue,
                      "correct": LETTERS[it["label"]],
                      "hint": h["hint_letter"], "outcomes": res})
    print(f"[supp {i+1}/{len(unf)}] idx={it['idx']} correct="
          f"{LETTERS[it['label']]} hint={h['hint_letter']} baseH={base_h} "
          f"baseU={base_u} real={res['real']['letter']} "
          f"rand={res['random0']['letter']}/{res['random1']['letter']} "
          f"content={res['content']['letter']}", flush=True)

# summarize over items where the no-CoT hinted baseline actually takes the hint
biased = [o for o in out_items
          if o["outcomes"]["baseline_hinted"] == o["hint"]]
def frac(cond, items):
    n = len(items)
    if n == 0:
        return None
    rev = sum(1 for o in items if o["outcomes"][cond]["letter"] == o["correct"])
    unch = sum(1 for o in items if o["outcomes"][cond]["letter"] == o["hint"])
    destr = sum(1 for o in items if not o["outcomes"][cond]["top1_is_letter"])
    return {"n": n, "reversion": rev / n, "unchanged": unch / n,
            "other_letter": (n - rev - unch) / n, "disruption": (n - rev - unch) / n,
            "destroyed_top1_not_letter": destr / n}

summary = {c: frac(c, biased) for c in
           ["real", "content"] + [f"random{s}" for s in cfg["random_seeds"]]}
out = {
    "note": "POST-HOC, not pre-registered. No-CoT direct-answer probe: copy "
            "shortcut removed. Summary restricted to items where the no-CoT "
            "hinted baseline equals the hinted letter.",
    "n_items": len(out_items),
    "n_nocot_baseline_takes_hint": len(biased),
    "n_nocot_unhinted_correct": sum(
        1 for o in out_items if o["outcomes"]["baseline_unhinted"] == o["correct"]),
    "summary_on_hint_taking_items": summary,
    "items": out_items,
    "wall_clock_s": round(time.time() - t0, 1),
}
json.dump(out, open(ROOT / "results/supp_nocot.json", "w"), indent=1)
m = json.load(open(ROOT / "results/metrics.json"))
m["phase3_supplementary_nocot"] = {k: v for k, v in out.items() if k != "items"}
json.dump(m, open(ROOT / "results/metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=2))
