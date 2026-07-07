"""Sanity anchor: multihop intermediate-concept readout on Qwen3.5-4B.

Replicates the paper's qualitative claim before any novel work: for
'Fact: The language spoken in the country where the Amazon River ends is',
the intermediate concept (Brazil) should appear in mid-layer J-lens readouts at
the final prompt position, and the answer (Portuguese) toward the top layers.
"""
import json, os, time, sys, platform
import torch, transformers, jlens

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.environ.get("JLENS_REPO", os.path.join(HERE, "..", "vendor", "jacobian-lens"))
OUT = os.path.join(HERE, "results", "qwen35_4b_readout.json")
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
MODEL = "Qwen/Qwen3.5-4B"
TOPK = 10
N_ITEMS = 8

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
t0 = time.time()
lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=LENS_FILE)
print(f"lens loaded {time.time()-t0:.0f}s; {len(lens.jacobians)} layers, d_model={lens.d_model}", flush=True)

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model = jlens.from_hf(hf, tok)
print(f"model on {device} {time.time()-t0:.0f}s", flush=True)

items = json.load(open(os.path.join(VENDOR, "data/experiments/probe-swap.json")))["items"][:N_ITEMS]
results = []
for it in items:
    t1 = time.time()
    lens_logits, model_logits, _ = lens.apply(model, it["prompt"], positions=[-1])
    inter_ids = tok.encode(" " + it["intermediate"], add_special_tokens=False)
    ans_ids = tok.encode(" " + it["answer"], add_special_tokens=False)
    per_layer = {}
    inter_layers, ans_layers = [], []
    for layer in sorted(lens_logits):
        top = lens_logits[layer][0].topk(TOPK).indices.tolist()
        per_layer[layer] = [tok.decode([t]) for t in top]
        if inter_ids and inter_ids[0] in top:
            inter_layers.append(layer)
        if ans_ids and ans_ids[0] in top:
            ans_layers.append(layer)
    results.append({
        "name": it["name"],
        "intermediate": it["intermediate"],
        "answer": it["answer"],
        "intermediate_top10_layers": inter_layers,
        "answer_top10_layers": ans_layers,
        "model_top5": [tok.decode([t]) for t in model_logits[0].topk(5).indices],
        "per_layer_top": {k: v for k, v in per_layer.items() if k % 4 == 0},
        "wall_s": round(time.time() - t1, 1),
    })
    print(f"{it['name']}: intermediate@{inter_layers} answer@{ans_layers} "
          f"model_top1={results[-1]['model_top5'][0]!r} ({results[-1]['wall_s']}s)", flush=True)

manifest = {
    "model": MODEL, "lens": LENS_FILE, "device": device,
    "torch": torch.__version__, "transformers": transformers.__version__,
    "platform": platform.platform(), "n_items": len(items),
    "total_wall_s": round(time.time() - t0, 1),
}
json.dump({"manifest": manifest, "results": results}, open(OUT, "w"), indent=1)
hits = sum(1 for r in results if r["intermediate_top10_layers"])
print(f"\nDONE: intermediate concept in top-{TOPK} at some layer for {hits}/{len(results)} items -> {OUT}", flush=True)
