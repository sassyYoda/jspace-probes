"""Part 2 fits (node GPU): refit lenses on new (model, corpus) cells.

  --fit qwen3-1.7b-chat    Qwen3-1.7B on ultrachat_200k user-turn text
  --fit gemma-2-2b-code    gemma-2-2b on codeparrot/codeparrot-clean-valid

100 prompts x 128 tokens, checkpoint_every=2 (resumable: re-run to resume).
GPU-heavy (the measured fits took 12-17 min each on an MI300X); run fits
sequentially, from the repo root:

  python universality-scaled/fit_corpus.py --fit qwen3-1.7b-chat
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens"))
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
import jlens  # noqa: E402

RESULTS = ROOT / "results"


def setup_logging(tag):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(RESULTS / f"fit_{tag}_log.txt")):
        h.setFormatter(fmt)
        root.addHandler(h)


def build_prompts(fcfg, tok, tag):
    """First n_prompts_target streamed docs whose first-128-token window is full."""
    pj = RESULTS / f"prompts_{tag}.json"
    if pj.exists():
        prompts = json.load(open(pj))["prompts"]
        logging.info("loaded %d cached prompts from %s", len(prompts), pj)
        return prompts
    from datasets import load_dataset

    ccfg = fcfg["corpus"]
    args_ds = [ccfg["hf_dataset"]] + ([ccfg["hf_config"]] if ccfg.get("hf_config")
                                      else [])
    ds = load_dataset(*args_ds, split=ccfg["split"], streaming=ccfg["streaming"])
    prompts, scanned = [], 0
    for ex in ds:
        scanned += 1
        if ccfg["user_turns_only"]:
            turns = [m["content"] for m in ex["messages"] if m["role"] == "user"]
            text = "\n\n".join(turns)
        else:
            text = ex[ccfg["text_field"]]
        ids = tok(text, truncation=True, max_length=ccfg["max_seq_len"],
                  add_special_tokens=False).input_ids
        if len(ids) >= ccfg["max_seq_len"]:
            # store the decoded first-128-token window so the fit's own
            # tokenize-and-truncate sees the same tokens
            prompts.append(tok.decode(ids))
        if len(prompts) >= ccfg["n_prompts_target"]:
            break
    json.dump({"dataset": ccfg["hf_dataset"], "split": ccfg["split"],
               "user_turns_only": ccfg["user_turns_only"],
               "docs_scanned": scanned, "prompts": prompts}, open(pj, "w"))
    logging.info("built %d prompts (scanned %d docs) -> %s", len(prompts), scanned, pj)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True,
                    choices=["qwen3-1.7b-chat", "gemma-2-2b-code", "gemma-2-2b-wiki"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    setup_logging(args.fit)

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    torch.manual_seed(cfg["seed"])
    fcfg = cfg["fits"][args.fit]
    mcfg = cfg["models"][fcfg["model"]]
    gcfg = cfg["fit"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.info("loading %s (%s, %s)", mcfg["hf_repo"], gcfg["dtype"], gcfg["device"])
    tok = AutoTokenizer.from_pretrained(mcfg["hf_repo"], revision=mcfg["hf_revision"])
    prompts = build_prompts(fcfg, tok, args.fit)  # before model load: fail fast

    hf = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_repo"], revision=mcfg["hf_revision"],
        dtype=getattr(torch, gcfg["dtype"])).to(gcfg["device"])
    model = jlens.from_hf(hf, tok)  # mutates hf in place; we own it
    logging.info("wrapped: %r", model)

    ckpt = RESULTS / f"fit_{args.fit}_checkpoint.pt"
    t0 = time.perf_counter()
    lens = jlens.fit(model, prompts,
                     source_layers=fcfg["source_layers"],
                     dim_batch=gcfg["dim_batch"],
                     max_seq_len=fcfg["corpus"]["max_seq_len"],
                     checkpoint_path=str(ckpt),
                     checkpoint_every=gcfg["checkpoint_every"])
    logging.info("fit complete: %r (wall %.1fs)", lens, time.perf_counter() - t0)
    out = ROOT / fcfg["lens_out"]
    lens.save(str(out))
    logging.info("saved lens -> %s", out)


if __name__ == "__main__":
    main()
