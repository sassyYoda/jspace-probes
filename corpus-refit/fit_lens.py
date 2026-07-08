"""Refit the Qwen3-1.7B Jacobian lens on a code corpus (Idea G mini / A confound).

Resumable: per-prompt checkpointing via jlens.fit(checkpoint_path=...). Re-running
this script resumes from results/fit_checkpoint.pt. The prompt list is materialized
once to results/prompts.json so resume indices stay valid across runs.

Usage:
  python fit_lens.py --bench     # time one prompt at dim_batch 8 and 16, print, exit
  python fit_lens.py             # budgeted fit (measures first 3 prompts, extrapolates)
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
sys.path.insert(0, str(VENDOR))
import jlens
from jlens.fitting import jacobian_for_prompt

RESULTS = ROOT / "results"
CKPT = RESULTS / "fit_checkpoint.pt"
PROMPTS_JSON = RESULTS / "prompts.json"
LOG = RESULTS / "fit_log.txt"


def setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG)):
        h.setFormatter(fmt)
        root.addHandler(h)


def build_prompts(cfg, tok):
    """First n_prompts_target streamed docs whose first-128-token window is full."""
    if PROMPTS_JSON.exists():
        prompts = json.load(open(PROMPTS_JSON))["prompts"]
        logging.info("loaded %d cached prompts from %s", len(prompts), PROMPTS_JSON)
        return prompts
    from datasets import load_dataset

    ccfg = cfg["corpus"]
    ds = load_dataset(ccfg["hf_dataset"], split=ccfg["split"],
                      streaming=ccfg["streaming"])
    prompts, scanned = [], 0
    for ex in ds:
        scanned += 1
        text = ex[ccfg["text_field"]]
        ids = tok(text, truncation=True, max_length=ccfg["max_seq_len"]).input_ids
        if len(ids) >= ccfg["max_seq_len"]:
            # store the decoded 128-token window, not the whole file, so the
            # fit's own tokenize-and-truncate sees the same 128 tokens
            prompts.append(tok.decode(ids))
        if len(prompts) >= ccfg["n_prompts_target"]:
            break
    json.dump({"dataset": ccfg["hf_dataset"], "split": ccfg["split"],
               "docs_scanned": scanned, "prompts": prompts},
              open(PROMPTS_JSON, "w"))
    logging.info("built %d prompts (scanned %d docs) -> %s",
                 len(prompts), scanned, PROMPTS_JSON)
    return prompts


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg, fcfg = cfg["refit_model"], cfg["fit"]
    if fcfg["device"] == "auto":
        fcfg["device"] = ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info("loading %s (%s, %s)", mcfg["hf_repo"], fcfg["dtype"], fcfg["device"])
    tok = AutoTokenizer.from_pretrained(mcfg["hf_repo"], revision=mcfg["hf_revision"])
    hf = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_repo"], revision=mcfg["hf_revision"],
        dtype=getattr(torch, fcfg["dtype"])).to(fcfg["device"])
    model = jlens.from_hf(hf, tok)  # mutates hf in place; we own it
    logging.info("wrapped: %r", model)
    return model, tok


def bench(model, prompts, cfg):
    fcfg = cfg["fit"]
    for db in (8, 16):
        t0 = time.perf_counter()
        _, seq_len, n_valid = jacobian_for_prompt(
            model, prompts[0], fcfg["source_layers"],
            dim_batch=db, max_seq_len=cfg["corpus"]["max_seq_len"])
        dt = time.perf_counter() - t0
        logging.info("BENCH dim_batch=%d: %.1fs/prompt (seq_len=%d, n_valid=%d)",
                     db, dt, seq_len, n_valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    setup_logging()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    torch.manual_seed(cfg["seed"])
    fcfg = cfg["fit"]

    model, tok = load_model(cfg)
    prompts = build_prompts(cfg, tok)

    if args.bench:
        bench(model, prompts, cfg)
        return

    fit_kwargs = dict(source_layers=fcfg["source_layers"],
                      dim_batch=fcfg["dim_batch"],
                      max_seq_len=cfg["corpus"]["max_seq_len"],
                      checkpoint_path=str(CKPT),
                      checkpoint_every=fcfg["checkpoint_every"])

    # Phase 1: time-budget measurement on the first bench_prompts prompts
    # (checkpointed, so this work is never repeated).
    n_bench = fcfg["bench_prompts"]
    already = 0
    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu", weights_only=True)
        already = state["next_idx"]
        logging.info("existing checkpoint: next_idx=%d", already)
    if already < n_bench:
        t0 = time.perf_counter()
        jlens.fit(model, prompts[:n_bench], **fit_kwargs)
        s_per_prompt = (time.perf_counter() - t0) / (n_bench - already)
    else:
        s_per_prompt = None  # resuming a run whose budget was already decided

    decision_file = RESULTS / "budget_decision.json"
    if decision_file.exists():
        n_fit = json.load(open(decision_file))["n_fit"]
        logging.info("resuming with previously decided n_fit=%d", n_fit)
    else:
        budget_s = fcfg["time_budget_hours"] * 3600
        projected = s_per_prompt * len(prompts)
        n_fit = len(prompts)
        if projected > budget_s:
            n_fit = max(cfg["corpus"]["n_prompts_floor"], int(budget_s / s_per_prompt))
            n_fit = min(n_fit, len(prompts))
        logging.info("budget: %.1fs/prompt, projected %.1fh for %d prompts "
                     "(budget %.1fh) -> fitting %d prompts",
                     s_per_prompt, projected / 3600, len(prompts),
                     fcfg["time_budget_hours"], n_fit)
        json.dump({"s_per_prompt": s_per_prompt, "n_fit": n_fit,
                   "projected_hours_100": projected / 3600},
                  open(decision_file, "w"))

    # Phase 2: the real fit (resumes from the phase-1 checkpoint).
    t0 = time.perf_counter()
    lens = jlens.fit(model, prompts[:n_fit], **fit_kwargs)
    logging.info("fit complete: %r (phase-2 wall %.1fs)", lens,
                 time.perf_counter() - t0)

    out = ROOT / cfg["comparison"]["refit_lens_file"]
    lens.save(str(out))
    logging.info("saved lens -> %s", out)


if __name__ == "__main__":
    main()
