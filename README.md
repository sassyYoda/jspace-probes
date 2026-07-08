# jspace-probes

Preliminary open-weights probes of the **J-space / global workspace** reported in
Anthropic's *Verbalizable Representations Form a Global Workspace in Language Models*
(Gurnee, Sofroniew, Lindsey et al., Transformer Circuits, July 2026). Every headline
result in that paper was measured on closed Claude models. This repo is a small,
runnable start on the question **which J-space claims are real, model-agnostic
properties of transformers, and which are Claude-specific or artifacts of the lens.**

These are early signal-check scripts, not a finished paper. They run on a single
consumer GPU or an Apple-Silicon laptop in seconds to minutes, using the pre-fitted
Jacobian lenses that [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens)
hosts for 38 open models, so no lens fitting is required to reproduce them.
Taken together the probes are converging on a unified picture: the J-space behaves
as a universal (family- and corpus-robust), ignition-gated, competitive buffer of
verbalizable content — and nothing else (probes for covert mechanisms and per-item
retention come up empty).

## What's here

| Dir | What it does | Headline |
|---|---|---|
| `harness/` | A validated J-space intervention harness (concept-swap and directional-ablation forward hooks) plus a causal validation on the paper's own 90 probe-swap items. | Swap-flip **58.6%** on baseline-correct items (paper: 54–70% on Claude) vs **1.7%** for angle-matched random controls, on Qwen3.5-4B. |
| `universality/` | Cross-family comparison of J-lens vector geometry (RSA / SVCCA) with the tokenizer-geometry confound baselines. | J-space relational geometry aligns across Qwen3 / Gemma-2 / GPT-2 at **RSA 0.75–0.87**, vs **0.05–0.61** for the unembedding baseline and **~0** shuffled floor. |
| `selectivity/` | J-space ablation with a dose-response in k and a task battery, with matched random and bottom-k controls. | Ablating active J-space contents is monotone in k and destroys rhyme, multi-hop reasoning, and single-hop facts (34–63 points beyond controls) while sparing MCQA/sentiment. Single-hop facts were predicted spared; the real axis is "answer-is-a-generated-content-word", not the paper's task split. |
| `sanity/` | A layer-by-layer J-lens readout that reproduces the paper's mid-layer intermediate-concept structure before any intervention. | Answers surface in top layers; intermediates in the mid-band. |
| `ignition/` | Presence-vs-stimulus dose-response on Qwen3.5-4B: 8-step prompt evidence ladders + graded j-vector injection with AICc/AIC model comparison, norm-matched random controls, and a paired competition probe. | Activation-side entry is step-like for **40/40** concepts by AIC (the stricter pre-registered AICc conjunction fails on the prompt-side arm — staircases, not smooth sigmoids), and competition is one-in-one-out: in 19/19 valid pairs the injected concept evicts the evoked one below the random control. |
| `ignition-gemma/` | Cross-family replication of `ignition/` on gemma-2-2b — same design, same pre-registered criteria, same controls, same 40 concepts. | The causal core replicates: sigmoid beats linear **40/40** by AIC and one-in-one-out competition holds in **27/27** valid pairs (specific eviction beyond the random control, sign-test p ≈ 7.5e-9); the Qwen-only "90% by AICc" headline does not replicate (27.5% — wider transitions meet AICc's small-sample penalty). |
| `corpus-refit/` | Refits the Qwen3-1.7B lens on Python code (codeparrot) and re-runs the universality comparison, to kill the shared-wikitext-fitting-corpus confound. | Universality survives the corpus swap: max RSA delta **0.069** at mid fractions, still 0.12–0.23 above the unembedding baseline — the shared-corpus confound is dead (one model, one alternative corpus so far). |
| `cot-faithfulness/` | Does mid-band J-space reveal what unfaithful CoT hides? Turpin-style hinted MMLU on Qwen2.5-7B-Instruct: detection AUCs (J-lens vs logit lens) + causal reversion by hint-direction ablation. | Boundary negative: hint *surface* tokens detect at AUC **1.00** (logit lens 0.43) but never-verbalized social-influence concepts sit at **0.46** (chance), and ablation reverts **0/16** biased answers — the workspace holds verbalized surface content, not the covert bias mechanism. |
| `capacity/` | Workspace-capacity psychophysics on Qwen3.5-4B: load-N word lists (N=2–64), J-lens occupancy against a matched non-listed-word null, yes/no recall d', plateau-vs-linear fits. | Boundary negative: per-word retention presence is **null** (listed-vs-control AUC 0.48, occupancy never rises above the false-positive floor) while recognition stays at **ceiling** through N≈32–64 — the capacity claim is not behaviorally testable with a token-probability readout. |

Each subdirectory has its own README with the exact method, controls, and numbers,
and a `results/` with the metrics JSON and figures behind the table above.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv venv
pip install -r requirements.txt

# The scripts read the paper's prompt-set JSON from a local clone of the
# Jacobian-lens reference repo (Apache-2.0). Point JLENS_REPO at it, or clone
# it to ./vendor/jacobian-lens (the default the scripts look for):
git clone https://github.com/anthropics/jacobian-lens vendor/jacobian-lens
```

Then, from the repo root:

```bash
python harness/validate_probe_swap.py     # causal validation (Qwen3.5-4B)
python universality/run.py                # cross-family RSA/SVCCA
python sanity/run_qwen_readout.py         # layer-by-layer readout
```

Device auto-detects CUDA / Apple MPS / CPU. Set `JLENS_REPO=/path/to/jacobian-lens`
if you cloned the reference repo elsewhere.

## Notes

- The pre-fitted lenses were fit on WikiText; `corpus-refit/` re-fits the Qwen3-1.7B
  lens on Python code and shows the universality result survives, killing the
  shared-fitting-corpus confound stated in `universality/README.md` (for one model
  and one alternative corpus; the full grid is future work).
- This is functional / access-consciousness territory only. Nothing here bears on
  phenomenal consciousness or moral status.

## Credit

The Jacobian lens, the `jlens` package, and the prompt-set data are Anthropic's
([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache-2.0).
Pre-fitted lenses via [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens).
This repo's own code is MIT-licensed.
