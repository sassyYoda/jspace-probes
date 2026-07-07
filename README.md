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

## What's here

| Dir | What it does | Headline |
|---|---|---|
| `harness/` | A validated J-space intervention harness (concept-swap and directional-ablation forward hooks) plus a causal validation on the paper's own 90 probe-swap items. | Swap-flip **58.6%** on baseline-correct items (paper: 54–70% on Claude) vs **1.7%** for angle-matched random controls, on Qwen3.5-4B. |
| `universality/` | Cross-family comparison of J-lens vector geometry (RSA / SVCCA) with the tokenizer-geometry confound baselines. | J-space relational geometry aligns across Qwen3 / Gemma-2 / GPT-2 at **RSA 0.75–0.87**, vs **0.05–0.61** for the unembedding baseline and **~0** shuffled floor. |
| `selectivity/` | J-space ablation with a dose-response in k and a task battery, with matched random and bottom-k controls. | Ablating active J-space contents is monotone in k and destroys rhyme, multi-hop reasoning, and single-hop facts (34–63 points beyond controls) while sparing MCQA/sentiment. Single-hop facts were predicted spared; the real axis is "answer-is-a-generated-content-word", not the paper's task split. |
| `sanity/` | A layer-by-layer J-lens readout that reproduces the paper's mid-layer intermediate-concept structure before any intervention. | Answers surface in top layers; intermediates in the mid-band. |

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

- The pre-fitted lenses were fit on WikiText; a corpus-robustness re-fit is future
  work, and the universality result's main open confound (shared fitting corpus) is
  stated in `universality/README.md`.
- This is functional / access-consciousness territory only. Nothing here bears on
  phenomenal consciousness or moral status.

## Credit

The Jacobian lens, the `jlens` package, and the prompt-set data are Anthropic's
([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache-2.0).
Pre-fitted lenses via [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens).
This repo's own code is MIT-licensed.
