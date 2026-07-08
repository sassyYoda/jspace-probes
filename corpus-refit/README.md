# Idea G (mini): kill test for Experiment A's shared-corpus confound

Experiment A (`../universality/`) found J-space relational geometry aligns
across model families (RSA 0.75–0.87 vs unembedding baseline 0.05–0.61). Its stated
confound: all three pre-fitted lenses share the wikitext-103 fitting corpus, so the
alignment could partly reflect shared corpus statistics rather than model-intrinsic
workspace structure.

**Kill test**: refit the Qwen3-1.7B lens on a maximally different corpus (Python
code), then re-run A's cross-model comparison with the code-corpus Qwen lens against
the other models' *wikitext* lenses. If the shared corpus was doing the work, the
cross-model RSA should collapse toward the unembedding baseline.

## Method

- **Refit**: `fit_lens.py` fits `jlens.fit()` (from the Jacobian-lens repo, located
  via `JLENS_REPO` or `vendor/jacobian-lens`) for
  Qwen/Qwen3-1.7B (bf16, MPS) on the first 100 documents of
  `codeparrot/codeparrot-clean-valid` (streaming) whose first-128-token window is
  full — 128-token sequences, matching the paper's fitting recipe.
  `bigcode/the-stack-smol` was the first choice but is gated on the Hub.
  Only the 9 source layers A's `frac_to_layer` maps fractions 0.1–0.9 onto for
  qwen3-1.7b are fitted ([3, 6, 8, 11, 14, 17, 20, 22, 25]) — the gradients come
  from the same backward passes, so this changes storage, not estimator.
  Per-2-prompt atomic checkpointing (`results/fit_checkpoint.pt`, ~150 MB) makes the
  run resumable; a first-3-prompts timing probe would have cut n_prompts to a
  convergence floor of 30 if the projection exceeded 6 h (it did not).
- **Comparison**: `compare.py` imports A's `run.py` unchanged (same 507 concepts,
  same fractions, same norm_scaled/raw variants, same unembed / shuffled /
  random-orthogonal baselines, seeds 0/[0,1,2]) and swaps only the qwen lens file.
  The untouched gemma-2-2b|gpt2-small pair doubles as an exact-reproduction sanity
  check. Pipeline validated end-to-end by running it with a 9-layer subset of the
  *wikitext* lens as a fake refit: reproduces A's numbers to 0.000.
- **Bonus (Idea G stability datapoint)**: same-model cross-corpus RSA — qwen
  wikitext lens vs qwen code lens on identical concepts/layers.

## Pre-registered read

At mid fractions (0.4–0.7), for both qwen pairs: if |RSA_code − RSA_wiki| ≤ 0.1 and
the code-lens RSA still clears the unembedding baseline by more than the shuffled
spread, the confound is dead. If RSA collapses toward the unembedding baseline, the
shared corpus was doing real work and A must be reframed.

## Reproduce

```bash
# from the repo root, deps installed (see top-level README)
python corpus-refit/fit_lens.py   # ~55 min on Apple MPS; resumable, re-run if interrupted
python corpus-refit/compare.py    # ~30 s
```

The fitted artifacts are too large for the repo and are not shipped (`*.pt` is
gitignored): the refit lens `results/qwen3-1.7b_codeparrot_lens.pt` (~72 MB) and
`results/fit_checkpoint.pt` (~150 MB) are regenerated exactly by `fit_lens.py`
(the fit is seeded and the 100-prompt list is pinned in `results/prompts.json`).
`compare.py` needs the refit lens, so run `fit_lens.py` first.

## Results (507 concepts, norm_scaled primary)

Fit: 100/100 codeparrot prompts, 37 s/prompt median on MPS (~62 min compute, run
concurrently with another MPS user). Convergence: per-prompt relative shift of the
running mean (`max_d_mean`) fell 0.65 (n=2) → 0.17 (n=10) → 0.065 (n=25) → 0.021
(n=75) → 0.020 (n=100), the expected ~1/n decay; per-prompt norm max‖J‖/√d median
9.6 with two heavy-tail outliers (61.7 at prompt 53, 39.3 at prompt 97) that
transiently bump the shift but wash out in the mean. `results/fit_convergence_summary.json`.

Cross-model RSA at mid fractions, code-corpus Qwen lens vs (A) wikitext lenses:

| pair | RSA unembed | lens | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---|---|---|---|---|---|
| qwen3-1.7b vs gemma-2-2b | 0.591 | wikitext (A) | 0.815 | 0.783 | 0.788 | 0.752 |
| | | **code (G)** | **0.777** | **0.741** | **0.790** | **0.734** |
| qwen3-1.7b vs gpt2-small | 0.608 | wikitext (A) | 0.873 | 0.868 | 0.871 | 0.793 |
| | | **code (G)** | **0.804** | **0.800** | **0.817** | **0.731** |

Max |delta| = 0.069 (qwen–gpt2 @0.4/0.5); every mid fraction within the 0.1
tolerance and 0.12–0.23 above the unembedding baseline (shuffled floor ~0 ± 0.02).
The untouched gemma-2-2b|gpt2-small pair reproduces A to 0.000 (sanity), and the
pipeline itself, fed a 9-layer subset of the wikitext lens as a fake refit,
reproduces A exactly. The `raw` variant collapses for Qwen pairs (0.06–0.08)
exactly as in A — a property of Qwen3's non-uniform final-norm weight, independent
of fitting corpus.

Same-model stability (qwen wikitext lens vs qwen code lens, same concepts/layers):
RSA 0.83–0.87 at fractions 0.1–0.5, rising to 0.98 by 0.9. The lens is
corpus-sensitive but far more corpus-stable than it is unembedding-like (code-lens
vs unembed RSA is only 0.41–0.62 at fractions ≤0.6). Idea G's first datapoint:
corpus swap perturbs J-space geometry by roughly the same margin (~0.13 RSA) at
mid layers as it perturbs the cross-model comparison, converging to
near-identical at late layers.

## Verdict

**The shared-corpus confound is dead.** Pre-registered criterion passes at all
mid fractions for both refit pairs: refitting the Qwen lens on Python code —
maximally different from wikitext — moves cross-model RSA by at most 0.07, leaving
it 0.12–0.23 above the unembedding baseline. The cross-model alignment Experiment A
found is a property of the models' J-spaces, not of the shared fitting corpus.
Honest caveats: (1) one model refitted, one alternative corpus, n=100 prompts —
the full Idea G grid (multiple corpora x models) remains for the funded phase;
(2) the small consistent negative deltas (−0.04 to −0.07) show the corpus
contributes a *little* shared structure, consistent with the same-model stability
numbers — corpus choice is a second-order effect, not the driver.
