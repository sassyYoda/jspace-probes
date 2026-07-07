# J-space intervention harness

Shared machinery for the J-space experiments: build j-vectors from a fitted
Jacobian lens (`j_v^l = normalize(J_l^T u_v)`), and intervene on the residual
stream via forward hooks on the decoder blocks.

- `jspace_interventions.py` — `JVecs` (j-vector construction + cache),
  `InterventionHooks` (context manager; `swap` / `ablate` / `noop` ops over a
  layer band and position range), `random_pairs` (angle-matched random control
  directions), `layer_band`, `noop_check`.
- `validate_probe_swap.py` — causal validation on the paper's 90 probe-swap
  items (`vendor/jacobian-lens/data/experiments/probe-swap.json`).
- `results/validation.json` — per-item outcomes and summary.

Swap: `h' = h + alpha * (h·ĵ_from)(ĵ_to − ĵ_from)`. Ablate: project out a
QR-orthonormalized direction set. Math in fp32, cast back to model dtype.
Hooks sit on `model.model.layers[l]`, whose output is the tensor the lens was
fitted on (same hook point as `jlens.hooks.ActivationRecorder`).

## Validation (Qwen/Qwen3.5-4B, mps, bf16)

Swap `j_intermediate -> j_swap_to` at all positions, alpha 1.0. Mid band =
lens layers 12-22 (fractions 0.4-0.75 of 31), early band = layers 2-6.
Flip = greedy continuation becomes `swap_answer`'s first token(s). Margin
shift = change in `logP(swap_answer) − logP(answer)` vs baseline.

| condition | flip rate (all 90) | flip rate (58 baseline-correct) | mean margin shift |
|---|---|---|---|
| real swap, mid band | 0.456 | 0.586 | +9.06 |
| real swap, mid, alpha 2.0 | 0.444 | 0.586 | +9.40 |
| real swap, early band | 0.211 | 0.276 | +2.60 |
| random swap, mid (seeds 0/1/2) | 0.044 each | 0.017 each | ~0.0 |

No-op hook check: identity intervention fires on every mid-band layer and
reproduces baseline logits bit-exactly (PASS). Baseline task accuracy 0.644.
Wall clock 68 s for 90 items x 6 conditions (~0.75 s/item).

Verdict: validated. Real-swap flips (58.6% on baseline-correct items) land in
the paper's 54-70% range and exceed the random control (1.7%) by ~34x; the
early band is markedly weaker than mid, as the paper predicts. The default
band worked, so the fallback band sweep never triggered.

Architecture notes: Qwen3.5-4B has 32 decoder blocks but the lens covers
layers 0-30 only; blocks are uniform `Qwen3_5DecoderLayer` modules despite the
hybrid linear/full attention pattern (full attention every 4th layer), and
each returns a bare hidden-states tensor, not a tuple.
