# Idea B cross-family replication — does the J-space ignite on gemma-2-2b?

Replication of the flagship ignition/competition experiment
(`../ignition/`, Qwen3.5-4B) on a second model family:
google/gemma-2-2b (mps, bf16, eager attention), lens
`gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt` from
`neuronpedia/jacobian-lens`. Same design, same pre-registered criteria, same
controls, same 40 concepts (all 40 + all 5 unrelated probes survive gemma's
tokenizer as single tokens with leading space — no top-up needed). Shared
machinery (ladders, fits, AICc/AIC, hooks, summaries) is imported from the Qwen
run's modules; only model-family-specific pieces are adapted (see Deviations).

Band recomputed as fractions 0.40-0.75 of the lens's 25 layers (0-24 of 26):
layers 10-18; injection at the band midpoint L14; downstream readout layers 15-18.
Mean content-token residual norm at L14 = 221.7.

## Results (side-by-side with Qwen3.5-4B)

Full metrics in `results/metrics.json`; curves in `results/ignition_curves.png`.
Wall clock 121 s (40 concepts, ~1,100 forward+readout passes).

| | Qwen3.5-4B arm 1 | gemma-2-2b arm 1 | Qwen3.5-4B arm 2 | gemma-2-2b arm 2 |
|---|---|---|---|---|
| sigmoid preferred (AICc, pre-reg) | 11/40 = 27.5% | 6/40 = 15% | **36/40 = 90%** | **11/40 = 27.5%** |
| sigmoid preferred (plain AIC, secondary) | 33/40 = 82.5% | 26/40 = 65% | 40/40 = 100% | 40/40 = 100% |
| median dAICc (lin - sig) | -11.5 | -19.0 | +14.0 | -3.4 |
| median 10-90% width / range (sig-preferred) | 0.11 | 0.11 | 0.21 | 0.28 |
| pre-reg conjunction (>60% AICc both arms + width) | FAIL (arm 1) | FAIL (both arms) | | |

| competition probe | Qwen3.5-4B | gemma-2-2b |
|---|---|---|
| valid pairs (Y evoked > 0.05) | 19/30 | 27/30 |
| one-in-one-out (X ignites, Y drops > 0.05) | 19/19 | 27/27 |
| coexistence cases | 0 | 0 |
| median X after injection | 0.98 | 0.63 |
| median Y after real X injection | ~0.000 | ~0.000 |
| median dY, real X / random control | -0.25 / -0.17 | -0.096 / -0.065 |
| Y below random control (specific excess) | 19/19, median -0.066, p ~ 4e-6 | 27/27, median -0.024, p ~ 7.5e-9 |

| controls | Qwen3.5-4B | gemma-2-2b |
|---|---|---|
| noop_check / alpha=0 bit-exact | PASS | PASS |
| random-direction max presence increase | 1.3e-4 | 3.1e-4 |
| unrelated-probe max increase under real injection | 2.4e-4 | 8.4e-6 |

## Verdict: the causal findings replicate; the arm-2 AICc pass does not

**Replicates cleanly.**
- **Activation-side ignition is direction-specific dose-response on gemma too.**
  Every one of the 40 concepts rises from a ~1e-7 baseline to 0.12-0.77 presence
  as alpha grows, while norm-matched random directions stay at baseline
  (max increase 3.1e-4) and the 5 unrelated probes stay flat (max 8.4e-6). By
  plain AIC the 4-param logistic beats linear for **40/40** concepts on both
  models (gemma dAIC 3.6-32.7, all > 2).
- **Competition (one-in-one-out) replicates and is stronger here**: 27/27 valid
  pairs show X ignition + Y eviction to ~0, zero coexistence, and the specific
  excess beyond the nonspecific random-injection corruption holds in 27/27 pairs
  (sign-test p = 2^-27 ~ 7.5e-9; Qwen: 19/19, p ~ 4e-6). Same signature: a
  specific eviction component on top of a large nonspecific one.
- **Arm-1 failure replicates**: prompt ladders are staircases on gemma exactly as
  on Qwen (15% vs 27.5% AICc; 65% vs 82.5% plain AIC; same median sig-preferred
  width 0.11). Prompt-side "strength" stays a characterization note.

**Does not replicate numerically.** The Qwen arm-2 headline — 90% sigmoid-preferred
by AICc — drops to 27.5% on gemma. This is a *sharpness* difference, not a shape
difference: gemma's transitions are ~60% wider (median 10-90% width 0.33 of the
alpha range over all concepts, 0.28 among sig-preferred, vs 0.21 on Qwen) and the
curves saturate more gradually, so with n=9 points the AICc small-sample penalty
(k=5 vs k=3 costs 19.2) swamps the RSS advantage even though the logistic is the
better model for literally every concept by plain AIC. This is the same
AICc-penalty failure mode the Qwen run diagnosed for its arm 1. Implication for
the paper: the robust cross-family claim is "activation-side presence is a
direction-specific, saturating (sigmoidal by AIC 40/40) dose-response with
one-in-one-out competition", not the pre-registered ">60% by AICc" bar, which is
knife-edge on transition sharpness and passed only on Qwen.

Other model differences worth noting: 4 gemma concepts (Canada, China, France,
India) are non-monotone — presence peaks at alpha 0.75-1.5 and *decays* at higher
alpha (over-driving degrades the readout), which never happened on Qwen within
the same alpha grid. Gemma's ignited plateaus are lower (median max ~0.58 vs ~1.0
on Qwen), and competition X ignites to 0.63 median rather than 0.98.

## Deviations from the Qwen run (all gemma-specific adaptations)

- **Band/injection recomputed by the same fractions** (0.40-0.75 of available lens
  layers): layers 10-18, inject at midpoint L14, read downstream 15-18 (Qwen:
  12-22, L17, 18-22).
- **BOS handling** (gemma prepends `<bos>`, a high-norm attention sink: L14 mean
  norm 392.8 with BOS vs 221.7 without): position 0 is excluded from the presence
  scan, from the mean-residual-norm used to scale injections, and from the
  competition "all positions" injection (starts at position 1). Qwen has no BOS,
  so this preserves the construct (content positions only).
- **Final logit softcap** (30·tanh(z/30)) applied before the readout softmax to
  match gemma's own output convention. It is monotone, so ranks and the
  sigmoid-vs-linear comparison are insensitive to it; only the absolute presence
  scale changes.
- **Readout norm**: presence uses the model's own final `RMSNorm` module, which
  applies gemma-2's (1+w) convention internally — the idea-A norm_scaled readout
  is automatic.
- **J-vectors stay raw-u** (`normalize(J^T u)`, the flagship harness convention).
  Measured head-to-head before the run: the (1+w)-norm-scaled direction is a much
  weaker steering vector on gemma (France, alpha=1 downstream presence: 0.05
  norm-scaled vs 0.62 raw), and raw-u is what the Qwen run injected, so raw-u
  maximizes construct equivalence. Recorded in the manifest.
- **Harness bug workaround**: `JVecs.token_id()` encodes *with* special tokens;
  on gemma the leading `<bos>` decodes non-empty, so it silently returns the BOS
  id and every j-vector becomes the BOS unembedding row (injection then does
  nothing: flat 1e-7 curves — caught in the pilot). All `jv.vec` calls here pass
  the verified single-token id explicitly. The Qwen run is unaffected (no BOS).
- `attn_implementation="eager"` (required for gemma-2's attention softcapping).
- Hysteresis probe skipped, same rationale as the Qwen run (not implementable
  prompt-side in a causal LM).

## Run

```bash
# from the repo root, deps installed (see top-level README)
python ignition-gemma/run.py             # full run, ~2 min
python ignition-gemma/run.py --limit 3   # pilot
```

(`ignition/` must be present: this script imports its fit/summary/ladder machinery.)

## Files

- `config.yaml` — model/lens/band/alphas/criterion (thresholds identical to Qwen).
- `run.py` — thin adaptation layer; imports ladders, fits, hooks and summaries
  from `../ignition/` (loaded as a module), defines only the
  gemma-specific presence readout and main loop; local copy of the figure code.
- `results/metrics.json` — manifest (incl. all adaptations), per-concept curves,
  fits (AICc + plain AIC), controls, competition pairs, criterion evaluation.
- `results/ignition_curves.png` — representative curves per arm + dAICc
  histograms + width-vs-dAICc scatters.
