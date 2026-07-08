# Idea B signal check — does the J-space ignite?

Hypothesis: concepts enter the J-space via
a threshold/competition process; presence vs stimulus strength should be sigmoidal
with a sharp inflection, not linear.

Model Qwen/Qwen3.5-4B (mps, bf16), lens `Qwen3.5-4B_jacobian_lens_n1000.pt`, harness
machinery from `../harness/jspace_interventions.py` (JVecs, InterventionHooks,
noop_check). Presence = max over mid-band layers 12-22 x all positions of the J-lens
softmax probability of the concept's leading-space token (norm-scaled readout). 40
concepts: the paper's `ignition.json` countries_12 + 28 concrete nouns, all verified
single-token with leading space.

## Method

**Arm 1 (prompt-side).** Deterministic 8-step evidence ladders (`ladders.py`): step 1
oblique base ("The place is nice."), steps 2-7 cumulatively add one attribute hint each
(category -> region/context -> fame -> landmark -> capital/near-giveaway), step 8 names
the concept ("France is the country in question."). Token-level assert: concept token
absent from steps 1-7. Presence + rank recorded per step.

**Arm 2 (activation-side).** Inject `alpha * mean_resid_norm(L17) * j_concept` (unit
j-vector from JVecs) at layer 17, last-position-onward, on the neutral carrier
"The weather today is mild and the streets are quiet." Alpha in {0, .125, .25, .5, .75,
1, 1.5, 2, 3}; presence read downstream at layers 18-22 only.

**Fits (pre-registered).** Per concept x arm: linear vs 4-param logistic, compared by
AICc (k=3 vs k=5 incl. sigma; multi-start `curve_fit`). Signal criterion: sigmoid
preferred (dAICc > 2) for >60% of concepts in BOTH arms, AND median 10-90% transition
width (ln81 * w) < 1/3 of the stimulus range in both arms.

**Rigor.** Harness `noop_check` PASS; alpha=0 injection reproduces baseline activations
bit-exactly (real and random direction). Controls: norm-matched random-direction
injection; 5 unrelated probe concepts tracked under real injection. Ladders
deterministic; the only sampling randomness (competition pairs) uses seeds {0,1,2}.

## Results

Full metrics in `results/metrics.json`; curves in `results/ignition_curves.png`.
Wall clock 135 s (40 concepts, ~1,100 forward+readout passes).

| | arm 1 (ladders) | arm 2 (injection) |
|---|---|---|
| sigmoid preferred (AICc, pre-reg) | 11/40 = 27.5% | 36/40 = 90% |
| median dAICc (lin - sig) | -11.5 | +14.0 |
| median 10-90% width / range (sig-preferred) | 0.11 | 0.21 |
| sigmoid preferred (plain AIC, secondary) | 33/40 = 82.5% | 40/40 |

**Verdict vs the pre-registered criterion: NO-SIGNAL.** Arm 2 passes cleanly (90%
sigmoid-preferred, median transition width 0.21 of the alpha range, well under 1/3);
arm 1 fails the >60% bar at 27.5%, so the conjunction fails.

Honest reading of the arm-1 failure: (a) arm-1 curves are staircases, not smooth
sigmoids — presence is flat, jumps when one specific hint lands (often the category or
capital hint), plateaus, then jumps again at explicit naming. A single 4-param logistic
cannot capture a double step, and AICc's small-sample penalty (n=8, k=5) is severe: the
plain-AIC secondary count is 33/40 sigmoid-preferred. So arm 1 is emphatically not
linear either (median linear fit is worse than the step fit for 33/40); the pre-reg
criterion fails on the specific parametric form, not on smoothness. (b) Prompt-side
"strength" is ordinal at best: cumulative ladders share prefixes, and the
max-over-positions measure inherits the prefix max, so plateaus are partly mechanical.
This is the pre-declared kill-criterion scenario (prompt-side and
activation-side constructs disagree): downgrade prompt-side strength to a
characterization note; the clean dose-response construct is activation-side alpha.

**Controls (both clean).** Random-direction injection (norm-matched) never produces
concept presence: max increase over all 40 concepts x alphas = 1.3e-4. The 5 unrelated
probes stay flat under real injection: max increase 2.4e-4, medians ~0. Ignition is
direction-specific, not global inflation.

**Competition probe (one-in-one-out, with caveat).** 30 pairs (10 per seed), carrier =
Y's step-7 ladder, inject X at alpha=1.0. 19/30 pairs valid (Y evoked > 0.05). In all
19, X ignites (median 0.98) and Y drops below threshold (median dY = -0.25, Y ends at
~0.000). Caveat: norm-matched random injection also depresses Y (median dY = -0.17,
Y ends at 0.066), so much of the raw drop is nonspecific corruption from all-position
injection. The specific excess is still consistent: real X injection drops Y below the
random control in 19/19 valid pairs (median excess -0.066, sign-test p ~ 4e-6), and Y
goes to ~0 only under real X. Signature: one-in-one-out with a specific component on
top of a large nonspecific one; no coexistence cases.

## Deviations and honest notes

- **Hysteresis probe: SKIPPED.** "Decreasing context accumulation within one prompt"
  is not implementable in a causal LM: context only accumulates, and prompt-side
  retraction leaves the hint tokens attended. No matched-evidence descending branch
  exists, so running it would fake the construct.
- **Competition injection is at ALL positions**, deviating from arm 2's
  last-position-onward convention. Measured reason: a prompt-evoked concept lives in
  the mid-band J-space only at its hint-token positions, never at the final position —
  for the France step-7 carrier + "The place I mean is", the model's next-token
  p(France) = 0.42 while mid-band p_last < 1e-4 (France appears at the last position
  only from layer 24 up). A last-position injection can therefore never contact a
  prompt-evoked competitor. This asymmetry (injected concepts readable downstream at
  the injected position; prompt-evoked ones only at their token positions) is itself a
  finding worth carrying forward.
- Arm-1 ladder hints were re-graded once after a 3-concept pilot (splitting
  "country + region" into two hints) because step 2 saturated the low range; this was
  stimulus calibration before the full run, not selection on fit outcomes.
- Arm-2 last-position secondary measure gives identical conclusions (90% preferred,
  width 0.21).
- 4 concepts fail sigmoid preference in arm 2 (e.g. sun, dAICc -12.9): shallow/late
  risers where the transition midpoint sits near alpha=3, so the sampled range covers
  only the lower knee.

## Run

```bash
# from the repo root, deps installed (see top-level README)
python ignition/run.py             # full run, ~2-3 min
python ignition/run.py --limit 3   # pilot
```

## Files

- `config.yaml` — model/lens/band/alphas/criterion (pre-registered thresholds).
- `ladders.py` — attribute tables and ladder builder; probes.
- `run.py` — both arms, controls, competition probe, fits, figure.
- `results/metrics.json` — manifest, per-concept curves, all fits + AICc, controls,
  competition pairs, criterion evaluation.
- `results/ignition_curves.png` — representative curves per arm + dAICc histograms +
  width-vs-dAICc scatters.
