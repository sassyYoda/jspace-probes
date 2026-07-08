# Idea J — workspace capacity psychophysics (signal check)

Does the J-space's asserted capacity limit ("a few dozen concepts") bind
behavior? Load-N lists, J-lens occupancy readout, yes/no recall, plateau fit.

Model Qwen/Qwen3.5-4B (mps, bf16); lens
`qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt`;
mid band = layers 12-22 (harness-validated).

## Method

- Prompts: `Remember this list: w1, ..., wN. Question: was X in the list?
  Answer (yes/no):` with N in {2,4,8,12,16,24,32,48,64}, 3 seeds x 12 prompts
  per (N, seed): half probe-present (probe position uniform over the list,
  recorded), half probe-absent (same family). Words are drawn from the paper's
  own capacity families (capacity.json), filtered to single-token-with-leading-
  space in the Qwen tokenizer. Survivors: names 409/505, surnames 480/603,
  countries 122/149, cities 185/256. (Deviation: capacity.json ships candidate
  pools only for these 4 families; the colors/animals/body/clothes families
  named in `targets_per_family` have no pools, so 4 families, cycled 3 prompts
  each per cell.)
- Recall: teacher-forced logit comparison at the final position (no trailing
  space): logsumexp{" yes"," Yes"} vs logsumexp{" no"," No"}. Accuracy and d'
  (log-linear corrected) vs N.
- Occupancy (primary, pre-registered): per listed word, presence = max softmax
  prob of the word's token over layers 12-22 x retention-region positions
  (from " Question" onward), with positions within +-1 of any literal
  occurrence of that token masked (the probe recurs in the question).
  Count of listed words above a fixed null threshold = 95th percentile of the
  identical measurement on 16 matched non-listed same-family words per prompt.
- Secondary (exploratory, added after the smoke test): "carry" presence = same
  readout over LIST-region positions strictly after the word's own position
  (own+-1 masked); its null = control words over the whole list region.
  "Self" presence at the word's own position is the positive control.
- No-hold control: identical lists in `Here are some words that appeared in a
  document: ... The document was long and nothing else about it was recorded.`
  (retention region length-matched, 12 vs 13 tokens); same readouts, own null.
- Fits: per seed, LS fit of count = min(N, C) vs count = a*N over per-prompt
  counts; AICc comparison (both models 1 param + variance, so equal penalty).
- CIs: bootstrap over prompts (2000 iters). Presence-recall link: point-
  biserial r between probe presence and recall correctness on probe-present
  items (raw + z-scored within N), permutation p.

Pre-registered signal criterion: (i) plateau model wins AICc for >=2/3 seeds
with C in [5,50]; AND (ii) recall d' pooled over N>=N* (first grid N >= pooled
C) is significantly below the N<=8 level (bootstrap 95% CI of the difference
excludes 0).

## Run

```bash
# from the repo root, deps installed (see top-level README)
python capacity/run.py            # full, ~10-20 min on Apple MPS
python capacity/run.py --smoke    # 8 prompts
```

Outputs: `results/records.json` (per-prompt raw), `results/metrics.json`,
`results/capacity_curves.png`.

## Verdict: NO-SIGNAL (pre-registered criterion fails; the pre-declared kill criterion is met)

Full run: 324 prompts (x2 with the no-hold control), 134 s wall on MPS.

- **Criterion (i) FAILS 3/3 seeds.** The above-threshold occupancy count does
  not saturate at a capacity — it never rises above the null in the first
  place. Counts grow ~0.03-0.06 per list item (seed slopes a = 0.029/0.055/
  0.034), i.e. at the 5% false-positive rate the 95th-percentile threshold
  guarantees by construction. Best-fit plateau C = 0.75/1.25/0.75 (pooled
  1.0), far below the pre-registered [5, 50] window, and the linear model
  wins AICc for every seed anyway. The direct check: listed-word retention
  presence vs the matched non-listed null has AUC 0.477 — per-word list
  content is **indistinguishable from same-family controls** in the retention
  region. This is the pre-declared kill criterion ("occupancy metric too noisy
  to define presence per concept"), not a measured capacity.
- **Criterion (ii) FAILS.** Recall is at ceiling through N=32 (accuracy
  1.000) and only begins to break at N=48 (0.972) / N=64 (0.917); d' pooled
  over N<=8 (4.72) vs N>=N* (4.66) does not differ (bootstrap 95% CI of the
  difference [-0.38, +0.37]). All behavioral errors are FALSE ALARMS on
  probe-absent items (hit rate 1.00 at every N; FA 0.17 at N=64) — the model
  never misses a listed word up to 64 items; it starts over-accepting lures.
- **Presence-recall correlation: undefined** — probe-present recall was
  162/162 correct, so the correctness variable has zero variance. Nothing to
  correlate; would need N >> 64 or harder lures to get misses.
- **Positive control passes**: presence at a word's own list position is
  strong (mean 0.62-0.70) and roughly flat in N — the lens and pipeline see
  the words fine where they are literally processed.
- **What the retention region actually holds** (why the metric nulls out):
  mid-band lens readout at post-list positions is dominated by local
  next-token content ("?", "____", punctuation), with the list words in the
  ~1e-4 tail alongside their same-family cousins. The exploratory "carry"
  metric (list-region positions after a word's own token) is likewise swamped
  by family-level anticipation: at commas the lens shows *many* family words
  (controls hit 0.01-0.05), so specific-word presence again fails to separate
  (AUC 0.53). Qwen3.5-4B evidently solves 64-item recognition without keeping
  per-word content linearly decodable as token probability in the mid-band
  J-space of the retention positions.
- **Directed modulation (no-hold control) is the one real positive**: mean
  per-word retention presence is 10-20x higher under the remember+question
  context (1.2-4.9e-4) than in the no-hold document framing (0.7-3.2e-5) at
  every N, CIs disjoint for N>=4. But since listed words do not beat
  same-family controls within either condition, this elevation is
  family/task-level activation, not word-specific retention.

Serial position: carry presence shows an apparent primacy gradient (first
half > last half at every N), but it is confounded — earlier words have more
subsequent positions to max over — so we do not claim eviction structure.
Retention-region serial curves are flat at the null level.

Bottom line: with this (pre-registered) occupancy
operationalization the capacity claim is not behaviorally testable — the
workspace readout has no per-item resolution in the retention region, while
behavior stays at ceiling until N~48-64 and degrades via false alarms. A
capacity psychophysics would need (a) a presence metric below the
token-probability readout (probing residuals directly), (b) tasks whose
retention region forces recall (generation, not yes/no), or (c) much larger N
or confusable lures to move behavior off ceiling.

Deviations from spec: capacity.json ships candidate pools for only 4 of the 8
named families (no colors/animals/body/clothes); carry/self metrics added
after the smoke test as diagnostics; per-N d' for N<=32 sits at the
log-linear-correction ceiling (3.88) because hits and correct rejections were
both perfect.
