# Selectivity: J-space ablation dose-response

Signal check for the claim that ablating a prompt's ACTIVE J-space contents
selectively degrades workspace-dependent tasks (multihop, two-step arithmetic,
rhyming) while sparing workspace-independent ones (MCQA, sentiment, single-hop
fact retrieval), with a graded dose-response in k, beyond matched controls.

Model Qwen/Qwen3.5-4B (bf16), pre-fitted wikitext J-lens (n1000), mid band
= lens layers 12-22 (the band validated in `../harness/`). Intervention
machinery is reused from the harness (`InterventionHooks.ablate`,
QR-orthogonalized projection removal, all positions); `noop_check` runs first.

## Result (headline)

The pre-registered clean double-dissociation criterion **fails as specified**,
and the reason is the interesting part. Ablating the top-k active J-space
concepts is monotone in k and drives accuracy far below matched random and
bottom-k controls on **rhyme, multi-hop reasoning, and single-hop fact
retrieval**, while leaving **MCQA and sentiment** untouched. Single-hop fact
retrieval was predicted (per the paper) to be spared; it collapses instead.

| task | base | real k20 | random k20 | real − random @ k20 |
|---|---|---|---|---|
| rhyme | 0.825 | 0.200 | 0.83 | **−0.63** |
| single-hop facts | 1.00 | 0.425 | 1.00 | **−0.58** |
| multi-hop reasoning | 0.633 | 0.278 | 0.61 | **−0.33** |
| arithmetic | 0.725 | 0.725 | 0.68 | +0.05 |
| sentiment | 0.950 | 0.900 | 0.94 | −0.04 |
| MCQA | 0.775 | 0.775 | 0.78 | 0.00 |

**Why single-hop facts collapse:** the top active concept for "the capital of
France is ___" is literally " Paris" (readout prob 0.63), i.e. the anticipated
answer. Ablating the active workspace contents removes it. The real predictor of
degradation is not the paper's workspace-dependent/independent split but
**whether the scored answer is a generated content word whose direction lives in
the active mid-band J-space** (rhyme, single-hop, multi-hop all qualify). MCQA
and sentiment are spared because they are scored over fixed option tokens
(A/B/C/D, pos/neg) that are never in the content-word concept set; arithmetic is
spared because the numeric answer does not surface as a top active concept
(consistent with the paper's arithmetic result failing to replicate on Qwen).

So the causal core holds strongly (a monotone, control-beating, 34-63 point
ablation effect), but the paper's task-cut is the wrong axis for a small model
scored on content words. The clean fix for the scaled study is to score every
task over a fixed option set, and to add matched single-hop vs multi-hop pairs
that isolate hop-count from answer-in-workspace. This is exactly the kind of
confound the funded robustness arm is designed to resolve.

## Per-prompt procedure

1. Clean forward; record mid-band residuals at all positions.
2. Rank vocab tokens by max J-lens softmax probability over (band layers x
   positions). Exclude: non-alphabetic/punctuation/digit tokens, single ASCII
   letters, ~110 English function words, tokens occurring in the prompt (by
   token id or word-boundary string match), tokenizer special tokens; dedupe by
   normalized string. Keep the top 500 as the "active pool".
3. Ablate j-directions (`normalize(J_l^T u_v)` per band layer) for
   - real: pool ranks 1..k
   - bottom-k: pool ranks 500-k+1..500 (active-but-weak control)
   - random: k iid gaussian directions per layer, seeds 0/1 (the ablate op
     QR-orthonormalizes, so ablated subspace dimension exactly matches real)
4. Re-score the task. k in {0, 5, 10, 20, 40}; k=0 is the shared baseline.

## Battery (scored by greedy teacher-forcing or logit comparison; no long generation)

| task | n | type | source |
|---|---|---|---|
| multihop | 90 | gen | probe-swap prompts, exact answer |
| arithmetic | 40 | gen | seeded (a+b)*c word problems, full digit match |
| rhyme | 40 | gen | hand-built "rhymes with X and means Y", quote/bold variants accepted |
| mcqa | 40 | logit A/B/C/D | MMLU all/validation, seed 0 |
| sentiment | 40 | logit pos/neg | SST-2 validation, seed 0 |
| singlehop | 40 | gen | capitals + one-hop facts, hand-built |

Prompts never end in trailing whitespace (readout-distortion gotcha); numeric
answers are teacher-forced through the space+digit token sequence.

Note one design asymmetry: for MCQA the answer string is present in the prompt,
so the prompt-literal exclusion prevents ablating the answer's own direction
there; for generative tasks the answer is not in the prompt and its
(anticipated) direction is ablatable. This is inherent to "ablate the active
workspace contents, not the input tokens".

## Pre-registered criterion (k=20)

Real minus random-control accuracy: workspace-dependent mean drop > 15 points
AND workspace-independent mean drop < 5 points.

## Run

```
# from the repo root, deps installed (see top-level README)
python selectivity/run.py            # full battery
python selectivity/run.py --limit 2  # pilot
```

Outputs: `results/metrics.json` (manifest, accuracy table, criterion verdict,
per-item flip records), `results/items.jsonl` (incremental, includes selected
concepts per item), `results/selectivity_dose_response.png`, `results/run.log`.
