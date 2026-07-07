# Idea A signal-check: cross-model universality of the J-space

Tests whether the relational geometry of J-lens vectors (j_v^l = J_l^T u_v, from
Anthropic's Jacobian lens) is more similar across model families than unembedding
geometry alone predicts, concentrated at mid-layer fractions.

## Method

- Models: qwen3-1.7b, gemma-2-2b, gpt2-small (pre-fitted lenses from
  `neuronpedia/jacobian-lens`, wikitext fit; no lens fitting here).
- Concepts: candidate pool of 559 concrete nouns / countries / colors / animals
  (`words.py` + countries from the vendored `capacity.json`); kept those where
  `" <word>"` is a single token in all three tokenizers (507 survive; threshold 120).
- Per model and layer fraction 0.1–0.9 (nearest lens layer): j_v = J_l^T u_v for
  all concepts, computed in fp32 (lenses are stored fp16; `jlens` upcasts on load),
  L2-normalized. The whole pipeline runs in two variants: `norm_scaled` (primary;
  u_v = tied-embedding unembedding row scaled elementwise by the final-norm weight,
  matching the lens readout softmax(W_U · norm(J_l h)); Gemma-2 uses the 1+w
  convention; gpt2's LayerNorm centering and bias are omitted — diagonal part only)
  and `raw` (bare unembedding rows, robustness check). Gemma-2's final logit softcap
  is a monotone scalar squash and irrelevant to direction geometry.
- Alignment: RSA = Spearman correlation between the upper triangles of the two
  models' concept-by-concept cosine-similarity matrices at matched fractions;
  SVCCA = mean canonical correlation after projecting each model's concept-by-d
  j-vector matrix to its top 50 PCs (concepts are the shared axis).
- Baselines: (a) unembedding-only RSA/SVCCA on u_v directly; (b) shuffled concept
  labels on one side, 3 permutations; (c) J_l replaced by a random orthogonal map,
  3 draws — note row-normalized cosine geometry is invariant to orthogonal maps,
  so (c) equals (a) by construction and serves as a numerical sanity check.

## Pre-registered criterion

For every model pair and every fraction in 0.4–0.7: (RSA_J − RSA_unembed) must be
positive and exceed the shuffled floor's spread (max − min over the 3 permutations).
A stricter post-hoc check (`strict_criterion` in metrics.json) compares the primary
J-space RSA against the *best* unembedding baseline across both variants.

## Reproduce

```bash
# from the repo root, with deps installed (see top-level README)
python universality/run.py
```

Seeds: global 0, baseline seeds [0, 1, 2] (config.yaml). Outputs land in
`results/` (metrics.json, rsa_by_layer.png). Only tokenizers, embedding/norm
tensors (fetched shard-wise via safetensors, no full model load), and lens files
are downloaded.

## Results (507 concepts, seeds 0/[0,1,2])

| pair (norm_scaled) | RSA unembed | RSA J @0.4/0.5/0.6/0.7 | shuffled floor |
|---|---|---|---|
| qwen3-1.7b vs gemma-2-2b | 0.591 | 0.815 / 0.783 / 0.788 / 0.752 | ~0.00 ± 0.02 |
| qwen3-1.7b vs gpt2-small | 0.608 | 0.873 / 0.868 / 0.871 / 0.793 | ~0.00 ± 0.02 |
| gemma-2-2b vs gpt2-small | 0.045 | 0.820 / 0.803 / 0.753 / 0.593 | ~0.00 ± 0.01 |

Pre-registered criterion: PASS for all 3 pairs. Strict criterion (vs best raw-u
baseline, 0.63–0.72): PASS at 0.4–0.6 for all pairs; FAIL at 0.7 for
gemma↔gpt2 (margin −0.035) and marginal for qwen↔gemma (0.028 vs spread 0.031).
The `raw` variant collapses for Qwen pairs (RSA_J ≈ 0.0–0.2) because Qwen3's
final-norm weight is strongly non-uniform (max/median ≈ 5.6) — dropping it makes
J^T u an unfaithful readout, which is why `norm_scaled` is primary.

## Interpretation (honest, 5 lines)

1. SIGNAL by the pre-registered criterion: J-space RSA beats the unembedding baseline by 0.16–0.77 at all mid fractions for all 3 pairs, far above the shuffled floor (~0 ± 0.02).
2. The signal survives the strictest baseline (best unembedding RSA over both norm conventions) at fractions 0.4–0.6 for all pairs, with margins 0.06–0.19.
3. At fraction 0.7 the strict margin dies for gemma↔gpt2 and is marginal for qwen↔gemma, so "concentrated at mid fractions" holds only as 0.2–0.6 elevation with late-layer decay, not a narrow mid peak.
4. SVCCA agrees directionally but weakly (J ≈ unembed ≈ 0.6–0.75), so the effect lives in relational (RSA) geometry, not in shared linear subspaces.
5. Main residual confound: all three lenses were fit on the same wikitext corpus, which could induce shared corpus-statistics structure; a different-corpus lens replication is the next check before believing universality.
