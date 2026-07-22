# Scaled ignition + competition — three families at 8-9B

GPU-phase scale-up of the flagship ignition result (`../ignition/`,
`../ignition-gemma/`). One config-parametrized runner (`run_scaled.py`), three
families run as separate processes on a single MI300X (2026-07-21):

| | Qwen3-8B | Gemma-2-9B | Llama-3.1-8B |
|---|---|---|---|
| lens (neuronpedia/jacobian-lens) | `qwen3-8b/.../Qwen3-8B_jacobian_lens.pt` | `gemma-2-9b/.../gemma-2-9b_jacobian_lens.pt` | `llama3.1-8b/.../Llama-3.1-8B_jacobian_lens.pt` |
| band (fractions 0.40-0.75) / inject / downstream | 14-25 / L20 / 21-25 | 16-30 / L23 / 24-30 | 12-22 / L17 / 18-22 |
| BOS prepend (excluded everywhere) | 0 | 1 | 1 |
| final logit softcap in readout | — | 30.0 | — |
| concepts surviving single-token filter | 80/80 (+5 probes) | 80/80 (+5) | 80/80 (+5) |
| wall clock (co-run, contended) | 1647 s | 1992 s | 1498 s |

Design deltas vs the 4B/2B signal checks: 80 concepts (original 40 + 40 same-family
extras, `concepts_extended.py`) for injection/competition; 15-point alpha grid
{0, 0.0625, ..., 3.0}; 3 carrier prompts (seed = carrier; carrier 0 is the original
"weather" sentence, carriers 1-2 new neutral sentences, all verified concept-free);
arm 1 = the original 40 ladders unchanged; full 20x20 competition matrix. Fits report
AICc AND plain AIC; the scaled PRIMARY criterion is the pre-registered WIDTH test:
median 10-90% transition width / alpha range < 1/3 over all arm-2 sigmoid fits
(concept x carrier). All machinery imported from `../harness` and `../ignition`;
presence definition unchanged.

## Arm 2 (activation-side injection) — five-way table

| | Qwen3.5-4B (signal check) | gemma-2-2b (signal check) | **Qwen3-8B** | **Gemma-2-9B** | **Llama-3.1-8B** |
|---|---|---|---|---|---|
| n fits (concepts x carriers) | 40 | 40 | 240 | 240 | 240 |
| sigmoid preferred, AICc | 90% | 27.5% | **100%** | **91.7%** | **100%** |
| sigmoid preferred, plain AIC | 100% | 100% | 100% | 95.8% | 100% |
| median 10-90% width / range | 0.21 | 0.33 (all) / 0.28 (pref) | **0.017** | **0.414** | **0.072** |
| WIDTH criterion (< 1/3, primary) | pass | fail | **PASS** | **FAIL** | **PASS** |
| per-carrier width (c0/c1/c2) | — | — | .023/.017/.013 | .389/.402/.442 | .069/.073/.074 |
| over-driven curves (of 240) | 0 (of 40) | 4 (of 40) | **0** | **1** (Portugal) | **0** |

The 15-point grid resolves the old AICc knife-edge: even gemma-9b is >60%
sigmoid-preferred by AICc now (2B failed at 27.5% purely from the n=9 small-sample
penalty). What actually separates families is transition SHARPNESS: Qwen/Llama
ignite step-like (widths 0.02-0.07 of the alpha range, transition midpoints near
alpha ~0.1), gemma-9b rises gradually (0.41, plateaus low — median ignited plateau
far below Qwen/Llama) and fails the width bar exactly as its 2B sibling trended.
Per-carrier summaries are near-identical within each model: carrier choice is not a
driver. Over-driving is essentially a gemma-family phenomenon (1 curve here, 4 on
2B; zero on Qwen/Llama even at alpha=3).

Arm 1 (unchanged 40 ladders) replicates the known staircase failure everywhere:
AICc 30% / 27.5% / 52.5% (Qwen3-8B / Gemma-2-9B / Llama-3.1-8B), plain AIC 77.5% /
75% / 92.5%. Prompt-side "strength" stays a characterization note.

## Competition matrix (20 residents x 20 injected, alpha=1.0, all content positions)

Residents = top-20 by evocability on "The story was about (the) Y." (evocable
>0.05: 58/80, 59/80, 80/80). Valid cells = off-diagonal with y_before > 0.05
(380/380 for all three). dY_specific = dY_real - mean(dY over 2 norm-matched
random-direction injections).

| | **Qwen3-8B** | **Gemma-2-9B** | **Llama-3.1-8B** |
|---|---|---|---|
| X ignites (dX > 0.05) | 380/380 | 251/380 | 380/380 |
| eviction (X up, Y down > 0.05) | **380/380 (100%)** | **250/380 (65.8%)** | **380/380 (100%)** |
| coexistence | 0 | 1 | 0 |
| median dY real / random | -0.94 / -0.94 | -0.12 / +0.05 | -0.93 / -0.07 |
| median dY_specific | +0.0001 | -0.167 | -0.860 |
| specific excess negative (sign test) | 110/380 (p=1.2e-16, WRONG direction — floor) | **380/380 (p=8.1e-115)** | **380/380 (p=8.1e-115)** |
| — same at secondary alpha=0.25 | **303/380 (p=8.4e-33)** | 375/380 (p=5.3e-104) | 246/380 (p=9.8e-9) |
| corr(dY_specific, cos(j_X,j_Y)) Spearman | +0.67 (a=1.0) | +0.07 | +0.55 (a=0.25) |

- **One-in-one-out generalizes**: at the pre-registered alpha=1.0, every valid cell
  on Qwen and Llama shows X ignition + Y eviction, zero coexistence; gemma-9b evicts
  in 2/3 of cells with its weaker ignition plateaus (X only reaches ~0.17 median).
- **Specificity**: Llama is the cleanest case ever measured — the random control
  barely moves Y (-0.07) while real X removes it (-0.93), so eviction is ~93%
  specific, 380/380 cells, p ~ 8e-115. Gemma also 380/380 (random actually *raises*
  Y slightly). On Qwen3-8B the alpha=1.0 matrix is floor-dominated (random alone
  already sends Y to ~0, exactly the nonspecific-corruption caveat from the 4B run,
  amplified): dY_specific is ~0 and its sign test lands significantly on the WRONG
  side (tiny positive residuals). The secondary alpha=0.25 matrix (added after the
  pilot exposed the floor; same protocol otherwise) de-floors it: 303/380 negative,
  p=8.4e-33. Report both honestly: the specific-excess claim holds on all three
  families, on Qwen only below the floor regime.
- **Similarity structure**: eviction is NOT stronger for similar concepts — the
  correlation runs the other way (where dY_specific is off-floor: rho +0.67 Qwen,
  +0.55 Llama): a *similar* injected X spares the resident's readout slightly
  (consistent with j-vector overlap partially supporting Y's token). Diagonal
  (X=Y) cells confirm: self-injection preserves the resident everywhere.

## Controls (all three families)

| | Qwen3-8B | Gemma-2-9B | Llama-3.1-8B |
|---|---|---|---|
| noop_check / alpha=0 bit-exact (3 carriers, real+random) | PASS | PASS | PASS |
| random-direction max presence increase (240 curves) | 4.0e-3 | 1.1e-3 | 3.5e-4 |
| unrelated-probe max increase under real injection | 9.6e-4 | 3.7e-3 | 1.7e-3 |
| tokenizer special-token behavior asserted | no BOS | 1 BOS, excluded | 1 BOS, excluded |

## Deviations and notes

- The three families were run concurrently on the one card. VRAM with all three:
  98 GB / 205 GB; GPU utilization was already 100% during co-running, so
  co-running stretched per-run wall clock (~25-33 min each alone by pilot
  estimate; 27/33/25 min co-run) but compressed total wall clock to ~75 min.
  NOTE: an unrelated process was also using the GPU throughout; utilization
  numbers are not cleanly attributable.
- **Secondary competition alpha=0.25** added after the Qwen pilot showed the
  norm-matched random control floors the resident at alpha=1.0 (deviation,
  clearly labeled; the pre-registered alpha=1.0 matrix is unchanged and
  primary). Files: `competition_matrix.png` (primary), `_a0.25.png`.
- Sub-alpha-0.0625 resolution is exhausted on Qwen3-8B: many transitions complete
  between the first two nonzero grid points (width_frac ~0.01-0.02 is partly an
  upper-bound artifact of fit extrapolation below the grid). The WIDTH pass is
  unaffected (widths are if anything overestimated by coarse sampling near 0),
  but a per-concept transition-point study at 8B should sample alpha in [0, 0.4].
- Llama-3.1-8B is gated on the Hub — reproduce with your own HF token; the lens
  is the base-model lens, not `-it`.
- All 80 extra-concept candidates survived all three tokenizers (no top-up needed);
  survivors recorded in each manifest.
- Checkpointing: per-unit (concept x carrier / resident x alpha) atomic JSON; all
  three runs completed without needing resume. `results/<tag>/checkpoint.json` is
  a runtime artifact (~1.6 MB each) and is not shipped; a fresh run recreates it.

## Run

```bash
# from the repo root, deps installed (see top-level README); an 8-9B model
# needs a large GPU (each run held 30-35 GB of VRAM)
python ignition-scaled/run_scaled.py                                  # Qwen3-8B
python ignition-scaled/run_scaled.py --config config_gemma2-9b.yaml   # Gemma-2-9B
python ignition-scaled/run_scaled.py --config config_llama3.1-8b.yaml # Llama-3.1-8B (gated; needs HF token)
python ignition-scaled/run_scaled.py --pilot 3                        # pilot (3 concepts)
```

Device auto-detects CUDA (incl. ROCm) / MPS / CPU; the measured runs used ROCm.

## Files

- `run_scaled.py` — config-parametrized runner (arms 1-2, competition matrices,
  controls, fits, figures, checkpointing).
- `config.yaml` / `config_gemma2-9b.yaml` / `config_llama3.1-8b.yaml` — per-family
  configs (BOS, softcap, attn implementation, alphas, criterion).
- `concepts_extended.py` — the 40 extra concepts + resident-carrier template.
- `results/<tag>/` — `metrics.json` (manifest, curves, fits, controls, competition),
  `competition_matrix.json`, `ignition_curves.png`, `competition_matrix.png`
  (+ `_a0.25.png`). Run logs, pilot smoke runs, and checkpoints are runtime
  artifacts, not shipped.
