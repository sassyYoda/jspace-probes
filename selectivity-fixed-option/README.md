# Fixed-option-scoring test of the answer-in-workspace reframe

GPU-phase scale-up of `../selectivity/` (Qwen3-8B + Gemma-2-9B, single MI300X).
Pre-registered BEFORE the full run; this file is the registration.

## Question

The selectivity signal check (`../selectivity/`) found that ablating a
prompt's top-k active mid-band J-space concepts destroys exactly the tasks
whose scored answer is a GENERATED CONTENT WORD (rhyme -0.63, single-hop
-0.58, multi-hop -0.33 beyond random controls) while fixed-option-scored tasks
(MCQA, sentiment) are spared — because for e.g. "the capital of France is" the
top active concept IS " Paris", so ablation removes the answer itself. The
paper's axis (task type: multi-hop "workspace-dependent", single-hop spared)
and ours (answer-in-workspace) come apart exactly when the same question is
scored both ways. This experiment scores the same questions both ways.

## Design

3 task types (singlehop facts, multihop facts, rhyme) x 60 underlying
questions each x 2 scoring formats:

- **GEN**: open completion, teacher-forced exact match (signal-check style).
- **FIXED**: 4-option multiple choice over letters A-D (correct answer + 3
  same-family distractors, seeded shuffle), scored by argmax over the
  ' A'..' D' letter-token logits at the last position. The answer's content
  word appears ONLY in the options list, which is part of the prompt, so the
  prompt-literal exclusion rule (rules 4+5 of the signal check's verbatim
  exclusion rule) keeps its direction OUT of the ablatable active set — the
  crux of the design. Per item we verify the answer direction is not in the
  selected top-k anyway ("leak": string equality / plural-stripped equality /
  len>=4 prefix relation against answer content words); leaked items are
  reported as a fraction and excluded from the primary FIXED cells
  (@noleak), with all-item accuracies reported alongside.

Ablation: top-k=20 active J-space concepts, mid band (fractions 0.40-0.75 of
lens layers), all positions (content positions only on BOS-prepending models).
Controls: matched random subspaces (2 seeds; QR-orthonormalized, dimension
exactly matches real) and bottom-k of the 500-deep filtered active pool.
Concept selection + exclusion rule copied verbatim from the signal check.

Items: multihop = 60 of the 68 probe-swap items with alphabetic content-word
answers (no digits, number words, or element symbols; deduped by prompt),
distractors hand-authored per item from the same family (other capitals,
languages, colors, ...; the paired swap_answer included where same-family).
Singlehop = 30 capitals + 30 one-hop facts (signal-check lists extended),
distractors same-family. Rhyme = signal-check 40 extended to 60; distractors
are other words that rhyme with the same cue (wrong meaning), so the FIXED
item still requires the rhyme+meaning computation. Everything seeded
(item_seed 0, crc32-keyed per-item rngs; option-label balance reported in the
manifest).

## Pre-registered predictions (evaluated at k=20, real minus mean-random drop)

- **OURS (answer-in-workspace)**: GEN drop >= 20 pts for ALL THREE task
  types; FIXED drop < 5 pts for ALL THREE, including multi-hop.
- **PAPER'S (task-type)**: multi-hop drops >= 15 pts in BOTH formats
  (workspace-dependent regardless of scoring); single-hop spared (< 5 pts)
  in both. (Under the paper's taxonomy rhyme is also workspace-dependent in
  both formats; reported but not part of its criterion.)

**The FIXED multi-hop cell decides.** Both predictions can fail (e.g. FIXED
multi-hop drops 5-15 pts): then neither axis is clean and we report the grid.

## RESULTS (both arms complete, 2026-07-21)

Accuracy drop under real top-20 ablation beyond mean matched-random control
(percentage points; FIXED cells leak-excluded — see below):

| task | Qwen3-8B GEN | Qwen3-8B FIXED | Gemma-2-9B GEN | Gemma-2-9B FIXED |
|---|---|---|---|---|
| single-hop | **+20.8** | 0.0 | **+55.0** | 0.0 |
| multi-hop  | **+31.7** | +3.6 | **+45.0** | +4.2 |
| rhyme      | +3.3 | +1.9 | +8.3 | 0.0 |

(bottom-k controls track random within a few points everywhere; full per-cell
accuracies in results*/metrics.json, per-item records in results*/items.jsonl.)

**Verdicts.** Both pre-registered conjunctions formally FAIL, asymmetrically:

- PAPER'S task-type axis fails decisively on both families: the decisive
  FIXED multi-hop cell is spared (+3.6 / +4.2 pts, needed >= 15), and
  single-hop — predicted spared — collapses in GEN (+20.8 / +55.0).
- OURS fails only on its GEN-side rhyme clause: rhyme GEN drops just +3.3 /
  +8.3 pts (needed >= 20) at 8-9B, unlike the 4B signal check where rhyme
  collapsed 63 pts. The FIXED half holds for all three tasks on both
  families (every FIXED cell < 5 pts).

The scoring-format axis is what survives: on the SAME single-hop and
multi-hop questions, switching from generated-answer scoring to fixed-option
scoring moves the ablation effect from 21-55 pts to 0-4.2 pts. Nothing about
"multi-hopness" makes a task workspace-dependent; having the scored answer
transit the ablatable workspace does.

**Rhyme-at-scale wrinkle.** Rhyme was the signal check's biggest casualty at
4B (-63 pts) but is nearly immune at 8-9B, despite the answer direction
sitting in the ablated top-20 in 27% (Qwen) / 45% (Gemma) of rhyme GEN items.
Larger models evidently recompute the rhyme target after mid-band ablation
rather than depending on the buffered copy — a capacity/redundancy effect
worth a panel in the paper, and the reason OURS as pre-registered does not
pass as a three-task conjunction.

**Answer-direction leak in FIXED items** (excluded from primary cells, noted):
Qwen 3.3% / 6.7% / 10% (single/multi/rhyme); Gemma 15% / 20% / 55%. Leaks are
morphological variants that slip the word-boundary prompt-literal rule
("grapes", "winters", "parisian", "australians"); Gemma leaks more because
its tokenizer surfaces plural/adjectival forms as single tokens. Including
leaked items changes no FIXED cell by more than 1.7 pts (all-item rows in
metrics.json) — even when the answer's direction IS ablated, fixed-option
scoring survives.

**Mechanism note.** In Gemma multi-hop GEN the answer token itself is in the
ablated top-20 only 6.7% of the time, yet accuracy drops 45 pts: the ablated
set instead carries the INTERMEDIATE hop (e.g. " France", " Italy" for
"capital of the country where Lyon is located"). Ablating the workspace copy
of the intermediate kills generation, while the same item scored over fixed
options is untouched.

Wall-clock: Qwen3-8B 1987 s (33 min, incl. 30-item pilot reuse), Gemma-2-9B
954 s (16 min), single MI300X co-running with the scaled ignition runs
(`../ignition-scaled/`).

## Run

```bash
# from the repo root, deps installed (see top-level README); 8-9B models
# want a large GPU
python selectivity-fixed-option/run.py --limit 5   # pilot, 5 items/cell
python selectivity-fixed-option/run.py             # full battery (resumes via per-item checkpointing)
python selectivity-fixed-option/run.py --analyze   # recompute metrics.json from items.jsonl
python selectivity-fixed-option/run.py --config config-gemma.yaml --out results-gemma   # Gemma-2-9B arm
```

Device auto-detects CUDA / MPS / CPU. Set `JLENS_REPO` if the Jacobian-lens
reference repo is not at `vendor/jacobian-lens`.

## Files

- `run.py` — both formats, ablation + controls, leak audit, verdicts.
- `tasks.py` — item batteries + hand-authored distractor families.
- `config.yaml` / `config-gemma.yaml` — per-family configs.
- `results/` (Qwen3-8B), `results-gemma/` (Gemma-2-9B) — `items.jsonl`
  (per-item records incl. ablated concepts and leak matches), `metrics.json`
  (manifest, 3x2 accuracy grid, criterion verdicts for both predictions).
  Run logs are runtime artifacts, not shipped.
