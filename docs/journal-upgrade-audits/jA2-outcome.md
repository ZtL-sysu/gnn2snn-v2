# JA2 Outcome Record — Input-Convention Diagnostic (machine-readable, corrected)

Evidence: `results/jA_conventions.json` (5 rows, audited official pipeline, pinned `spikinggcn36`
env, seed 0, per-row scratch dirs + config hash recorded), `logs/jA_conventions5.log`.
All rows re-measured through `ReadData.conv_graph` (the Split-I feature path:
`mat = normalize_adj(A)@normalize_adj(A)@X`), the transform inserted at the documented stage.

## The five rows (Cora, audited pipeline)

| convention | acc | role |
|---|---|---|
| raw bag-of-words, self-loops | **0.515** | faithful cell (manuscript "50.0") |
| row-normalised features | **0.130** | collapse (manuscript "13.0") |
| row-normalised → max-rescale encoder input | **0.130** | conventional Poisson fix does **not** rescue (manuscript "Also 13.0") |
| raw bag-of-words, no self-loops | **0.515** | **correction** — unchanged from faithful; self-loops are not decisive |
| row-normalised, no self-loops | **0.130** | **this is the original "fourth combination" 13.0 — mislabelled in the old probe** |

## Correction to the manuscript (A-2 resolution)

The delivered text says "removing the self-loops as well, a fourth combination, gives 13.0 again".
The machine-readable re-measurement shows raw-without-self-loops holds 51.5%; the 13.0 value
belongs to *row-normalised* without self-loops. The journal text is corrected accordingly.

## Corrected causal statement (stronger than the original)

Row-normalisation is the decisive failure cause: every row-normalised variant lands at 13.0%
(chance-level-ish for 7 classes ≈ 14.3%), and the conventional max-rescale fix does not rescue it,
because rescaling a row-normalised encoder input restores the range but not the per-node scale the
Poisson sampler needs. Raw bag-of-words is robust to the self-loop choice (51.5% both ways).
