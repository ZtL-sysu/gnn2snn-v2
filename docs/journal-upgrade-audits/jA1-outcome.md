# JA1 Outcome Record — Transportability vs Measured Seed Variance (Paper A)

Evidence: `logs/jA1_spikegcl_multiseed.log` (8 cells, seeds 2023–2026), `results/wp10_audit.json`,
`third_party/spikinggcn_official/wp13_results.json`, `audits/jA_replay-checksum.json` (pass).

## Measured seed spreads (4 seeds each)

- SpikeGCL Cora: mean 86.56, std 0.13 pp, range 86.38–86.68.
- SpikeGCL CiteSeer: mean 76.70, std 1.24 pp, range 75.32–78.26.
- SpikingGCN PubMed: per-seed [59.6, 63.1, 70.3] (seed 0 derived from the registered 3-seed mean
  0.6433 and wp13's seeds 1/2 values 0.631/0.703 — arithmetic identity shown), std 5.37 pp.
- Cora seed-2023 cell 0.8662 reproduces the audited cell exactly; CiteSeer seed-2023 re-run gave
  0.7826 vs the audited 0.7728 — a 1.0 pp same-nominal-seed rerun discrepancy, recorded as part of
  the audit's sensitivity evidence (not silently reconciled).

## JA1 criterion (pre-registered: split gaps ≥3× measured seed spread; PubMed re-scope candidate)

| graph | Split I/II gap | seed spread (std) | gap / (3× spread) | verdict |
|---|---|---|---|---|
| Cora | 8.0 pp | 0.13 pp | 20.5 | **pass — transportability dominates noise** |
| CiteSeer | 5.1 pp | 1.24 pp | 1.37 | marginal pass (above the bar; flag as marginal) |
| PubMed | 11.7 pp | 5.37 pp | 0.73 | **fails the bar → re-scoped** (split choice and seed noise are the same order; not deleted) |

Consequence per the frozen rule: the transportability finding is restated per dataset — strongly
supported on Cora, marginal on CiteSeer, and on PubMed the split gap and the seed variance are
confounded (a finding in itself, reported as such). No headline claim uses PubMed's split gap alone.

---

## CORRECTION (2026-11-19, P9 integrity review M3)

The multiplier table above mixed noise sources: the split gaps (8.0/5.1/11.7 pp) are SpikingGCN's
Split I/II differences, but the Cora/CiteSeer noise rows used **SpikeGCL's** seed spreads. The
source-consistent recomputation (SpikingGCN's own measured spreads where available):

| graph | split gap | matching noise source | gap / s.d. | gap / (3 s.d.) | corrected verdict |
|---|---|---|---|---|---|
| Cora | 8.0 pp | SpikingGCN 5-seed s.d. **1.97 pp** (per-seed 48.1–52.6) | 4.06× | **1.35×** | marginal (was: strong 20.5×) |
| CiteSeer | 5.1 pp | SpikingGCN n=1 → **SpikeGCL 4-seed s.d. 1.24 pp as labelled proxy** | 4.11× | 1.37× | marginal (proxy noise) |
| PubMed | 11.7 pp | SpikingGCN 3-seed s.d. 5.37 pp (derived) | 2.18× | 0.73× | confounded (unchanged) |

The correct headline: **all three split gaps exceed their matching seed noise by 2–4× the s.d.,
but only marginally clear the pre-registered 3σ bar (1.35×/1.37×), and PubMed does not (0.73×).**
No dataset-level "strong" verdict stands. The manuscript, abstract and packet fact A-transport are
corrected accordingly; the original table is retained above as the record of what was reviewed.
