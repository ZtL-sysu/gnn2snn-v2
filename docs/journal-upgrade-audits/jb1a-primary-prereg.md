# JB1a Sweep-Free Latency Selection — Primary-Claim Pre-Registration (R4 pivot)

Date: 2026-11-18. Frozen before any JB1a pilot data. Supersedes JB1's binding allocation target
(which closes as a regime-mapped negative: weight-derived rules R1/R2 both failed their frozen
pilots — see `correction-ledger.md`; the ≥4-window regime holds the band at 8–29% saving on 3/8
pilot cells and is reported as a diagnostic regime map, not as a capability claim).

## Procedure (one-shot, no sweeps)

1. **Calibration (training/calibration graphs only):** fit the constant c on {cora, citeseer} so
   that `T_cert = clip(ceil(c · G_max · sqrt(H_active) / Δ_med_train), 1, 64)` reproduces the
   empirical minimal uniform T reaching the ANN−1.5 pp band on each calibration graph's
   **train-split** accuracy. ≤3 c variants logged; the variant is chosen on calibration graphs
   only, then frozen.
2. **One-shot deployment on a new graph:** compute `T_cert` from its trained weights and its
   **train-split** margins (`Δ_med_train` = median top1−top2 train margin). Run the standard
   uniform conversion (`core.convert`) at exactly that T, 7 seeds. No graph-specific tuning,
   no retries.
3. **Reporting-only sweep (after the one-shot is locked):** the integer grid T∈{1..64} is run to
   report `T*` (empirical minimal uniform T at band) and `κ = T_cert/T*`. The sweep may not feed
   back into T_cert.

## Binding targets (fresh, baseline-calibrated; do not lower)

- **strong (binding):** on ≥2/3 *reachable* held-out configs ({chameleon, squirrel,
  ogbn-arxiv} × the standard 2-layer protocol; reachability per the existing train/val rule),
  the one-shot T_cert reaches the ANN−1.5 pp band (test accuracy reported once) **and**
  κ ≤ 1.25 (near-minimal steps) **and** T_cert ≤ 64.
- **minimum:** band reached on ≥2/3 reachable configs with κ ≤ 2.
- **failure:** band missed on >1/3 reachable configs, or median κ > 2 → the claim is removed and
  the negative result reported (expected-consistent with `wp7_predict` required-T 2/12 transfer
  failure; that risk history is part of the claim text).

## Falsifiers and references

- shuffled-c control: c permuted across calibration graphs must calibrate strictly worse;
- oracle-tuned T* is a reference row only, never the claim;
- the allocation regime map (windows ≥4 vs <4) is reported alongside, unchanged.

## Budget

Calibration ≤1 h; one-shot runs 3 graphs × 7 seeds × seconds-minutes (arxiv the largest);
reporting sweep ≤2 h. Total ≤4 h. No test-set tuning; all constants above frozen now.
