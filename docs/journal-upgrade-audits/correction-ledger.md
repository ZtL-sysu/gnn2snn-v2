# Correction Ledger — E-B1 allocation-rule repair candidates

Generated with `ccf-idea-optimizer` (standard repair mode), 2026-11-18. All candidates stay inside
the original method family (graph ANN→SNN conversion; identity-derived per-layer windows; staged
engine). Binding target unchanged: ≥20% total-update saving vs best uniform T* on ≥2/3 reachable
held-out configs at the ANN−1.5 pp band. No test-set tuning; training-split margins only. Planned
claims only.

## Diagnosis (normalized)

- **Problem:** the gain-equalized rule `T_l = ceil(κ·(G_l/G_max)·T*)` starves low-gain layers to
  windows {1,2,3}; at those windows the identity's midpoint-quantization term degrades faster than
  its RMS estimate (discrete rate alphabet {0, 1/T, ...}, per-node tail errors), and the margin
  certificate covers only 41–68% of nodes — tail flips cost 2–5.6 pp on 5/8 pilot cells.
- **What already works:** staged engine exact (identity 0.0); moderate windows (≥4) hold the band
  at 8–29% savings on 3/8 pilot cells; headroom G_max/Ḡ up to 2.6.

## Candidates

### R1 — floor-augmented, theory-derived (RECOMMENDED)

`T_l = max(T_min, ceil(κ·(G_l/G_max)·T*))`, with `T_min` derived from the identity's own
quantization bound on **training-split** margins: smallest T such that
`G_max·θ·√H_active/(2T) ≤ 0.5·Δ_p10_train` (p10 train margin quantile; the p10 choice is a
pre-registered strictness constant, not tuned). Mechanism: the floor bounds the per-layer
quantization tail that the equalized propagated term cannot see. Falsifier: floor-only uniform
control (all layers at T_min) must lose to the floor+gain rule on the saving axis at comparable
accuracy — otherwise the gain term is dead weight and the claim re-scopes to "floor suffices".
Expected saving under unchanged target: pilot's ≥4-window cells held band at 8–29%; L=3
`[16,6]` ≈ 25%, L=4 `[16,6,6]` ≈ 31% → ≥20% on ≥2/3 reachable is plausible but not guaranteed
(honest failure signal stands). Cost: 1 re-pilot (2 seeds × 2 datasets, ≤30 min) + full campaign.

### R2 — per-layer sufficiency calibration (absolute κ_l)

`T_l = min{T : G_l·ĉ_l(T)·√H_active/Δ_p01_train ≤ 1}` where `ĉ_l(T)` is the measured train-node
logit perturbation per unit at window T (calibrated on training splits). Mechanism: absolute
per-layer sufficiency against the 1% tail, not relative equalization. Falsifier: shuffled margins
across nodes must calibrate worse. Risk: conservative inflation (κ_l > 1) may push savings below
20% → then JB1a becomes primary and the allocation claim reports the bounded negative. Cost: one
calibration sweep (≤1 h) + campaign.

### R3 — hybrid coarse+fine two-phase schedule

Phase 1: every layer runs `T_min` (midpoint-corrected); phase 2: only layers with the largest
gain-weighted residual (statically derived from G_l and train-activation variance) extend to their
R1 windows. Mechanism: coarse resolution first, fine resolution only where the identity says it
pays. Falsifier: phase-2 selection shuffled across layers. Accounting is the pinned ΣT_l. Cost:
engine change + campaign (highest engineering risk of the three).

### R4 — pivot to JB1a (fallback)

Weight-derived latency *selection* as the primary capability; allocation reported as a
regime-bounded negative (fails below window 4, holds above). No new engineering.

## Ranking (likelihood of meeting the binding target without test-set tuning)

**R1 > R2 > R3 > R4.** R1's floor is derived from the same identity the claim is built on, is the
cheapest, and preserves the gain term's falsifier. R2 is the safe conservative backup; R3 has the
best ceiling but the most new machinery; R4 is the honest fallback.

## Cycle outcomes (appended)

- **R1 (floor-augmented) re-pilot = FAIL** (`logs/jB1_repilot_r1.log`): the pre-registered floor
  formula pinned `G_max` (not per-layer `G_l`) against the p10 margin and clipped to the 64 cap on
  every cell (`T_min=64`) — the rule degenerated to uniform-64 (−300% "saving"; gain term
  untestable: `gain_term_beats_floor_only=False`). Honest no-go; constants untouched; verdict
  `escalate-R2-or-pivot` per the pre-registration.
- **R2 (per-layer absolute sufficiency) constants frozen before its pilot** (audits
  `jB1-rule-v2-prereg.md` addendum below): `T_l = min{T in 1..64 : G_l·θ·√(H_act·d_l)/(2T) ≤
  0.5·Δ_p01_train}`, d_l = width of the matrix immediately downstream of layer l (clipped at
  H_act), Δ_p01_train = 1st percentile of train-node top1−top2 margin. Pass-to-full (4-seed
  cora/citeseer L∈{3,4}): ≥5/8 cells in the 1.5 pp band of staged-uniform at ≥15% saving.
  Escalation → R4 pivot (JB1a primary; allocation regime-bounded negative).
- **R2 re-pilot = FAIL** (`logs/jB1_repilot_r2.log`, 16 cells): the worst-case certificate
  over-provisions on every cell — all T_l clip at the 64 cap (savings −300% = unreachable), gain
  term untestable. This is consistent with the known 3.2–6.7× bound conservatism; no constants
  were adjusted. Per the frozen ladder: **R4 pivot executed** — JB1a sweep-free selection becomes
  the primary claim with a fresh pre-registered binding target (`audits/jb1a-primary-prereg.md`,
  feasibility-contract v6); the allocation claim closes as a regime-mapped negative (windows ≥4
  hold the band at 8–29% saving on 3/8 v1-rule cells; windows <4 fail 2–5.6 pp). R3 was not in
  the frozen ladder and is recorded as untried, not silently skipped.
