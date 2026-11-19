# Correction Loop State — E-B1 prototype gate failures

Opened: 2026-11-18 at P4 (E-B1 prototype stage). Trigger: prototype gate verdict
`stop-and-return` (twice: v1 engine semantics, v2 allocation-rule floor behavior).

## Cycle 1 — engine semantics (class: implementation)

- Evidence: prototype v1 (`logs/jB1_prototype.log`, `results/jB1_prototype.json`): lockstep-freeze
  semantics under-count heterogeneous-window rates (acc_alloc 0.10–0.30 vs acc_unif 0.29–0.74,
  catastrophic); output identity internally exact (1e-7) — the check didn't cover the semantic
  requirement.
- Repair: one local engine fix — **staged semantics** (`convert_staged`: per-layer window with
  constant per-step input `A(rate_{l-1} @ M_l) + const + θ/(2T_l)`, exactly the core layer-1
  pattern with X → decoded previous rate; the identity carries with E_l = count_l/T_l − relu(B_l)).
  Lockstep-freeze retained as a labelled negative control.
- Result: prototype v2 (`logs/jB1_prototype_v2.log`) — collapse gone; alloc vs staged-unif gaps
  −0.3 to −5.6 pp; identity 0.0 exactly.

## Cycle 2 — allocation-rule floor behavior (class: mechanism) — OPEN

- Evidence: v2 per-cell (alloc − staged-unif) at savings 8–45%:
  within 1.5 pp band on 3/8 cells (T_l patterns with min window ≥ 4); outside on 5/8 cells — all
  failing cells have min T_l ∈ {1,2,3} (patterns [16,3,1], [16,2], [16,2,1]). The gain-equalized
  rule `T_l ∝ G_l` starves the smallest-gain layers below the window where the identity's
  midpoint-quantization term behaves like its RMS estimate; the tail (per-node errors on the 32–59%
  of nodes the margin certificate does not cover) flips decisions.
- The strong target and floors are NOT lowered. The rule is incomplete: it equalizes the propagated
  term `G_l/T_l` but omits a per-layer resolution floor for the quantization term.
- Repair candidate to evaluate (via `ccf-idea-optimizer` per the loop): floor-augmented rule
  `T_l = max(T_min, ceil(κ·(G_l/G_max)·T*))` with T_min derived from training-split margins only
  (e.g. smallest T with θ/(2T) ≤ 0.1 × median train margin), plus a floor-only uniform control
  (all layers at T_min) as the new falsifier. Fallback pivot: JB1a as primary, allocation reported
  as a bounded negative result.

## Status

Cycle 2 open. No claim text changes until the repaired rule re-pilots; expected_strong stays
binding; every failed cell stays in the record.
