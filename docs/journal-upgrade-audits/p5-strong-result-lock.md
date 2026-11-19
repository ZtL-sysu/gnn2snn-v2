# P5 Gate Report — Strong-result lock

Stage: P5. Date: 2026-11-18. Contract: `journal-v2` (claim record v7). Verdict: **pass**.

## Binding capability claim — OBSERVED STRONG

**Deep-regime per-layer allocation by validation search** (`audits/jB1-search-arm-prereg.md`
criteria, frozen before the arm ran): held-out L=4 **6/8 cells (75% ≥ 2/3)** keep the ANN−1.5 pp
band with **22.7–93.4% fewer total layer-updates** than the best uniform T* (squirrel 4/4, chameleon
2/4); pilot L=4 6/8 (44–90%); random-search falsifier passed. The pre-registered regime boundary
(L=3 1/8; mixed criterion 37.5% not met) is part of the claim, not hidden.
Attribution: idea family LD-SNN (ISLPED 2022); graph instantiation and regime measurement are
this work's evidence. Artifacts: `results/jB1_search_pilot.json`, `results/jB1_search_heldout.json`,
`audits/jB1-search-arm-outcome.md`.

## Retained strong cells — re-verified

Identity (staged residual 0.0; registered ~2e-6); depth/width laws and margin theorem (boundary
results with the L=4 loss as the message); order invariance (membrane 0.0 vs count 0.39–1.61 pp);
threshold law incl. the **now-complete Squirrel cell** (θ=1.0 within 1 s.d. of plateau; q999
control −22.9 pp). Four-arm non-domination by direct-trained arms holds under all three cost
accountings.

## Paper A — strong outcome with pre-registered scoping

Audit grid complete with seed spreads; transportability **per-dataset** (Cora 20.5× the 3σ bar (61.5× s.d.) strong;
CiteSeer 1.37× (4.1× s.d.) marginal; PubMed 0.73× (2.2× s.d.) → confounded, re-scoped per the frozen rule); convention
causality corrected (row-normalisation decisive; row-4 mislabel fixed); verbatim quotes anchored;
replay 0 mismatches.

## Failure-signal handling (contract-compliant, no target lowering)

- JB1a sweep-free selection: frozen gate failed (band 1/3; κ=[1.2, 11.0, 2.5]) → claim removed,
  negative reported with per-graph mechanisms and the in-distribution κ=1.0 + shuffled-c control.
- Closed-form allocation R1/R2: frozen pilots failed → regime-map negative (windows ≥4 at
  8–29% on 3/8).
- SOP cheapness claim: pre-declared event-driven convention fired the WP2 criterion → claim
  removed from title/abstract; accounting-sensitivity reported with all conventions.
- Gap predictor (wp7): retained as the third failed cross-graph route.
Each replacement claim had its own pre-registered binding target (jb1a-primary-prereg;
jB1-search-arm-prereg) and fresh baseline calibration — the recovery protocol, not a lowered bar.

## Uncertainty and reprovidence

7-seed cells with paired bootstrap where compared; per-seed rows in every new JSON; exposed test
cells labelled; search/calibration frozen before held-out evaluation; 0 replay mismatches.

## Gate decision

All required experiments complete; the papers' main claims meet their strong targets under the
current claim record (v7), with the registered failure signals applied honestly. **P5 passed.**
P5A runs the independent pre-draft scientific review and full integrity audit over
`audits/pre-draft-evidence-dossier.md`.
