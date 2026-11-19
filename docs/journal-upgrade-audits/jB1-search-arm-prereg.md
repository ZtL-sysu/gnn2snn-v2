# Search-Based Per-Layer Allocation Arm — Pre-Registration (F2-mandated comparator)

Date: 2026-11-18, frozen before the arm runs. This is the LD-SNN-style comparator required by the
F2 novelty closure (`audits/ccfa-novelty-map-E-B1.md`, baseline arm list) and never yet executed.
It answers the question the closed-form negatives leave open: **can per-layer windows found by a
validation-only search deliver the step savings on graphs, and how much?** Attribution is explicit:
the idea family is prior art (LD-SNN, ISLPED 2022); the graph instantiation and its measurement
are this work's evidence. It does not revive any "first per-layer allocation" wording.

## Procedure (frozen)

- Engine: staged per-layer simulation (`convert_staged`), same as the closed-form arms.
- Search: per model, coordinate descent over per-hidden-layer windows T_l ∈ {1,2,3,4,6,8,12,16},
  maximizing **validation** accuracy at fixed total-update budget; 2 rounds; the search never
  sees test labels.
- Final evaluation: the chosen allocation is run once on the test split; reported once.
- Comparator: staged uniform at T* (smallest uniform T at band, integer grid 1..64), same models.

## Binding-style criterion (arm-level, pre-registered)

- **arm success:** test accuracy within the 1.5 pp band of staged-uniform at ≥20% fewer total
  layer-updates on ≥2/3 of the pilot configs ({cora, citeseer} × L∈{3,4} × seeds 0..3 = 16 cells);
  then confirm on held-out graphs (chameleon, squirrel).
- **arm partial:** saving ≥20% on ≥1/2 cells.
- **arm failure:** savings indistinguishable from zero or band misses everywhere → the search
  family does not transfer to graphs either, and the paper reports all three routes
  (predictor / closed-form / search) as failed with measured mechanisms.

## Falsifiers

- random-search allocation at the same number of search evaluations must do strictly worse;
- test-oracle allocation is reported as an upper reference only.

## Budget

≤1 h pilot; ≤2 h held-out confirmation; no new binding claim is created unless the arm succeeds,
in which case it is named in the claim record as a validated procedure with prior-art attribution.
