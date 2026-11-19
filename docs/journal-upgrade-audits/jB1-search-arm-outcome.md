# Search-Based Per-Layer Allocation — Outcome Record (positive capability, deep regime)

Evidence: `results/jB1_search_pilot.json`, `results/jB1_search_heldout.json`, logs
`jB1_search_pilot.log` / `jB1_search_heldout.log`. Preregistered in
`audits/jB1-search-arm-prereg.md` (procedure, criteria and falsifiers frozen before any run).
Attribution: the per-layer-window idea family is prior art (LD-SNN, ISLPED 2022; PASCAL 2025);
the graph instantiation, the validation-search protocol and the regime measurement are this
work's evidence. No "first per-layer allocation" wording.

## Headline result (pre-registered criteria)

- **Held-out confirmation (chameleon, squirrel × L=4 × seeds 0–3): 6/8 cells (75%) meet the band
  with ≥20% total-update savings — `arm_success: True`.** Savings 48.4–93.4%; four cells beat the
  uniform comparator in accuracy as well (squirrel 4/4 in-band, chameleon 2/4).
- Pilot L=4 subset (cora/citeseer × L=4): 6/8 cells, savings 44.5–89.8%.
- Pilot L=3 subset: 1/8 cells — little headroom where uniform T* ≤ 16.
- Random-search falsifier (8 random allocations per cell, same budget): search beats or ties
  random with fewer updates on ≥ half the cells in both runs (`random_falsifier_ok: True`).
- The mixed L∈{3,4} pilot criterion (≥2/3 of 16) was **not** met (37.5%) — reported as the
  regime boundary, not hidden. The capability lives in the deep regime.

## Why the deep regime (connection to the paper's laws)

At L=4 the depth law (`G^0.95`) makes the uniform requirement explode (T* = 64 on 6/8 held-out
cells): every layer is held to the worst layer's budget. Per-layer windows found by a
validation-only search release the low-gain layers from that budget — exactly the equalization the
closed-form rule attempted, but found by search where the weight-only certificates (R1/R2) failed
their sufficiency calibration. At L=3 the uniform budget is already compact (T* ≤ 16), so the
same search finds almost nothing to release.

## Claim record (created because the arm succeeded per prereg)

**Validated procedure (deep-regime capability; deep-regime re-scope confirmed at L=4 only, since
the pre-registered mixed L∈{3,4} pilot criterion was not met at 37.5%):** for graph ANN→SNN
conversion at L=4 (3 hidden layers), a validation-only coordinate-descent search over per-layer
windows preserves the ANN−1.5 pp
band with 22.7–93.4% fewer total layer-updates than the best uniform budget on held-out heterophilic
graphs (6/8 cells; squirrel 4/4), beating a random-search control. Shown on Cora/CiteSeer pilot
and confirmed on chameleon/squirrel; per-seed rows in the result JSONs. Scope: the tested GCN
family, L=4, the named graphs; the search costs a small validation sweep per model (the practical
trade the procedure buys).

## Honest boundary (kept in the paper)

- chameleon confirmation is mixed (2/4); the claim says "≥2/3 held-out cells", not "all".
- L=3 shows no meaningful headroom; the claim is scoped to the deep regime.
- Closed-form and certificate routes (JB1 allocation, JB1a selection) and the gap predictor
  (wp7) all failed their held-out gates; the search is the one route that works, at the cost of a
  per-model validation sweep. The identity/regime map explains where the headroom is.
