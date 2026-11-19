# P10 Revision Log — execution of `p9-concern-ledger.md`

Stage: P10. Date: 2026-11-19. Every row names the concrete passage edited. All changes are
evidence-led; no claim was strengthened. Both trees recompiled with 0 errors after the edits.

## Major items — Paper B (RB-M1..M6)

| id | action | where it landed |
|---|---|---|
| RB-M1 (allocation headline scope) | REWRITE | abstract: "six of eight held-out four-layer configurations --- the deep regime only: the pre-registered mixed-depth screen was not met (37.5%), and the search beats or ties a random-allocation control on the majority of cells"; intro capability bullet carries the same scope |
| RB-M2 (intro unbacked 21-offset grid) | REWRITE | intro contribution bullet: "the train-mask-selected bias grid (step 0.05) in all nine graph-by-latency cells" (code-verified wording) |
| RB-M3 (anchor balance) | KEEP | abstract now reports both anchors: SpikeGraphormer 74.0 and SpikeGCL 86.6 ("external anchors, not audited rows") |
| RB-M4 (conclusion 2-23x qualifier) | KEEP | conclusion: "under the static-update accounting --- the ordering is convention-conditional (sec:accounting)" |
| RB-M5 (related-work "closed-form and training-free") | REWRITE | related work: "derived in closed form from an exact error identity --- and where that derivation fails to transfer across graphs, the working procedure is a validation-only window search rather than the formula" |
| RB-M6 (remaining minors) | KEEP/ROUTED | depth-fragment deduplicated; gain range now "≈420–710"; abstract L≥3 → L=3 (measured scope); q999 saturation direction corrected (quantisation-side collapse, not saturation); numeric fixes below |

## Major items — Paper A (RA-M1..M5)

| id | action | where it landed |
|---|---|---|
| RA-M1 (census basis + title) | KEEP | title → "An Audit of Every Code-Releasing Method in a Defined Survey"; Methods gains "The surveyed population" paragraph (seven-method survey selection, Nov-2026 repo checks, bounded claim) |
| RA-M2 (failure-mode-3 misdescription) | REWRITE | abstract + conclusion + results: "silently mix the two splits across their baseline rows (a GCN row consistent with Split I, a SpikingGCN row consistent with Split II)" |
| RA-M3 (noise-source mixing) | REWRITE + CORRECTION | `audits/jA1-outcome.md` correction appendix; new Results subsection `sec:transport` with the source-consistent table (Cora 1.35x, CiteSeer 1.37x proxy-labelled, PubMed 0.73x); abstract and conclusion rewritten to the marginal/confounded verdicts; packet fact A-transport corrected |
| RA-M4 (criterion vs classification) | REWRITE | Methods: graded scale (reproduction / reproduction with stated deficit / disagreement); SpikeGCL Cora = reproduction with 0.8-point deficit, stated as such in abstract |
| RA-M5 (convention counts) | KEEP | unified to five variants everywhere |

## Major items — cross-paper (RX-M1..M12)

| id | action | where it landed |
|---|---|---|
| RX-M1 (grid duplication beyond anchors) | REMOVE_NONSCIENTIFIC+ | orphan `tables/tab15.tex` deleted (never compiled; duplicated A's grid); division-of-evidence statement now true as written |
| RX-M2 (tab caption split contradiction) | REMOVE_NONSCIENTIFIC+ | resolved with the deletion of the orphan table |
| RX-M3 (threshold Squirrel absent) | KEEP | threshold subsection: Squirrel cell added (plateau 0.75–1.25, θ=1.0 inside one s.d.); abstract "five graphs including two heterophilic holdouts"; packet fact B-threshold updated |
| RX-M4-M6 (unregistered contribution facts) | BLOCK_EVIDENCE → repaired | packet B gains B-closedform-scoring (105 cells, 0.08/0.40, L=3 3.5), B-ranking (ρ=0.967/4.6), B-ensemble (+1.21/+1.06/+1.11), B-anchor-external (74.0/86.6) |
| RX-M7 (80.5 anchor) | NARROW | covered by B-anchor-external; Paper A keeps its "context only, not an audited row" framing |
| RX-M8-M9, M12 | KEEP | (81.0) parenthetical dropped; "to within a point" → "to within two points"; SpikeGCL row values kept with "consistent with" phrasing |
| RX-M10-M11 | MERGE | covered by RB-M5 and RB-M4 |

## Numeric minors

- 0.66/0.64 vs 0.6837: Pareto passage aligned to the registered series ("their bests 0.6837 and 0.6511").
- transposed 4.2/4.4 → "4.4 and 4.2 points below the band, about twelve times the floor"; "the baseline" → "the rate-coded arm" (antecedent fixed).
- graph counts: threshold scan now five graphs (two holdouts) consistently in text and abstract.

## Disposition of remaining editorial nits

Copyedit-register items flagged as `REMOVE_NONSCIENTIFIC` in the ledger (spelling, Delta
conventions, caption style, minor phrasing) are carried to P19 (`scientific-paper-writing` /
humanizer diagnosis) with their named locations; no scientific content depends on them.

## Recompiles

`tex-journal/main.pdf` and `tex_audit-journal/main.pdf` rebuilt after all edits: 0 errors each.
