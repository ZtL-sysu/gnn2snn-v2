# Deviation Audit — Paper A / Paper B vs. Guidance Lineage, Fable 5.2 Material, and f-guidance-builder-v1

Date: 2026-11-18. Run root: `f-research-v1-run-20261118-181848-journal-upgrade/`.
Review execution: independent-agent read audits (Paper A, Paper B) plus owner verification of every
high-severity item cited below. Fable 5.2 material is external planning input; its claims were used
only where independently verified.

## 1. Authority map (what each paper is actually contracted to)

| source | what it covers | status |
|---|---|---|
| `contracts/guidance.md` (v1, in prior run) | original method: neighborhood temporal calibration + TPP, flood-graph domain | **superseded** — method hypothesis falsified; guidance-v3 header records v1→v2→v3 lineage |
| `auto-paper/20261104-060416/predictive-laws-gnn2snn/guidance.md` | Paper B, builder-complete contract: identity, predictive laws, 4-arm Pareto, order invariance | active for Paper B's claims; strong target **partially failed** (see §2) |
| `guidance-v3.md` (snapshotted in `inputs/`) | Paper B final claim set after failure repositioning; Non-goals; 3 deferred items | active; deferred items still open (see §2) |
| `execution-contract-v2/v3.md` (prior run) | two-paper output set, Paper A E&D reformat, WP15 threshold round | Paper A's effective scope contract |
| Fable 5.2 GitHub/Notion snapshots | a **different** research direction (dynamic vision graphs) + literature corrections | comparison input only, not a contract |

Key structural finding: **Paper A has no f-research-v1 guidance file of its own.** It was created
inside execution-contract-v2's "two manuscripts" authorization as an audit/E&D paper. Its claims are
disciplined and evidence-backed, but its scope exists only at execution-contract level. For journal
submission this run supplies a proper guidance contract for Paper A (in `guidance.md`).

## 2. Paper B vs. its guidance contracts

### 2.1 Binding strong-result cells (verified against result JSONs this run)

| predictive-laws guidance `expected_strong` | measured | verdict |
|---|---|---|
| prediction MAE ≤ 1.0 pp on held-out cells, required-T exact ≥ 70% | `wp7_predict.json`: LOCO MAE 4.64 pp, LODO 5.41 pp, required-T exact 2/12 (16.7%); `_verdict: strong_pass=False, failure=True` | **FAILED — and per the guidance's own failure signal the claim was removed from title/abstract and repositioned**; delivered title "Spectral Gain, Activation Mass, and an Order-Invariant Readout" reflects this; failure is reported in the manuscript (confirmed by terminal review's limitation list) |
| conversion reaches ANN−1.5 pp with fewer SOPs than every external arm on ≥ 2/3 graphs | `wp2_pareto.json`: `graphs_fewer_sop=2/3`, `strong_pass=True`, 0 graphs dominated | **MET** |
| membrane spread ≤ 0.01 pp with count spread ≥ 0.3 pp in same cells | `wp6a_misalignment.json`: membrane 0.0 pp; count 0.39–1.61 pp | **MET** |

Conclusion: Paper B's delivered claims (identity, laws, Pareto, order-invariant readout, threshold
law) are consistent with the evidence; the one guidance binding target that failed is honestly
recorded and repositioned rather than fudged. This is compliant with the research-recovery rule
(versioned retarget to guidance-v3 claims, prior failure preserved). **No numeric overclaim found in
these headline cells.** The deviation is real but lawful: journal-v1 must not silently restore the
failed "predictive MAE ≤ 1 pp" claim.

### 2.2 Open deviations and deferred promises

| # | item | source | severity | needed action |
|---|---|---|---|---|
| B-1 | **No new positive capability** (M-B1): WP15 validates a configuration already used; review says +1 score needs an identity-derived method, e.g. per-layer latency allocation from G | `ccfa-skill-review-B-wp15-final.md:73-77` | major (Significance) | **new experiment** (journal E-B1): identity/margin-derived per-layer step allocation, preregistered, held-out graphs, matched-accuracy fewer-steps vs uniform T |
| B-2 | External architecture (SGC+512-wide) beats conversion on Cora/CiteSeer; guidance-v3 deferred the **capacity-aligned** re-comparison ("同宽同传播阶数") | `guidance-v3.md:48` ("还欠的三件事" #3); `wp2_sgc512.json` is 512-wide vs our 16-wide | major (claim robustness) | **new experiment** (E-B2): matched width/propagation-order external comparison |
| B-3 | Squirrel held-out cell incomplete (2/7 thresholds, `infeasible-with-evidence`) | review M-B3 | moderate | **new experiment** (E-B3): cheap protocol (7 trainings, one per seed) to complete the held-out cell |
| B-4 | Two-layer transductive scope (M-B2, +0.5): depth boundary established, not relaxed; second architecture family absent (`arch_generality.json` is operator variants, not a second family) | review M-B2 | moderate | evaluate at P1A; execute only if mechanism and budget hold (E-B4 optional) |
| B-5 | 22 pages / 19 tables vs journal limits (M-B4); abstract second paragraph dense; "A practitioner might instead..." conversational | review M-B4 + writing concerns | moderate (presentation) | journal restructuring in P6–P16 |
| B-6 | **Novelty wording contradicted by known precedents (HIGH)**: `tex/sections/intro.tex:28` says "For graphs, that second route has not been formulated" — but One-Spike SNN (2025, training-free Text-GCN conversion), PMSM (2025, tactile graph), and Lan et al. ICCV'23 (point-cloud, max→mean + retrain) exist; `hwang2024onespike` in `refs.bib:520` is cited only as vision "phase coding" (identity of that entry vs the Fable Text-GCN record must be resolved). Related work's "no published analysis of ... conversion error" is correctly scoped, but the intro claim is not | **high** (reviewer-fatal if unaddressed) | `inputs/fable/10-claude-audit.md`, `09-paper-ideas.md`; `tex/sections/intro.tex:28`; `related.tex:109` | P12/P14: narrow claim to "no error analysis for message passing", cite precedents with their limitation clauses, resolve One-Spike entry identity |
| B-8 | **Table 3 vs Table 1 bias provenance (MED-HIGH)**: tab3 "method as reported (default)" = 0.8021/0.6880/0.7684 = the *grid-searched* bias column of `beta_star_s7.json`, while Table 1 "FTBC β*" = 0.8026/0.6907/0.7740 is the closed-form `star` column; tab3 footnote "bit-identical to Table 1" is false as printed (up to 0.56 pp on PubMed); ablation deltas (readout −3.68/−6.24/−0.01 etc.) are computed against the grid series | **medium-high** (numeric integrity) | `tables/tab3.tex`; `results/beta_star_s7.json`; `experiments.tex:324-347` | E-B5: resolve which bias variant each delta measures against; fix footnote/numbers or recompute deltas; registry row for both variants |
| B-9 | **Per-channel bias claim contradicted (MED)**: "performs identically to the scalar one in all nine cells" vs `ablate_s7.json` (per-channel minus scalar = −0.30/+0.66/+0.19 pp at T=4; −5.89 pp PubMed T=1; identical only at T=8) | medium | `experiments.tex:332-335`; `results/ablate_s7.json` | W-B wording fix with measured deltas |
| B-10 | **T=8 win overclaim (MED)**: "exceeds the directly trained spiking arm on all three citation graphs" vs the paper's own floor rule listing PubMed T=8 (+0.20 pp = 0.5× floor) under "What does not survive" | medium | `experiments.tex:533-535`, `297-298` | W-B: "two of three robustly; numerically positive on the third" |
| B-11 | **Depth non-goal contract drift (HIGH, reconcilable)**: `guidance-v3.md:42` Non-goals include "深度>2", while the manuscript's depth law reports L=3/L=4 boundary results (4-layer loses 21 pp). The predictive-laws guidance (later, builder-complete) explicitly requires the depth law; the correct reading of the v3 non-goal is "no depth>2 *capability* claim" | high if unreconciled | `guidance-v3.md:42`; predictive-laws traceability "Depth law" | journal-v1 guidance explicitly authorizes the depth **boundary** law (negative result), forbids any depth>2 capability claim — recorded here as the reconciliation |
| B-12 | sgc512 superiority scope inconsistent (MED): abstract "two graphs" (correct) vs intro contribution 8 and §pareto naming Cora only | low-medium | `wp2_pareto.json`; `experiments.tex:169-173` | W-B wording alignment to "two graphs" |
| B-13 | tightness range drift + rounding (LOW): 1.3–4.1× vs 1.4–2.3× (`theory_tightness.json` overall 1.33–4.08, FTBC-only 1.33–2.33; the 3.2–6.7× prose figure is reproducible only over T∈{4,8} cells of `bound_over_actual_median` — state the subset); "420–710" vs 417.7–708.5; E[U]=0.481 unbacked in JSONs (nearest 0.4796/0.4831 — state aggregation); −3.68 vs −3.69 rounding drift | low | `experiments.tex:463-464`; `discussion.tex:37` | W-B: pick subset, state aggregation, fix rounding |
| B-15 | **New-claim novelty boundary (found at P1B F2 gate)**: per-layer timestep allocation in ANN→SNN conversion is established prior art (PASCAL arXiv:2505.01730; MT-SNN Front. Neurosci. 2026; LD-SNN ISLPED 2022 owns the update-savings phenomenon) — the E-B1 claim must be the closed-form gain/margin derivation + freeze realization + graph setting, never "first per-layer allocation" | high (wording fatal if unaddressed) | `audits/ccfa-novelty-map-E-B1.md` | re-scoped in `guidance.md`/contract v4; PASCAL-adapted + LD-SNN-search + isolation arms added to E-B1 |
| B-14 | 4 Overfull boxes remain (tab19 caption 21.88 pt, method 4.41 pt, tab14, tab18) despite guidance-v3 deferred item 2 | low (presentation) | `tex/main.log` | W-B journal typesetting |
| B-7 | Anonymous "companion" cross-citation is a double-blind compliance item owned by author | review X1 | compliance | declaration-mode author-managed; record, do not infer |

### 2.3 Fable 5.2 material vs Paper B

- Fable's core direction (dynamic vision graphs: dynamic kNN + max aggregation, training-free) is
  **not** Paper B's scope; Fable itself lists "fixed graph, linear aggregation, low-timestep
  conversion" as a direction with existing precedents (One-Spike SNN, PMSM) — i.e. it judges Paper
  B's *method family* to be populated. Paper B survives this only because its contribution is the
  error identity / laws / readout analysis, not "first graph conversion". The delivered related work
  already uses the safe phrasing ("no published analysis of ... conversion error") and cites
  One-Spike SNN (`hwang2024onespike` in `tex/refs.bib:520`). **Deviation to close: B-6.**
- Fable's energy-accounting caution ("static-graph SNN may not save energy") is already respected:
  guidance-v3 Non-goals forbid energy claims and the manuscript reports spike/SOP proxies only. **No
  deviation.**
- Fable's temporal-misalignment distinction (layer-shuffle vs neighbor-phase) matches guidance-v3's
  section 2 boundary (ICML'25 TPP distinguished). **No deviation.**

## 3. Paper A vs. its contract and evidence

Paper A's contract-level source is execution-contract-v2/v3 (E&D contribution type), not
guidance-v3. Against its own claims and the audit evidence, independent review found:

| # | finding | severity | evidence | needed action |
|---|---|---|---|---|
| A-1 | **SpikeGCL spread sentence wrong on three counts**: (i) "Both differences sit inside the reported spread" — Cora gap −0.78 pp vs published ±0.6 pp is NOT inside; (ii) the quoted "audited-code spread 0.48–0.53 Cora" is backed but **misattributed** — it is SpikingGCN Cora's own 5-seed spread (`wp10_audit.json` per_seed [0.515,0.481,0.494,0.484,0.526]), not SpikeGCL's (both SpikeGCL cells are n=1, no spread); (iii) "0.60–0.70 on PubMed" is unbacked (SpikingGCN PubMed n=3 stores no per-seed values; SpikeGCL PubMed was never run) — integrity audit M2 corrected the first-pass audit note | **high** | `tex_audit/sections/results.tex:76-82`; `results/wp10_audit.json` (verified this run) | **new experiment** (E-A1): multi-seed SpikeGCL runs → measured SpikeGCL spread; re-attribute or remove the SpikingGCN spread sentence; store per-seed values for every multi-seed cell; compare to published ±0.6 only where our spread is absent |
| A-2 | **Input-convention count inconsistent**: abstract/intro/conclusion say "four plausible input conventions"; results lists three bullets + "a fourth combination"; the 13.0 diagnostic values have no dedicated machine-readable artifact (`wp13_results.json`/`wp14_citeseer.json` not in `results/`) | **high** | `tex_audit/sections/abstract.tex:9-11` vs `results.tex:25-41` | **new artifact** (E-A2): input-convention diagnostic JSON (4 rows: convention, seeds, values, log paths) + consistent counting in prose |
| A-3 | **"Complete auditable population" overstated**: population = 7 methods from this paper's related work, not a field-wide search protocol | **high** | `tables/census.tex:4-15`; `results.tex:108-121` | writing fix: bound the claim ("complete among the seven surveyed methods") + documented search window/terms in protocol |
| A-4 | Single-run cells (SpikeGraphormer public split, both SpikeGCL cells, SpikingGCN CiteSeer) lack seed variance | medium | `tables/audit.tex` | **new experiment** (E-A1 extension): 3–5 seeds where runtime permits (WSL `spikg` env verified alive this run) |
| A-5 | SpikeGCL year inconsistency (2024 in text vs 2023 arXiv in `refs.bib:295-301`) | medium | — | writing/citation fix at P12 |
| A-6 | 80.5 "companion" number weakly traceable inside Paper A (anonymous "under review" ref) | medium | `results.tex:123-130` | keep but strengthen provenance wording, or drop from comparison |
| A-7 | "authors' own script / zero changes" too absolute vs two-line patch + hardcoded path recorded in protocol table | low | `tables/protocol.tex:4-11` | wording fix |
| A-8 | Split-mixing inference paraphrases rather than quotes source split descriptions | low-medium | `results.tex:90-106` | analysis addition: verbatim split quotes table (E-A3) |
| A-9 | Independent parse/replay check: JSON→table traced, but raw-log→JSON derivation not independently recomputed | medium | — | **new artifact** (E-A4): replay/checksum script |

### 3.1 Fable 5.2 material vs Paper A

- Fable's "failure census" (candidate 1) is a *visual-graph conversion* failure census (neighbor
  flips, max-winner changes); Paper A is a *reproduction audit of trained spiking GNNs*. Adjacent in
  spirit (both ask "what actually reproduces/works"), different objects. **No scope conflict**, but
  Paper A's intro should not imply it covers conversion-method failures.
- Fable's audit-of-Claude corrections (e.g. "graph networks have never been converted" is false)
  reinforce Paper A's own discipline about census wording (A-3).
- SGNNBench is already cited in both bibliographies. **No deviation.**

## 4. Against the f-guidance-builder-v1 standard

- `predictive-laws-gnn2snn/guidance.md` (Paper B) is builder-complete: all 9 required sections,
  CCFA startup packet, feasibility contract, traceability checklist, claim/finding separation. **The
  paper's deviations are from this contract's targets (§2), not from the builder's format.**
- `guidance-v3.md` is a condensed claim contract (not builder-format) but was produced as a versioned
  correction record; its Non-goals and evidence boundaries match builder discipline (no energy/SOTA
  overclaim, explicit "we are not aware of" scoping). Its three deferred items (figure-1 quality,
  overflow fixes, capacity-aligned comparison) — items 1–2 were done in the delivered PDFs; **item 3
  is B-2 above**.
- Paper A's missing guidance-level contract is the largest structural deviation from the builder
  standard. This run's `guidance.md` supplies it before any new experiment (P0/P1).

## 5. Quantified deviation summary

- **Paper B: 3 high-severity items** (B-6 novelty wording vs known precedents, B-11 depth non-goal
  contract drift — reconcilable and reconciled here, B-1 capability gap), 1 medium-high numeric
  provenance (B-8), 3 medium (B-3, B-9, B-10), 2 low-medium (B-5, B-12), 2 low (B-13, B-14), plus
  B-4 (optional) and B-7 (author compliance). Headline numeric claims verified correct by both the
  independent audit and owner spot-checks; one binding guidance target legitimately failed and was
  repositioned.
- **Paper A: 3 high-severity claim/provenance defects** (A-1, A-2, A-3), 3 medium evidence gaps
  (A-4, A-6, A-9), 2 low wording/citation (A-5, A-7), 1 analysis addition (A-8).
- **Cross-cutting: 0 fabricated numbers found.** Every defect found is a wording/provenance/scope
  defect or a missing experiment, not a falsified measurement.

## 6. Experiment consequences (journal E-programme)

| id | paper | action | replaces / closes |
|---|---|---|---|
| E-B1 | B | per-layer (margin/G-derived) step allocation + weight-derived latency selection, preregistered + falsifiers | B-1 (M-B1) |
| E-B2 | B | capacity-aligned external architecture comparison | B-2 (guidance-v3 deferred #3) |
| E-B3 | B | Squirrel completion, cheap 7-training protocol | B-3 (M-B3) |
| E-B4 | B | second architecture family (optional, P1A gate) | B-4 (M-B2) |
| E-B5 | B | resolve Table 1/3 bias provenance (grid vs closed-form), recompute or restate ablation deltas | B-8 |
| E-A1 | A | multi-seed SpikeGCL/SpikingGCN CiteSeer cells → measured spread | A-1, A-4 |
| E-A2 | A | input-convention diagnostic artifact (4 rows, logs) | A-2 |
| E-A3 | A | verbatim source-split quote table | A-8 |
| E-A4 | A | raw-log→table replay/checksum script | A-9 |
| W-A | A | census bounding, year consistency, "own script" wording, companion provenance | A-3, A-5, A-6, A-7 |
| W-B | B | intro novelty narrowing + precedent citations (One-Spike identity check, PMSM, Lan, DCGS); per-channel-bias and T=8 wording; sgc512 scope; tightness/rounding; abstract density; register; overfull/table restructuring | B-6, B-9, B-10, B-12, B-13, B-14, B-5 |

Both independent read-audits (Paper A, Paper B) completed; every high-severity row above is
owner-verified against the named JSON/TeX sources before inclusion.
