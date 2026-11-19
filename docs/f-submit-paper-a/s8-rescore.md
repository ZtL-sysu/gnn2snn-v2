# S8 Re-score — Paper A ("What Reproduces in Spiking Graph Neural Networks? An Audit of Every Code-Releasing Method in a Defined Survey")

- **Stage:** S8 (re-score after S7 humanizer/revision edits; f-submit-v1).
- **Run root:** `tex_audit-journal/f-submit-v1-run-20261119-003907-paperA`.
- **Inputs read:** `module-quality-review.md` (frozen S6 scores, thresholds, named defects),
  S7-changed sources `sources/sections/{intro,results,conclusion}.tex`, dependent modules
  `sources/sections/{protocol,discussion,abstract}.tex`, `sources/tables/protocol.tex`, and the
  rubric `[WORKSPACE]\.claude\skills\f-submit-v1\references\module-quality-rubric.md`.
- **Frozen comparator thresholds (unchanged; not moved to obtain a pass):** Title 80, Abstract 82,
  Introduction 80, Related Work 74, Method 84, Experiments 86, Discussion 84, Conclusion 82,
  Figures/Tables 84. Comparator-score uncertainty remains +/-5 (S6 statement).
- **S7 change scope (confirmed by file mtimes: `intro.tex`/`results.tex`/`conclusion.tex` edited
  01:56-01:57; `protocol.tex`/`discussion.tex`/`abstract.tex`/`related.tex` unchanged since 00:39):**
  introduction (baseline-row articulation), results (five explicit convention bullets, baseline-row
  articulation, three-sigma bar anchor), conclusion (patch-count fix + baseline-row articulation).
- **Re-score scope:** introduction, results/experiments, conclusion (changed) + method (score
  depended on them: D-M1 count drift spans Method<->Results; D-M3 bar anchor spans
  Method/Abstract/Results/Conclusion). Frozen pre-edit scores: introduction 83, method 82,
  experiments 84, conclusion 80. All other modules unchanged; their S6 verdicts stand.

## 1. Named S6 defect closure verification (searched in current `sources/` text)

| Defect | Module | Status in current text | Evidence |
|---|---|---|---|
| D-I1 "same baseline" mis-statement | Introduction | **CLOSED** | `intro.tex` L40 now "quote $81.5$ and $87.7$ for baseline rows from the same comparison source without saying which split they mean"; contribution bullet L62-63 "baseline rows cited at $81.5$ and $87.7$ from the two splits of one comparison source, with no label". No "same baseline" phrasing remains in `intro.tex`. Consistent with the registered fact (GCN row vs SpikingGCN row). |
| D-M1 input-variant count drift | Method | **OPEN** | `protocol.tex` L53 still "We report the three combinations we tried in \secref{sec:sgcn}" vs results L28 "re-measured five of them", results L31-43 five explicit bullets, results L55 "any of the five measured input variants", abstract L12 "five-variant input-convention re-measurement", intro L24-25 "five input variants". The Results side is now internally consistent at five; the Method-side "three combinations" was not touched, so the drift persists (3 vs 5) and is now sharper. |
| D-M2 process-narration heading | Method | **OPEN** | `protocol.tex` L47 heading "What we verified about the data, and why it took a while." unchanged. |
| D-M3 unanchored pre-registered 3-sigma bar | Method | **PARTIALLY CLOSED** | Anchor added at the use-site: `results.tex` L142-144 "Against our three-sigma bar --- fixed in the audit protocol before the multi-seed runs, as the pre-registration rule for this comparison ---". Registration artifact ("the audit protocol") and temporal precedence are now named. Still missing: the verdict-scale definition (three verdict bands) and any definition in Method "How numbers are read" (`protocol.tex` L42-45 unchanged); results/abstract/conclusion use "marginal"/"fail" without a defined band scale. |
| D-E1 draft-correction narration | Experiments | **OPEN (override-class)** | `results.tex` L90-93 "an earlier draft attributed a spread measured on SpikingGCN's Cora seeds ($0.48$--$0.53$) to this pipeline, which was a mis-attribution, and quoted a PubMed range that no run of this pipeline produced (SpikeGCL's paper reports no PubMed cell)" — verbatim defect text, unedited. |
| D-E2 diagnostic-history aside | Experiments | **OPEN** | `results.tex` L41-42 "this is the combination an earlier diagnostic recorded as the ``fourth combination''; it was the normalisation, not the self-loops." — unchanged (now inside the fifth convention bullet). |
| D-E3 "same baseline" lead sentence | Experiments | **CLOSED** | `results.tex` L111-112 now "the two later papers we audited draw baseline rows from the two different splits of the same comparison source"; L113-120 identify the rows correctly (GCN $81.5$ "consistent with the Split~I family"; SpikingGCN $87.7$ "consistent... with SpikingGCN's \emph{Split~II} numbers"), with the "consistent with" hedging retained. |
| D-E4 "happy ending" | Experiments | **OPEN** | `results.tex` L65-66 "If the audit stopped here it would be a reproduction study with a happy ending." — unchanged. |
| D-E5 "we would welcome either" | Experiments | **OPEN** | `results.tex` L56 "A container image or a commit hash would settle it in an hour, and we would welcome either." — unchanged. |
| D-E6 repeated defensive negation | Experiments | **OPEN** | `results.tex` L89-90 "recorded as part of the sensitivity evidence rather than reconciled away" and L137 "recorded rather than reconciled" — both unchanged. |
| D-D1 "Both numbers are correct" gloss | Discussion | **OPEN (module unchanged by S7)** | `discussion.tex` L19-20 verbatim: "the later tables cite $81.5$ and $87.7$ without saying which. Both numbers are correct." Now also contradicts the corrected articulation in intro/results/conclusion (which identify two different baseline rows, GCN vs SpikingGCN). |
| D-D2 conversational falsifiability register | Discussion | **OPEN (module unchanged by S7)** | `discussion.tex` L73-75 "for you ... we would like to know ... we have mis-set a flag" — unchanged. |
| D-C1 "compatibility patches two of the three required" | Conclusion | **CLOSED** | `conclusion.tex` L6 now "(disclosing the compatibility patches the one non-zero-change pipeline required)". Count reconciles with `protocol.tex` L34-35 ("for two of the three pipelines the answer is zero") and `tables/protocol.tex` (SpikeGraphormer 2 lines; others 0): exactly one pipeline with non-zero code change. Minor non-blocking note: "patches" (plural) vs the caption's "one two-line patch" — the caption also lists two change items (torch in-place rule + hardcoded path), so the plural is defensible; tighten to "the compatibility patch" in a later polish if desired. |
| D-T1 tab:audit caption | Figures/Tables | Not in re-score scope (module unchanged, S6 PASS stands; non-blocking). |

**S7 additions verified as described by the workflow:** results L31-43 carries five explicit
convention bullets (raw+loops 51.5, raw no-loops 51.5, row-norm 13.0, row-norm+rescale 13.0,
row-norm no-loops 13.0) matching "five of them"; results L142-144 carries the three-sigma anchor;
intro L40/L62-63, results L111-120, conclusion L16-19 carry the baseline-row articulation
(conclusion states it in the S6-recommended form: "(a GCN row consistent with Split~I, a SpikingGCN
row consistent with Split~II)").

## 2. Re-scored modules

### Introduction — 83 -> 84 / 80 — PASS
The sole S6 blocker (D-I1 contradiction) is closed; the contribution bullet and the failure-mode-3
paragraph now state the registered fact (two distinct baseline rows quoted from one two-split
comparison source without labels). Chain strength unchanged (problem->gap->contribution, enumerated
findings, audit-deliverable contributions, scope paragraph). +1 reflects the corrected positioning/
selling-point consistency; no new defect introduced. The bare "This is not a citation error; both
numbers are correct" (L41-42) survives but is now framed by the rows articulation and reads as
"both quoted values are genuine source-table values", matching results L124-126 which S6 did not
flag; not a defect. 84 >= 80, no open contradiction, no override item.

### Method (dependent) — 82 -> 82 / 84 — FAIL
Module text unchanged by S7. D-M1 remains open (protocol.tex L53 "three combinations" contradicts
the now-consistent five-variant enumeration in Results/Abstract/Intro — a count contradiction, the
S6 blocking item). D-M2 remains open (process-narration heading). D-M3 partially closed at the
use-site (results anchor names the artifact and timing) but the verdict-band scale is still
undefined anywhere and Method "How numbers are read" still contains no bar definition. 82 < 84 and
D-M1 is an open contradiction. Fixes remain those prescribed at S6: delete "three combinations" and
say "five input variants" (or the five-entry phrase), retitle the data-provenance heading, add the
one-sentence bar/band definition (now ideally cross-referencing the results anchor).

### Experiments / Results — 84 -> 85 / 86 — FAIL
Genuine gains from S7: five explicit convention bullets resolve the results-internal count ambiguity
(old "four plausible input conventions") and raise specificity; D-E3 closed (rows articulation,
"consistent with" hedging kept); the three-sigma anchor adds a missing evidence callback at the
point of use. Remaining defects: **D-E1 (override-class draft-correction narration, L90-93)**,
D-E2 (diagnostic-history aside, L41-42), D-E4 ("happy ending", L65-66), D-E5 ("we would welcome
either", L56), D-E6 (both defensive-negation instances, L89-90 and L137). D-E1 alone is an
override-class item and cannot pass on score alone; the score also stays below the 86 threshold
(borderline under +/-5, as at S6). The remaining fixes are the S6-prescribed passage rewrites,
verbatim still applicable.

### Conclusion — 80 -> 82 / 82 — PASS
D-C1 closed: the patch-count callback now reconciles with Method and Table 2 under every reading
("the one non-zero-change pipeline" = the single pipeline with non-zero lines changed). The
failure-mode-3 sentence now carries the exact rows articulation (GCN row consistent with Split I,
SpikingGCN row consistent with Split II). Genre template intact (bounded restatement + per-dataset
verdict + one actionable recommendation; population bound restated); no new numbers introduced;
transportability magnitudes consistent (see closure). 82 >= 82; no open contradiction; no override
item. (Score reflects minimal recovery for removing the sole named defect plus the improved
articulation; the number density remains a style observation, not a defect.)

## 3. Core-claim closure re-run (Gates N1/N2)

- **Central question:** are published spiking-GNN accuracies commensurable, and what makes a number
  portable? — unchanged, and now articulated consistently at all use-sites.
- **Core claim:** no — three measured failure modes; the binding property is split transportability,
  stated per dataset at measured magnitudes (1.35x/1.37x/0.73x the pre-registered 3-sigma bar).
- **Forward trace:** title -> abstract (three graded verdicts + per-dataset transportability) ->
  intro (question, three findings, four contributions) -> method (replay protocol, graded scale,
  one-flag split comparison) -> results (audit grid, split effect, split-mixing with verbatim split
  definitions, transportability, census, side-by-side). Holds.
- **Backward trace:** conclusion (bounded census, per-dataset verdict, run-the-code) -> discussion
  (transportability-not-variance, remedies, threats) -> results (sigma ratios) -> method (pinned
  environments, sample sizes) -> stated gap. Holds, with one blemish: `discussion.tex` L19-20
  (D-D1) still frames $81.5$/$87.7$ as the same baseline's split values, contradicting the now-
  corrected articulation in intro/results/conclusion. The contradiction lives in Discussion's own
  text (D-D1, unchanged by S7, already a named S6 defect); it does not invalidate the Introduction/
  Results/Conclusion statements, which are the supported ones.
- **Transportability magnitudes consistent for A:** 4.06/3 = 1.35x, 4.11/3 = 1.37x, 2.18/3 = 0.73x
  — the sigma ratios (results L140-142) and bar multiples (abstract L30-32, results L143-145,
  conclusion L25-28) agree exactly across abstract/intro/results/conclusion; "two to four standard
  deviations" phrasing is identical in abstract and conclusion. Supporting magnitudes agree
  (50.0/80.7/64.3/77.6/70.6/72.5; 84.7/84.8+/-0.7/74.0/10.7 pp; 86.6/87.4/77.3/77.6; 8.0/5.1/11.7
  pp; five-to-twelve points; 3 of 7; 81.5 GCN~Split-I / 87.7 SpikingGCN~Split-II). The 3-sigma bar
  is now anchored (results L142-144), removing the unanchored-trust reading at the use-site.
- **"Allocation capability recalled in intro+conclusion for B":** not applicable — this run root
  contains only Paper A (`sources/` = one manuscript); no Paper-B module map exists here. No
  allocation-capability claim appears in this manuscript; nothing to verify for B in this run.
- **Status:** core chain closed end-to-end for Paper A with consistent magnitudes; the
  failure-mode-3 articulation is now correct in abstract/intro/results/conclusion; open items are
  the Method-side count drift (D-M1), the Method heading (D-M2), the bar's verdict-band definition
  (D-M3 residue), five Results-register/narration defects (D-E1 override + D-E2/D-E4/D-E5/D-E6),
  and Discussion's untouched D-D1/D-D2. No core claim lacks evidence; no conclusion-only claim.

## 4. Professionalism re-scan (changed modules only)

- Manuscript-production narration: still present in Results (D-E1 draft-correction, L90-93; D-E2
  diagnostic history, L41-42). Introduction and Conclusion: none.
- Conversational verdicts / rhetorical register: still present in Results (D-E4 L65-66, D-E5 L56).
- Repeated defensive negation: still present in Results (D-E6, L89-90 and L137).
- Unsupported trust claims: the "pre-registered" bar is now anchored at its use-site (artifact +
  timing, results L142-144); the band-scale definition remains missing (D-M3 residue) but no longer
  reads as an unanchored process-trust claim.
- No new defect introduced by the S7 edits; no stylistic complaint recorded on the sound passages.

## 5. Summary table

| Module | Before | After | Threshold | Pass | Blocking defect(s) remaining |
|---|---:|---:|---:|---|---|
| Introduction | 83 | 84 | 80 | PASS | — (D-I1 closed) |
| Method | 82 | 82 | 84 | FAIL | D-M1 count drift (open); D-M2 heading (open); D-M3 residue (bands undefined) |
| Experiments | 84 | 85 | 86 | FAIL | D-E1 override narration (open); D-E2, D-E4, D-E5, D-E6 (open); D-E3 closed |
| Conclusion | 80 | 82 | 82 | PASS | — (D-C1 closed; plural "patches" non-blocking) |

Unchanged modules (S6 verdicts stand): Title 87/80 PASS; Abstract 84/82 PASS (journal word limit
pending, journal-routed); Related Work 86/74 PASS; Discussion 82/84 FAIL (D-D1, D-D2 open — not
edited by S7; D-D1 now also contradicts the corrected articulation elsewhere); Figures/Tables
85/84 PASS (D-T1 named, non-blocking).

**Verdict: needs-round-2.** Two modules (Method, Experiments) remain below threshold and carry open
named defects, including one override-class item (D-E1). All remaining fixes are passage-level,
prescribed at S6, and preserve numbers, citations, and claim strength; nothing is
blocked-needs-author-input. Introduction and Conclusion now pass at their frozen thresholds.
