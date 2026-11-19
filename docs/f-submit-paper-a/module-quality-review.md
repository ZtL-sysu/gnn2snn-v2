# Module Quality Review — Paper A ("What Reproduces in Spiking Graph Neural Networks? An Audit of Every Code-Releasing Method in a Defined Survey")

- **Stage:** S6 (module-by-module quality scoring against frozen comparators; f-submit-v1).
- **Run root:** `tex_audit-journal/f-submit-v1-run-20261119-003907-paperA` (the task text's path
  placeholders rendered as `undefined`; the run root was resolved from the workflow journal
  `wf_994aa292-b43` — this agent is `s6:paperA`, spawned after the paperA S5 digest).
- **Inputs read (frozen):**
  - Manuscript: `sources/main.tex` + `sources/sections/{abstract,intro,related,protocol,results,discussion,conclusion}.tex` + `sources/tables/{audit,protocol,split,census}.tex`.
  - Comparators: `comparator-literature-table.md` (10 verified papers, frozen 2026-11-19).
  - S5 digest: `comparator-section-patterns.md` (per-module transferable patterns + access-honesty record).
  - Rubric: `[WORKSPACE]\.claude\skills\f-submit-v1\references\module-quality-rubric.md`.
  - Gates: `[WORKSPACE]\.claude\skills\f-submit-v1\references\narrative-and-prose-gates.md` (N1/N2/N3, P1, P2).
  - Context: `manuscript-module-map.md`, `narrative-consistency-map.md`, `submission-profile.md`,
    `reviews/p9-review-paperA.md`, `reviews/p9-integrity-two-papers.md`, `reviews/s1a-dispositions.md`.

## 1. Scoring method and uncertainty

Each module is scored on the rubric's 100-point dimensions (purpose clarity 15, structure 10,
specificity 15, evidence/citation support 20, positioning 10, concision 10, tone 10, selling-point
closure 10; title uses the adapted version). The operational threshold per module is the
**fourth-highest of the 10 comparator module scores** (top-40% gate).

**Uncertainty statement.** Comparator module scores are estimates: the S5 digest inspected abstracts
and section organizations (several bodies only via preprint/repository deposit — see its
"Per-comparator inspection record"), not full typeset bodies for all ten. Estimated comparator scores
carry roughly **±5 points**; derived thresholds carry the same ±5. Borderline module calls (Method,
Experiments, Conclusion) could flip on score at the low end of threshold uncertainty but are in each
case independently blocked by a named contradiction or professionalism-override defect, which does
not flip. The threshold is an operational comparison, not a calibrated acceptance probability.

## 2. Estimated comparator module scores (10 papers) and thresholds

Scores estimated from the S5 digest, fetched abstracts, and section trees (venue genre taken into
account). Sorted high→low; threshold = 4th-highest.

| Comparator (short) | Title | Abstr | Intro | RelW | Method | Exper | Disc | Concl | Tables |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kapoor & Narayanan 2023 (Patterns) | 88 | 88 | 88 | 72 | 84 | 86 | 88 | 84 | 82 |
| Snelleman et al. 2026 (Patterns) | 80 | 86 | 84 | 76 | 90 | 86 | 86 | 84 | 88 |
| So et al. 2023 (NMI) | 78 | 82 | 78 | 72 | 84 | 84 | 84 | 82 | 80 |
| Heirman & Bittremieux 2024 (NMI) | 78 | 80 | 76 | 72 | 82 | 82 | 82 | 80 | 82 |
| Bosma et al. 2023 (MIDL/PMLR) | 74 | 78 | 78 | 72 | 84 | 86 | 80 | 82 | 84 |
| Nijdam et al. 2024 (TMLR) | 76 | 80 | 78 | 74 | 82 | 82 | 84 | 80 | 84 |
| Lell & Scherp 2023 (CD-MAKE) | 76 | 78 | 76 | 76 | 76 | 82 | 82 | 82 | 84 |
| Li et al. 2023 (NeurIPS D&B) | 84 | 82 | 82 | 78 | 82 | 88 | 82 | 80 | 88 |
| SGNNBench 2025 (arXiv) | 74 | 78 | 78 | 70 | 80 | 84 | 78 | 78 | 86 |
| NeuroBench 2025 (Nat Commun) | 80 | 76 | 80 | 72 | 80 | 78 | 80 | 78 | 82 |
| **4th-highest (threshold)** | **80** | **82** | **80** | **74** | **84** | **86** | **84** | **82** | **84** |

## 3. Module verdicts (Paper A)

Pass rule: `score >= threshold` AND no open contradiction (N1/N2 closure) AND no
professionalism-override defect. All defects below are concrete and passage-level; no stylistic
churn was recorded.

### Title — 87 / 80 — PASS
Question + population bound ("An Audit of Every Code-Releasing Method in a Defined Survey") matches
the genre's two-move pattern (S5 digest §1) at or above the strongest comparators (K&N, Li et al.).
Specific, discoverable, non-hype. No defect.

### Abstract — 84 / 82 — PASS
Full chain present: commensurability gap → bounded corpus accounting → instrument → three graded
verdicts → conditional transportability headline (1.35x/1.37x/0.73x) → artifact → boundary
statement. Matches digest §2 patterns (corpus accounting first, graded per-claim verdicts,
conditional headline, instrument at end). Defects: none concrete. **Uncertainty:** the de-TeXed
abstract runs ~400 words; a hard journal word/character limit cannot be checked because the target
journal is author-managed/unresolved (`submission-profile.md`). If a venue with a <=250-word limit is
selected, the abstract fails readiness until compressed; this is a journal-routed item, not a prose
defect.

### Introduction — 83 / 80 — FAIL (contradiction; score would otherwise pass)
Strong problem→gap→contribution chain, enumerated findings, contributions as audit deliverables,
early scope paragraph (digest §3 patterns all present). **Blocking defect:**
- D-I1 "same baseline" mis-statement (`intro.tex` L40-42 "The two later papers we audit quote $81.5$
  and $87.7$ for the same baseline"; L62-63 contribution bullet "the same baseline cited at $81.5$
  and $87.7$ with no label"): contradicts the paper's own abstract and Sec. 5.4 evidence — 81.5 is a
  **GCN** row in SpikeGraphormer's table, 87.7 a **SpikingGCN** row in SpikeGCL's table; two
  different baselines drawn from the same two-split source tables (registered fact A-mixing;
  previously flagged as P9-M2 and still present in the frozen sources). **Fix:** reword both
  passages to "two baseline rows copied from the same two-split source tables and quoted without a
  split label (a GCN row consistent with Split I, a SpikingGCN row consistent with Split II)".
Per the rubric ("a module also fails when the narrative map shows ... contradiction"), the module
fails despite 83 >= 80.

### Related Work — 86 / 74 — PASS
Four thematic buckets (spiking-GNN census, protocol-sensitivity of conversion work, benchmark/split
conventions, reproducibility as measured), explicit closest-prior-work distinction ("re-runs" vs
"audits" — a precise scope-boundary move, digest §4), scope closure. No "prior work cannot solve"
claims beyond cited evidence. No defect.

### Method (protocol.tex) — 82 / 84 — FAIL
Strengths: surveyed-population accounting, "The rule" (own script/pinned environment, hypothesis
ordering), graded disagreement scale defined before results, held-fixed/varied discipline, log-parsed
numbers, per-cell sample sizes, "What the audit cannot see" (digest §5 patterns: sampling
accounting, graded rubric, restate-and-pin, statistical apparatus). Defects:
- D-M1 Input-variant count drift (`protocol.tex` L53-54 "We report the three combinations we tried"
  vs five variants in abstract/intro/results L28 "re-measured five of them"; results L52 "four
  plausible input conventions") — three incompatible counts of one measurement (P9-M5, still
  present). **Fix:** delete "three combinations"; define once "five input variants under four
  convention entries (raw features with and without self-loops counted separately)" and use that
  phrase in Method and Sec. 5.1 (including L52).
- D-M2 Process-narration heading (`protocol.tex` L47 "What we verified about the data, and why it
  took a while.") — conversational audit chronology in a heading (P2/override-adjacent). **Fix:**
  retitle "Feature conventions and data provenance (SpikingGCN)" and drop the duration framing; the
  paragraph body (library-array verification, Poisson-encoder consequence) is legitimate method and
  stays.
- D-M3 Unanchored "pre-registered three-sigma bar" (used in abstract L31, results L138-139,
  conclusion L26; never defined or anchored in the Method — no registration artifact, date, or
  verdict-scale definition). For an audit paper whose thesis is protocol labelling this is a missing
  evidence callback. **Fix:** add one sentence to "How numbers are read" defining the 3-sigma verdict
  bar, its three verdict bands, and naming the pre-registration artifact.

### Experiments / Results — 84 / 86 — FAIL
Evidence architecture is strong (per-pipeline subsections, delta table with n + spreads, controlled
one-flag split comparison with single attributable cause, split-mixing with verbatim definitions,
seed-noise transportability with per-dataset verdicts, census with eligibility accounting,
side-by-side with companion value labelled context-only — digest §6 patterns largely met).
Defects:
- D-E1 Manuscript-production narration / draft-correction chronology (`results.tex` L87-90 "an
  earlier draft attributed a spread measured on SpikingGCN's Cora seeds ($0.48$--$0.53$) to this
  pipeline, which was a mis-attribution, and quoted a PubMed range that no run of this pipeline
  produced") — professionalism override class (statements about corrected bugs; an earlier draft is
  not part of the published record). **Fix:** state the facts directly: "the $0.48$--$0.53$ spread is
  measured on SpikingGCN's Cora seeds and does not apply to this pipeline; SpikeGCL's paper reports
  no PubMed cell" — delete the earlier-draft clause.
- D-E2 Manuscript-production narration (`results.tex` L39-40 "this is the combination an earlier
  diagnostic recorded as the ``fourth combination''") — internal diagnostic history. **Fix:** delete
  the aside; keep "it was the normalisation, not the self-loops".
- D-E3 "same baseline" lead sentence (`results.tex` L108 "the two later papers we audited cite the
  same baseline on different splits") — mislabels the two rows established two sentences later (GCN
  vs SpikingGCN); same defect family as D-I1. **Fix:** "the two later papers lift baseline rows from
  the same two-split source tables without labelling the split".
- D-E4 Conversational verdict (`results.tex` L62-63 "If the audit stopped here it would be a
  reproduction study with a happy ending."). **Fix:** "If the audit stopped here, the result would be
  a straightforward reproduction."
- D-E5 Conversational aside (`results.tex` L53 "and we would welcome either"). **Fix:** "A container
  image or a commit hash would settle the question."
- D-E6 Repeated defensive negation (`results.tex` L86-87 "recorded as part of the sensitivity
  evidence rather than reconciled away" and L132-133 "recorded rather than reconciled"; cf. intro
  L26-27 and Method "What we deliberately did not do") — four instances where one evidence boundary
  suffices (rubric professionalism list). **Fix:** keep the Method paragraph as the single statement;
  rewrite both Results instances as plain fact ("...differed by $1.0$ points from the audited cell").
D-E1 alone is an override-class item, so the module cannot pass on score alone; the score (84 vs 86)
is also below threshold (borderline under ±5 threshold uncertainty).

### Discussion — 82 / 84 — FAIL
Content coverage is complete (remedy per failure mode, reader guidance, structural exposure,
companion division of evidence, threats to the audit with self-critical items, evaluative scope with
stated assumptions, broader impact, falsifiability). Defects:
- D-D1 "Both numbers are correct" gloss (`discussion.tex` L19-22 "the later tables cite $81.5$ and
  $87.7$ without saying which. Both numbers are correct.") repeats the same-baseline
  mis-statement (D-I1/D-E3); the evidence supports "each value is consistent with one of the two
  splits of the source tables" (P9-M2 item). **Fix:** replace the two sentences with: "the later
  tables quote rows from those two-split tables without saying which split each row belongs to; each
  value is consistent with one of the two splits, and that is precisely why the choice has gone
  unnoticed."
- D-D2 Conversational register in the falsifiability paragraph (`discussion.tex` L72-76 "for you ...
  our environment is wrong and we would like to know ... we have mis-set a flag") — P2 conversational
  verdict / second-person address. **Fix:** "If the shipped SpikingGCN configuration reproduces
  $80.7$ on Cora, the environment pin is wrong and the discrepancy should be reported; if the
  public-split SpikeGraphormer configuration reproduces $84.8$, the split flag was mis-set."

### Conclusion — 80 / 82 — FAIL
Per-dataset verdict and run-the-code recommendation match the genre template (digest §8: bounded
restatement + per-unit verdict + one actionable recommendation; population bound restated). Defects:
- D-C1 Contradictory callback (`conclusion.tex` L5-6 "disclosing the compatibility patches two of the
  three required") vs Method L33-35 and Table 2 ("for two of the three pipelines the answer is zero"
  lines of author code changed; one two-line patch for SpikeGraphormer) — the counts do not reconcile
  under any reading of "patch". **Fix:** "(disclosing the path and configuration adjustments each
  pipeline required and the one two-line compatibility patch)".
The rest of the conclusion restates many numbers but introduces none new; the density is a style
observation, not a recorded defect.

### Figures / Tables — 85 / 84 — PASS
Four tables cover the genre's core evidence classes (digest §9): reported-vs-reproduced deltas with
n (`tab:audit`), pinned-environment/patch accounting (`tab:protocol`), split-registry-style one-flag
comparison (`tab:split`), census with eligibility accounting (`tab:census`). Captions carry the
conventions; multi-seed extensions correctly kept in text. Named defect (non-blocking):
- D-T1 Caption column-semantics ambiguity (`tables/audit.tex`): "Pub. I"/"Pub. II" are defined as
  SpikingGCN's two splits but hold any paper's single reported value — SpikeGraphormer's $84.8\pm0.7$
  is its own random class-balanced split, not Split I, and the caption does not say so. For a paper
  whose thesis is "label your split", the headline table under-labels one. **Fix:** add one caption
  clause ("for papers reporting a single split the value is their own published setting;
  SpikeGraphormer's is a random class-balanced split of Split-I size, not the fixed Split I") or
  rename the columns "published (setting as shipped)".
Note: `sources/figures/` holds figure files that no section of this manuscript references (they
appear to be companion/Paper-B artifacts); the manuscript text is tables-only. Artifact-hygiene
item for packaging, not a module defect.

## 4. Core-claim closure (Gates N1/N2)

- **Central question:** are published spiking-GNN accuracies commensurable, and what makes a number
  portable?
- **Core claim:** no — three measured failure modes; the binding property is split transportability,
  stated per dataset at measured magnitudes (1.35x/1.37x/0.73x the pre-registered 3-sigma bar).
- **Forward trace:** title (audit of every code-releasing method) -> abstract (three graded verdicts
  + per-dataset transportability) -> intro (question, three findings, four contributions) -> method
  (replay protocol, graded scale, one-flag split comparison) -> results (audit grid, split effect,
  split-mixing, transportability, census, side-by-side). Holds.
- **Backward trace:** conclusion (bounded census, per-dataset verdict, run-the-code) -> discussion
  (transportability-not-variance, remedies, threats) -> results (sigma ratios) -> method (pinned
  environments, sample sizes) -> stated gap. Holds.
- **Magnitudes consistent** across abstract/intro/results/conclusion (50.0/80.7/64.3/77.6/70.6/72.5;
  84.7/84.8+/-0.7/74.0/10.7 pp; 86.6/87.4/77.3/77.6; 8.0/5.1/11.7 pp; 4.06/4.11/2.18 sigma;
  1.35x/1.37x/0.73x; 3 of 7). Supporting and negative findings (PubMed confounded cell, same-seed
  1.0 pp delta, bounded census) carry their warranted local interpretation only — correct per the
  closure rule; no forced callbacks required.
- **Open items at variance with the narrative map's "none open after P10":** D-I1/D-E3/D-D1 (the
  third failure mode's "same baseline" headline is contradicted by its own evidence rows) and D-M1
  (count drift). The S1A disposition records P9-M2/M5 as closed at P10, but the frozen `sources/`
  snapshot still contains both; treat as open and repair in the humanizer/revision pass.
- **Status:** core chain closed end-to-end with consistent magnitudes; corrections required at the
  supporting-finding articulation (failure mode 3), one methodological anchor (pre-registered bar),
  and two peripheral counts. No core claim lacks evidence; no conclusion-only claim.

## 5. Professionalism override scan (every module)

- Manuscript-production narration: FOUND in Method (D-M2 heading) and Results (D-E1, D-E2) — named,
  fixes above.
- Debugging/correction chronology: D-E1 (draft-correction), D-E2 (diagnostic history).
- Journal/editor/reviewer-facing language: none.
- Conversational verdicts / rhetorical register: D-E4, D-E5, D-D2 — named.
- Repeated defensive negation: D-E6 — named.
- Material caveat placement (Gate P1): threats paragraph in Discussion covers environment drift,
  under-powered cells, Cora-centring, own-code discount; per-cell qualifications sit with results
  (Method "Sample size, stated per cell"); no misplaced or hidden limitation found.
- Unsupported trust/novelty/deployment claims: the "pre-registered" bar (D-M3) is the one
  unanchored process-trust claim; no novelty/deployment overclaim found ("a census, not a sample" is
  bounded by construction and restated as such).

## 6. Humanizer decisions (per module)

1. Title: **no defect**.
2. Abstract: **no defect** (journal word limit pending — journal-routed, not a humanizer item).
3. Introduction: **named defect — D-I1 "same baseline" mis-statement** (fix as in §3).
4. Related work: **no defect**.
5. Method: **named defects — D-M1 count drift, D-M2 process-narration heading, D-M3 unanchored
   pre-registered bar** (fixes as in §3).
6. Experiments: **named defects — D-E1 draft-correction narration, D-E2 diagnostic-history aside,
   D-E3 same-baseline lead sentence, D-E4 "happy ending", D-E5 "we would welcome either", D-E6
   repeated defensive negation** (fixes as in §3).
7. Discussion: **named defects — D-D1 "both numbers are correct" gloss, D-D2 conversational
   falsifiability register** (fixes as in §3).
8. Conclusion: **named defect — D-C1 "compatibility patches two of the three" contradictory
   callback** (fix as in §3).
9. Figures/tables: **named defect — D-T1 tab:audit caption column-semantics** (fix as in §3).

All fixes are passage-level and preserve numbers, citations, and claim strength; `sources/` is
immutable in this run, so fixes are queued for the revision/humanizer stage.

## 7. Summary table

| Module | Score | Threshold (4th of 10) | Pass | Blocking defect(s) |
|---|---:|---:|---|---|
| Title | 87 | 80 | PASS | — |
| Abstract | 84 | 82 | PASS | — |
| Introduction | 83 | 80 | FAIL | D-I1 contradiction (same baseline) |
| Related work | 86 | 74 | PASS | — |
| Method | 82 | 84 | FAIL | D-M1 count drift (also D-M2, D-M3) |
| Experiments | 84 | 86 | FAIL | D-E1 override narration (also D-E2..D-E6; score below threshold) |
| Discussion | 82 | 84 | FAIL | D-D1 contradiction (also D-D2) |
| Conclusion | 80 | 82 | FAIL | D-C1 contradictory callback |
| Figures/tables | 85 | 84 | PASS | D-T1 named, non-blocking |

Comparator-score uncertainty: +/-5 per cell (estimates from digest + abstracts + section trees;
several comparator bodies accessed only via preprint/deposit). Under threshold uncertainty Method
(82 vs 84), Experiments (84 vs 86), and Conclusion (80 vs 82) are borderline on score but each is
independently blocked by a named contradiction or override defect. Introduction fails on the
contradiction despite 83 >= 80.
