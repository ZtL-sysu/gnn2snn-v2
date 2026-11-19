# S6 — Module Quality Review (Paper B, conversion)

**Run root:** `[WORKSPACE]\auto-paper\20261028-171500\Spiking-GNN-Conversion\f-research-v1-run-20261118-181848-journal-upgrade\tex-journal\f-submit-v1-run-20261119-003907-paperB`
**Manuscript:** `sources/main.tex` + `sections/` (abstract, intro, related, method, experiments, discussion, conclusion), `tables/tab1–tab14,tab16–tab19`, `sources/figures/` (9 figures)
**Frozen comparators:** `comparator-literature-table.md` (10 papers, frozen 2026-11-19)
**S5 digest:** `comparator-section-patterns.md` (produced 2026-11-19 01:29; full-text section-level inspection of all 10 comparators)
**Rubric:** `[WORKSPACE]\.claude\skills\f-submit-v1\references\module-quality-rubric.md`
**Gates:** `[WORKSPACE]\.claude\skills\f-submit-v1\references\narrative-and-prose-gates.md`
**Review date:** 2026-11-19. Sources are read-only this stage (`sources/` immutable); every fix below is named, not applied.

---

## 0. Method, threshold rule, and uncertainty

Per the rubric's top-40% gate: score the same module in all 10 comparators, sort high-to-low, set the
threshold to the **4th-highest** comparator score, pass only if `user_score >= threshold`. The
threshold is an operational comparison, not a calibrated acceptance probability, and it is not moved
to obtain a pass. A module can also fail on the narrative gates (orphan / contradiction / missing
evidence callback / conclusion claim unprepared) or the professionalism override regardless of score.

**Comparator score basis and uncertainty.** Comparator module scores are *estimated* from the S5
digest (section skeletons as found, per-module transferable patterns, abstract move analysis) and the
abstract-level evidence quoted there; no comparator full text was re-read in this S6 pass (WebFetch
remains network-blocked per the digest's access record). **Estimated comparator scores carry roughly
±5–7 points of uncertainty per module**; fourth-highest thresholds move accordingly. Scores near the
threshold are marked threshold-sensitive below. Two comparators deserve an explicit estimation
caveat: SGNNBench is an unrefereed preprint (venue disputed in the frozen table) and Spiking PointCNN
is a mid-tier MDPI journal paper — their module scores were set below the conference-core group;
alternatives (treating the claimed NeurIPS slot as real) would raise the SGNNBench scores by ~3–5
points on Experiments/Figures.

Rubric dimensions (100-point): purpose clarity 15 · structure/flow 10 · specificity 15 · evidence 20 ·
positioning 10 · concision 10 · tone/polish 10 · selling-point consistency/closure 10. Title uses the
adapted version (specificity, discoverability, field fit, non-hype).

---

## 1. Comparator module score estimates (10 per module) and thresholds

Scores are listed comparator-by-comparator in the S5 digest's own terms where possible
(QCFS/Offset-Spikes/SpikeZIP-TF/DCGS/SpikeGCL = conference-core; Adaptive Calibration = strong AAAI;
SpikingGCN = older IJCAI with appendix-related-work artifact; MK-SGN = solid journal, distillation
scope; SGNNBench = preprint benchmark; Spiking PointCNN = mid-tier journal).

| module | 10 estimated comparator scores (sorted high→low) | 4th-highest threshold | uncertainty |
|---|---|---:|---|
| Title | SpikeGCL 85, DCGS 83, QCFS 82, Offset Spikes 80, SpikeZIP-TF 78, AdaptiveCal 76, SGNNBench 74, SpikingGCN 72, MK-SGN 71, PointCNN 70 | **80** | ±5 |
| Abstract | QCFS 87, Offset Spikes 85, SpikeZIP-TF 84, DCGS 83, SpikeGCL 82, AdaptiveCal 80, SpikingGCN 76, SGNNBench 75, MK-SGN 72, PointCNN 68 | **83** | ±5 |
| Introduction | QCFS 85, Offset Spikes 84, SpikeGCL 83, DCGS 82, SpikeZIP-TF 81, AdaptiveCal 79, SpikingGCN 77, SGNNBench 76, MK-SGN 73, PointCNN 70 | **82** | ±6 (threshold-sensitive) |
| Related work | SpikeGCL 84, DCGS 83, AdaptiveCal 80, SpikeZIP-TF 78, Offset Spikes 78, QCFS 76, MK-SGN 74, SGNNBench 72, PointCNN 72, SpikingGCN 68 | **78** | ±6 |
| Method | QCFS 90, Offset Spikes 88, DCGS 86, SpikeZIP-TF 84, SpikeGCL 82, AdaptiveCal 80, PointCNN 75, MK-SGN 74, SpikingGCN 72, SGNNBench 70 | **84** | ±5 |
| Experiments | SpikeZIP-TF 86, QCFS 85, DCGS 84, Offset Spikes 84, SGNNBench 83, SpikeGCL 82, AdaptiveCal 80, SpikingGCN 76, MK-SGN 75, PointCNN 70 | **84** | ±5 |
| Discussion | QCFS 75, SpikeGCL 73, SGNNBench 72, DCGS 65, SpikeZIP-TF 65, Offset Spikes 63, AdaptiveCal 63, SpikingGCN 62, MK-SGN 60, PointCNN 60 | **65** | ±7 (most comparators merge Discussion into Conclusion or appendix; scores reflect absence) |
| Conclusion | QCFS 78, SpikingGCN 76, DCGS 75, SpikeGCL 74, SpikeZIP-TF 73, Offset Spikes 73, AdaptiveCal 72, PointCNN 70, MK-SGN 70, SGNNBench 70 | **74** | ±5 |
| Figures/tables | SpikeZIP-TF 85, QCFS 83, Offset Spikes 82, SpikeGCL 82, SGNNBench 80, DCGS 80, AdaptiveCal 78, SpikingGCN 74, MK-SGN 74, PointCNN 70 | **82** | ±5 (threshold-sensitive) |

---

## 2. Module scores, verdicts, and named defects

Defect IDs D1–D14 are listed with passage-level fixes in §4. "Pass" = score >= threshold AND no
narrative/professionalism hard gate fired.

### 2.1 Title — 84 / threshold 80 → **PASS**
Question-title + object-subtitle matches the field's framing-paper convention (digest §1 endorses both
moves); names task (graph-to-spiking conversion) and three analytic objects; non-hype. Deductions:
"Activation Mass" is never defined as a named quantity in the body (terminology drift, D14); title is
long. Threshold-sensitive margin: +4.

### 2.2 Abstract — 78 / threshold 83 → **FAIL**
The complete chain is present with configuration scoping and honest negatives (the digest confirms
the moves have precedent cover), and the P10 scoping repair (deep-regime-only + failed mixed-depth
screen + "beats or ties ... on the majority") is in place. It fails on the comparator-facing
dimensions: **concision** (648 words, 15+ numeric endpoints — the comparators' 5-move spine carries
2–4 headline numbers), **tone** (slogan-like negative-parallelism opening, conversational verdict
"That is the sense in which the error is not noise", a `\citep` inside the abstract). Defects D1, D2,
D3. Fail is robust to comparator-estimation error unless the comparator abstract estimates are ≥5
points too high.

### 2.3 Introduction — 82 / threshold 82 → **FAIL (narrative hard gate)**
The problem→gap→mechanism→artefacts→contributions→boundaries chain is excellent, the gap statement
is precise and narrowly scoped ("in graph conversion"), the three-integer-artefact paragraph is the
kind of specificity few comparators match, and the early boundaries paragraph is a strength. Score
equals the threshold (threshold-sensitive). **Hard gate:** the abstract's constructive-decision
promise (deep-regime per-layer window allocation) has **no contribution bullet** — the forward trace
title/abstract promise → Introduction contribution breaks for a core claim (Gate N2), and the
three-route transfer-negative map is absent from the contribution list (P9-RB-M2 residue; the
module map's "five contribution bullets incl. deep-regime capability + negative map" does not match
the 9 bullets actually in `intro.tex`). Structural deduction: 9 bullets vs the field's 3–4 (digest
condition: keep claim-dense — satisfied, so this is a deduction, not a defect). Defects D4 (closure),
D5 (conversational aside).

### 2.4 Related work — 86 / threshold 78 → **PASS**
The strongest module by positioning discipline: per-paragraph differentiators, contrastive taxonomy
("What is inherited and what is not"; misalignment vs quantisation), explicit prior-art attribution
for per-layer allocation ("We therefore claim no priority"), both competitor classes, boundary
conditions stated as boundaries. Deductions (positioning 7/10, evidence 16/20): **3 of the 10 frozen
comparators are never cited** — SpikeZIP-TF (whose exact/lossless-equivalence claim the digest names
as the strongest identity claim to differentiate), Spiking PointCNN (closest graph-niche prior art),
MK-SGN (only verified GCN→SGN conversion-adjacent work) — against the S5 digest's requirement that
each frozen comparator carry an explicit positioning sentence; and the intro's "point-cloud
conversion attempts that first replace max pooling with averaging" is an uncited prior-art claim
(D6). Minor register item D6b. Passes on score with margin +8.

### 2.5 Method — 89 / threshold 84 → **PASS**
Named-object chain (identity → quantiser lemma → closed-form bias → decoder → theorem → recursion →
prediction → complexity) follows the digest's define→diagnose→fix template; the identity is boxed
and exact, the phase-randomisation assumption is verified rather than asserted, the theorem carries
its scope conditions, and deployment complexity is stated. Deductions: no notation table despite a
law-heavy symbol set (digest field-standard aid — structural note, not a prose defect); repeated
templated transitions (D8). Passes with margin +5.

### 2.6 Experiments — 89 / threshold 84 → **PASS**
On evidence discipline this exceeds every comparator: seven seeds everywhere, paired bootstrap
against a measured run-to-run floor, pre-registered screens reported when they fail, negative results
with controls attached (ensemble control, shuffled-scale ablation, threshold controls), codebase
anchors labelled as anchors. Deductions: concision (15 subsections, repeated scope sentences),
tone (repeated honesty-signalling / defensive contrast, a writing-process aside, a redundant
pre-registration parenthetical — D7), templated transitions (D8), and the 10-table `\input` dump
that scrambles table numbering (D13, recorded under figures/tables). Passes with margin +5.

### 2.7 Discussion — 88 / threshold 65 → **PASS**
A real journal Discussion where most comparators have none: variance-vs-bias interpretation with
certificate-share evidence, tightness analysis with a proposed-but-untried improvement and the reason
for not trying it, a boundary paragraph per threat (Gate P1 placement done properly), outlook with
concrete future work and the transfer failure framed as an open problem with measured failure modes.
Deductions: templated-transition tail (D8), honesty signalling ("One honest limitation", D7 tail).
Large margin over the threshold because comparators merge this content into conclusions/appendices.

### 2.8 Conclusion — 77 / threshold 74 → **FAIL (professionalism override)**
Recalls the identity + laws claim and its evidence boundary well, introduces no new result, and the
limits-as-findings paragraph is strong. But three defects fire the override and the closure gate:
(a) **one-sided anchor callback** — "the one we could place beside our own ... sits below the
conversion rather than above it" contradicts the paired "brackets rather than ranks" framing of
sec:audit and of the abstract (SpikeGCL's 86.6 is above every arm); P9-RB-M4 residue (the abstract
was repaired at P10, the conclusion sentence was not); (b) **cost claim without its accounting
convention** — "at matched accuracy they cost 2–23× fewer synaptic operations" states the cheapness
claim as a property of the conversion, which sec:accounting explicitly forbids ("the cheapness claim
is therefore stated with its convention attached"); P9-RB-M6 residue; (c) **no recall of the core
constructive claim** — the deep-regime allocation capability (a full abstract paragraph, sec:allocation,
discussion outlook) is absent from the conclusion (Gate N3). Plus conversational self-assessment
("we think, more useful than the claim we set out to make") and a four-fold "does not buy" anaphora
(D12). Defects D9, D10, D11, D12. Score 77 would pass the threshold; the override forces the fail.

### 2.9 Figures/tables — 82 / threshold 82 → **PASS (threshold-sensitive)**
Figure order follows the digest's argument-order convention (landscape fig1 → accuracy/latency and
spike curves → ablation → theory check → prediction failure → Pareto → allocation regime → transfer
negatives); captions are self-contained; tab:ablate and tab:pareto carry the digest-endorsed
footnotes for non-comparable configurations and re-implementation caveats. Deductions (D13): ten
table environments are declared in a single `\input` dump (experiments.tex, end of file), so table
numbers do not follow argument order and the central result (tab:fidelity) compiles as Table 8;
**tab:capacity (tab11) is an orphan** — compiled but never `\tabref`'d (P9 residue), and its caption
lead ("the accuracy gap does not grow with it") overstates the measured partial cancellation of
sec:width; **four figures (fig:pred, fig:pareto, fig:allocation, fig:transfer) are labelled but never
referenced in prose**. Exactly at the threshold; sensitive to ±5 comparator estimation error.

---

## 3. Closure trace for core claims (Gate N2/N3)

Per `narrative-consistency-map.md`, the core claim is: *identity + laws characterise conversion error
exactly within a configuration; the deep-regime validation-search allocation is the working
constructive capability (prior-art attributed).* Supporting/diagnostic/negative findings need only
their warranted local interpretation.

**Core claim A — identity + laws (including depth law, order-invariant readout, threshold pinning,
ranking-only predictor): closure COMPLETE.**
- Forward: title (spectral gain, order-invariant readout) → abstract (identity + consequences with
  scoping) → intro bullets 1–7 → method (eq:identity, eq:recursion, eq:sigma, thm:margin) →
  experiments (sec:closed 105-cell check, sec:theory gap exactly 0, sec:depth G^0.95/r=0.990,
  sec:order 0.000 pp, sec:threshold five-graph scan, sec:ranking ρ=0.967/4.6 pts with the failed
  mechanism ablation).
- Backward: conclusion recalls identity, bias, readout, theorem, depth law with the same numbers and
  scope. No conclusion-only claim inside this spine. Residual: title term "Activation Mass" is not
  the body's vocabulary (D14 — terminology drift, Gate N1, non-blocking for evidence closure).

**Core claim B — deep-regime per-layer window allocation: closure INCOMPLETE.**
- Abstract (full paragraph, correctly scoped: L=4-only, failed mixed-depth screen 37.5%, prior art
  attributed) → **Introduction: MISSING** (no contribution bullet; only a passing gap mention of
  "per-layer window allocation" at intro.tex:32; Gate N2 forward link broken) → experiments
  sec:allocation (procedure, 22.7–93.4% on 6/8 held-out L=4 cells, random control, failed screen) →
  discussion outlook (practical guidance) → **Conclusion: MISSING** (Gate N3 recall absent).
- Fix (named, not applied): add one contribution bullet in intro.tex mirroring the abstract's
  scoped sentence; add one recall sentence in conclusion.tex ("Where depth makes uniform budgets
  explode, a validation search over per-layer windows preserves accuracy with 22.7–93.4% fewer
  layer-updates on six of eight held-out four-layer configurations; the idea family is prior art and
  the measurement is where its headroom lives.").
- Note: `manuscript-module-map.md` claims "five contribution bullets incl. deep-regime capability +
  negative map" and method "allocation procedures"; the actual sources have 9 bullets without the
  capability and no allocation subsection in method (the procedure is stated inside sec:allocation).
  The S1 map does not match the source snapshot; the allocation procedure itself is reproducible as
  written in sec:allocation, so method-side operationalization counts as present-but-in-results.

**Supporting/diagnostic/negative findings — local interpretation warranted and present.**
Order invariance (sec:order with two-sided test), threshold law (sec:threshold with two controls),
ensemble negative (sec:ensemble with trained-ANN control), transfer negatives (sec:transfer with
pre-registered gates and failure mechanisms), accounting sensitivity (sec:accounting with all three
conventions shown), width/margin partial cancellation (sec:width "only partly right") — each carries
its local interpretation; none is forced into abstract/intro/conclusion, per the rubric. No orphan
results and no unprepared conclusion claims beyond the two conclusion residues D9/D10 and the
missing recall D11.

---

## 4. Concrete defects and passage-level fixes

Prose/rhetoric defects (Gate P2 / professionalism classes) and structural defects (gates N1–N3) are
both concrete; no stylistic churn is proposed. Sources are immutable this stage — fixes are for the
revision stage.

**D1 — Abstract: clause-stacked endpoint inventory (P2 class).** `sections/abstract.tex` (whole
abstract, 648 words): 15+ numeric endpoints (0.08/105 cells, 3.5, 0.000/1.6, G^0.95/r=0.990/21,
22.7–93.4%, 37.5%, ρ=0.967, 4.6, κ=1.2/11.0/2.5, 74.0/86.6, seven-seed scan, two-of-three, ...) in
four 100+-word sentences against the comparators' 2–4-headline-number spine.
*Fix:* compress to the 5-move spine retaining four flagships (identity agreement 0.08 pp over 105
cells; depth law G^0.95, r=0.990; deep-regime 22.7–93.4% on 6/8 held-out four-layer configurations
with the failed screen clause kept; ranking ρ=0.967 but level off by 4.6 points), move the remaining
numbers to their result sections; target ~200–250 words pending the journal's hard limit.

**D2 — Abstract: slogan-like negative parallelism + conversational verdict (P2 class).**
`abstract.tex:2–3` "The error ... is not noise. It is a deterministic, closed-form function ...",
"an identity, not a bound", and `abstract.tex:10` "That is the sense in which the error is not noise,
and it has a boundary."
*Fix:* "The conversion error of a graph-to-spiking conversion is a deterministic, closed-form
function of the trained weights ..." and "The identity holds exactly for the two-layer conversion;
at L=3 the closed form fails by 3.5 points because ..." — delete the "That is the sense ..." verdict.

**D3 — Abstract: citation inside abstract (tone/polish).** `abstract.tex:49` `\citep{companion}`.
Most journals prohibit references in abstracts; the hard limit is unadjudicable while the venue is
author-pending.
*Fix:* "a companion paper reports the full audit of every spiking-GNN method that publishes code"
(no `\citep`); route the reference to sec:audit.

**D4 — Introduction: missing core-claim callback (Gate N2, structural).** `intro.tex:69–133` — the
9-item contribution list has no bullet for the abstract's constructive decision (deep-regime
per-layer window allocation) and no bullet for the transfer-negative map (P9-RB-M2 residue).
*Fix:* add one bullet: "**A deep-regime per-layer window allocation.** Where uniform timesteps are
held to the worst layer's budget, a validation search preserves accuracy with 22.7–93.4% fewer
layer-updates on six of eight held-out four-layer configurations; the pre-registered mixed-depth
screen was not met, and the idea family is established prior art (\secref{sec:allocation})."; extend
bullet 7 with one clause naming the three failed weight-only transfer routes (\secref{sec:transfer}).

**D5 — Introduction: conversational aside (P2 class).** `intro.tex:130–131` "and, the point we
consider most diagnostic, the method degrades in the \emph{predicted} way when ...".
*Fix:* delete the aside: "and the method degrades in the predicted way when the aggregation operator,
the threshold, the width or the decoder is changed."

**D6 — Related work/intro: frozen comparators unpositioned (positioning/citation defect).**
SpikeZIP-TF (exact/lossless equivalence — the digest's named identity-claim to differentiate), Spiking
PointCNN (closest graph-niche conversion framework), MK-SGN (GCN→SGN distillation, ≠ weight
transfer) appear in `comparator-literature-table.md` but in no citation; `intro.tex:30` "point-cloud
conversion attempts that first replace max pooling with averaging" is an uncited prior-art claim.
*Fix (three passage-level insertions):* (i) attach the point-cloud conversion citation at intro.tex:30
and differentiate scope (point-cloud X-Conv vs message-passing aggregation); (ii) at related.tex:43
("the machinery now extends to transformers") add SpikeZIP-TF with the differentiation sentence
("exact activation-quantisation equivalence for transformers; our identity is a graph error
identity, and the two 'conversion is exact' claims are exact about different objects"); (iii) at the
graph-precedent sentence (related.tex:125–135 or intro.tex:28–32) add MK-SGN as distillation-based
GCN→SGN transfer, explicitly not weight-transfer conversion.

**D6b — Related work: conversational meta-register (P2 class, minor).** `related.tex:68` "Because
the bias mechanism is not ours, it is worth being blunt about the boundary."
*Fix:* "Because the bias mechanism is not ours, the boundary is as follows:" (or fold into the
Inherited/Not-inherited list without the meta-sentence).

**D7 — Experiments: repeated defensive/honesty signalling + writing-process aside (P2 class,
professionalism "repeated defensive negation").** Sites: `experiments.tex:57–59` "the version of this
idea that would have been tempting to write up" (writing-process narration); `experiments.tex:137–138`
"We report it because a pre-registered miss is worth more than a post-hoc hit"; `experiments.tex:133`
redundant parenthetical "(the prediction was recorded before the measurement and is part of the
released record)" duplicating the preceding clause; `experiments.tex:263` "We report these cells
rather than hiding them"; `experiments.tex:300` "rather than as wins"; `experiments.tex:379` "Where
the claim is thin, we say so"; `experiments.tex:408–409` "reporting the reversal is the honest
alternative"; plus "The honest counterweight" (169) and intro "an honest negative answer". The
paper's evidence-boundary discipline is a strength; the self-narration about it is the defect.
*Fix:* keep one direct boundary statement per claim site and delete the rest: ensemble closing
sentence → "We report this as a negative result with its control; the positive reading is not
supported."; sec:ranking → "We report the pre-registered prediction and its outcome." (drop the
parenthetical and the "worth more" sentence); sec:iso opening → drop "Where the claim is thin, we
say so" and keep the measured statement; keep exactly one "no energy claim" statement at the
sec:iso efficiency paragraph and let the other five "not joules/no energy claim" instances carry
only their local label.

**D8 — Method/Experiments/Discussion: repeated templated transition (P2 class).** The "Two/Three
things X" template recurs ~10 times: method.tex:67 "Three things follow.", :89 "Three regimes
matter", :118 "Two consequences are worth separating.", :145 "Two boundaries on that claim.";
experiments.tex:150 "Three things follow.", :478 "Two readings of this row are possible and only one
is intended.", :499 "Two things change and one does not.", :572 "Two repairs suggest themselves, and
both fail."; discussion.tex:70 "Two things are worth saying about that.", :110 "Two directions follow
directly."
*Fix:* vary each into a topic-led sentence ("The identity has three consequences.", "The claim needs
two boundaries.", "Width changes two things.", "Both natural repairs fail.", "The row supports two
readings.") or fold into the first substantive clause.

**D9 — Conclusion: one-sided anchor callback (professionalism override — material caveat omitted;
P9-RB-M4 residue).** `conclusion.tex:31` "the one we could place beside our own ... sits below the
conversion rather than above it" — sec:audit and the abstract frame the two released pipelines as a
pair that "brackets rather than ranks" (74.0 below, 86.6 above); the conclusion quotes only the
favorable one and in the singular.
*Fix:* "and the two released pipelines that bracket our result — one below the conversion, one
self-supervised encoder above it — show that published numbers cannot be used as baselines without
checking."

**D10 — Conclusion: cost claim without its accounting convention (professionalism override;
P9-RB-M6 residue).** `conclusion.tex:19–20` "at matched accuracy they cost $2$--$23\times$ fewer
synaptic operations than a trained spiking network" — sec:accounting requires the cheapness claim to
carry its convention (the 2–23× range is the static-update accounting; under event-driven counting a
linear-readout external arm is cheaper at T=4 on Cora/CiteSeer).
*Fix:* "under the static-update accounting, at matched accuracy they cost 2–23× fewer synaptic
operations than the trained arms on two of three graphs — under event-driven counting a
linear-readout external arm is cheaper at low latency (\secref{sec:accounting})."

**D11 — Conclusion: missing recall of the core constructive claim (Gate N3).** `conclusion.tex` —
the deep-regime allocation capability (abstract paragraph 3; sec:allocation) is never recalled.
*Fix:* add one sentence to paragraph 2 or 4 as given in §3 (scoped to L=4, prior art attributed).

**D12 — Conclusion: conversational self-assessment + negative-parallelism anaphora (P2 class).**
`conclusion.tex:34` "What survives is narrower and, we think, more useful than the claim we set out
to make"; `conclusion.tex:31` "we state them as findings rather than as disclaimers"; the four-fold
"Conversion does not buy ... / Nor does it buy ..." anaphora (24–31).
*Fix:* "What survives is narrower and more useful:" (delete "we think" and the process reference);
replace "we state them as findings rather than as disclaimers" by stating the finding directly;
keep the "does not buy" frame for at most two items and state the remainder as direct findings
("Depth fails predictably ...", "Only the membrane decoder is order-invariant ...").

**D13 — Figures/tables: cross-reference and ordering defects.**
(a) Ten table environments declared in one `\input` dump (`experiments.tex`, the
`\input{tables/tab1}`–`tab11` block) so numbering does not follow argument order and tab:fidelity
compiles as Table 8. *Fix:* move each `\input` to just before its first `\tabref` site.
(b) `tables/tab11.tex` (`tab:capacity`) is compiled but never `\tabref`'d; its caption lead ("the
accuracy gap does not grow with it because the logit margin grows too") overstates sec:width's
measured partial cancellation (gap grows −0.17→−2.01 on PubMed, peaks −2.71 on CiteSeer).
*Fix:* reference it at `experiments.tex:512–520` (the margin paragraph) and align the caption with
the measured statement ("the effects roughly cancel on Cora and do not cancel on CiteSeer or
PubMed"), or drop the table from the build.
(c) `fig:pred`, `fig:pareto`, `fig:allocation`, `fig:transfer` are labelled but never referenced.
*Fix:* add `\figref{fig:pred}` at experiments.tex:94, `\figref{fig:pareto}` at experiments.tex:145,
`\figref{fig:allocation}` at experiments.tex:675, `\figref{fig:transfer}` at experiments.tex:707.

**D14 — Title/Conclusion: terminology drift (Gate N1).** The title object "Activation Mass" is never
defined as a named quantity; the body's vocabulary is "activation ceiling" (abstract, sec:threshold),
"saturated mass" (sec:generality), "activation scale(s)" (sec:threshold, sec:transfer), with "activation
mass" appearing only in conclusion.tex:38 and discussion.tex:100 (pruning). The conclusion claim
"the two quantities ... its activation mass and its cumulative spectral gain ... can both be read off
the trained weights" therefore points at an unnamed quantity.
*Fix:* either define "activation mass" once in sec:lemma1/sec:threshold (as the activation ceiling /
saturated-mass quantity fixed during ANN training) and use it consistently in abstract, sec:threshold
and conclusion; or align title and conclusion to the body's existing term.

---

## 5. Humanizer decisions (Gate P2) — one per module

1. **title — no defect** (P2 scan clean; D14 recorded as an N1 terminology defect with fix).
2. **abstract — named defect:** clause-stacked endpoint inventory with slogan-like negative-parallelism
   opening and conversational verdict (D1, D2); also citation-in-abstract (D3).
3. **introduction — named defect:** conversational aside "the point we consider most diagnostic" (D5).
   (D4 is a closure gap, not a humanizer item.)
4. **related work — named defect:** conversational meta-register "it is worth being blunt about the
   boundary" (D6b). (D6 is a positioning/citation gap, not a humanizer item.)
5. **method — named defect:** repeated templated transition "Two/Three things …" (D8).
6. **experiments — named defect:** repeated defensive/honesty signalling and writing-process aside
   "tempting to write up" (D7), plus templated transitions (D8).
7. **discussion — named defect:** honesty signalling "One honest limitation" plus templated-transition
   tail (D7/D8).
8. **conclusion — named defect:** conversational self-assessment "we think" and negative-parallelism
   anaphora "does not buy … / rather than as" (D12). (D9–D11 are professionalism/closure items
   recorded separately.)
9. **figures-tables — no defect** (captions clean on the P2 scan; presentation defects D13 recorded
   separately).

Per Gate P2, any flagged passage must later be humanized with named defect attached and re-verified
for terminology, equations, numbers, citations, claim strength and formal register; the
writing-adapter integration is not invoked at S6 (no text edited).

---

## 6. Summary table

| module | score | threshold (4th of 10) | pass | blocking condition |
|---|---:|---:|---|---|
| title | 84 | 80 | PASS | — (D14 terminology, non-blocking) |
| abstract | 78 | 83 | **FAIL** | score < threshold (concision, tone) |
| introduction | 82 | 82 | **FAIL** | Gate N2: core-claim callback missing (D4) |
| related work | 86 | 78 | PASS | — (D6 must-fix before S8) |
| method | 89 | 84 | PASS | — |
| experiments | 89 | 84 | PASS | — |
| discussion | 88 | 65 | PASS | — |
| conclusion | 77 | 74 | **FAIL** | professionalism override (D9, D10) + Gate N3 (D11) |
| figures/tables | 82 | 82 | PASS | threshold-sensitive (D13 must-fix before S8) |

**Blocked cases:** none (`blocked-needs-author-input` not triggered — every defect above is repairable
by editing passages within the current evidence; no new experiments or author facts required).
Author-managed items (venue, keywords, declarations) remain pending and are out of S6 scope; the
abstract hard word/char limit cannot be adjudicated until the journal is selected (affects D1 target
length only).
