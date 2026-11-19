# S8 — Re-score after S7 revision (Paper B, conversion)

**Run root:** `[WORKSPACE]\auto-paper\20261028-171500\Spiking-GNN-Conversion\f-research-v1-run-20261118-181848-journal-upgrade\tex-journal\f-submit-v1-run-20261119-003907-paperB`
**Scope:** re-score the S7-changed modules (abstract, introduction, method, conclusion) plus the module whose
score depended on them (title, via D14). Frozen comparator thresholds and the 4th-of-10 top-40% gate are
carried over unchanged from `module-quality-review.md` (S6); thresholds are not moved. Unchanged modules
(related work, experiments, discussion, figures/tables) are not re-scored; their S6 defects are re-verified
below only to record closure status.
**Rubric:** `[WORKSPACE]\.claude\skills\f-submit-v1\references\module-quality-rubric.md`
**Frozen pre-edit scores (S6):** title 84, abstract 78, introduction 82, method 89, conclusion 77.
**Review date:** 2026-11-19. Sources audited in place.

---

## 0. What S7 actually changed (verification against the source)

S7's revision log (`s7-revision-log.md`) records 5 edits: intro capability bullet (D4), conclusion
allocation recall (D11), conclusion accounting qualifier (D10), conclusion balanced anchors (D9), and the
method activation-mass definition (D14). Source inspection confirms exactly those three files were touched
(`intro.tex`, `method.tex`, `conclusion.tex`, mtimes 01:53–01:54) and **no other module was edited**.

**Coverage gap:** the log's claim "Every S6 defect mapped to a concrete edit" is not borne out — the table
covers 5 of the 14 named S6 defects (D4, D9, D10, D11, D14), and only 3 of those 5 are fully closed (see §1).
In particular the **abstract was not edited at all** despite being a FAIL module in S6: `sources/sections/abstract.tex`
still carries the import-time timestamp (00:39), and every S6-flagged string is verbatim at S6's cited lines
(lines 2–3 slogan opening, line 6 "an identity, not a bound", line 10 "That is the sense in which the error is
not noise", line 49 `\citep{companion}`). (The task brief's "S7 changed: abstract" is a provenance artifact of
abstract.tex being listed in the pipeline-state S7 evidence block; no edit reached the file.)

## 1. Named-defect closure audit (D1–D14, searched in current sources)

| defect | S6 class | status in current text | evidence |
|---|---|---|---|
| D1 abstract endpoint inventory | P2 (score) | **NOT closed** | abstract verbatim; ~640 words (S6: 648), 15+ numeric endpoints unchanged |
| D2 slogan opening + conversational verdict | P2 (score) | **NOT closed** | abstract.tex:2 "is not noise. It is…", :6 "an identity, not a bound", :10 "That is the sense…" verbatim |
| D3 citation inside abstract | tone/polish | **NOT closed** | abstract.tex:49 `\citep{companion}` verbatim |
| D4 intro core-claim callback (Gate N2) | structural | **CLOSED** | intro.tex:124–128 new bullet 9 with the deep-regime capability (22.7–93.4%, six of eight held-out four-layer cells, failed screen disclosed) **and** the three-route transfer-failure clause (P9-RB-M2 residue cleared). Minor residual: the prescribed "idea family is established prior art" clause was not added to the bullet (attribution present in abstract:37–38, related work "We therefore claim no priority", and sec:allocation's citations — non-blocking) |
| D5 intro conversational aside | P2 (humanizer) | **NOT closed** | intro.tex:135–136 "and, the point we consider most diagnostic, …" verbatim |
| D6 frozen comparators unpositioned | positioning | **NOT closed** (module unchanged) | SpikeZIP-TF / Spiking PointCNN / MK-SGN still uncited anywhere; intro.tex:30 point-cloud conversion claim still uncited (D6 fix (i) targeted a changed module, not applied) |
| D6b related meta-register | P2 | **NOT closed** | related.tex:68 "it is worth being blunt about the boundary" verbatim |
| D7 experiments honesty signalling + writing-process aside | P2 (override class "repeated defensive negation") | **NOT closed** | experiments.tex:59 "tempting to write up", :137 "worth more than a", :169 "The honest counterweight", :263 "rather than hiding them", :300 "rather than as wins", :379 "Where the claim is thin, we say so", :410 "honest alternative"; discussion.tex:50 "One honest limitation"; intro.tex:93 "an honest negative answer" — all verbatim |
| D8 templated "Two/Three things …" transitions | P2 | **NOT closed** | method.tex:67 "Three things follow.", :89 "Three regimes matter", :118 "Two consequences are worth separating.", :145 "Two boundaries on that claim." (plus 6 sites in experiments/discussion) verbatim |
| D9 conclusion one-sided anchor callback | **professionalism override** | **PARTIAL — not closed** | balanced pair added at conclusion.tex:26–28 (74.0 below at T=16, 86.6 above every arm, labelled external) — the material caveat is now in the Conclusion and matches sec:audit's "brackets rather than ranks". But the S6-flagged claim-site sentence survives verbatim at conclusion.tex:39: "the one we could place beside our own, measured from its authors' code on the public split, sits below the conversion rather than above it" — still one-sided at the claim "published numbers cannot be used as baselines without checking", and its singular "the one" now conflicts with the module's own two-anchor sentence and with sec:audit ("this paper uses only two of the resulting numbers … as an external anchor"). S6's prescribed fix (replace the clause with the bracketing statement) was not applied |
| D10 conclusion cost claim without convention | **professionalism override** | **CLOSED** | conclusion.tex:18–21 now reads "…2–23× fewer synaptic operations than a trained spiking network on two of three graphs under the static-update accounting --- the ordering is convention-conditional (\secref{sec:accounting})…". Verified against sec:pareto (2× Cora vs direct arm, 23× CiteSeer vs published architecture, PubMed tie → "two of three") and sec:accounting ("the cheapness claim is therefore stated with its convention attached") |
| D11 conclusion missing allocation recall (Gate N3) | structural | **CLOSED** | conclusion.tex:22–25 "in the deep regime … a validation-only per-layer window search keeps the accuracy band with 22.7–93.4% fewer layer-updates on six of eight held-out four-layer cells (\secref{sec:allocation}), while the weight-only routes fail to transfer (\secref{sec:transfer})". Minor residual: the prescribed "the idea family is prior art" clause omitted (mitigated as in D4) |
| D12 conclusion conversational self-assessment + "does not buy" anaphora | P2 (humanizer) | **NOT closed** | conclusion.tex:32 "we state them as findings rather than as disclaimers"; :33–39 four-fold anaphora ("does not buy energy" / "does not buy depth" / "does not even buy freedom" / "Nor does it buy a free ensemble"); :43 "narrower and, we think, more useful" — all verbatim |
| D13 figures/tables cross-references | presentation | **NOT closed** (module unchanged) | still zero `\tabref{tab:capacity}` and zero prose `\figref` for fig:pred/fig:pareto/fig:allocation/fig:transfer in experiments.tex; table `\input` dump ordering unchanged |
| D14 title terminology drift (Gate N1) | structural | **CLOSED** | method.tex:223–225 now defines the title object: "The active hidden width $H$ is what the title calls the \emph{activation mass}: the number of active units carrying quantisation error", feeding eq:sqrtH and eq:sigma; the conclusion's "its activation mass and its cumulative spectral gain … can both be read off the trained weights" (conclusion.tex:46–47) now resolves to a named quantity consistent with "Everything in \eqnref{eq:sigma} except the single scalar $c$ is computed from the trained weights and the graph". Minor residual: discussion.tex:100 "pruning half the hidden units by activation mass" still uses the term as a per-unit ranking attribute (out-of-scope module; non-blocking) |

**S8-found regression (introduced by the S7 edit):** at conclusion.tex:28–30 "They train one model instead of
five, and they degrade in the way the theory predicts …" — the subject is the converted networks (one training
run vs five latencies of direct training), but the inserted external-anchor sentence now sits immediately
before it, so "They" most naturally attaches to SpikeGraphormer/SpikeGCL. One-clause fix: restate the subject
("The conversion trains one model instead of five, and the converted networks degrade …") or move the anchor
sentence to the end of the paragraph. This is a flow/coherence deduction, not a gate.

## 2. Re-scored modules (frozen thresholds; top-40% gate unchanged)

### 2.1 Title — 84 → **86** / threshold 80 → **PASS** (was PASS)
D14 closed: the title's third object now has a body definition cross-referenced back to the title
("what the title calls"), and the conclusion callback resolves. Deduction remaining: long title
(pre-existing, priced). Margin +6.

### 2.2 Abstract — 78 → **78** / threshold 83 → **FAIL** (unchanged)
No S7 edit reached the file (§0). D1, D2, D3 all verbatim present. The S6 fail was on the comparator-facing
dimensions (concision: ~640 words and 15+ numeric endpoints against the comparators' 2–4-headline-number
spine; tone: slogan-like negative-parallelism opening and conversational verdict; citation-in-abstract).
Nothing changed, so the score is unchanged and the fail stands. The S6 note holds: the fail is robust to
comparator-estimation error unless the comparator abstract estimates are ≥5 points too high.

### 2.3 Introduction — 82 → **84** / threshold 82 → **PASS** (was FAIL on Gate N2)
D4 closed: the capability bullet (intro.tex:124–128) restores the title/abstract promise → Introduction
contribution forward link for core claim B, with deep-regime scoping, the failed mixed-depth screen disclosed,
and the three transfer-failure routes named (the negative map now appears in the contribution list). The
selling-point consistency/narrative closure dimension is restored. Unchanged deductions, already priced in
the 82: D5 conversational aside (intro.tex:135–136), D6(i) uncited point-cloud claim (intro.tex:30), D7-tail
"an honest negative answer" (intro.tex:93), 10 bullets vs the field's 3–4 (the 10th is the mandated fix, no
new deduction). Minor residual not priced twice: bullet 9 omits the prescribed prior-art clause
(non-blocking; attribution lives in abstract/related/sec:allocation). Score equals threshold +2; the S6
threshold-sensitivity flag (±6) remains — on the frozen threshold this passes.

### 2.4 Method — 89 → **90** / threshold 84 → **PASS** (was PASS)
D14's definition (method.tex:223–225) is scientifically consistent with the surrounding math: eq:sqrtH
($\lVert E_1\rVert \approx \kappa\sqrt{H}/T$) and eq:sigma ($\sigma_n = cG\sqrt{H}\lVert a_n\rVert_2 r_L/T$)
use exactly the active width the definition names, and the conclusion's "two quantities … read off the trained
weights" now points at the two weight-readable factors of eq:sigma ($\sqrt{H}$ and $G$). Specificity and
closure gain ~1 point. Unchanged deductions, already priced: D8 templated transitions (4 sites in method),
no notation table (structural note). Margin +6.

### 2.5 Conclusion — 77 → **82** / threshold 74 → **FAIL (professionalism override persists)** (was FAIL on override + Gate N3)
Repairs landed: D10 closed (cost claim now carries its static-update accounting convention and the
convention-conditional clause; verified against sec:pareto and sec:accounting — magnitudes exact), D11 closed
(Gate N3 recall present with sec:allocation/sec:transfer pointers and identical magnitudes), D9 partially
repaired (the balanced 74.0/86.6 pair added at conclusion.tex:26–28, matching sec:audit's "brackets rather
than ranks"). Score-wise these restore the closure and evidence dimensions (77 → 82; the score would clear
the threshold). **But the module still fails:** S6 fired the professionalism override on the D9 passage and
that passage is verbatim unedited at conclusion.tex:39 — the claim-site evidence for "published numbers
cannot be used as baselines without checking" is still the favorable instance alone, in the singular, and it
now contradicts the module's own two-anchor sentence ("the one we could place beside our own" vs "SpikeGraphormer
… and SpikeGCL …"). An override-class flagged passage that S6 named and prescribed a fix for, left verbatim,
keeps the override alive; D12 (the named humanizer defect) is likewise verbatim ("we think" :43, "findings
rather than as disclaimers" :32, four-fold "does not buy" :33–39), and Gate P2's "must later be humanized"
is unmet. Plus the S8-found "They train one model instead of five" antecedent regression (§1). Round-2 fix is
small: apply the S6-scribed replacement for the :39 clause (or delete it — the pair is now stated at :26–28),
humanize D12 per the S6 fix, and restate the "They" subject.

## 3. Core-claim closure re-run (Gate N2/N3)

**Core claim A — identity + laws (depth law, order-invariant readout, threshold pinning, ranking-only
predictor): closure COMPLETE (D14 residual now cleared).**
- Forward: title (spectral gain, activation mass, order-invariant readout — all three objects now defined in
  the body) → abstract → intro bullets 1–8 → method (eq:identity, eq:recursion, eq:sigma, thm:margin,
  eq:sqrtH) → experiments (sec:closed, sec:theory, sec:depth, sec:order, sec:threshold, sec:ranking) →
  conclusion ¶1 (identity, bias, readout, theorem, depth law with G^0.95, r=0.990 over nine cells).
- Backward: conclusion ¶4 recalls the ranking magnitudes (ρ=0.967, 4.6 points) and the activation-mass /
  spectral-gain pair with the same scope as the abstract. No conclusion-only claim in the spine.
- **Transportability/transfer magnitudes consistent:** κ = 1.0 in-distribution and κ ∈ {1.2, 11.0, 2.5} on
  held-out graphs (experiments.tex:715–717) match the abstract's "(κ=1.2,11.0,2.5)" and its "exact
  in-distribution yet misses the accuracy band on two of three held-out graphs"; the conclusion's "the
  weight-only routes fail to transfer" carries no conflicting magnitude. All other A-spine numbers agree
  across modules (0.08 pp / 105 cells; 3.5 at L=3; 0.000 vs 1.6; G^0.95, r=0.990; 21 points at T=8).

**Core claim B — deep-regime per-layer window allocation: closure NOW COMPLETE** (was INCOMPLETE at S6).
- Abstract ¶3 → **intro.tex:124–128 (new bullet 9)** → experiments sec:allocation (procedure, 22.7–93.4%
  fewer total layer-updates on six of eight held-out L=4 configurations, random-allocation control, failed
  mixed-depth screen 37.5%) → discussion outlook → **conclusion.tex:22–25 (new recall)**.
- Magnitudes identical at all four sites: 22.7–93.4%, six of eight held-out four-layer (chameleon 2/4,
  squirrel 4/4 per experiments.tex:680–681), screen not met at 37.5% (experiments.tex:690, abstract.tex:36).
- Prior-art attribution intact where it carries weight (abstract.tex:37–38 "The idea family … is established
  prior art"; related work "We therefore claim no priority"; sec:allocation cites ramesh2025pascal,
  guo2026mixed, nallathambi2022ldsnn). Residual polish: the intro bullet and the conclusion recall omit the
  attribution clause the S6 fix text carried — one clause each, non-blocking.

**Title-to-conclusion chain:** complete for both cores. One chain-level residue: the conclusion's stale
one-sided anchor clause (conclusion.tex:39) is the only passage in the chain that no longer matches the
paired "brackets rather than ranks" framing used by sec:audit, the abstract, and conclusion ¶2.

## 4. Summary table

| module | before | after | threshold | pass | blocking condition |
|---|---:|---:|---:|---|---|
| title | 84 | 86 | 80 | PASS | — (D14 closed) |
| abstract | 78 | 78 | 83 | **FAIL** | score < threshold; D1/D2/D3 untouched by S7 |
| introduction | 82 | 84 | 82 | PASS | — (D4 closed; D5/D6(i)/D7-tail deductions remain) |
| method | 89 | 90 | 84 | PASS | — (D8 deduction remains) |
| conclusion | 77 | 82 | 74 | **FAIL** | professionalism override persists (D9 passage verbatim at :39); D12 unhumanized |

**Unchanged modules not re-scored (S6 scores stand):** related work 86/78 PASS (D6/D6b open), experiments
89/84 PASS (D7/D8 open), discussion 88/65 PASS (D7/D8 open), figures/tables 82/82 PASS-threshold-sensitive
(D13 open). Note: S6 marked D6 and D13 "must-fix before S8"; both are still open in modules outside this
rescore's scope and must be handled in round 2.

**Blocked cases:** none. Every open defect is repairable by passage-level editing within the current evidence;
no new experiments or author facts required. Author-managed items (venue, keywords, declarations) unchanged.

## 5. Verdict: **needs-round-2**

Round-2 minimum edit set (ordered by severity):
1. Abstract D1–D3: compress to the 5-move spine with the four flagships (per S6 D1 fix), rewrite the
   slogan opening/verdict (D2), drop `\citep{companion}` from the abstract (D3). This is the only module
   failing on score against the frozen comparator threshold.
2. Conclusion D9 residue: replace the conclusion.tex:39 clause with the S6-scribed bracketing sentence (or
   delete it); then humanize D12 per the S6 fix; fix the "They train one model instead of five" antecedent.
3. Intro D5 aside deletion; intro.tex:30 point-cloud citation (D6(i)); optional prior-art clause in bullet 9.
4. Out-of-scope open items: D6 (three comparator positionings + MK-SGN/SpikeZIP-TF/PointCNN citations),
   D6b, D7 (experiments/discussion honesty-signalling set), D8 (10 templated transitions), D13 (table
   ordering, tab:capacity orphan, four unreferenced figures), D14-minor (discussion.tex:100 usage).
