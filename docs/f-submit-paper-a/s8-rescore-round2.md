# S8 Re-score, Round 2 — Paper A

- **Stage:** S8 round-2 verification after the second S7 revision; same-agent role-pass, read-only verification.
- **Run root:** `tex_audit-journal/f-submit-v1-run-20261119-003907-paperA`.
- **Inputs:** round-1 `s8-rescore.md`; current `sources/`; frozen `module-quality-review.md`; `s7-revision-log.md`; rubric `[WORKSPACE]\.claude\skills\f-submit-v1\references\module-quality-rubric.md`; narrative/prose gates and S7/S8 convergence rules.
- **Re-score scope:** only Method and Experiments/Results, as assigned. Frozen round-1 scores are Method 82 and Experiments 85.
- **Frozen operational thresholds:** Method 84; Experiments 86. These remain the fourth-highest of the ten frozen comparator scores. Comparator/threshold uncertainty remains approximately +/-5 and is not a calibrated acceptance probability. A numeric score at or above threshold does not override a professionalism defect.
- **Current source binding (SHA-256):**
  - `sections/protocol.tex`: `8777122e2669bce144e6189191525b35efbec291d7b05bbc199ad0b917cef8fa`
  - `sections/results.tex`: `164645c4938cd88dfe022c6ddbc52736d43b7ec101a7d00d84bf3ebac2b778ce`
  - `sections/abstract.tex`: `ce5db31c14ede503b550ed02a1b2f8b83a927f48443c65cbb7b296cf72ab6aaf`
  - `sections/intro.tex`: `8be4861e333ea226c01390f230771b26ce75cc874d135d93853a5573ae09881b`
  - `sections/discussion.tex`: `5b7672bab1408e11b565dbaa38ce2aca8bb08ac011d0503fcace19479fc07ebe`
  - `sections/conclusion.tex`: `12678304e08f81e4221dd2d1c7a2daa29a3a370e50ad2ba4cd6fca7134e56095`
  - `main.tex`: `264c5aff10ff7d1c7a12f8bead2251fc3fd7daf6efda004bb6d5c4ae6a2dd422`

## 1. Named-defect closure verification

Searches covered the current `sources/**/*.tex`, followed by full reads of `protocol.tex` and `results.tex`. The revision log is not treated as evidence of closure where the current TeX disagrees with it.

| Defect | Round-2 status | Current-source evidence |
|---|---|---|
| **D-M1 three-vs-five count drift** | **CLOSED** | `protocol.tex` L56-57 now says "the five input variants measured". This agrees with `results.tex` L28 ("re-measured five of them"), the five bullets at L31-43, `abstract.tex` L12, and `intro.tex` L24-25. No "three combinations" instance remains in the TeX sources. |
| **D-M2 process-narration heading** | **CLOSED** | `protocol.tex` L50 is now the scientific heading "What the data path controls." The former "why it took a while" duration/chronology framing is absent. |
| **D-M3 band scale / Method anchor** | **CLOSED** | `protocol.tex` L21-27 states the graded comparison verdicts (reproduction, reproduction with a stated deficit, disagreement); L42-44 now explicitly defines the transportability band as three seed standard deviations of the matching pipeline and states that the rule was fixed before the multi-seed runs. `results.tex` L142-145 retains the audit-protocol/pre-registration anchor and attaches each verdict to its exact bar multiple (1.35x, 1.37x, 0.73x). Thus the round-1 residue—a missing Method-side operational definition—is repaired; "marginal" and "fail" are not left as unquantified verdicts. |
| **D-E1 draft-correction narration** | **CLOSED** | `results.tex` L90-93 now directly states that the 0.48--0.53 spread belongs to SpikingGCN, not SpikeGCL, and that SpikeGCL has no PubMed cell. The "earlier draft", "mis-attribution", and correction-chronology wording is gone. |
| **D-E2 fourth-combination process narration** | **OPEN — untouched** | `results.tex` L41-42 still reads "recorded in an earlier diagnostic as the ``fourth combination''; the cause is the normalisation, not the self-loops." This is the exact internal-diagnostic-history defect. The S7 log claims it was reworded, but the current source does not implement that claim. |
| **D-E4 "happy ending"** | **CLOSED** | `results.tex` L63-66 now ends the counterfactual with "it would report a clean reproduction." No "happy ending" wording remains. |
| **D-E5 "we would welcome either"** | **CLOSED** | `results.tex` L55-56 now says only "A container image or a commit hash would settle the question." The conversational appeal is absent. |
| **D-E6 rather-than-reconciled defensive negation** | **PARTIALLY CLOSED; counted as REMAINING** | The first site is repaired: `results.tex` L87-90 now says "recorded as part of the sensitivity evidence" without "rather than reconciled away". The second site remains verbatim at L136-137: "audited cell, recorded rather than reconciled." One of the two named Results instances therefore survives. |

### Defect-count and convergence record

- Round-1 named-target burden: **8** defects (Method: D-M1, D-M2, D-M3; Experiments: D-E1, D-E2, D-E4, D-E5, D-E6). Partially closed D-M3 was counted as remaining in round 1.
- Round-2 burden: **2** named defects remain: **D-E2** and **D-E6**. D-E6 improved but is counted as remaining until both named instances are removed.
- **Closed this round: 6** (D-M1, D-M2, D-M3, D-E1, D-E4, D-E5).
- This is a real reduction from 8 to 2. The rule requiring a stop after two consecutive rounds without fewer named defects is therefore **not triggered**. Round 3 is warranted for the two narrow Results passages, not a wholesale rewrite.

## 2. Re-scored modules

### Method — 82 -> **86 / 84 — PASS**

D-M1 removes the cross-section numerical contradiction; D-M2 removes the process-chronology heading; D-M3 now supplies the Method-side transportability-band definition and pre-run timing, completing the evidence callback for the 3-sigma comparisons. The existing strengths remain intact: bounded surveyed population, authors-own-entry-point protocol, held-fixed/varied design, explicit sample sizes, log-derived values, and stated audit boundaries.

The four-point gain reflects restored technical precision, evidence anchoring, tone, and selling-point consistency. No named professionalism-override or count contradiction remains in this module. The score exceeds the frozen operational threshold by two points; threshold uncertainty is unchanged and is not being used to move the bar.

### Experiments / Results — 85 -> **86 / 86 — FAIL (professionalism override)**

Closing D-E1 removes the most serious draft-correction chronology and replaces it with the actual sensitivity evidence. Closing D-E4 and D-E5 improves academic register and concision. These changes justify a one-point numeric gain to the frozen threshold.

The module nevertheless **cannot pass**: D-E2 is still manuscript-production/diagnostic-history narration, an override-class defect under the rubric, and D-E6 still has one defensive "rather than reconciled" instance. Numeric equality with 86 is necessary but not sufficient. The residual defects are local and do not require changing any result, number, citation, or claim.

**Required round-3 edits only:**

1. At `results.tex` L41-42, delete "recorded in an earlier diagnostic as the ``fourth combination'';" and retain the scientific attribution: "the cause is the normalisation, not the self-loops."
2. At `results.tex` L136-137, replace "audited cell, recorded rather than reconciled." with the plain factual "audited cell."

## 3. Core-claim closure spot-check (N1/N2)

- **Central question:** whether published spiking-GNN accuracies are commensurable and what makes a number portable.
- **Core claim:** the audit identifies three measured failure modes; transportability is split- and dataset-dependent, with per-dataset evidence rather than a blanket effect.
- **Forward trace:** title's bounded audit -> abstract's three findings and per-dataset transportability -> Introduction's gap and contributions -> Method's replay protocol, five-variant diagnostic, controlled one-flag comparison, graded scale, and 3-sigma band -> Results' audit grid, split-mixing evidence, transportability ratios, and bounded census. **Holds after round 2.**
- **Backward trace:** Conclusion's bounded census and per-dataset verdict -> Discussion's interpretation and threats -> Results' measured gaps/noise -> Method's execution and measurement rules -> stated commensurability gap. **The core chain remains traceable.** The already-named Discussion defects D-D1/D-D2 are unchanged and outside this two-module re-score; in particular D-D1 remains a whole-paper articulation blemish, so this spot-check is not a declaration that the entire paper passes S8.
- **Numerical spot-check:** Cora 8.0/1.97 = 4.06 sigma and 4.06/3 = 1.35x; CiteSeer 5.1/1.24 = 4.11 sigma and 4.11/3 = 1.37x; PubMed 11.7/5.37 = 2.18 sigma and 2.18/3 = 0.73x. The abstract, Results, and Conclusion retain these values and the PubMed confounded-cell boundary. No new conclusion-only claim or stronger causal wording was introduced.
- **Supporting/negative findings:** five input variants now agree across Method/Results/Abstract/Introduction; row-normalisation remains the supported local attribution; the same-seed 1.0-point sensitivity result remains stated locally. D-E2/D-E6 are professionalism residues, not missing evidence for the core claim.

## 4. Professionalism re-scan of the named modules

- **Method:** no remaining named process heading, count contradiction, or unanchored-bar defect.
- **Experiments:** manuscript-production narration remains at D-E2; defensive negation remains at the second D-E6 site. No old D-E1 draft-correction text, D-E4 conversational verdict, or D-E5 conversational appeal remains.
- No numbers, citations, protected facts, limitations, or declarations were edited by this verification.

## 5. Summary and verdict

| Module | Frozen round-1 score | Round-2 score | Frozen threshold | Gate | Remaining named defects |
|---|---:|---:|---:|---|---|
| Method | 82 | **86** | 84 | **PASS** | None of D-M1/D-M2/D-M3 |
| Experiments | 85 | **86** | 86 | **FAIL** | D-E2 open; D-E6 partially closed but still present at L137 |

**Verdict: needs-round-3.** Method now passes. Experiments reaches the numerical comparator threshold but fails the independent professionalism override. The defect count fell from 8 to 2, so this is not a no-progress stopping case. Only the two quoted Results passages require correction before the affected Experiments re-score and final dependent checks.
