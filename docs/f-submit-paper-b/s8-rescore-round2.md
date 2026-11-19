# S8 round 2 — verification and re-score (Paper B, conversion)

**Run root:** `[WORKSPACE]\auto-paper\20261028-171500\Spiking-GNN-Conversion\f-research-v1-run-20261118-181848-journal-upgrade\tex-journal\f-submit-v1-run-20261119-003907-paperB`  
**Review date:** 2026-11-19  
**Review execution:** same-agent role-pass; current sources inspected directly. No manuscript source was edited in this verification step.  
**Rubric:** `[WORKSPACE]\.claude\skills\f-submit-v1\references\module-quality-rubric.md`  
**Frozen round-1 scores:** abstract 78, introduction 84, method 90, conclusion 82.  
**Frozen fourth-highest-of-10 thresholds:** abstract 83, introduction 82, method 84, conclusion 74. These operational comparator thresholds are unchanged; they are not calibrated acceptance probabilities.

Scope is limited to the four named modules and their round-2 target defects. Related work, experiments, discussion, figures/tables, and title are not re-scored. Experiments and discussion are read only for the core-claim closure spot-check.

## 0. Current-source identity

The checks below bind to these exact files:

| source | SHA-256 |
|---|---|
| `sources/sections/abstract.tex` | `14cb55a6a87ed989c0d430ec0134bbb14c6f8d5de98a12bc316d4ebca9104ba2` |
| `sources/sections/intro.tex` | `38d01e5ab8ad8340bb44049f6f244ba8896757a5fefaa80c1e83514bddfd7a7a` |
| `sources/sections/method.tex` | `225f348eb0e34b529db5e3ec50c5c729848dc9f15bd038334b6bf4c3dec3438b` |
| `sources/sections/conclusion.tex` | `5028da6f5738420a28372b57401518b696cc0535298c36e0e7d32ce354492361` |

## 1. Named-defect closure audit

| module / item | round-2 status | current-source evidence |
|---|---|---|
| Abstract **D1 — density / clause-stacked endpoint inventory** | **NOT CLOSED** | The abstract is still approximately 600 lexical words after simple LaTeX stripping (round 1: approximately 640; S6: 648), far above the S6 target of approximately 200–250 words. It still carries well over 15 numeric endpoints, including 0.08/105 cells, 3.5 at L=3, G^0.95/r=0.990/21 points at T=8, 22.7–93.4%/six-of-eight, 74.0/T=16, and 86.6 (`abstract.tex:2–43`). The four-flagship compression prescribed for D1 was not achieved; only a modest shortening occurred. |
| Abstract **D2 — slogan opening / conversational verdict** | **CLOSED** | The opening is now the direct statement “The error of a graph-to-spiking conversion is a deterministic, closed-form function of the trained weights” (`abstract.tex:2–3`). “is not noise. It is…”, “an identity, not a bound”, and “That is the sense in which…” are absent. |
| Abstract **D3 — citation inside abstract** | **CLOSED** | There is no `\citep`/`\citet` in `abstract.tex`. The companion audit is mentioned without a citation at `abstract.tex:42–43`, as prescribed. |
| Introduction **D5 — conversational aside** | **CLOSED** | “the point we consider most diagnostic” is absent. The sentence now directly states that the method degrades in the predicted way when operator, threshold, width, or decoder changes (`intro.tex:134–137`). |
| Introduction **D6(i) — uncited point-cloud prior-art claim** | **CLOSED LOCALLY** | The point-cloud/max-pooling assertion has been removed, so the uncited claim no longer exists. The graph-precedent sentence now names cited text-classification and tactile-graph conversions (`intro.tex:28–31`). This closes only the introduction sub-item assigned here; it does not assert closure of the separate related-work comparator-positioning items. |
| Introduction **D7 tail — “an honest negative answer” honesty label** | **CLOSED** | The phrase is absent. The corresponding contribution is headed “A depth law, and a negative result” and states the measured outcome directly (`intro.tex:92–101`). |
| Introduction **D4 residual — prior-art clause in the constructive-contribution bullet** | **CLOSED** | Bullet 9 now explicitly says “the idea family is established prior art; the graph instantiation and regime measurement are ours,” while retaining 22.7–93.4%, six of eight held-out four-layer cells, the failed mixed-depth screen, and transfer failures (`intro.tex:123–128`). The round-1 core-callback closure remains intact. |
| Method **D8 — repeated templated “Two/Three things…” transitions** | **CLOSED IN METHOD** | The four flagged sites no longer use the flagged template: the identity discussion proceeds through substantive First/Second/Third consequences (`method.tex:66–77`); the quantiser regimes are tied directly to the implementation (`:89–96`); the bias discussion uses “Two consequences separate cleanly” (`:118–126`); and the readout uses the S6-sanctioned topic-led form “The claim has two boundaries” (`:145–152`). This is no longer the repeated empty “Two/Three things X” pattern. No claim is made here about the separate experiments/discussion instances. |
| Conclusion **D9 — stale one-sided external-anchor clause** | **CLOSED** | The singular favorable-only clause is gone. The conclusion now gives both anchors, 74.0 below and 86.6 above (`conclusion.tex:26–28`), and its claim-site sentence says “the two anchors… bracket the conversion rather than rank it” (`:42–44`), matching `experiments.tex:188–194`. |
| Conclusion **D12 — conversational self-assessment + anaphora** | **PARTIAL — NOT CLOSED** | The four-fold anaphora is reduced to two “buy” formulations, and the remaining decoder/ensemble findings are stated more directly (`conclusion.tex:32–44`), satisfying that subpart. However, self-narration remains in “we state them as findings” (`:32`), “We also report a prediction we could not make good on” (`:46`), and “What survives is narrower and more useful than the claim we set out to make” (`:48–49`). The first and last retain exactly the kind of reporting-process/self-assessment framing that the S6 fix instructed be replaced by direct findings; only “we think” and “rather than as disclaimers” were removed. |
| Conclusion **round-1 antecedent regression — “They train one model…”** | **CLOSED** | The subject is now explicit: “The converted networks train one model instead of five, and they degrade…” (`conclusion.tex:28–30`). It can no longer attach to SpikeGraphormer/SpikeGCL in the preceding sentence. |

**Target-defect count:** 11 items entered round 2; **9 closed, 2 remain open** (D1 and D12). D1 and D12 both improved partially, but partial wording change is not counted as closure. This is a real reduction from 11 to 2, so the rule requiring a stop after two consecutive rounds without fewer named defects is **not triggered**.

## 2. Frozen-threshold re-scores

| module | before | after | frozen threshold | gate | rationale |
|---|---:|---:|---:|---|---|
| **abstract** | 78 | **80** | 83 | **FAIL** | D2 and D3 restore tone/polish (+2 overall). D1 remains the dominant concision/structure defect: approximately 600 words and an endpoint inventory rather than the comparator-aligned five-move, four-flagship spine. The small reduction in length does not justify closing D1 or clearing the threshold. |
| **introduction** | 84 | **87** | 82 | **PASS** | D5 and the D7 tail remove the local conversational/honesty-signalling deductions; removal of the uncited point-cloud assertion repairs citation support; the prior-art clause completes D4’s attribution residual. Core-claim B remains closed. The pre-existing ten-item contribution-list density remains priced, so this is not scored as defect-free. |
| **method** | 90 | **91** | 84 | **PASS** | D8 is closed at all four method sites without changing equations, terminology, citations, or claim strength (+1 tone/flow). The previously priced absence of a notation table remains unchanged. |
| **conclusion** | 82 | **85** | 74 | **FAIL (P2/professionalism closure)** | D9’s bracketing callback and the explicit antecedent repair evidence consistency and flow; the reduced anaphora improves tone. The numeric score clears the frozen threshold, but D12 is only partially repaired. The remaining conversational self-assessment/reporting-process passages keep the named P2/professionalism defect open, so score alone cannot pass the module. |

No comparator identity, comparator score, or threshold was changed. The conclusion’s fail is therefore not a threshold failure; it is a hard prose/professionalism-gate failure. The abstract fails on score and still has its named density defect.

## 3. Core-claim closure spot-check (N1–N3)

### Core claim A — exact identity and weight-readable conversion laws: **CLOSED**

- **Forward chain:** title (`main.tex:57–58`) → abstract identity/depth/readout promises (`abstract.tex:2–26`) → introduction contributions 1–8 (`intro.tex:69–122`) → method identity, membrane readout, margin theorem, recursion, activation mass, and spectral-gain predictor (`method.tex:59–77`, `:128–152`, `:154–195`, `:197–268`) → experiments.
- **Evidence agreement:** the depth result is consistently G^0.95, r=0.990 over nine cells and a 21-point deficit at L=4/T=8 (`experiments.tex:555–570`; `discussion.tex:66–68`; `conclusion.tex:12–15`). Order invariance is operationalized and measured at 0.000 pp membrane spread versus up to 1.6 pp for count decoding (`experiments.tex:588–604`). The bounded predictor claim remains ranking-only: rho=0.967 but level error 4.6 points (`experiments.tex:100–125`; `discussion.tex:70`; `conclusion.tex:46–52`).
- **Backward chain:** the conclusion recalls the same identity, readout, depth law, ranking limitation, activation mass, and spectral gain; no stronger absolute-prediction or cross-graph-transfer claim appears. Abstract/introduction/conclusion strengths are compatible even though the abstract now omits some numeric details.

### Core claim B — deep-regime validation-search per-layer allocation: **CLOSED**

- **Forward chain:** abstract (`abstract.tex:28–38`) → introduction bullet 9 with explicit prior-art attribution (`intro.tex:123–128`) → validation-only coordinate-search procedure and held-out evaluation (`experiments.tex:672–694`).
- **Evidence agreement:** all sites retain 22.7–93.4% fewer total layer-updates on six of eight held-out L=4 configurations. The experiments retain the failed mixed-depth screen at 37.5% and the random-allocation control (`experiments.tex:679–690`); the abstract discloses the failed screen without repeating its numeric percentage.
- **Interpretation and backward callback:** discussion limits weight-only certificates to in-configuration statements and recommends validation search/recalibration for new graphs (`discussion.tex:121–128`). Conclusion recalls the same 22.7–93.4%/six-of-eight result and the weight-only transfer failures (`conclusion.tex:22–25`). No conclusion-only capability or incompatible magnitude was introduced.

### External-anchor consistency: **CLOSED**

The abstract, experiments, and conclusion now all use the two-sided 74.0/86.6 bracket. The conclusion no longer contradicts its own paired-anchor sentence, and the repaired explicit subject removes the round-1 antecedent ambiguity.

**Spot-check verdict:** both core claims have complete, evidence-bounded forward and backward chains. The two remaining defects are presentation/prose defects, not newly discovered evidence orphans or contradictory core callbacks.

## 4. Verdict and minimum round-3 work

**Verdict: `needs-round-3`.** Introduction and method pass. Abstract and conclusion do not. There is no evidence/input blocker and no stopping-rule trigger: the named target-defect count fell from **11 to 2**.

1. **Abstract D1:** perform the actual five-move compression to approximately 200–250 words, retaining the four S6 flagship evidence groups and moving the remaining endpoint inventory to results. Preserve the now-clean opening and citation-free abstract.
2. **Conclusion D12:** replace the remaining reporting-process/self-assessment clauses with direct scientific findings. In particular, remove “we state them as findings,” “could not make good on,” and “than the claim we set out to make”; retain the already repaired two-sided anchor and explicit converted-network subject.

Out-of-scope round-1 defects in related work, experiments, discussion, and figures/tables are neither re-scored nor declared closed by this report.
