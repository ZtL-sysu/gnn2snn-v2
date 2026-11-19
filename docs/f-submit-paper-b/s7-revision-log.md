# S7 Revision Log — Paper B (round 1 of budget 3)

Every S6 defect mapped to a concrete edit; facts/numbers/citations/protected sources unchanged.

| S6 defect | action | landed at |
|---|---|---|
| Core B missing from intro contribution list | inserted capability bullet (deep-regime scope + failed mixed screen disclosed + transfer-failure pointer) | sources/sections/intro.tex |
| Core B missing from conclusion recall | added recall sentence with sec:allocation/sec:transfer pointers | sources/sections/conclusion.tex |
| Conclusion convention-less 2-23x claim | added static-accounting qualifier + convention-conditional clause | sources/sections/conclusion.tex |
| One-sided anchor callback in conclusion | both anchors added (74.0 below at T=16; 86.6 above all arms), labelled external | sources/sections/conclusion.tex |
| Title term "Activation Mass" undefined in body | defined at eq:sqrtH as the active hidden width carrying quantisation error | sources/sections/method.tex |

Post-edit checks: numbers unchanged (22.7-93.4, 6/8, 37.5%, 74.0/86.6); cross-refs sec:allocation/sec:accounting/sec:transfer resolve; register preserved; sources rebuilt with 0 errors (build-B-s7.log).


## Round 2 (S8 round-1 defects)

| defect | action |
|---|---|
| D1 abstract density (15+ endpoints) | rewritten: central numbers only (0.08/105, 3.5, 21, 22.7-93.4, 6/8, 74.0/86.6); secondary values qualitativeised |
| D2 slogan parallelism | removed "is not noise. It is..." / "an identity, not a bound" / "That is the sense..." |
| D3 citation in abstract | \citep{companion} removed; "a companion paper reports..." |
| D4 capability bullet prior-art clause | added to the intro capability bullet |
| D5 conversational aside | removed ("and, the point we consider most diagnostic") |
| D6 uncited point-cloud claim | clause deleted (verified precedents One-Spike/PMSM retained) |
| D7 "an honest negative answer" | -> "a negative result" |
| D8 templated transitions (method x4) | reworded to direct consequence statements |
| D9 stale one-sided anchor clause | -> "the two anchors ... bracket the conversion rather than rank it" |
| D12 honesty signalling / anaphora | "as findings"; "does not buy" 4x -> 2x; "we think" removed |
| S8 regression (antecedent) | "The converted networks train one model instead of five" |

Round-2 rebuild: 0 errors (build-B-r2.log).
