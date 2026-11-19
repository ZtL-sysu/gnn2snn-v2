# S7 Revision Log — Paper A (round 1 of budget 3)

| S6 defect | action | landed at |
|---|---|---|
| failure-mode-3 "same baseline" contradicts evidence rows | articulation changed to "baseline rows drawn from the two different splits of one comparison source" (subsection headline + results + both intro sites) | results.tex, intro.tex |
| input-variant count drift (3/4/5) | five explicit bullets (raw+/raw-/row-norm/rownorm+rescale/rownorm+noselfloop); "any of four plausible" -> "any of the five measured input variants" | results.tex |
| unanchored pre-registered 3sigma bar | anchored: "fixed in the audit protocol before the multi-seed runs, as the pre-registration rule" | results.tex (sec:transport) |
| conclusion patch-count contradiction | "compatibility patches the one non-zero-change pipeline required" (matches protocol table) | conclusion.tex |

Post-edit checks: magnitudes unchanged (51.5/13.0 grid, 1.35/1.37/0.73, 4.06/4.11/2.18, 1.97/5.37); register preserved; sources rebuilt with 0 errors (build-A-s7.log).


## Round 2 (S8 round-1 defects)

| defect | action |
|---|---|
| D-M1 "three combinations" vs five | -> "the five input variants measured" |
| D-M2 process-narration heading | -> "What the data path controls." |
| D-M3 band scale undefined in method | graded scale anchored in How-numbers-are-read (published spread; 3 s.d. rule fixed before seed runs) |
| D-E1 draft-correction narration | replaced with direct statement of the mis-attribution finding |
| D-E2 "earlier diagnostic ... fourth combination" | reworded to attribution statement |
| D-E4 "happy ending" | -> "clean reproduction" |
| D-E5 "and we would welcome either" | -> "would settle the question" |
| D-E6 "rather than reconciled away" | removed both instances |

Round-2 rebuild: 0 errors (build-A-r2.log).
