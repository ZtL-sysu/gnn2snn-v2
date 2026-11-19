# P6 Outlines — both manuscripts (canonical six sections; local-executor substitution for ARC stage 16)

## Paper B — "When Does Graph-to-Spiking Conversion Work? Spectral Gain, Activation Mass, and an Order-Invariant Readout"

1. **Introduction.** The practitioner's blind choice (depth/width/T chosen without signal); the
   question stated; prior-art boundary (per-layer allocation is established — PASCAL/MT-SNN/LD-SNN;
   graph *methods* exist — One-Spike/PMSM/Lan; what is missing is analysis of graph conversion
   error); four contribution bullets (identity+laws; order-invariant readout; deep-regime allocation
   capability with attribution; negative map of weight-only decisions + accounting sensitivity).
   Purpose: state the question and the comparison that answers it. Evidence: B-identity..B-search-arm.
2. **Related Work.** Trained spiking GNNs; ANN→SNN conversion theory (FTBC/TPP inheritance and
   boundary); per-layer/mixed-timestep allocation prior art (PASCAL, MT-SNN, LD-SNN with their
   derivation sources contrasted); graph-conversion precedents; misalignment vs quantisation
   vocabulary. Purpose: place the contribution against the correct nearest work. Evidence:
   B-search-arm attribution; novelty map.
3. **Method.** L-layer recursion and the exact identity (+ proof sketch); layer-1 quantisation law;
   margin criterion and what it does/does not certify (41–68% node coverage stated); membrane
   readout and order invariance; depth/width corollaries; staged per-layer simulation semantics;
   the allocation procedures (validation-search = the working one; closed-form attempts and their
   failure modes in brief, detailed in Experiments). Purpose: give the analysis object and both
   constructive procedures. Evidence: B-identity, B-margin, B-order.
4. **Experiments and Results.** Protocol and floors; laws in-sample (depth/width/threshold incl.
   Squirrel completion); the deep-regime capability (search arm: 6/8 held-out L=4 at 22.7–93.4%
   saving, random falsifier, regime boundary L=3 + failed mixed criterion); the negative map
   (closed-form R1/R2, JB1a transfer, predictor) with mechanisms; accounting-sensitivity
   (three conventions, WP2 criterion outcome); external-arm repair + Pareto non-domination; bias
   provenance footnote correction. Purpose: answer each question with the measured comparison.
   Evidence: B-search-arm, B-regime, B-jb1a, B-closed-form-alloc, B-predictor, B-accounting,
   B-truewidth, B-biasprov, B-pareto-trained, B-threshold.
5. **Discussion.** What the laws license (in-configuration design) and what they do not (cross-graph
   latency decisions from weights alone — three routes measured); why the deep regime is where
   allocation headroom lives (uniform T* explodes per the depth law); accounting as a methodological
   hazard in SNN-efficiency comparisons; deployment guidance (validation search is the working
   procedure; its cost); hypotheses labelled (why margin scales fail to transfer).
   Purpose: interpret the pattern and its supported use. Evidence: B-regime, B-jb1a, B-accounting.
6. **Conclusion.** The identity and laws answer "when does conversion work" within a configuration;
   the allocation procedure answers "what can be computed" in the deep regime; the negative map
   marks the open problem (transferable certificates). No efficiency claim. Purpose: answer the
   opening question. Evidence: B-identity, B-search-arm, B-jb1a.

## Paper A — "What Reproduces in Spiking Graph Neural Networks? An Audit of Every Method That Publishes Code"

1. **Introduction.** Quoted accuracies assumed commensurable; the audit question; three failure
   modes previewed (dataset-dependent reproduction; split-flag sensitivity; unlabelled split
   mixing); contribution = executable audit protocol + bounded census + transportability verdict
   per dataset. Purpose: question + comparison. Evidence: A-grid, A-splitflag, A-mixing.
2. **Related Work.** Trained spiking GNNs (the seven-method survey, bounded); reproducibility
   practice; split conventions in the audited papers (verbatim anchors); explicitly NOT the
   training-free conversion failure census. Purpose: define the surveyed population.
   Evidence: A-scope.
3. **Method.** Audit protocol (own script, pinned environment, own split; reproduction vs stress
   test distinguished); census boundary (3 of 7 code-releasing; search-window wording); replay
   protocol; seed policy. Purpose: make the audit executable by a reader. Evidence: A-replay, A-scope.
4. **Experiments and Results.** The audit grid (with n and spreads); SpikingGCN dataset-dependence;
   SpikeGraphormer split flag (10.7 pp); SpikeGCL reproduction and its seed spreads; the split-mixing
   analysis with verbatim quotes; the five-variant convention diagnostic (corrected causality);
   transportability verdict per dataset (20.5×/1.37×/0.73× the 3σ bar; PubMed confounded);
   replay closure; same-seed sensitivity note. Purpose: each failure mode with its measurement.
   Evidence: A-grid, A-splitflag, A-mixing, A-conventions, A-transport, A-seeds, A-replay.
5. **Discussion.** Transportability, not variance, is the binding property; what a portable number
   requires (split label + environment + seeds); why individual numbers stay internally consistent
   (checklist-invisible failures); scope boundaries. Purpose: interpret. Evidence: A-finding, A-transport.
6. **Conclusion.** Within the surveyed population, no number is portable without its split; three
   concrete failure modes measured; the protocol is executable and fully replayed. Purpose: answer.
   Evidence: A-transport, A-replay.

## Evidence maps

Both outlines trace to `writing/paper*-packet.json` facts only; administrative records, review
dialogue and ledger text stay outside drafting context per the writing contract.
