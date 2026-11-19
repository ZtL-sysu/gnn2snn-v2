# Manuscript Module Map — Paper B (conversion)

- title: "When Does Graph-to-Spiking Conversion Work? Spectral Gain, Activation Mass, and an Order-Invariant Readout"
- abstract: scoped story (identity -> laws -> deep-regime capability -> transfer negatives -> accounting sensitivity); anchors 74.0/86.6 labelled external.
- keywords: absent (journal field pending).
- introduction: blind-choice problem; analysis-level gap with prior-art attribution (PASCAL/MT-SNN/LD-SNN; One-Spike/PMSM graph precedents); five contribution bullets incl. deep-regime capability + negative map.
- related work: trained spiking GNNs; conversion theory; per-layer timesteps prior art with derivation-source contrasts; graph precedents; misalignment vs quantisation.
- method: L-layer identity + proof; layer-1 quantisation law; margin criterion (41-68% node coverage stated); membrane readout invariance; staged per-layer semantics; allocation procedures.
- experiments/results (sec:exp): protocol; ensemble; closed-form; ranking; pareto (accounting-conditional); audit anchors; fidelity; baselines; ablation; iso-accuracy; generality; theory; scale; width; threshold (5 graphs); depth; order; NEW accounting sensitivity (sec:accounting); deep-regime allocation (sec:allocation, fig8); transfer negatives (sec:transfer, fig9).
- discussion: laws licensed in-configuration; transfer open problem; accounting hazard; deployment guidance (validation search).
- conclusion: bounded answer (identity+laws; deep-regime procedure; open transfer); no efficiency claim.
- figures: fig1 framework; fig2-7 data (accepted base); fig8 allocation regime; fig9 transfer.
- tables: tab1-14, tab16-19 (tab15 orphan removed at P10); tab3 footnote corrected.
- references: 61 eligible verified (citation-gate-B PASS).

## Narrative map (central claim spine)

- central question: when does conversion of a trained GNN preserve accuracy, and what latency decisions can weights alone support?
- candidate core claim: identity + laws characterise conversion error exactly within a configuration; the deep-regime validation-search allocation is the working constructive capability (prior-art attributed).
- supporting/diagnostic: threshold law; order invariance; accounting sensitivity; truewidth repair; bias provenance.
- negative findings (first-class): sweep-free selection transfer failure (kappa [1.2, 11.0, 2.5]); closed-form allocation regime map; gap predictor level failure.
- visible orphans/contradictions: none open after P10 (P9 majors closed; numbers packet-mapped).
- protections/declarations: `sources/` immutable; declarations author-managed pending.
