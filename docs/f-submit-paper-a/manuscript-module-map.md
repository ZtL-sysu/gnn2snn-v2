# Manuscript Module Map — Paper A (audit)

- title: "What Reproduces in Spiking Graph Neural Networks? An Audit of Every Code-Releasing Method in a Defined Survey"
- abstract: three failure modes; graded SpikeGCL wording; five-variant conventions; per-dataset transportability (1.35x/1.37x/0.73x the bar).
- keywords: absent (journal field pending).
- introduction: commensurability problem; three findings; executable-audit contribution; five input variants statement.
- related work: bounded seven-method survey context; trained spiking GNNs; explicitly not the conversion failure census.
- method: surveyed-population definition; rule (own script/pinned env); graded disagreement scale; entry-point/patch disclosure; replay protocol.
- experiments/results (sec:exp): audit grid (n + spreads); SpikingGCN dataset-dependence; SpikeGraphormer split flag; SpikeGCL graded reproduction; split-mixing with verbatim quotes; transportability (sec:transport); census (bounded); side-by-side anchor paragraph (companion 80.5 labelled context-only).
- discussion: transportability-not-variance; portable-number requirements; checklist-invisible failures.
- conclusion: bounded census; per-dataset verdict; run-the-code recommendation.
- figures/tables: audit, census, split, protocol tables; seed data in text.
- references: 57 eligible verified (citation-gate-A PASS).

## Narrative map (central claim spine)

- central question: are published spiking-GNN accuracies commensurable, and what makes a number portable?
- candidate core claim: no - three measured failure modes; the binding property is split transportability, stated per dataset at measured magnitudes.
- supporting: convention causality (row-normalisation decisive); replay closure; graded reproduction scale.
- negative/boundary findings: PubMed confounded cell; same-seed 1.0 pp rerun delta; census bounded to the survey.
- visible orphans/contradictions: none open after P10 (noise-source correction applied).
- protections/declarations: `sources/` immutable; declarations author-managed pending.
