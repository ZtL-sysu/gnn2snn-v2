import json, hashlib
from pathlib import Path
root = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade")

def ev(*rels):
    out = []
    for r in rels:
        p = root / r
        out.append({"path": r, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return out

MODEL_B = "GCN-family graph ANN-to-SNN conversion (Paper B)"
MODEL_A = "Spiking-GNN reproduction audit (Paper A)"

def R(id, statement, metric, value, unit, agg, unc, scope, evid, model, strength="main", **kw):
    f = {"id": id, "kind": "result", "statement": statement, "status": "observed_verified",
         "model": model, "condition": "as specified in the named artifact",
         "metric": metric, "value": value, "unit": unit,
         "aggregation": agg, "uncertainty": unc, "measurement_scope": scope,
         "claim_strength": strength, "evidence": evid}
    f.update(kw)
    return f

sj = {
    "abstract": "question, design, main findings, meaning",
    "introduction": "specific unresolved question and the comparison used to answer it",
    "methods": "scientific operations, conditions, measurement scope",
    "results": "principal comparison, effect, uncertainty, adverse findings first-class",
    "discussion": "interpretation of the pattern/trade-off; supported use; hypotheses labelled",
    "conclusion": "answer the opening question with supported main evidence",
    "local-revision": "replace the defective passage; KEEP returns identical bytes",
}

briefB = {
    "question": "When does ANN-to-SNN conversion of a trained graph network preserve its accuracy, and what latency/step decisions can a practitioner make from the trained weights alone?",
    "answer": "Within a configuration, conversion error is an exact identity governed by spectral gain and quantisation mass, giving validated laws and an order-invariant readout; across graphs, weight-only latency decisions do not transfer (three routes fail with measured mechanisms), while a validation-searched per-layer window allocation preserves accuracy with 22.7-93.4% fewer layer-updates in the deep regime (L=4).",
    "audience": "Neuromorphic / graph-learning researchers and practitioners deploying converted spiking GNNs.",
    "domain_bridge": "The graph aggregation operator turns conversion error into an exactly characterisable object (identity), which the paper then uses to explain when latency can and cannot be reduced.",
    "priorities": ["identity and laws", "deep-regime allocation capability with prior-art attribution", "negative map of weight-only decisions", "cost-accounting sensitivity", "scope honesty"],
    "terminology": {"G_l": "downstream gain product of layer l", "layer-updates": "total simulated per-layer timesteps incl. output", "band": "within 1.5 pp of the staged-uniform comparator at T*", "deep regime": "L=4 (3 hidden layers), where uniform T* explodes"},
    "section_jobs": sj,
    "style": "formal journal register; no process narration; limitations as positive evidence boundaries",
    "journal_requirements": "unresolved (author-managed); complete and proportionate; six numbered sections",
    "author_constraints": "declaration_mode author-managed; no efficiency claims in title/abstract; no first-per-layer-allocation wording",
    "study_type": "empirical",
    "paragraph_plan": "results organised by question: identity; within-configuration prediction; cross-graph transfer; computable/deployable decisions",
    "qualification_placement": "regime boundary (L=3) in the capability subsection and discussion; prior-art attribution in related work and the capability subsection",
}

factsB = [
    R("B-identity", "The per-layer conversion error satisfies the exact identity B_hat_l - B_l = (1-alpha) A_hat (E_{l-1} M_l); the staged realisation gives max residual 0.0 (prior-run register ~2e-6).", "max_abs_residual", "0.0", "float", "single check", "float-level exact", "staged engine, this run", ev("results/jB1_prototype.json"), MODEL_B),
    R("B-depth-law", "Error amplifies as G^0.95 (r=0.990) with depth; a four-layer model loses about 21 pp at T=8, reported as a boundary result.", "exponent_and_r", "0.95 / 0.990", "dimensionless", "fitted across L=2..4 cells", "prior-run registered", "GCN family, L in {2,3,4}", ev("inputs/paper-B-baseline.pdf"), MODEL_B),
    R("B-width-law", "Layer-1 quantisation error scales with sqrt(H): width is the lever depth is not.", "norm_ratio_spread", "under 6% spread", "ratio", "7 seeds", "prior-run registered", "widths 16..128", ev("inputs/paper-B-baseline.pdf"), MODEL_B),
    R("B-margin", "The margin criterion is 30/30 sound and 3.2-6.7x conservative (bound/actual median over T in {4,8}).", "counterexamples_and_conservatism", "0 / 3.2-6.7x", "count / ratio", "30 checks", "subset stated", "tested configurations", ev("inputs/paper-B-baseline.pdf"), MODEL_B),
    R("B-order", "Membrane readout is order-invariant (0.0 pp spread) while count readout spreads 0.39-1.61 pp in the same cells.", "spread_pp", "0.0 vs 0.39-1.61", "pp", "2x2 cells", "per-cell", "Cora/CiteSeer, IF/TPP", ev("inputs/registered/wp6a_misalignment.json"), MODEL_B),
    R("B-threshold", "theta=1.0 lies within one seed s.d. of the accuracy plateau on all four graphs including the completed Squirrel cell (plateau 0.75-1.25); the q999 adaptive control loses 22.9 pp on Squirrel.", "acc_vs_theta", "plateau 0.75-1.25; q999 delta -22.9", "pp", "7 seeds", "seed s.d.", "Squirrel held-out cell", ev("results/jB3_squirrel_complete.json"), MODEL_B),
    R("B-search-arm", "Validation-searched per-layer windows (idea family LD-SNN; graph instantiation here) keep the band with 22.7-93.4% fewer total layer-updates than best uniform T* on 6/8 held-out L=4 cells (squirrel 4/4, chameleon 2/4); random-search falsifier passed; criterion inclusive at 20.0%.", "update_saving_fraction", "0.227-0.934", "fraction", "6/8 held-out cells", "per-seed rows", "held-out chameleon/squirrel, L=4, seeds 0-3", ev("results/jB1_search_heldout.json", "results/jB1_search_pilot.json"), MODEL_B, attribution="LD-SNN (ISLPED 2022) idea family; PASCAL 2025 / MT-SNN 2026 prior art"),
    R("B-regime", "The capability is a deep-regime re-scope: the pre-registered mixed L in {3,4} pilot criterion was not met (37.5%); L=3 has no headroom (1/8 inclusive).", "fraction_band_and_20pct", "0.375 mixed; L=3 1/8", "fraction", "16 pilot cells", "inclusive 20.0% criterion", "pilot configs", ev("results/jB1_search_pilot.json"), MODEL_B, claim_strength="boundary"),
    R("B-jb1a", "Weight-only sweep-free latency selection fails to transfer: in-distribution kappa=1.0 exact (shuffled-c falsifier passed); held-out band 1/3, kappa [1.2, 11.0, 2.5]; pre-registered failure signal fired.", "band_hits_and_kappa", "1/3; [1.2, 11.0, 2.5]", "count / ratio", "3 held-out graphs", "one-shot reporting", "held-out chameleon/squirrel/arxiv", ev("results/jB1a_select.json", "audits/jB1a-shuffled-c.json"), MODEL_B, claim_strength="negative-with-mechanism"),
    R("B-closed-form-alloc", "Closed-form weight-derived allocation rules fail their frozen pilots (R1 floor degenerates to uniform-64; R2 certificate clips at the 64 cap); windows >= 4 hold the band at 8-29% saving on 3/8 pilot cells, windows < 4 lose 2-5.6 pp.", "band_and_saving_by_window", "8-29% (>=4); -2.0..-5.6 pp (<4)", "pp / fraction", "4 frozen pilots", "pre-registered gates", "cora/citeseer, L in {3,4}", ev("audits/correction-ledger.md"), MODEL_B, claim_strength="negative-with-mechanism"),
    R("B-predictor", "The gap predictor required-T transfers 2/12 (LOCO MAE 4.6 pp): a third failed cross-graph weight-only route.", "required_t_exact_and_mae", "2/12; 4.6", "count / pp", "12 cells", "prior-run registered", "held-out cells", ev("inputs/paper-B-baseline.pdf"), MODEL_B, claim_strength="negative-with-mechanism"),
    R("B-accounting", "Cost rankings are accounting-dependent: under the pre-declared event-driven SynOps the F-to-C linear-readout arm dominates at T<=4 on Cora and CiteSeer (WP2 failure criterion fires), while static-update accounting has conversion cheapest on 3/3; non-domination by direct-trained arms holds under all conventions.", "verdict_per_convention", "event: 2 dominations; static: 3/3 cheapest", "count", "3 graphs", "conventions pre-declared", "three accountings", ev("results/jB2c_sop_verdict.json", "audits/sop-convention-predeclaration.md"), MODEL_B),
    R("B-truewidth", "The published-architecture arm hidden-width parameter was inert (single F-to-C layer); true-width re-implementations show a working knob (10.4 pp spread) under three cost accountings.", "acc_spread_across_H", "0.104", "pp", "18 cells", "7 seeds", "3 graphs x 3 widths", ev("results/jB2b_truewidth.json", "audits/p3-baseline-findings.md"), MODEL_B, claim_strength="secondary"),
    R("B-biasprov", "Every ablation delta is attributed to the grid-bias default series (printed -3.68 vs computed -3.69); the default row is not bit-identical to the closed-form beta* row (max gap 0.56 pp).", "delta_attribution", "6/6 printed deltas matched", "count", "6 checks", "rounding noted", "ablation table", ev("results/jB5_bias_provenance.json"), MODEL_B, claim_strength="integrity"),
    R("B-pareto-trained", "Conversion is never dominated by the direct-trained spiking arms at matched accuracy across the tested grid.", "dominated_graphs", "0", "count", "3 graphs", "registered + recheck", "four-arm grid", ev("results/jB2c_sop_verdict.json"), MODEL_B),
    R("B-spikecost", "At matched accuracy conversion emits 17-83% more spikes than trained arms in some cells; no energy claims (static proxies and event counts only).", "spike_delta", "+17% to +83%", "fraction", "matched cells", "registered", "iso-accuracy pairs", ev("inputs/paper-B-baseline.pdf"), MODEL_B, claim_strength="boundary"),
]

briefA = {
    "question": "Are published spiking-GNN accuracies commensurable across papers, and what makes a reported number portable?",
    "answer": "No: an executable audit of every code-releasing method in a bounded seven-method survey finds dataset-dependent reproduction failure, split-flag sensitivity worth 10.7 pp, and unlabelled split mixing in comparison tables; the binding property is split transportability, which dominates seed noise on Cora (61.5x s.d.), is marginal on CiteSeer (4.1x), and is confounded on PubMed (2.2x).",
    "audience": "Benchmark/evaluation researchers and practitioners assembling spiking-GNN comparison tables.",
    "domain_bridge": "A number measured under one split is not a baseline for another; the audit makes the split label and the reproduction environment part of the claim.",
    "priorities": ["executability of the audit protocol", "per-dataset transportability honesty", "unlabelled split mixing as a field-level hazard", "convention causality", "replay closure"],
    "terminology": {"Split I": "20 labelled instances per class (verbatim)", "Split II": "8:2 train:test, 20% of training for validation (verbatim)", "transportability": "split gap vs measured seed spread", "3sigma bar": "pre-registered criterion gap >= 3x seed s.d."},
    "section_jobs": sj,
    "style": "formal journal register; census bounded to the survey; verbatim quotes for split definitions",
    "journal_requirements": "unresolved (author-managed); complete and proportionate; six numbered sections",
    "author_constraints": "declaration_mode author-managed; no rankings of audited methods; bounded census wording",
    "study_type": "empirical",
    "paragraph_plan": "results by failure mode: code-vs-paper; split flag; unlabelled mixing; input conventions; transportability verdict per dataset",
    "qualification_placement": "census bound in methods; per-dataset scoping in transportability subsection and conclusion",
}

factsA = [
    R("A-grid", "Three of seven surveyed methods ship code and form the auditable census; SpikingGCN reproduces on CiteSeer (70.6 vs 72.5) but fails dataset-dependently elsewhere (Cora 50.0 vs 80.7, 5 seeds, per-seed 0.481-0.526; PubMed 64.3 vs 77.6, 3 seeds, s.d. 5.37 pp).", "acc_repro_vs_published", "see statement", "pp", "3-5 seeds", "seed s.d.", "three datasets", ev("inputs/registered/wp10_audit.json", "logs/jA1_spikegcl_multiseed.log"), MODEL_A),
    R("A-splitflag", "SpikeGraphormer reproduces on its own split (84.7 vs 84.8 plus/minus 0.7, 3 runs) and drops to 74.0 on the public split: one flag is worth 10.7 pp.", "acc_by_split", "84.7 / 74.0", "pp", "3 runs / 1 run", "published s.d.", "their split vs public", ev("inputs/registered/wp10_audit.json"), MODEL_A),
    R("A-mixing", "Comparison tables quote the same baseline on two splits without labels: one paper GCN row 81.5 is consistent with the Split-I family; another SpikingGCN row 87.7/77.7/87.5 is consistent with Split II (88.7/77.6/89.3); split definitions quoted verbatim (arXiv:2205.02767).", "cited_vs_split_values", "81.5 / 87.7 rows", "pp", "table rows", "verbatim quotes", "literature tables", ev("results/jA_splitquotes.md", "inputs/registered/wp10_audit.json"), MODEL_A),
    R("A-conventions", "Input conventions decide the SpikingGCN Cora cell: raw bag-of-words 51.5% with or without self-loops; every row-normalised variant collapses to 13.0% including the conventional max-rescale fix; the old fourth-row 13.0 was the row-normalised no-self-loop variant (mislabel corrected).", "test_acc_5_variants", "0.515/0.130/0.130/0.515/0.130", "fraction", "5 variants", "seed 0", "Cora, audited pipeline", ev("results/jA_conventions.json"), MODEL_A),
    R("A-transport", "Split transportability vs seed noise is per-dataset: Cora 8.0 pp gap = 20.5x the 3sigma bar = 61.5x seed s.d. (strong); CiteSeer 5.1 pp = 1.37x bar = 4.1x s.d. (marginal); PubMed 11.7 pp = 0.73x bar = 2.2x s.d. (confounded, re-scoped per pre-registration). SpikeGCL seed spreads: Cora 0.13 pp, CiteSeer 1.24 pp (4 seeds each).", "gap_vs_seed_spread", "see statement", "pp ratio", "4 seeds", "per-dataset verdict", "three datasets", ev("audits/jA1-outcome.md", "logs/jA1_spikegcl_multiseed.log"), MODEL_A),
    R("A-replay", "Every audit-table number recomputes from raw logs: replay/checksum audit 0 mismatches.", "mismatches", "0", "count", "full table", "exact", "all audited cells", ev("audits/jA_replay-checksum.json"), MODEL_A, claim_strength="integrity"),
    R("A-seeds", "A same-nominal-seed CiteSeer re-run differed by 1.0 pp from the audited cell (0.7826 vs 0.7728): recorded sensitivity evidence.", "rerun_delta", "1.0", "pp", "1 comparison", "single pair", "CiteSeer", ev("logs/jA1_spikegcl_multiseed.log"), MODEL_A, claim_strength="boundary"),
    R("A-scope", "The census covers the seven methods of this survey (three with code); no field-wide claim; this audit covers trained spiking GNNs, not training-free conversion methods.", "census_bound", "3 of 7", "count", "survey", "bounded wording", "survey-defined population", ev("audits/jA2-outcome.md"), MODEL_A, claim_strength="boundary"),
    R("A-finding", "Each individual number is internally consistent: a variance checklist would catch none of these failures; the binding property is transportability, not variance.", "failure_mode_taxonomy", "3 modes", "count", "audit", "qualitative", "all cases", ev("audits/jA_replay-checksum.json", "audits/jA1-outcome.md"), MODEL_A),
]

for name, brief, facts in [("paperA-packet.json", briefA, factsA), ("paperB-packet.json", briefB, factsB)]:
    d = {"schema_version": 1, "brief": brief, "facts": facts}
    (root / "writing" / name).write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("built", name, len(facts), "facts")
