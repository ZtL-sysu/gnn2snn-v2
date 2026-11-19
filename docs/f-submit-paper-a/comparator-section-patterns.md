# Comparator Section-Pattern Digest — Paper A ("What Reproduces in Spiking Graph Neural Networks? An Audit of Every Code-Releasing Method in a Defined Survey")

- **Purpose:** Extract transferable organization and evidence-presentation patterns from the 10 frozen comparators (`comparator-literature-table.md`, frozen 2026-11-19), by manuscript module, to guide the audit manuscript. Comparator conclusions are NOT requirements for this manuscript; their data, methods, and scale are study-specific.
- **Inspection method:** abstracts and section organizations inspected via web retrieval against publisher pages (nature.com, cell.com/ScienceDirect, proceedings.mlr.press, PMC/Europe PMC), arXiv abs+HTML pages, bioRxiv, ChemRxiv, OpenAIRE, and institutional repositories. (WebFetch was network-blocked in this run; pages were retrieved by direct HTTP download and parsed — same sources.) Access limits per paper are recorded in "Per-comparator inspection record" below and stated honestly: several published bodies were paywalled or bot-walled and were mapped from open preprint/deposit versions of the same study; one (So et al.) is mapped from its own bioRxiv preprint, not the published body.
- **Distinction used throughout:** *transferable* = argument order, evidence-presentation, and scoping moves that carry to a reproducibility/evaluation audit of trained spiking GNNs; *study-specific* = their datasets, methods, domain motivations, numerical scales.

---

## 1. Title module

Transferable patterns:

- **Genre tag or question framing announces audit intent before the abstract.** "Reusability report:" prefix (So et al.; Heirman & Bittremieux — NMI genre tag), "Reproducibility Study Of ..." (Nijdam et al.), pitfall+remedy framing "Current Pitfalls and New Benchmarking" (Li et al.), concept-noun "The cost of reproducibility in artificial intelligence" (Snelleman et al.), problem-question framing (Kapoor & Narayanan). Paper A's question title + "An Audit of Every Code-Releasing Method in a Defined Survey" is the same two-move pattern: question + population bound.
- **Scope carried in the title itself** ("in machine-learning-based science", "on Large-scale Graph") — a scoping move, not decoration.

Study-specific (do not copy): domain nouns (metabolite MS, oncology, link prediction), benchmark name-coinings, their method names.

## 2. Abstract module

Transferable patterns:

- **Corpus accounting before any verdict.** Kapoor & Narayanan: "17 fields ... 294 papers". Snelleman et al.: "918 publications between 2022 and 2024 at top AI venues". SGNNBench: "9 state-of-the-art SGNNs across 18 datasets". Bosma et al.: three diagnostic + two segmentation tasks. Paper A: bounded seven-method survey / every code-releasing method — count first.
- **Graded, per-claim verdicts instead of one pass/fail.** Nijdam et al.: "partially reproduce one of the original three claims and fully substantiate the other two" — a claim ledger in two sentences. So et al.: graded reproducibility verdict ("while not reaching the same level of superiority ... able to confirm the superiority") plus a distinct reusability verdict. Heirman & Bittremieux: "encountering minor challenges but largely succeeding". Matches Paper A's graded SpikeGCL wording and per-dataset verdicts.
- **Conditional headline findings stated with their condition.** Lell & Scherp: gains hold "if the train-test split is randomized" — the moderator appears in the abstract. Paper A's split-mixing and transportability magnitudes (1.35x/1.37x/0.73x the bar) belong at this explicitness level.
- **Negative findings and failure modes get abstract space.** Snelleman et al.: "6.72% of implementation repository links were broken", award papers not less costly. Bosma et al.: false significance "in 15% of comparisons". Kapoor & Narayanan: corrected re-evaluation collapses the claim ("do not perform substantively better than decades-old LR models").
- **Instrument/artifact stated at abstract end.** Kapoor & Narayanan: "model info sheets, which we introduce". Li et al.: "Our implementation and data are available". Nijdam et al.: GitHub link. So et al.: "We developed new resources".
- **Structured Background/Objectives/Methods/Results/Conclusions variant exists** (Snelleman et al., structured abstract preserved from its ICLR version into the Patterns article) — a legitimate option for a structured journal abstract; unstructured comparators still follow the same internal order: gap → population → instrument → verdict → recommendation.

Study-specific: their numbers, domains, and their single illustrative case study (civil war prediction; CASMI 2022; PDTC cohorts).

## 3. Introduction module

Transferable patterns:

- **Gap stated as a missing-instrument claim, not a missing-method claim.** SGNNBench: "no systematic benchmark ... To bridge this gap, we present SGNNBench". NeuroBench: field "lacks standardized benchmarks". Kapoor & Narayanan: "we systematically investigate reproducibility issues". Paper A analogue: no systematic audit of trained spiking GNNs; commensurability problem.
- **Enumerated findings preview.** Li et al. preview three pitfalls as (1)(2)(3) already in the abstract/introduction; Kapoor & Narayanan preview survey → taxonomy → instrument → case study as an explicit roadmap. Paper A's "three findings" + "five input variants" statements follow this move.
- **Defensive scoping paragraph available early.** Kapoor & Narayanan carry a subsection literally titled "Why do we call it a reproducibility crisis?" — pre-empting definitional objections inside the results flow. The transportable version for Paper A is a short definitional defense of "reproducibility audit" vs "replication" placed early (their subsection is late in Results; the move can go earlier).
- **Contributions listed as audit deliverables** (taxonomy, instrument, case-study correction in Kapoor & Narayanan; framework tracks in NeuroBench; benchmark + design-space study in SGNNBench), not as method novelties.

Study-specific: their domain-importance paragraphs (precision oncology, untargeted MS, energy-efficient GNNs).

## 4. Related work module

Transferable patterns:

- **Three viable placements; all keep the section short and scope-serving.** (a) Front-loaded section with 2-3 thematic buckets (Lell & Scherp §2: flat-minima methods / GNNs; Nijdam et al. fold prior work into "Scope of reproducibility" + Methodology). (b) Appendix demotion for a benchmark/audit paper whose contribution is the instrument: SGNNBench puts "Related Work" in Appendix G (Energy-saving GNNs / SNNs / SGNNs) so main text opens with the benchmark. (c) Inline/late: Kapoor & Narayanan carry "Prior work on model cards and reporting standards" inside the instrument section and broader positioning in Discussion ("Beyond leakage: Perspectives"); Snelleman et al. place "Related work" after Methods (Patterns house style). NeuroBench has no separate related work — the Introduction carries it.
- **Scope-boundary move: say what this is NOT.** Nijdam et al.'s "Scope of reproducibility" section explicitly lists which claims are in and out. Paper A's "explicitly not the conversion failure census" is the same move and can sit in related work or scope.
- **Preliminaries-as-related-work option** (Li et al. §2: Task Formulation / Link Prediction Methods / Datasets and Experimental Settings) — useful when the audited population needs defining before the audit (Paper A: trained spiking GNNs, seven-method survey context).

Study-specific: their citation neighborhoods and bucket taxonomies.

## 5. Method module (audit protocol / "Protocol" section)

Transferable patterns:

- **Sampling/accounting subsection with inclusion-exclusion arithmetic.** Snelleman et al. Methods: "Paper sampling" — 1,061 screened → 143 theoretical excluded → 918 analysed. Kapoor & Narayanan: "Systematic search of predictive modeling literature in civil war research" inside the case study. Paper A's surveyed-population definition (every code-releasing method in a defined survey) is this move.
- **Graded rubric defined before results.** Snelleman et al.: five cost dimensions, each a numbered subsection in Methods (Implementation / Data / Configuration / Experimental / Expertise cost), each with objective characteristics scored 1-10, plus "Assessing the cost" and an inter-rater sub-study ("Second review", ICC reported; expertise dimension dropped after low agreement — a rubric-reliability disclosure). Paper A's graded disagreement scale and entry-point/patch disclosure map directly.
- **Instrument is a first-class artifact.** Kapoor & Narayanan devote a full results block to "Model info sheets for detecting and preventing leakage" (prior work → scientific arguments → theory of change → limitations of the instrument itself). Paper A's replay protocol deserves the same: definition, why-it-surfaces-failures, and its own limitations subsection.
- **Restate-and-pin the original protocol before deviations.** Heirman & Bittremieux: "following the tool's original training and testing protocols" then list encountered deviations. So et al.: reconstruct the original four stages (feature extraction → training → fine-tuning → validation) and audit stage by stage. Nijdam et al. §3: restate Graphair + datasets + setup + hyperparameters + computational requirements.
- **Statistical apparatus specified in Methods, not left implicit.** Bosma et al.: "Statistical Analysis" subsection (bootstrap/permutation-test procedure spelled out step-wise in an appendix); Li et al.: unified "same dataset and hyperparameter search settings" fair-comparison rule stated before any table.
- **Computational resources subsection is standard in reproduction genre** (Nijdam et al. §3.6 "Computational requirements", with GPU-hours appendix; Bosma et al. §2.4 "Computational Resources"; SGNNBench Appendix A.2).

Study-specific: their rubric dimensions (documentation cost, medical-imaging tasks), their pipelines.

## 6. Experiments / Results module

### 6a. Audit grid (n + spreads)

- **Organize by research question, land each as a numbered observation.** SGNNBench: RQ1 effectiveness → RQ2 efficiency → RQ3 ablation, each closing with "Obs. 1..5" observation boxes — a digestible audit-grid spine. Li et al.: pitfall-by-pitfall results blocks.
- **One table per evidence class, same row population.** SGNNBench: accuracy split into homophily / heterophily / dynamic tables (T2-T5) with identical method rows; efficiency table T6; ablation T7. Consistent row axes let readers scan one column down a failure mode.
- **Reported-vs-reproduced delta table is the genre's core table.** Nijdam et al. Table 2 ("Comparison of the results reported by the original authors with those obtained by us"); Li et al. Table 3 ("Comparison of ours and the reported results for GCN and GAE"); their Appendix D "Reported vs. Our Results on ogbl-ddi". Paper A's side-by-side anchor paragraph and audit grid use this pattern.
- **Mark re-implemented numbers explicitly** — SGNNBench: "∗ denotes our implementation" in table captions. Ties into Paper A's entry-point/patch disclosure.
- **Spreads reported as their own evidence object.** Li et al. Table 6 ("Mean model standard deviation" under both settings); Lell & Scherp relegate SD tables (12-13) to appendix but reference them from every mean table. Bosma et al. make run-to-run spread the entire subject. Paper A's "n + spreads" grid columns follow the Li et al. convention.

### 6b. Split-mixing (split as experimental factor)

- **Table the split registry itself.** Lell & Scherp Table 2 ("Dataset splits": fixed "Planetoid" vs "ra-pl" random vs "622" 60/20/20) — the split variants are documented with naming legend before any result.
- **Show conclusion reversals across split regimes, then condition the headline.** Lell & Scherp Tables 3-4 report mean accuracy per split AND dataset; §6.3 "Influence of Dataset Splits" narrates the reversal (2-point gains only under randomized splits; fixed Planetoid biased); abstract and conclusion carry the conditional. This is the direct template for Paper A's split-mixing evidence.
- **Name the inherited split bias and cite its source** ("Following Shchur et al. ... fixed splits like 'Planetoid' are biased") — attribution of the split critique, then your own measured consequence.
- **Li et al. pitfall (2)** ("a lack of a unified data split and evaluation metric") is the problem-statement version of the same axis.

### 6c. Conventions (protocol choices as causal factors)

- **Present convention variants as controlled comparisons and report rank inversions.** Li et al.: three pitfalls (weak baselines / inconsistent splits+metrics / unrealistic negative sampling) each shown to invert method rankings; results tables re-run under both "existing setting" and new setting so the convention effect is visible as table-vs-table delta.
- **Design-space ablation of conventions.** SGNNBench §2 "Design Space" (coding scheme, membrane-potential rectifier, etc.) then RQ3 ablation; Heirman & Bittremieux ablation finding "some algorithmic complexities may not significantly enhance the performance". Convention choices are experiment factors, not footnotes — the template for Paper A's row-normalisation-decisive claim.
- **Metric-definition audit as first-class** (Li et al. Appendix B "Evaluation Metrics" and the "unrealistic negative sampling" pitfall; NeuroBench's whole two-track metric taxonomy is the standards-side version).

### 6d. Transportability (beyond the original evaluation)

- **Two-block results split: reproducing the original vs beyond the original.** Nijdam et al. §4.1 "Results reproducing original paper" / §4.2 "Results beyond original paper" — the cleanest structural template for Paper A's replay-vs-transportability split.
- **External-dataset evaluation section with its own verdict.** Heirman & Bittremieux "Reusability": apply to CASMI 2022 unseen data, separate findings; So et al. "Reusability": new training sets (CTRP, gCSI, GDSC2) then escalating contexts (cell lines → PDX → two clinical trials), per-context verdicts.
- **Transport friction is itself a finding.** So et al.: hyperparameters "can not seamlessly be used for another" dataset — "Re-tuning ... must occur anytime a new dataset ... is introduced"; drug-recovery attrition (32/50). Paper A's per-dataset transportability magnitudes (1.35x/1.37x/0.73x) are the same genre of measured friction.
- **Readout-alignment move when contexts differ** (So et al. binarize heterogeneous drug-response readouts to R/NR before cross-context comparison) — the abstract pattern is: define the commensuration transformation before comparing across contexts. Paper A's five input variants / convention alignment is the analogous transformation.

### 6e. Census / bounded population results

- **Census tables carry eligibility accounting, not just counts.** Snelleman et al. Table 1 (reviewed vs excluded counts per venue, exclusions in brackets); Kapoor & Narayanan's survey (17 fields, 294 papers) with the taxonomy table as the census's coding frame. Broken-link rate (6.72%) shows a census can include artifact-health columns — directly parallel to Paper A's bounded census of code-releasing methods.

## 7. Discussion module

Transferable patterns:

- **Genre-standard meta-subsections in reproduction studies.** Nijdam et al. §5.1 "What was easy and what was difficult" + §5.2 "Communication with original authors" — transparent process accounting; the TMLR reproducibility-study house pattern. Portable as "what replayed easily vs what needed intervention" plus any author contact.
- **Limitations separated from verdicts.** Lell & Scherp §6.6 "Assumptions and Limitations"; Snelleman et al. standalone "Limitations of the study" section before Conclusion; Kapoor & Narayanan "Limitations of model info sheets" nested in the instrument section. Paper A's bounded-census caveat and confounded-cell notes fit here.
- **Key insights numbered/subsectioned** (Lell & Scherp §6.1 "Key Insights", then one subsection per moderator: splits §6.3, transductive-vs-inductive §6.4) — the discussion walks each moderator found in results, one at a time. Matches Paper A's transportability-not-variance and checklist-invisible-failures moves.
- **Implications framed as requirements on future numbers.** So et al.: containerized execution, DOI-versioned datasets (ORCESTRA), capsule-archived workflows. Snelleman et al.: checklists, appendices, low-cost-as-award-criterion. Heirman & Bittremieux: "community-wide efforts in benchmarking, transparency and data sharing". Paper A's "portable-number requirements" and "run-the-code recommendation" are this move.
- **Infrastructure-fix-to-failure-mode pairing.** Every observed failure gets an attached remedy in Discussion (So et al.: code-cloning pitfalls → Code Ocean capsule; Kapoor & Narayanan: leakage types → model info sheets; Li et al.: pitfalls → HeaRT protocol).

Study-specific: their domain implications (precision oncology deployment, metabolomics community, GNN training practice).

## 8. Conclusion module

Transferable patterns:

- **Short, bounded restatement + per-unit verdict + one actionable recommendation; no new numbers.** Bosma et al. (2 sentences: variation substantial, must be accounted for); Lell & Scherp (bounded claim + "always use weight averaging" recommendation); Li et al. (pitfalls recap + new setting); SGNNBench (one paragraph). Kapoor & Narayanan: "Conclusions" separate from "Discussion". Paper A: bounded census; per-dataset verdict; run-the-code recommendation — same three moves.
- **The population bound is restated in the conclusion** ("in a defined survey" / "our survey of 918 publications") so the closing claim cannot be over-read as universal.

## 9. Tables module

Transferable patterns (table inventory by evidence class):

| Evidence class | Comparator exemplar | Pattern |
|---|---|---|
| Reported-vs-reproduced deltas | Nijdam T2; Li T3 (+ Appendix D) | Original column next to reproduced column; deltas narrated in text |
| Split registry | Lell T2 | One row per split variant with naming legend; result tables keyed to it |
| Audit grid (n + spreads) | Li T1-T2, T4-T5, T6; SGNNBench T2-T5 | Stable method rows across tables; rank highlighting (top-3); spread table separate |
| Convention/ablation | SGNNBench T7; Heirman (ablation figures) | Convention variant as table factor |
| Reimplementation marking | SGNNBench T2-T5 ("∗ denotes our implementation") | Footnote/asterisk convention in caption |
| Census + eligibility accounting | Snelleman T1 | Counts with exclusions in brackets per venue/stratum |
| Rubric definition / guidelines | Snelleman T4 ("data collection guidelines") | The instrument itself as a table |
| Reliability | Snelleman T2 (ICC with 95% CI) | Inter-rater statistics table when human judgement codes artifacts |
| External-validation correlation | Snelleman T3 (Kendall vs reproduction success) | Instrument validity evidence |
| Hyperparameters as evidence | Nijdam T7-T9; Heirman T1-T4; SGNNBench T8 | Hyperparameter tables double as reproducibility proof; Heirman's caption convention "Values not specified in the supplementary material are ..." marks gaps explicitly |
| Dataset statistics | all | Row per dataset: n, features, splits, steps (SGNNBench T1 with T time-steps) |
| Efficiency | SGNNBench T6 (model size / memory / energy) | Orthogonal axis kept in its own table |

Transferable rule: one table per evidence class, shared row axis, captions carry the conventions (ranking marks, implementation marks, gap marks). Study-specific: their exact metrics and datasets.

## 10. References / availability apparatus module

Transferable patterns:

- **Named availability blocks in fixed order:** "Data availability" + "Code availability" (Nature family: So et al., Heirman & Bittremieux, NeuroBench); "Resource availability" umbrella with Lead contact / Materials availability / Data and code availability (Cell Press: Kapoor & Narayanan, Snelleman et al.); inline link in abstract (Nijdam, Li et al.).
- **Artifact identity is citable, not just linked:** Code Ocean capsule DOI (So et al.: 10.0260/CO.7288809.v2), Zenodo dataset DOI (Snelleman et al.: 10.5281/zenodo.21032062), GitHub + license (Li et al.: HeaRT MIT; Nijdam: graphair-reproducibility). Paper A's replay protocol should pin artifact identities the same way.
- **Provenance/license appendix:** Li et al. Appendix J "Dataset Licenses" — data legality as an appendix item.
- **Process declarations:** Snelleman et al. carry "Declaration of generative AI and AI-assigned technologies" and "Declaration of interests" (Patterns house style); peer-review metadata blocks in Nature family.
- **Checklist appendix genre:** Li et al. §7 "Checklist" (NeurIPS D&B) — the venue checklist as a visible section; Snelleman et al. recommend checklist requirements from their findings. For Paper A the analogous move is the checklist-invisibility argument anchored to these checklist instruments.

---

## Per-comparator inspection record (access honesty)

| Comparator | What was inspected | Access route | Limits |
|---|---|---|---|
| Kapoor & Narayanan 2023 (Patterns) | Published abstract + full section tree | Europe PMC full-text XML (PMC10499856); arXiv preprint 2207.07048 available as fallback | cell.com/ScienceDirect bot-walled (Cloudflare); PMC XML gives published section structure faithfully |
| Snelleman et al. 2026 (Patterns) | Published abstract + full section tree + table captions | Europe PMC full-text XML (PMC13674308) | cell.com/ScienceDirect JS-shell/Cloudflare; OpenReview version (forum vPN4ay2wyu, ICLR 2026 submission) challenge-walled; structured-abstract wording cross-checked via Lacuna mirror of the OpenReview record |
| So et al. 2023 (NMI Reusability Report) | Abstract (published) + body structure | nature.com abstract page; body via bioRxiv preprint of the same study (2023.07.06.547938) | NMI published body paywalled; published-version headings not directly verified — section names above (Reproducibility / Reusability / Discussion / Data and Code Availability) are the preprint's, of the same work |
| Heirman & Bittremieux 2024 (NMI Reusability Report) | Abstract (published) + full body incl. tables 1-4 | UAntwerp repository deposit (docstore d:irua:26014, accepted manuscript, 17 pp.) via OpenAIRE | NMI published body paywalled; ChemRxiv PDF Cloudflare-walled; accepted-manuscript section names may differ slightly from typeset version |
| Bosma et al. 2023 (MIDL/PMLR 227) | Abstract + full section tree | proceedings.mlr.press bosma24a page + PDF | none material (open access) |
| Nijdam et al. 2024 (TMLR) | Abstract + full section tree + captions | arXiv 2409.00421 abs + HTML | arXiv version; TMLR-accepted version may differ in minor headings |
| Lell & Scherp 2023 (CD-MAKE) | Abstract + full section tree + captions | arXiv 2306.09121 abs + HTML | arXiv version; Springer LNCS version may differ in minor headings |
| Li et al. 2023 (NeurIPS D&B) | Abstract + full section tree + captions | arXiv 2306.10453 abs + HTML | arXiv version; proceedings version may differ in minor headings |
| SGNNBench 2025 | Abstract + full section tree + captions | arXiv 2509.21342 abs + HTML | preprint; no peer-reviewed venue confirmed (per frozen table) |
| NeuroBench 2025 (Nature Communications) | Abstract + full section tree | nature.com open-access full text | none material (open access) |

## Usage note for the manuscript

- All comparator findings, numbers, and recommendation emphases above are cited as *patterns of organization and evidence presentation only*. Nothing here requires Paper A to adopt any comparator's conclusions, coverage, or scale.
- The strongest module-level templates for Paper A: Nijdam et al.'s scope/results-split/discussion structure; Lell & Scherp's split registry + conditioned conclusions; Li et al.'s reported-vs-reproduced delta tables + convention-inversion evidence; Snelleman et al.'s sampling accounting + rubric Methods + reliability sub-study; So et al. / Heirman & Bittremieux's reproduce-then-transport two-block results; SGNNBench's RQ-grid with observation boxes and implementation marking; Kapoor & Narayanan's corpus-scoped census + instrument-with-own-limitations.
