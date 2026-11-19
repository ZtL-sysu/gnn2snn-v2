# Paper B — Comparator Section-Pattern Digest

**Manuscript:** "When Does Graph-to-Spiking Conversion Work? Spectral Gain, Activation Mass, and an Order-Invariant Readout" (Paper B, ANN-to-SNN conversion analysis for graphs)
**Frozen comparator set:** `comparator-literature-table.md` (10 verified papers, frozen 2026-11-19)
**Digest date:** 2026-11-19
**Purpose:** extract transferable *organization* and *evidence-presentation* patterns by manuscript module. Comparator conclusions are NOT requirements for this manuscript (see "Do-not-import" at the end). Study-specific choices (their data, models, scale) are marked as such.

---

## 0. Inspection method and access limits (honest record)

- **WebFetch** was attempted on `www.mdpi.com` and `arxiv.org` and was **blocked** (domain-verification failure), consistent with the frozen table's note that WebFetch was network-blocked in the verification pass. **WebSearch was not needed** because full texts were obtained directly.
- Retrieval routes actually used (curl through the local proxy):
  | paper | abstract source | section-structure source | access limit |
  |---|---|---|---|
  | Spiking PointCNN (Electronics 2024) | publisher PDF, p.1 | publisher PDF, 15 pp (`res.mdpi.com` copy; `www.mdpi.com` HTML/PDF endpoints = Akamai Access Denied) | PDF is owner-password restricted; text extracted with pymupdf; layout-based section numbering verified |
  | SGNNBench (arXiv:2509.21342) | export.arxiv.org API (v1 = latest) | arXiv LaTeXML HTML v1 | full main text + appendices visible |
  | SpikingGCN (IJCAI 2022) | export.arxiv.org API (v2 = latest) | arXiv PDF v2 (IJCAI camera-ready), 12 pp | arXiv HTML/ar5iv unavailable for 2022 papers; structure read from PDF text |
  | SpikeGCL (ICLR 2024) | export.arxiv.org API (latest abstract) | arXiv LaTeXML HTML **v1** | latest is v2; structure read from v1 — camera-ready may differ slightly |
  | MK-SGN (Neurocomputing 2025/26) | export.arxiv.org API (v5 abstract) | arXiv LaTeXML HTML **v1** | latest is v5; structure read from v1; journal version not fetched |
  | QCFS (ICLR 2022) | export.arxiv.org API | arXiv PDF v1 (ICLR camera-ready), 20 pp | section headings recovered from PDF text |
  | Offset Spikes (ICLR 2023) | export.arxiv.org API | arXiv PDF v1 (ICLR camera-ready), 19 pp | as above |
  | SpikeZIP-TF (ICML 2024) | export.arxiv.org API | arXiv LaTeXML HTML v1 | PMLR version not fetched |
  | Adaptive Calibration (AAAI 2025) | export.arxiv.org API | arXiv LaTeXML HTML v1 | AAAI OJS version not fetched |
  | DCGS (ICML 2025) | export.arxiv.org API (v3 abstract) | arXiv LaTeXML HTML **v1** | latest is v3; structure read from v1 |
- Not inspected: OpenReview review threads, supplementary material beyond arXiv appendices, code repositories. Figure/table inventories come from captions only. All 10 comparators reached **section-level** inspection; no comparator was abstract-only.

### Section skeletons as found (abridged)

- **Spiking PointCNN:** 1 Introduction (roadmap paragraph + 3 bullets) · 2 Related Works (2.1 CNNs on point clouds, 2.2 SNN) · 3 Methods (3.1 Spiking X-Conv Module [3.1.1 LIF neuron, 3.1.2 X-Transformation Conv, 3.1.3 Spiking X-Conv], 3.2 adjustable surrogate gradient, 3.3 all-purpose ANN-to-SNN converting framework [3.3.1 time steps, 3.3.2 spiking neuron model, 3.3.3 training methodology]) · 4 Experiments and Results (4.1 settings, 4.2 ablations [components / surrogate k / time steps], 4.3 comparison with SOTAs) · 5 Conclusions.
- **SGNNBench:** 1 Intro · 2 Design Space (GNN / spiking neuron / coding scheme / membrane rectifier) · 3 Benchmarking Over Effectiveness (RQ1) · 4 Benchmarking Over Efficiency (RQ2) · 5 Ablation (RQ3) · 6 Conclusion · Appendices A–G (settings, time-step effect, theoretical energy, pre-linear layer, supplementary ablations + significance test, **Limitation**, **Related Work in Appendix G**). Numbered `Obs. 1–5` takeaways after experiment blocks.
- **SpikingGCN:** 1 Introduction · 2 Spiking Graph Neural Networks (preliminaries + method merged; run-in paragraphs: Graph Convolution / Representation Encoding / Charge-Fire-Reset / Model Feasibility Analysis) · 3 Experiments (four stated objectives: node classification / active learning / neuromorphic energy / extensions) · 4 Conclusions · Acknowledgments · References · **Appendix** (organization paragraph; A **Related Work in appendix**; B training; C extra experiments).
- **SpikeGCL:** 1 Intro ("Present work." + 4 bullets) · 2 Related Work · 3 Preliminaries (problem formulation; SNNs) · 4 SpikeGCL (4.1 grouping, 4.2 binarizing, 4.3 overall framework, 4.4 blockwise surrogate gradient) · 5 Theoretical guarantees (informal Theorem 1 in main text) · 6 Experiments · 7 Conclusion · Appendices A–F (full theory+proofs, vanishing gradients, algorithms, **D Discussion [D.1 complexity, D.2 Limitation, D.3 societal impact]**, settings, extra results).
- **MK-SGN:** 1 Intro · 2 Related Work · 3 Methodology (3.1 skeleton spiking coding + base SGN, 3.2 SMF module + spiking attention GCN, 3.3 GCN-to-SGN knowledge distillation [soft-label; inner-layer]) · 4 Experiments (4.1 datasets, 4.2 implementation, **4.3 theoretical energy consumption calculation**, 4.4 ablation [one sub-subsection per component], 4.5 results) · 5 Conclusion.
- **QCFS:** 1 Intro · 2 Preliminaries · **3 Conversion Error Analysis** (decomposes into clipping / quantization / unevenness error) · **4 Optimal ANN-SNN Conversion** (4.1 clip-floor, 4.2 clip-floor-shift + Theorem "expected error is zero", 4.3 training algorithm) · 5 Related Work (**after** the method) · 6 Experiments (6.1 source-ANN sanity check, 6.2 SOTA comparison, 6.3 with/without shift, 6.4 effect of quantization steps L) · 7 Discussion and Conclusion · Appendix (proofs, configs, extra tables).
- **Offset Spikes:** 1 Intro (4 numbered contribution items) · 2 Related Works · 3 Preliminaries (3.1 neuron models, 3.2 ANN-SNN conversion) · 4 Methods (4.1 offset spike definition + distribution, 4.2 judge errors via residual membrane potential, 4.3 eliminate via shifting initial membrane potential + optimal shift proof, 4.4 iterative property) · 5 Experiments (5.1 effectiveness, 5.2 SOTA comparison, 5.3 effect of inference time-step rho, 5.4 effect of iterative optimization) · 6 Conclusions · Appendix.
- **SpikeZIP-TF:** 1 Intro · 2 Background and Related Works (merged; run-in: Spiking Neurons / Learning Methods / Transformer-based SNNs) · 3 Methods (3.1 ST-BIF+ dynamics, 3.2 architecture [SESA, Spike-Softmax, Spike-LayerNorm], **3.3 complexity analysis**) · 4 Experiments (4.1 setup [CV + NLP], 4.2 results per benchmark, 4.3 training cost, **4.4 power estimation on neuromorphic hardware**, 4.5 ablation [accuracy vs time-steps], **4.6 equivalence inspection via experiments**) · 5 Conclusion · Impact Statement · Appendices A1–A4 (extra results, spike counting, full equivalence proofs, hyperparameter tables A3–A12).
- **Adaptive Calibration:** 1 Intro · 2 Related Work (run-in: spiking neuron / burst-firing neurons / conversion + calibration) · 3 Adaptive Calibration (**Motivation** first, then Preliminary; AdaFire neuron; Sensitivity Spike Compression; Input-aware Adaptive Timesteps) · 4 Experiment (effectiveness of AdaFire / SSC / IAT as three subsections + ablation) · 5 Conclusion · Appendices.
- **DCGS:** 1 Intro · 2 Related Works (**2.1 rate-based conversions, 2.2 temporal-coding conversions** — contrastive taxonomy) · 3 Preliminaries (multi-threshold neuron; rate coding) · 4 Method (4.1 differential coding [Definition 4.1 + Theorems 4.2–4.5 across layers], 4.2 optimal threshold for ReLU [Assumption → Lemma → Theorem ladder]) · 5 Experimental Results (5.1 SOTA comparison, **5.2 energy estimation and result analysis**, 5.3 effectiveness of differential coding, 5.4 effectiveness of threshold iteration) · 6 Conclusion · Impact Statement · Appendices A–M (algorithm, every proof in its own appendix, neuron model, extra tables).

---

## 1. Title

**Transferable patterns**
- **Name the analytic object, subtitle the scope.** Conversion-theory titles put the mechanism first and the claim/scope second: "Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency SNNs" (QCFS), "Bridging the Gap between ANNs and SNNs by Calibrating Offset Spikes", "Differential Coding for Training-Free ANN-to-SNN Conversion", "Adaptive Calibration: A Unified Conversion Framework of SNN". Paper B's "Spectral Gain, Activation Mass, and an Order-Invariant Readout" follows the same object-naming move.
- **Question/claim titles are used for framing papers:** "A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets SNNs" (SpikeGCL); benchmark scoping via "A Holistic Evaluation of X on Y" (SGNNBench). Paper B's "When Does Graph-to-Spiking Conversion Work?" is the question-title variant — consistent with the field's framing-paper convention.
- Method-name portmanteau titles (SpikeGCL, SpikeZIP-TF, MK-SGN) presuppose a coined system; **do not import** for an analysis manuscript without a proposed model.

**Study-specific:** their coined model names and task suffixes ("for Skeleton-based Action Recognition", "on Large-scale Graph").

## 2. Abstract

**Transferable patterns**
- **Universal 5-move spine:** (1) domain promise (SNN = efficient alternative) → (2) concrete failure/gap ("performance gap remains"; "requires long time-steps"; "compute unsustainable") → (3) named object + mechanism ("we define an offset spike..."; "we propose QCFS..."; "we present SGNNBench to quantify progress") → (4) theory move when applicable ("we prove that the expected conversion error is zero" (QCFS); "theoretical guarantees... comparable expressiveness" (SpikeGCL); "derive the optimal shifting distance and prove" (Offset Spikes)) → (5) quantified headline over stated datasets ("83.82% ImageNet / 93.79% SST-2" (SpikeZIP-TF); "67.12% top-1 at 6 time-steps" (Offset Spikes); "energy savings up to 70.1/60.3/43.1% on CIFAR-10/100/ImageNet" (Adaptive Calibration)).
- **Name the error and its rank early.** Error-taxonomy abstracts declare the dominant error in the abstract itself: QCFS "ascribe them to clipping error, quantization error, and unevenness error... unevenness error... has been neglected"; Adaptive Calibration "the Unevenness Error — the primary source of error of converted SNNs". Paper B's abstract correctly names its objects (identity, laws, activation mass) — keep the rank/ordering claims literal and configuration-scoped.
- **Analysis/benchmark abstracts state their dissection dimensions explicitly:** SGNNBench "from multiple perspectives, including effectiveness, energy efficiency, and architectural design... 9 SGNNs across 18 datasets... reveal the often-overlooked energy bottlenecks". Abstract-level honest negative ("often-overlooked bottlenecks", SpikeGCL "largely overlooked" burden) is an accepted move; Paper B's transfer-negatives sentence has precedent cover.
- **Scoping clause in the abstract:** SpikeGCL "making balanced trade-offs between efficiency and performance" — a one-clause scope limiter. Paper B's "labelled external" anchors (74.0/86.6) follow the same honesty convention.

**Study-specific:** "achieves state-of-the-art" + energy-reduction headline claims (universal in comparators) — an analysis paper must not inherit these (see Do-not-import).

## 3. Introduction

**Transferable patterns**
- **Move order:** domain stakes → the competing route and its dilemma → gap ("To bridge this gap..." SGNNBench; "the dilemma of ANN-SNN conversion is..." QCFS) → mechanism sentence → contribution bullets → optional roadmap.
- **Bullet taxonomy observed:** first bullet = new problem or new error object (SpikeGCL "We believe our work is timely..."; QCFS "go deeper into the errors... ascribe them to..."; Offset Spikes "introduce the concept of offset spike... one additional (or one less) spike"); second = mechanism with theorem-level claim ("prove that the expected conversion error... is zero"; "derive the optimal shifting distance and prove"); third = training/algorithmic device; fourth = empirical anchor with one flagship number. Paper B's five bullets (incl. deep-regime capability + negative map) exceed the usual 3–4 but each carries one claim — keep them claim-dense, not scope-dense.
- **Flagship numbers come with their accounting inline:** Offset Spikes "67.12% top-1 on ImageNet with only 6 time-steps (4 time-steps for calibration and 2 time-steps for inference)" — the budget breakdown in parentheses is exactly the accounting-conditional move Paper B needs for T/allocation numbers.
- **Novelty positioning sentence:** "To the best of our knowledge, this is the first time..." (QCFS, Offset Spikes, MK-SGN "for the first time"). For a contested-claim area (Paper B vs QCFS/SpikeZIP/Adaptive Calibration), attribute firsts explicitly and narrowly.
- **Roadmap paragraph (optional):** Spiking PointCNN inserts "To summarize... (Section 2.1)... (Section 4.2)..." before its bullets. House-style for MDPI; usable but optional for a journal submission.

**Study-specific:** their problem origins (point clouds, skeleton action, NLP) and dataset mentions in bullets.

## 4. Related Work

**Transferable patterns**
- **Three placement options exist in the wild:** early Section 2 (SpikeGCL, MK-SGN, DCGS, Offset Spikes, Adaptive Calibration); **after the method** (QCFS places Related Work as Section 5, after its theory/method Sections 2–4 — the "give the reader the analysis first" move suited to claim-first analysis papers); **fully deferred to appendix** (SpikingGCN Appendix A with an "Organization of the Appendix" preamble; SGNNBench Appendix G). For Paper B (journal, related work already positioned as a claim-axis map), the early-section placement with per-paragraph differentiators is right; QCFS's late placement shows the alternative if the identity claim needs to arrive before the literature.
- **Contrastive taxonomy headings:** DCGS splits "2.1 Rate-based ANN to SNN Conversion / 2.2 Temporal Coding Conversions" — the taxonomy itself encodes the axis of novelty. Paper B's "misalignment vs quantisation" contrast should get the same heading-level visibility.
- **Run-in background paragraphs** (SpikeZIP-TF "Background and Related Works" with bold run-ins "Spiking Neurons. / Learning Methods of SNN / Transformer-based SNNs."; Adaptive Calibration "Spiking Neuron Model. / Burst-firing Neurons. / ANN-to-SNN conversion and SNN Calibration"): each paragraph = one substrate concept the method will use. Good for the per-layer-timestep prior-art derivation-source contrasts.
- **Per-paragraph closer convention:** every thematic paragraph ends with what prior art does *not* cover (implicit in all ten; SGNNBench "no systematic benchmark to explore the basic design principles" is explicit). This is where the QCFS zero-error / SpikeZIP equivalence / Adaptive-Calibration unevenness claims must be cited *by claim location* (abstract + theorem + experiment subsection) so Paper B's differentiation is checkable.

**Study-specific:** SpikingGCN's appendix-related-work is an IJCAI page-limit artifact — do not copy in a journal paper.

## 5. Method

**Transferable patterns**
- **Notation table up front** (QCFS Table 1 "Summary of notations"; SpikeZIP-TF Table 2; SpikingGCN Table 8 in appendix). A symbols table early is a field-standard aid for a law-heavy paper like Paper B.
- **Named-object chain: define → diagnose → fix → refine.** Offset Spikes Section 4 is the template: 4.1 define the error object + show its distribution (Fig 1), 4.2 diagnostic criterion (judge via residual membrane potential), 4.3 remedy with theorem (optimal shift, provable), 4.4 derived property (iterative refinement). Paper B's identity → layer-1 law → margin criterion → readout invariance → staged semantics → allocation chain is the same shape; each subsection should carry one named quantity and one statement (definition/lemma/criterion).
- **Error decomposition precedes remedy.** QCFS: Section 3 "Conversion Error Analysis" decomposes the error into clipping / quantization / unevenness *before* Section 4's fix, and each fix term in the activation function maps 1:1 to a decomposed error. Paper B's "error identity" presentation should keep decomposition-then-identity order so the law terms visibly cancel/partition the error.
- **Theorem statements in main text, proofs in appendix; informal theorem in main text** (SpikeGCL "Theorem 1 (Informal)" in Section 5, full Theorem 2 + proofs in Appendix A; DCGS states Definition/Theorems 4.1–4.5 in text with *all* proofs in Appendices C–I; SpikeZIP-TF puts equivalence lemmas in Appendix A3). The informal-theorem device is directly usable for Paper B's identity/laws statements.
- **Assumption → lemma → theorem ladder with visible scope conditions** (DCGS 4.2: Assumption 4.6 → Definition 4.7 → Lemmas 4.8–4.9 → Theorem 4.10). For configuration-scoped laws, stating assumptions as numbered objects is the field norm for making scope auditable.
- **Motivation subsection with empirical error-share evidence before the model** (Adaptive Calibration Section 3 "Motivation" + Fig 3 "the unevenness error dominates conversion loss... percentage of three main conversion errors"). The "name the dominant term and show its share" device suits Paper B's activation-mass/spectral-gain laws.
- **Complexity/analysis subsection inside the method** (SpikeZIP-TF 3.3 Complexity Analysis; SpikingGCN "Model Feasibility Analysis" with probability bounds). A dedicated "what the law costs/implies" subsection is standard.
- **Method subsections mirror named modules 1:1** (MK-SGN 3.2 SMF / 3.3 KD; each with its own figure). For Paper B: each law gets exactly one subsection name that figures/tables can reference.

**Study-specific:** their neuron/activation inventions (QCFS clip-floor-shift, ST-BIF+, AdaFire, MT neuron) — Paper B proposes analysis, not a neuron; do not imitate "module + figure" packaging for non-module contributions.

## 6. Experiments/Results (including accounting / allocation / transfer subsections)

**Transferable patterns**
- **Setup block before any result:** datasets, baselines, implementation, metrics (MK-SGN 4.1/4.2; SGNNBench 3.1 "Datasets / Baselines" run-ins; SpikingGCN "Datasets." run-in). Paper B's protocol-first ordering matches.
- **Accounting subsection that defines the counter model before numbers:** MK-SGN "4.3 Theoretical Energy Consumption Calculation" precedes results; SGNNBench 4.1 setup + Appendix C "Theoretical Energy Consumption"; SpikeZIP-TF "4.4 Power Estimation on Neuromorphic Hardware" (Table 8); SpikingGCN Tables 3–4 (operations + energy); SpikeGCL Table 2 (parameter size + theoretical energy). All state their counting convention; SGNNBench makes the sensitivity itself a finding ("often-overlooked energy bottlenecks"). **Transfer to Paper B `sec:accounting`:** one fixed counting definition + explicit "accounting-conditional" labels on every energy/passthrough number, plus a sensitivity demonstration — this is the strongest precedent set in the comparator field.
- **Timestep-sweep table layout:** rows = methods (converted + directly-trained), columns = T = 2/4/8/16/32/64/≥512 with ANN anchor column (QCFS Table 2; Offset Spikes Table 1). Footnotes flag non-comparable configs (QCFS: "RTS and SNNC-AP use altered ResNet-18, while ours use standard ResNet-18"). Use this grid for Paper B's T sweeps / iso-accuracy tables.
- **Per-contribution effectiveness subsection + ablation rows:** every claimed mechanism receives one "effectiveness of X" subsection and one ablation-table block (Adaptive Calibration: Effectiveness of AdaFire / SSC / IAT + Table 5; DCGS 5.3 differential coding + 5.4 threshold iteration; MK-SGN 4.4.1–4.4.5 one sub-subsection per component; Spiking PointCNN 4.2.1–4.2.3). Map 1:1 onto Paper B's laws/diagnostics (closed-form check, threshold law, order, truewidth).
- **Allocation precedent:** search/allocation procedures are presented as (i) a pseudocode block (DCGS Algorithm 1 threshold iteration; SpikeGCL Algorithms 1–2), (ii) a sensitivity/Pareto figure (Adaptive Calibration Fig 4 "Pareto Frontier Searching"; Fig 4a–b sensitivity per layer), and (iii) a regime curve family (SGNNBench Fig 2/3 time-step influence on performance/memory/latency). **Transfer to Paper B `sec:allocation` (fig8):** procedure as algorithm-style steps + regime-map figure + prior-art attribution for the validation-search part + scope conditions.
- **Transfer/generality precedent:** multi-domain tables (Adaptive Calibration 2D/3D/event/detection/segmentation; SpikeZIP-TF CV+NLP; SpikingGCN extensions; SpikeGCL Tables 6–7 across encoders and neuron types). Paper B `sec:transfer` inverts this into a *negative* map — keep the same per-regime table/figure grammar but label failure modes explicitly; negative-first presentation is rare in the field (only SGNNBench's Obs. honesty and SpikeGCL's Appendix D.2 limitation approach it), so state the negative claim as a finding, not an apology.
- **RQ + numbered-observation pattern (SGNNBench):** section titles are research questions (RQ1/RQ2/RQ3) and each experiment block ends with a bolded "Obs. N." takeaway. Highly scannable; recommended as bolded takeaway lines at the end of Paper B's results subsections.
- **Equivalence/claim-inspection subsection:** SpikeZIP-TF "4.6 Equivalence Inspection via Experiments" runs a direct empirical check of the theoretical equivalence; QCFS 6.1 sanity-checks the source ANN before any conversion number. **Transfer:** a "law-check" pass (predicted vs observed error, fig5/fig6) placed before SOTA-comparison tables, exactly as Paper B's closed-form/theory subsections do.
- **Two baseline classes:** conversion peers *and* directly-trained SNN peers (QCFS 6.2 lists both; Offset Spikes Table 2 "Comparison with other types of SNN training methods"). Keep both classes in Paper B's baselines.
- **Table styling conventions:** bold best / gray runner-up (SpikeZIP-TF Table 4); gap shading (SpikeGCL Table 1); "top-1/2-3 highlighted" (SGNNBench Tables 2–5); significance tests in appendix (SGNNBench E.1, Fig 5).

**Study-specific:** their datasets (ImageNet/CIFAR/NTU-RGB+D/ModelNet40 vs our citation graphs), model families, and absolute accuracies.

## 7. Discussion

**Transferable patterns**
- Conference papers merge discussion into conclusion (QCFS "7 Discussion and Conclusion") or defer it to appendix (SpikeGCL Appendix D: complexity / limitation / societal impact). A journal paper can afford a real Discussion; the content patterns to borrow:
  - **Trade-off interpretation with the hyperparameter named** (QCFS: "Choosing the proper L is a trade-off between the accuracy at low time-steps... and model capacity").
  - **Boundary/deployment implication** (Offset Spikes: low-T degradation "hampers the practical applications of SNNs to neuromorphic chips" → their fix reopens deployment).
  - **Limitation paragraph as first-class** (SpikeGCL D.2; SGNNBench Appendix F "Limitation"). Paper B's four discussion moves (laws licensed in-configuration; transfer open; accounting hazard; deployment guidance) each match one of these precedents; the accounting-hazard move parallels SGNNBench's energy-bottleneck honesty.

**Study-specific:** ICLR societal-impact paragraphs (checklist artifacts).

## 8. Conclusion

**Transferable patterns**
- **One paragraph, three moves:** restate the contribution ("we present X..."), summarize demonstrated results *scoped to tested settings*, end with one open/future sentence (SpikingGCN: applicability demo "we have not gone into depth... leave this design in the future work"; all ten end with a future or impact sentence).
- **Bounded-claim device:** SpikingGCN explicitly scopes a weak experiment ("the purpose of this experiment is to demonstrate applicability... not the focus of the current paper"). Paper B's bounded answer (identity + laws; deep-regime procedure; open transfer; **no efficiency claim**) is a *stronger* version of this device and is a differentiator — no comparator manages to close without an efficiency claim; keep it that way.
- No new numbers in conclusions (universal).

## 9. Figures / Tables

**Transferable patterns (figure order = argument order)**
1. **Landscape/concept figure first:** comparison of prior paradigm vs proposed framing (SpikeGCL Fig 1 conventional GCL vs SpikeGCL; MK-SGN Fig 1 GCN vs SGN; SpikeZIP-TF Fig 1 SOTA landscape scatter + Fig 2 conversion pipeline). = Paper B fig1 framework.
2. **Error-diagnostic figures before fix figures:** Offset Spikes Fig 1 offset-spike distribution per layer → Fig 2 mechanism → Fig 3 before/after; Adaptive Calibration Fig 3 error-share pie/bars; QCFS Figs 1–2 error decomposition and SNN-vs-ANN output. = Paper B fig5 theory / fig6 prediction grammar: *show the error object empirically, then the law that explains it.*
3. **Architecture figure with numbered callouts** matching method subsections (Spiking PointCNN Fig 6 "complete structure"; SpikeGCL Fig 2; MK-SGN Fig 2 with teacher/student).
4. **Regime/curve families:** accuracy-vs-T multi-panel (SpikeZIP-TF Fig 5 a–d across NLU/CV/model-size/quantization; SpikeGCL Fig 7; SGNNBench Figs 2–3 T influence on performance/memory/latency). = Paper B fig2/fig3 and fig8 allocation regime; multi-panel with per-dataset panels is the norm.
5. **Ablation bar/curve figures** (Spiking PointCNN Figs 3–4 surrogate-gradient k; Adaptive Calibration Figs 6–7; SpikeGCL Fig 6 gradient norms).
- **Algorithm blocks** for procedures (SpikeGCL Alg 1–2; DCGS Alg 1–2) — consider for Paper B's allocation procedures.
- **Table inventory pattern:** notations table → dataset statistics → SOTA comparison with T columns (accuracy) → separate or combined accounting table (DCGS Table 2 "accuracy and energy ratio"; SpikeGCL Table 2; SGNNBench Table 6 model size/memory/energy) → per-component ablation → hyperparameter/search-space tables in appendix (SpikeZIP-TF Tables A3–A12; SGNNBench Table 8). Paper B's tab1–19 already follows this inventory; keep footnoted non-comparable configs (tab3 precedent).

## 10. References

**Transferable patterns**
- Every baseline named in a comparison table is cited with year and venue at first mention ("HC from Rathi et al. (2019), RMP from Han et al. (2020)..." QCFS 6; Offset Spikes same). 
- Code availability as footnote or abstract sentence (all ten with public code; SpikingGCN footnote 1; SGNNBench arXiv comment field).
- Field sizes: conference papers ~40–60 entries, MDPI journal ~25–30. The frozen table's 10 papers must each appear with an explicit positioning sentence (which claim of theirs we differentiate).
- Novelty sentences are anchored to citations ("first... (to the best of our knowledge)"), not asserted bare.

---

## Cross-cutting argument spines

1. **Conversion-theory spine (QCFS → Offset Spikes → SpikeZIP-TF → Adaptive Calibration → DCGS):** promise → low-T failure → *named error taxonomy* → formal error object → decomposition/diagnosis → remedy with theorem → SOTA at low T → derived property (iterative / general / exact). Paper B's spine (question → identity → laws → deep-regime capability → transfer negatives → accounting sensitivity) is the analysis-genre analog; the comparators validate the order *definition before law before procedure before evidence*, and validate that negative/limiting results can sit at the end as findings (SGNNBench Obs.; SpikeGCL D.2).
2. **Graph-niche spine (SpikingGCN → SpikeGCL → Spiking PointCNN → MK-SGN → SGNNBench):** graph promise → efficiency burden → adaptation route (spike coding / conversion framework / distillation) → benchmark tasks → energy accounting. Paper B's related work should present these five as the route map and mark where structure-preserving conversion claims must be differentiated (esp. Spiking PointCNN's modular conversion and MK-SGN's distillation-≠-weight-transfer caveat).
3. **Claim-location discipline:** the strongest comparators state their headline claim three times at three altitudes — abstract sentence, intro bullet with proof claim, theorem/experiment subsection with the number. Paper B's error-identity claim should be findable at the same three altitudes, each time with its configuration scope attached.
4. **Accounting-conditional honesty is emerging as a field norm** (SGNNBench energy bottlenecks; SpikeZIP-TF power estimation caveats "time-step of SpikeGPT larger than 50, therefore power is less than 0.234" — bound-style reporting). Paper B's `sec:accounting` is ahead of the field here; present it as a protocol contribution, with bound-style footnotes where the counter model is uncertain.

## Do-not-import (study-specific or wrong-genre for Paper B)

- Comparator **conclusions are not requirements**: zero expected error (QCFS), exact/lossless equivalence (SpikeZIP-TF), dominant-unevenness-error (Adaptive Calibration), 98% energy reduction (MK-SGN) — these are their claims over their configurations; Paper B must position against them, not inherit or refute them as obligations.
- Do not inherit SOTA/efficiency headline claims ("state-of-the-art", "ultra-low", "green/environment-friendly") — Paper B's conclusion explicitly makes **no efficiency claim**; keep every energy number accounting-conditional.
- Do not copy mechanism packaging (coined neuron names, module+figure kits) for what are analysis results.
- Do not copy dataset/model scale (ImageNet VGG/ResNet/Transformer) as evidence weight; graph-config evidence is the scope.
- MDPI roadmap paragraphs, ICLR societal-impact sections, and appendix-only related work are venue artifacts, not transferable argument structure.
- SGNNBench's claimed NeurIPS status and ICLR "CCF-A" labels remain disputed in the frozen table — do not cite venue prestige arguments as evidence in the manuscript.
