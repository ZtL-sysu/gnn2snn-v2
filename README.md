# gnn2snn-v2

> **中文简介**：本仓库归档两篇面向期刊投稿的 **免训练 GNN→SNN 转换与脉冲图神经网络复现审计** 完整稿件（LaTeX 源码 + 编译 PDF）、全部注册实验结果 JSON、偏差与投稿准备审计文档、实验协议与写作管线代码，以及作为来源记录保留的外部规划素材（`fable-material/`）。两篇稿件均处于**预投稿（pre-submission）状态**，作者与期刊字段待填，尚未向任何期刊/会议提交。云端另含 `gtmc-grid` 实验分支，记录后续 GTMC 转换校准与因子解码对比网格代码。

Two journal-track manuscripts on **training-free conversion of Graph Neural Networks (GNNs) to Spiking GNNs** and **reproducibility auditing of spiking GNNs**, together with their registered evidence JSONs, deviation & submission-readiness audits, experiment/writing code, and archived external planning materials.

- **Status:** Pre-submission drafts — **not submitted anywhere**; author, affiliation, venue, and disclosure fields remain author-pending.
- **Related Repositories:**
  - [`ZtL-sysu/phd-snn`](https://github.com/ZtL-sysu/phd-snn) — Parent PhD research workspace covering the full multi-campaign GNN→SNN exploration (10-idea sweep, LR-C05, CERT-C11, POS-C13, Fable 5.2 experiments, cross-agent collaboration logs).
  - [`ZtL-sysu/gnn2snn`](https://github.com/ZtL-sysu/gnn2snn) — Earlier superpixel visual-graph GNN→SNN conversion archive (`MNIST`/`CIFAR10` fixed GCN & GatedGCN fidelity archive).
  - [`ZtL-sysu/phd-dongtaitu`](https://github.com/ZtL-sysu/phd-dongtaitu) — Offline Event-Driven Spiking Dynamic Graph Networks (EDSGN) doctoral proposal implementation.

---

## 1. Branches

| Branch | Purpose | Head Scope |
|---|---|---|
| `main` (default) | Frozen journal-track archive of **Paper A** & **Paper B** | Revised LaTeX + PDFs, `docs/`, `code/{engine,protocols,writing}/`, `results/*.json`, `fable-material/` |
| `gtmc-grid` | Follow-up closed-form conversion & validation-guarded calibration grid | Adds `gtmc_grid.py`, `gtmc_repair.py`, `faithful_baselines.py`, `factorial_decode.py`, `matched_conversion.py`, `val_guard.py`, and unbiased multi-spike TPP emission in `core.py` |

To inspect the follow-up calibration grid experiments locally:

```bash
git checkout gtmc-grid
```

---

## 2. Manuscripts & Verified Key Findings

### Paper B — Conversion Mechanism, Deep-Regime Allocation & Negative Controls (`paper-b-conversion/`)

- **Title:** *When Does Graph-to-Spiking Conversion Work? Spectral Gain, Activation Mass, and an Order-Invariant Readout*
- **Deliverables:** [`paper-b-conversion/main.tex`](paper-b-conversion/main.tex), [`paper-b-conversion/revised-manuscript.pdf`](paper-b-conversion/revised-manuscript.pdf), [`paper-b-conversion/main.pdf`](paper-b-conversion/main.pdf)
- **Core Contributions & Evidence-Backed Claims:**
  1. **Exact-Error Identity & Spectral Gain Law:** Exact conversion residual decomposition for staged ANN-to-SNN graph conversion, characterizing sensitivity to graph spectral gain, layer depth, width, activation mass, and order-invariant membrane readout.
  2. **Deep-Regime Per-Layer Timestep Allocation (Pre-Registered Positive Arm):** Validation-search per-layer timestep allocation (idea lineage attributed to LD-SNN / PASCAL / MT-SNN; graph instantiation and regime boundaries newly measured). In the deep regime ($L=4$), achieves held-out accuracy within $\text{ANN} - 1.5\text{pp}$ on **6/8 cells** (including 4/4 on Squirrel, beating random allocation controls) while reducing layer-updates by **22.7%–93.4%** (48%–93% on the primary deep search arm). In the shallow regime ($L=3$), headroom collapses (1/8 cells).
  3. **Three Pre-Registered Negative Results on Cross-Graph Transfer:**
     - Closed-form allocation rules (`R1`/`R2`) fail their frozen acceptance criteria.
     - Calibration-free selector (`jB1a`) achieves exact $\kappa = 1.0$ on calibration graphs but fails cross-graph transfer (held-out $\kappa \in [1.2, 11]$, Chameleon outside band), establishing an *"in-graph characterizable, non-transferable across graphs"* diagnostic finding.
     - Gap predictor (`wp7`) fails its original $\text{MAE} \le 1\text{pp}$ target and is explicitly preserved as a negative result.
  4. **SOP-Accounting Sensitivity Analysis:** Demonstrates that apparent efficiency rankings reverse under different Synaptic Operations (SOP) accounting conventions (e.g., event-driven vs. static accounting, and correcting a discovered mismatch where an external baseline `hidden=512` argument was previously unused in an SGC+linear model).

### Paper A — Reproducibility Audit of Spiking GNNs (`paper-a-audit/`)

- **Title:** *What Reproduces in Spiking Graph Neural Networks? An Audit of Every Code-Releasing Method in a Defined Survey*
- **Deliverables:** [`paper-a-audit/main.tex`](paper-a-audit/main.tex), [`paper-a-audit/revised-manuscript.pdf`](paper-a-audit/revised-manuscript.pdf), [`paper-a-audit/main.pdf`](paper-a-audit/main.pdf)
- **Core Contributions & Evidence-Backed Claims:**
  1. **Exhaustive Code-Releasing Survey Census:** Audits every code-releasing spiking GNN method within a defined seven-method survey scope, separating dataset-dependent reproduction gaps (**SpikingGCN**), a ~10-point split discrepancy (**SpikeGraphormer**), and two verified reproductions (**SpikeGCL**).
  2. **Unlabelled Split Mixing & Per-Dataset Transportability:** Documents split-protocol mixing across published comparison tables and quantifies cross-split transportability ratios against consistent measured seed noise (**1.35×** Cora, **1.37×** CiteSeer, **0.73×** PubMed — triggering pre-registered narrowing on PubMed).
  3. **Five-Variant Input-Convention Causal Diagnosis (`jA_conventions`):** Isolates how input feature normalization, self-loops, and adjacency conventions drive reported accuracy swings, backed by exact replay verification (`jA_replay.py`, 0 mismatches).

---

## 3. Repository Structure

```text
gnn2snn-v2/
├── README.md                          # This guide
├── .gitignore                         # Excludes datasets, weights, build artifacts, caches
├── paper-a-audit/                     # Paper A final LaTeX sources, tables, figures, compiled PDFs
│   ├── main.tex                       # Paper A root TeX
│   ├── refs.bib                       # Verified bibliography
│   ├── main.pdf / revised-manuscript.pdf
│   ├── sections/                      # abstract, intro, related, protocol, results, discussion, conclusion
│   ├── tables/                        # audit.tex, census.tex, protocol.tex, split.tex
│   ├── figures/                       # fig1–fig7 + per-dataset accuracy-energy plots
│   └── input-manifest.json            # File provenance & hash manifest
├── paper-b-conversion/                # Paper B final LaTeX sources, tables, figures, compiled PDFs
│   ├── main.tex                       # Paper B root TeX
│   ├── refs.bib                       # Verified bibliography
│   ├── main.pdf / revised-manuscript.pdf
│   ├── sections/                      # abstract, intro, related, method, experiments, discussion, conclusion
│   ├── tables/                        # tab1–tab14, tab16–tab19
│   ├── figures/                       # fig1–fig9 (incl. fig8_allocation_regime, fig9_transfer)
│   └── input-manifest.json            # File provenance & hash manifest
├── docs/
│   ├── journal-upgrade-audits/        # Deviation audit, preregistrations, outcome logs, novelty map, correction ledger
│   ├── f-submit-paper-a/              # Paper A submission-readiness reviews, comparator table, rescore logs, final audit
│   └── f-submit-paper-b/              # Paper B submission-readiness reviews, comparator table, rescore logs, final audit
├── code/
│   ├── engine/                        # Core GNN/SNN training, conversion & baseline engines
│   ├── protocols/                     # Experiment runners for Paper A & Paper B arms (jA*, jB*, p2_smoke)
│   └── writing/                       # Citation gate, evidence packets, flat TeX builders & repo staging script
├── results/                           # 12 registered JSON result artifacts backing manuscript tables & claims
└── fable-material/                    # Archived external planning snapshot (GitHub + Notion export; provenance only)
```

---

## 4. Registered Evidence & Audit Index

### Experiment Results (`results/`)

| File | Paper | What It Records |
|---|---|---|
| `results/p2_smoke.json` | A & B | Environment smoke test & exact conversion identity residual check (`0.0` residual on Cora) |
| `results/jA_conventions.json` | A | Five-variant input/self-loop/normalization convention causal grid |
| `results/jB_baseline_recheck.json` | B | Recheck of registered baselines uncovering the unused `hidden=512` SGC+linear bug |
| `results/jB1_prototype.json` | B | Closed-form allocation prototype & headroom/κ stop-loss gate |
| `results/jB1_search_pilot.json` | B | Validation-search per-layer timestep allocation (pilot graphs) |
| `results/jB1_search_heldout.json` | B | Validation-search per-layer timestep allocation (held-out graphs: 6/8 deep cells) |
| `results/jB1a_select.json` | B | Calibration-free selector (`jB1a`) calibration vs. held-out transfer outcomes (negative result) |
| `results/jB2_capacity.json` | B | Width/capacity probe across `H=16/32/512` confirming identical outputs on the legacy SGC arm |
| `results/jB2b_truewidth.json` | B | Repaired true-hidden-width (`512/32/16`) MLP/GCN comparison with 3-convention SOP accounting |
| `results/jB2c_sop_verdict.json` | B | Final SOP-accounting sensitivity verdict (event-driven vs. static accounting) |
| `results/jB3_squirrel_complete.json` | B | Complete Squirrel heterophilous graph conversion and allocation matrix |
| `results/jB5_bias_provenance.json` | B | Table 1 / Table 3 bias-term provenance audit |

### Key Audit & Governance Documents (`docs/`)

- [`docs/journal-upgrade-audits/deviation-audit.md`](docs/journal-upgrade-audits/deviation-audit.md) — Full claim/guidance deviation audit prior to the journal upgrade.
- [`docs/journal-upgrade-audits/ccfa-novelty-map-E-B1.md`](docs/journal-upgrade-audits/ccfa-novelty-map-E-B1.md) — Prior-art map covering LD-SNN (ISLPED 2022), PASCAL (arXiv:2505.01730), and MT-SNN (2026).
- [`docs/journal-upgrade-audits/jB1-search-arm-prereg.md`](docs/journal-upgrade-audits/jB1-search-arm-prereg.md) & [`jB1-search-arm-outcome.md`](docs/journal-upgrade-audits/jB1-search-arm-outcome.md) — Pre-registration and outcome record for the validation-search allocation arm.
- [`docs/journal-upgrade-audits/sop-convention-predeclaration.md`](docs/journal-upgrade-audits/sop-convention-predeclaration.md) — Pre-declared SOP accounting formulas.
- [`docs/journal-upgrade-audits/p5-strong-result-lock.md`](docs/journal-upgrade-audits/p5-strong-result-lock.md) & [`p20-p21-delivery.md`](docs/journal-upgrade-audits/p20-p21-delivery.md) — Final claim lock and clean-build delivery verification.
- [`docs/f-submit-paper-a/final-submission-audit.md`](docs/f-submit-paper-a/final-submission-audit.md) & [`docs/f-submit-paper-b/final-submission-audit.md`](docs/f-submit-paper-b/final-submission-audit.md) — Submission-readiness audits (S0–S10 complete; author/venue fields pending).

---

## 5. Follow-Up Calibration Branch (`gtmc-grid`)

The `gtmc-grid` branch extends `code/engine/` with a systematic shared-checkpoint conversion grid (tested on Cora, CiteSeer, PubMed, and heterophilous graphs across multiple seeds and latency windows $T \in \{1, 2, 4, 8\}$):

- `code/engine/core.py` — Adds unbiased multi-spike Two-Phase Probabilistic (TPP) emission.
- `code/engine/gtmc_grid.py` — 11-arm shared-checkpoint comparison (`ann`, `oracle`, `naive`, `threshbal`, `weightnorm`, `rts`, `snnc_ap`, `tpp`, `permute`, `gtmc`, `gtmc_tpp`).
- `code/engine/gtmc_repair.py` — Activation-scale threshold + closed-form bias + membrane readout repair arms (`gtmc_wn`, `gtmc_wn_tpp`).
- `code/engine/faithful_baselines.py` — Faithful percentile weight normalization (`weightnorm_faithful`), matched readout (`matched_readout`), and gauge-normalized GTMC (`gtmc_gauge`).
- `code/engine/factorial_decode.py` — $2 \times 2 \times 2$ factorial design crossing threshold (`1` vs. `p99`), bias (`0` vs. $\theta/(2T)$), and decoding (`raw` vs. threshold-aware $q \cdot s$) with a real oracle assertion.
- `code/engine/matched_conversion.py` — Constant-current count-formula comparison & per-channel unlabeled logit calibration.
- `code/engine/val_guard.py` — Validation-constrained calibration (`guarded` and `direct` arms + validation-fit class biases) with pinned dependency hashes.

---

## 6. Reproducing & Building

### Environment

- Python 3.10+ with **PyTorch** (CUDA or Apple Silicon MPS supported) and **PyTorch Geometric** (`torch_geometric`), `numpy`, `scipy`.
- Datasets (not checked in; auto-downloaded by PyG / OGB on first run):
  - Planetoid citation graphs: `Cora`, `CiteSeer`, `PubMed`
  - Heterophilous graphs: `Chameleon`, `Squirrel` (Geom-GCN splits)
  - Large-scale citation graph: `ogbn-arxiv` (OGB)

### Compiling the Manuscripts (TeX Live / `latexmk`)

Both manuscripts use self-contained LaTeX preambles (no external venue `.sty` required):

```bash
# Compile Paper A
cd paper-a-audit
latexmk -pdf -interaction=nonstopmode main.tex

# Compile Paper B
cd ../paper-b-conversion
latexmk -pdf -interaction=nonstopmode main.tex
```

### Running Experiment Protocols

See [`code/protocols/journal-commands.md`](code/protocols/journal-commands.md) and script docstrings in `code/protocols/` for exact pinned CLI invocations:

```bash
# Smoke test & exact-error identity residual check
python code/protocols/p2_smoke.py

# Paper A five-variant convention grid & replay audit
python code/protocols/jA_conventions.py
python code/protocols/jA_replay.py

# Paper B allocation, true-width capacity & SOP sensitivity suites
python code/protocols/jB1_search_arm.py
python code/protocols/jB2b_truewidth.py
python code/protocols/jB2c_sop_verdict.py
```

---

## 7. Exclusions & Provenance Policy

- **Intentionally excluded:** Raw/processed graph datasets (Planetoid, Geom-GCN, OGB, PyG caches), trained model checkpoints/weights, LaTeX build intermediates, credentials/tokens, and local logs containing host-specific absolute paths (all checked-in text files were path-scrubbed to `[WORKSPACE]` and verified with 0 secret matches).
- **Provenance notice for `fable-material/`:** Contains an archived snapshot of earlier external planning materials (`github-gnn2gsnn/` + 13-page / 115-entry Notion export). It is retained strictly for provenance; its claims were **not** independently verified and are superseded by the audited experiments and manuscripts above.

## 8. License & Citation Status

All research artifacts in this repository are published openly for transparency, auditability, and peer review. Manuscripts are pre-submission drafts whose reported numbers trace 1-to-1 to the checked-in `results/*.json` evidence files.
