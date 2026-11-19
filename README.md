# gnn2snn-v2

Two journal-track manuscripts on **training-free conversion of graph neural networks to spiking GNNs**, with their evidence, experiment code, and research-planning materials.

Manuscripts are **not submitted anywhere**; author/venue fields are left pending.

## Contents

| Path | What it is |
|---|---|
| `paper-b-conversion/` | Paper B — *conversion* manuscript: final revised LaTeX sources + compiled PDF (`revised-manuscript.pdf`). Exact-error identity for staged conversion, spectral gain law, deep-regime per-layer timestep allocation (validation-search arm, held-out 6/8 cells within ANN−1.5pp with 22.7–93.4% fewer layer-updates), plus three pre-registered negative results on cross-graph transferability and an SOP-accounting sensitivity analysis. |
| `paper-a-audit/` | Paper A — *audit* manuscript: final revised LaTeX sources + compiled PDF. Audit of every code-releasing spiking-GNN method in a defined seven-method survey: dataset-dependent reproduction gap (SpikingGCN), a split worth ~10 points (SpikeGraphormer), two reproductions (SpikeGCL), unlabelled split mixing in comparison tables, per-dataset transportability against measured seed noise, and a five-variant input-convention diagnosis. |
| `docs/journal-upgrade-audits/` | Deviation audit vs. the original guidance, pre-registration records, outcome records for each experiment arm (including negative ones), correction ledger, and delivery notes. |
| `docs/f-submit-paper-a/`, `docs/f-submit-paper-b/` | Journal submission-readiness artifacts: verified comparator literature tables, module quality reviews, revision logs, re-scoring rounds, and final submission audits. |
| `code/protocols/` | Experiment scripts for both papers (conversion engine with staged per-layer windows, allocation rules, capacity/SOP probes, convention grid, replay check, search arm). |
| `code/writing/` | Evidence-packet build scripts and packets used by the writing pipeline. |
| `code/engine/` | Core training/evaluation engine dependencies from the original experiment run. |
| `results/` | Registered result JSONs backing the papers' claims. |
| `fable-material/` | Earlier external planning material for this direction: a GitHub research-material snapshot (`github-gnn2gsnn/`) and a Notion export (`notion-export/`, 13 pages / 115 literature entries). Kept as provenance; its claims were **not** independently verified and were superseded by the audited experiments above. |

## What is intentionally excluded

- **Datasets** (Planetoid raw files, OGB, PyG caches) — download from the original sources instead.
- **Model weights / checkpoints**.
- **Credentials, tokens, logs with local absolute paths** (local paths in included text files are scrubbed to `[WORKSPACE]`).
- Build intermediates (`.aux`, `.log`, …) and per-run scratch directories.

## Reproducing

The code expects a PyTorch + PyTorch-Geometric environment and the standard citation datasets (Cora, CiteSeer, PubMed, and the heterophilous graphs used in Paper B). See `code/protocols/` headers and `docs/journal-upgrade-audits/` for the pinned conditions and per-arm command records. Result JSONs in `results/` are the claim-backing evidence; the papers' tables map to them one-to-one.

## License / status

Research artifacts published for transparency and review. Papers are pre-submission drafts; numbers in the manuscripts are backed by the checked-in evidence files.
