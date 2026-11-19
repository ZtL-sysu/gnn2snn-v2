# Journal-Upgrade Command Registry (P2)

Interpreter: `[WORKSPACE]/miniconda3/envs/dyg-cold/python.exe` (run with `-I`). Prior-run engine
and loaders are imported read-only from
`[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261028-174053-gnn-snn-conv/`
(`protocols/graph_data.py`, `experiments/core.py`). New scripts are written under this run root's
`protocols/` and emit to `results/` + `logs/`. WSL legacy runs use
`wsl -d Ubuntu-22.04 -- ~/miniconda3/envs/spikg/bin/python`.

| item | command (to be run in P3/P4) | script status | budget |
|---|---|---|---|
| E-B1 prototype | `python -I protocols/jB1_prototype.py --seeds 0 1 --datasets cora citeseer --L 3 4` | to write at P3 (engine freeze extension) | ≤1 h |
| E-B1 full | `python -I protocols/jB1_allocation.py --seeds 0..6 --datasets chameleon squirrel --L 3 4 --grid 1..64` (arxiv phase after pilot) | to write after prototype gate | 6–10 h |
| E-B1 falsifiers | same script `--arm uniform_cert --arm reversed --arm oracle` | included | included |
| E-B1a | `python -I protocols/jB1_select.py --kappa-report` | to write | ≤1 h |
| E-B2 capacity match | `python -I protocols/jB2_capacity.py --K 2 --H 16 32 --seeds 0..6` (adapts prior `protocols/spikinggcn_baseline.py` flags) | adapt | 2–3 h |
| E-B3 Squirrel θ | `python -I protocols/jB3_squirrel.py --train-per-seed-once --theta 0.75 0.9 1.0` (rework of prior `experiments/wp15_confirm.py` 49-train design) | adapt | ≤2 h |
| E-B5 provenance | `python -I protocols/jB5_bias_provenance.py` (reads `beta_star_s7.json`, `ablate_s7.json`) | to write | ≤1 h |
| E-A1 multi-seed | WSL: `bash third_party/run_spikegcl.sh --seeds 3..7`; legacy: `spikinggcn36` CiteSeer seeds 2..5; SpikeGraphormer public-split seeds 2..4 | adapt existing shell entry points | ≤4 h |
| E-A2 conventions | `python -I protocols/jA_conventions.py --emit results/jA_conventions.json` | to write | ≤1 h |
| E-A3 split quotes | documentary (no command) | — | ≤1 h |
| E-A4 replay | `python -I protocols/jA_replay.py --logs <legacy logs> --tables tex_audit/tables` | to write | ≤1 h |
| smoke | `python -I protocols/p2_smoke.py` | **done** (`results/p2_smoke.json`) | 10 s |

Ordering: E-B1 prototype gate → E-B1 full + falsifiers → E-B1a → E-B2 → E-B3 → E-B5; E-A1..A4 in
parallel batches (CPU/WSL, no GPU contention). All runs append registry rows; no run overwrites a
`results/*.json` from the protected prior run.
