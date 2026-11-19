#!/usr/bin/env bash
# E-A1: multi-seed cells for the Paper A audit grid (run at P4, after P3/P3A gates).
# SpikeGCL supports --seed (default 2023); each seed is a separate README-command run.
# Runs inside WSL Ubuntu-22.04 with the pinned spikg env. Output: wsl logs + parsed JSON.
set -uo pipefail
REPO=[WORKSPACE]/third_party/SpikeGCL
DATA=[WORKSPACE]/third_party/SpikeGraphormer/datasets
SRC=[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261028-174053-gnn-snn-conv/results/.data
OUT=[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade/logs
PY="$HOME/miniconda3/envs/spikg/bin/python"
LOG="$OUT/jA1_spikegcl_multiseed.log"

mkdir -p "$DATA/Planetoid/Cora/raw" "$DATA/Planetoid/Citeseer/raw"
cp -n "$SRC"/ind.cora.* "$DATA/Planetoid/Cora/raw/" 2>/dev/null || true
cp -n "$SRC"/ind.citeseer.* "$DATA/Planetoid/Citeseer/raw/" 2>/dev/null || true

cd "$REPO" || exit 1
: > "$LOG"
# seeds 2023 (README default, reproduces the audited cell) + 2024..2026
for seed in 2023 2024 2025 2026; do
  echo "### Cora seed=$seed" >> "$LOG"
  "$PY" -u main.py --dataset Cora --threshold 5e-4 --outs 2 --T 64 --bn --epochs 5 \
    --seed "$seed" --root "$DATA/" >> "$LOG" 2>&1
  echo "exit=$?" >> "$LOG"
  echo "### Citeseer seed=$seed" >> "$LOG"
  "$PY" -u main.py --dataset Citeseer --threshold 5e-3 --T 32 --bn --epochs 5 \
    --seed "$seed" --root "$DATA/" >> "$LOG" 2>&1
  echo "exit=$?" >> "$LOG"
done
grep -E "###|Final_Acc|exit=" "$LOG"
