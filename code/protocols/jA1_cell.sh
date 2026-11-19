#!/usr/bin/env bash
# One SpikeGCL audit cell: jA1_cell.sh <Dataset> <seed>  (runs inside WSL)
set -uo pipefail
DS="$1"; SEED="$2"
REPO=[WORKSPACE]/third_party/SpikeGCL
DATA=[WORKSPACE]/third_party/SpikeGraphormer/datasets
PY="$HOME/miniconda3/envs/spikg/bin/python"
LOG=[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade/logs/jA1_spikegcl_multiseed.log
mkdir -p "$DATA/Planetoid/$DS/raw"
cp -n [WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261028-174053-gnn-snn-conv/results/.data/ind.* "$DATA/Planetoid/$DS/raw/" 2>/dev/null || true
cd "$REPO" || exit 1
echo "### $DS seed=$SEED" >> "$LOG"
TH=5e-4; T=64; EP=5
if [ "$DS" = "Citeseer" ]; then TH=5e-3; T=32; fi
"$PY" -u main.py --dataset "$DS" --threshold "$TH" --outs 2 --T "$T" --bn --epochs "$EP" \
  --seed "$SEED" --root "$DATA/" >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
