#!/usr/bin/env bash
# E-A1 resume driver: one WSL invocation per audit cell; skips cells with Final_Acc already logged.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/../logs/jA1_spikegcl_multiseed.log"
for cell in "Cora 2023" "Cora 2024" "Cora 2025" "Cora 2026" "Citeseer 2023" "Citeseer 2024" "Citeseer 2025" "Citeseer 2026"; do
  ds=${cell% *}; seed=${cell#* }
  if grep -A40 "### $ds seed=$seed" "$LOG" 2>/dev/null | grep -q "Final_Acc"; then
    echo "SKIP $ds $seed"; continue
  fi
  echo "RUN $ds $seed"
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash "[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade/protocols/jA1_cell.sh" "$ds" "$seed"
  sleep 2
done
echo "ALL CELLS ATTEMPTED"
grep -E "###|Final_Acc|exit=" "$LOG" | tail -30
