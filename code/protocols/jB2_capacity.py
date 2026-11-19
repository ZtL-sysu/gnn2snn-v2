#!/usr/bin/env python3
"""E-B2: capacity-aligned external formulation baseline (JB3, planned claim).

Runs the SpikingGCN-formulation arm (SGC K-step propagation + one spiking layer) at matched
capacity K=2, H in {16, 32}, same split/7-seed protocol as the registered 512-wide arm, so the
capacity confound named in guidance-v3 deferred item 3 is measured rather than argued.
Neutral framing: tests whether the external lead is a capacity effect. Any outcome is reportable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PRIOR = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
             r"f-research-v1-run-20261028-174053-gnn-snn-conv")
RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
           r"f-research-v1-run-20261118-181848-journal-upgrade")
sys.path.insert(0, str(PRIOR / "protocols"))
sys.path.insert(0, str(PRIOR / "experiments"))
import graph_data as G  # noqa: E402
import snn_gnn_experiment as E  # noqa: E402
import spikinggcn_baseline as SB  # noqa: E402

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"stage": "E-B2-capacity-match", "device": dev, "K": 2,
           "protocol": "external formulation under our protocol; not a published-number reproduction",
           "cells": []}
    for name in ["cora", "citeseer", "pubmed"]:
        d = G.load_ds(name, str(PRIOR / "results" / ".data"))
        adj = E.normalize_adj(d["adj"]).tocsr()
        X = d["features"].astype(np.float32).copy()
        if d["row_normalize"]:
            rs = X.sum(1); rs[rs == 0] = 1.0; X = X / rs[:, None]
        Xp = SB.propagate(adj, X, 2)
        mx = Xp.max()
        if mx > 1:
            Xp = Xp / mx
        tr = torch.tensor(np.where(d["train_mask"])[0], dtype=torch.long)
        va = torch.tensor(np.where(d["val_mask"])[0], dtype=torch.long)
        te = np.where(d["test_mask"])[0]
        C = int(d["labels"].max()) + 1
        for hidden in [16, 32]:
            for T in [4, 8, 16]:
                accs = []
                for sd in range(7):
                    acc, _ = SB.train_sgc_snn(Xp, d["labels"], tr, va, te, C, T, sd,
                                              hidden=hidden, dev=dev)
                    accs.append(acc)
                rec = {"dataset": name, "hidden": hidden, "T": T,
                       "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                       "acc_per_seed": accs}
                out["cells"].append(rec)
                print(f"[{name}] K=2 H={hidden} T={T} acc={rec['acc_mean']:.4f}", flush=True)
    (RUN / "results" / "jB2_capacity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", RUN / "results" / "jB2_capacity.json")

if __name__ == "__main__":
    main()
