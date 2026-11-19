#!/usr/bin/env python3
"""P3 baseline applicability + reproduction recheck (Paper B registered arms).

Re-runs a small slice of the registered conversion arm through the protected prior-run engine and
compares against `results/wp2_pareto.json`; records input hashes for applicability. Read-only
against the prior run; outputs land in this run root.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PRIOR = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
             r"f-research-v1-run-20261028-174053-gnn-snn-conv")
RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
           r"f-research-v1-run-20261118-181848-journal-upgrade")
sys.path.insert(0, str(PRIOR / "protocols"))
sys.path.insert(0, str(PRIOR / "experiments"))
import core  # noqa: E402

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

reg = json.loads((PRIOR / "results" / "wp2_pareto.json").read_text(encoding="utf-8"))
out = {"stage": "P3-baseline-recheck",
       "applicability": {p: sha(PRIOR / "results" / p)
                         for p in ["wp2_pareto.json", "baselines_s7.json", "wp6a_misalignment.json"]},
       "cells": []}

HID = 16
for name, ts in [("cora", [4, 8]), ("citeseer", [4])]:
    data = core.load(name, root=str(PRIOR / "results" / ".data"))
    for T in ts:
        accs = []
        for sd in range(3):
            m = core.train_ann(data, hidden=HID, n_layers=2, seed=sd)
            r = core.convert(data, m, T=T)
            accs.append(r["acc"])
        got = float(np.mean(accs))
        want = float(reg["arms"]["conversion"][name]["acc"][str(T)])
        delta = got - want
        out["cells"].append({"dataset": name, "T": T, "seeds": 3, "rerun_mean": got,
                             "registered_mean": want, "delta_pp": delta * 100,
                             "registered_std": reg["arms"]["conversion"][name]["std"][str(T)],
                             "within_2x_std": abs(delta) <= 2 * reg["arms"]["conversion"][name]["std"][str(T)] + 1e-12})
        print(name, T, f"rerun={got:.4f} reg={want:.4f} delta={delta*100:+.2f}pp", flush=True)

out["verdict"] = ("pass" if all(c["within_2x_std"] for c in out["cells"]) else "review")
(RUN / "results" / "jB_baseline_recheck.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("verdict:", out["verdict"])
