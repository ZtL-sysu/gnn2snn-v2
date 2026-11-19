#!/usr/bin/env python3
"""E-B3: complete the Squirrel held-out threshold cell (M-B3 repair).

Fixes the WP15 design flaw: train each seed's ANN ONCE (7 trainings), then sweep theta per ANN
(conversion is cheap) — 7 trainings instead of 49. Emits per-seed registry rows for the JA4 replay
chain. Same THETAS / T / protocol as `wp15_confirm.py`; no claim depends on this cell completing
(overrun -> infeasible-with-evidence with partial rows kept).

Output: results/jB3_squirrel_complete.json (+ per-seed rows in the file).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PRIOR = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
             r"f-research-v1-run-20261028-174053-gnn-snn-conv")
RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
           r"f-research-v1-run-20261118-181848-journal-upgrade")
sys.path.insert(0, str(PRIOR / "experiments"))
sys.path.insert(0, str(PRIOR / "registry"))
import core  # noqa: E402

T = 8
SEEDS = 7
THETAS = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
DS = "squirrel"


def q999(model):
    b = np.concatenate([np.ravel(x) for x in model["Bs"]])
    return float(np.quantile(np.abs(b), 0.999))


def main():
    t0 = time.time()
    out = {"stage": "E-B3", "dataset": DS, "T": T, "seeds": SEEDS, "thetas": THETAS,
           "design_fix": "train-once-per-seed, sweep theta per ANN (7 trainings, not 49)",
           "rows": [], "theta_scan": {}, "adaptive": {}}
    data = core.load(DS, root=str(PRIOR / "results" / ".data"))
    models = {}
    for sd in range(SEEDS):
        models[sd] = core.train_ann(data, hidden=16, n_layers=2, seed=sd)
        print(f"trained seed {sd} ({time.time()-t0:.0f}s)", flush=True)
    for th in THETAS:
        accs, spikes = [], []
        for sd in range(SEEDS):
            r = core.convert(data, models[sd], T=T, theta=th)
            accs.append(r["acc"]); spikes.append(r["spikes_per_node"])
            out["rows"].append({"dataset": DS, "method": f"theta{th}", "t_steps": T,
                                "hidden": 16, "seed": sd, "ann_ref": models[sd]["acc_test"],
                                "acc": r["acc"],
                                "gap_pp": (r["acc"] - models[sd]["acc_test"]) * 100,
                                "spikes": r["spikes_per_node"], "theta": th})
        out["theta_scan"][str(th)] = {"acc": float(np.mean(accs)), "std": float(np.std(accs)),
                                      "spikes": float(np.mean(spikes))}
        out["q999"] = float(np.mean([q999(models[sd]) for sd in range(SEEDS)]))
        print(f"theta {th}: acc {np.mean(accs):.4f} +- {np.std(accs):.4f}", flush=True)
    ada, fixed = [], []
    for sd in range(SEEDS):
        th = q999(models[sd])
        ra = core.convert(data, models[sd], T=T, theta=th)
        rf = core.convert(data, models[sd], T=T, theta=1.0)
        ada.append(ra["acc"]); fixed.append(rf["acc"])
        out["rows"].append({"dataset": DS, "method": "adaptive_theta", "seed": sd,
                            "acc": ra["acc"], "theta_used": th, "fixed_acc": rf["acc"]})
    out["adaptive"] = {"adaptive": float(np.mean(ada)), "fixed": float(np.mean(fixed)),
                       "delta_pp": float((np.mean(ada) - np.mean(fixed)) * 100)}
    out["wall_seconds"] = round(time.time() - t0, 1)
    out["status"] = "complete" if len(out["theta_scan"]) == len(THETAS) else "infeasible-with-evidence"
    (RUN / "results" / "jB3_squirrel_complete.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("status:", out["status"], f"wall={out['wall_seconds']}s")


if __name__ == "__main__":
    main()
