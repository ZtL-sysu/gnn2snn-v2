#!/usr/bin/env python3
"""JB1a arxiv phase (pre-registered config #3). Same frozen procedure: c is taken verbatim from
results/jB1a_select.json (calibration done on cora/citeseer only); one-shot T_cert on ogbn-arxiv;
reporting-only integer sweep afterwards. Output merges into results/jB1a_select.json as an
append-only arxiv row + recomputed verdict."""
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

BAND = 0.015
SEEDS = 7


def delta_med_train(data, model):
    H_act = model["Hs"][-1]
    adj = data["adj"]
    B2 = np.asarray(adj @ (H_act @ model["Ms"][-1]).astype(np.float32)) + model["b_out"]
    B2 = B2[data["train"]]
    part = np.partition(-B2, 1, axis=1)
    return float(np.median(-part[:, 0] + part[:, 1]))


def main():
    sel = json.loads((RUN / "results" / "jB1a_select.json").read_text(encoding="utf-8"))
    c_frozen = float(sel["c_frozen"])
    print("c_frozen =", c_frozen, flush=True)
    data = core.load("ogbn-arxiv", root=str(PRIOR / "results" / ".data"))
    print("arxiv loaded N=%d F=%d" % (data["N"], data["F"]), flush=True)
    models = [core.train_ann(data, hidden=16, n_layers=2, seed=sd) for sd in range(SEEDS)]
    print("trained %d ANNs" % SEEDS, flush=True)
    m0 = models[0]
    dm = delta_med_train(data, m0)
    G = float(np.linalg.norm(m0["Ms"][-1], 2))
    H_act = m0["Hs"][-1]
    H_active = float((H_act > 0).sum() / max(H_act.shape[0], 1))
    T_cert = int(np.clip(int(np.ceil(c_frozen * G * np.sqrt(H_active) / dm)), 1, 64)) if dm > 0 else 64
    accs = [core.convert(data, m, T=T_cert)["acc"] for m in models]
    anns = [m["acc_test"] for m in models]
    band_hit = float(np.mean(accs)) >= float(np.mean(anns)) - BAND
    ann0 = m0["acc_test"]
    t_star = None
    for T in range(1, 65):
        if core.convert(data, m0, T=T)["acc"] >= ann0 - BAND:
            t_star = T
            break
    row = {"dataset": "ogbn-arxiv", "heldout": True, "T_cert": T_cert,
           "acc_mean": float(np.mean(accs)), "ann_mean": float(np.mean(anns)),
           "band_hit": band_hit, "T_star_report": t_star,
           "kappa": (T_cert / t_star) if t_star else None, "seeds": SEEDS}
    sel["one_shot"].append(row)
    held = [r for r in sel["one_shot"] if r["heldout"]]
    reach = [r for r in held if r["T_star_report"] is not None]
    hit = [r for r in reach if r["band_hit"]]
    kappas = [r["kappa"] for r in reach if r["kappa"]]
    strong = (len(reach) > 0 and len(hit) >= (len(reach) + 1) // 2 and
              all(k <= 1.25 for k in kappas) and all(r["T_cert"] <= 64 for r in reach))
    minimum = len(hit) >= (len(reach) + 1) // 2 and all(k <= 2 for k in kappas)
    sel["verdict"] = {"heldout_reachable": len(reach), "band_hits": len(hit), "kappas": kappas,
                      "strong_pass": bool(strong), "minimum_pass": bool(minimum),
                      "status": "planned-claim-result; reported once after rule freeze (incl. arxiv phase)"}
    (RUN / "results" / "jB1a_select.json").write_text(
        json.dumps(sel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("arxiv row:", row, flush=True)
    print("verdict:", sel["verdict"], flush=True)


if __name__ == "__main__":
    main()
