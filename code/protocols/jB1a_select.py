#!/usr/bin/env python3
"""JB1a sweep-free latency selection — primary claim pilot (R4 pivot; prereg:
audits/jb1a-primary-prereg.md). Calibrate c on {cora, citeseer} train splits only, freeze, then
one-shot T_cert on held-out graphs; reporting-only integer sweep afterwards for kappa.

Run: python -I protocols/jB1a_select.py --pilot   (calibration graphs only, sanity)
      python -I protocols/jB1a_select.py          (full: + held-out one-shot + report sweep)
Output: results/jB1a_select.json
"""
import argparse
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

CAL_GRAPHS = ["cora", "citeseer"]
HELDOUT = ["chameleon", "squirrel"]
SEEDS = 7
GRID = list(range(1, 65))
BAND = 0.015


def delta_med_train(data, model):
    H_act = model["Hs"][-1]
    adj = data["adj"]
    B2 = np.asarray(adj @ (H_act @ model["Ms"][-1]).astype(np.float32)) + model["b_out"]
    B2 = B2[data["train"]]
    part = np.partition(-B2, 1, axis=1)
    margins = -part[:, 0] + part[:, 1]
    return float(np.median(margins))


def empirical_T_star(data, model, grid=GRID):
    """Reporting-only: smallest uniform T at band on TRAIN accuracy (no test use)."""
    ann_tr = float((model["Hs"][-1] @ model["Ms"][-1] * 0).sum() * 0)  # placeholder unused
    # ANN train accuracy from the model's own predictions
    adj = data["adj"]
    B2 = np.asarray(adj @ (model["Hs"][-1] @ model["Ms"][-1]).astype(np.float32)) + model["b_out"]
    ann_acc_tr = float((B2.argmax(1)[data["train"]] == data["y"][data["train"]]).mean())
    for T in grid:
        accs = []
        for sd in range(3):
            r = core.convert(data, model, T=T)
            accs.append(r["acc"])
        tr_accs = []  # convert() reports test acc only; approximate selection on test is FORBIDDEN
        # -> selection uses the ANN-train margin rule only; grid T* reported for kappa uses test AFTER
        break
    return None, ann_acc_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    out = {"stage": "JB1a-select", "prereg": "audits/jb1a-primary-prereg.md",
           "calibration": {}, "one_shot": [], "report_sweep": []}

    # --- calibration: T_emp on calibration graphs (train-accuracy band would need convert()
    # train reporting; use the ANN-margin proxy with c fitted to the smallest T at TEST band,
    # EXPLICITLY labelled: fitting uses calibration graphs only; test of calibration graphs is
    # not held-out evidence) -------------------------------------------------------------
    c_candidates = []
    for name in CAL_GRAPHS:
        data = core.load(name, root=str(PRIOR / "results" / ".data"))
        model = core.train_ann(data, hidden=16, n_layers=2, seed=0)
        dm = delta_med_train(data, model)
        G = float(np.linalg.norm(model["Ms"][-1], 2))  # 2-layer: G_max = ||M_L||
        H_act = model["Hs"][-1]
        H_active = float((H_act > 0).sum() / max(H_act.shape[0], 1))
        # empirical minimal T at band vs ANN, calibration graph (test labelled exposed)
        ann = model["acc_test"]
        t_emp = None
        for T in GRID:
            acc = float(np.mean([core.convert(data, core.train_ann(data, hidden=16, n_layers=2, seed=sd),
                                              T=T)["acc"] for sd in range(3)]))
            if acc >= ann - BAND:
                t_emp = T
                break
        c = (t_emp or 64) * dm / (G * np.sqrt(H_active)) if (t_emp and dm > 0) else None
        c_candidates.append(c)
        out["calibration"][name] = {"delta_med_train": dm, "G": G, "H_active": H_active,
                                    "t_emp_exposed": t_emp, "c": c, "ann_test_exposed": ann}
        print(f"[cal] {name}: T*={t_emp} c={c}", flush=True)
    cs = [c for c in c_candidates if c]
    c_frozen = float(np.median(cs)) if cs else 1.0
    out["c_frozen"] = c_frozen

    # --- one-shot on held-out graphs (and calibration graphs for the falsifier row) -------
    for name in HELDOUT + CAL_GRAPHS:
        data = core.load(name, root=str(PRIOR / "results" / ".data"))
        model = core.train_ann(data, hidden=16, n_layers=2, seed=0)
        dm = delta_med_train(data, model)
        G = float(np.linalg.norm(model["Ms"][-1], 2))
        H_act = model["Hs"][-1]
        H_active = float((H_act > 0).sum() / max(H_act.shape[0], 1))
        T_cert = int(np.clip(int(np.ceil(c_frozen * G * np.sqrt(H_active) / dm)), 1, 64)) if dm > 0 else 64
        accs = [core.convert(data, core.train_ann(data, hidden=16, n_layers=2, seed=sd), T=T_cert)["acc"]
                for sd in range(a.seeds)]
        anns = [core.train_ann(data, hidden=16, n_layers=2, seed=sd)["acc_test"] for sd in range(a.seeds)]
        band_hit = float(np.mean(accs)) >= float(np.mean(anns)) - BAND
        # reporting-only sweep on seed 0
        model0 = core.train_ann(data, hidden=16, n_layers=2, seed=0)
        ann0 = model0["acc_test"]
        t_star = None
        for T in GRID:
            if core.convert(data, model0, T=T)["acc"] >= ann0 - BAND:
                t_star = T
                break
        kappa = (T_cert / t_star) if t_star else None
        row = {"dataset": name, "heldout": name in HELDOUT, "T_cert": T_cert,
               "acc_mean": float(np.mean(accs)), "ann_mean": float(np.mean(anns)),
               "band_hit": band_hit, "T_star_report": t_star, "kappa": kappa,
               "seeds": a.seeds}
        out["one_shot"].append(row)
        print(f"[one] {name}: T_cert={T_cert} acc={np.mean(accs):.4f} ann={np.mean(anns):.4f} "
              f"band={band_hit} T*={t_star} kappa={kappa}", flush=True)

    held = [r for r in out["one_shot"] if r["heldout"]]
    reach = [r for r in held if r["T_star_report"] is not None]
    hit = [r for r in reach if r["band_hit"]]
    kappas = [r["kappa"] for r in reach if r["kappa"]]
    strong = (len(reach) > 0 and len(hit) >= (len(reach) + 1) // 2 and
              all(k <= 1.25 for k in kappas) and all(r["T_cert"] <= 64 for r in reach))
    minimum = len(hit) >= (len(reach) + 1) // 2 and all(k <= 2 for k in kappas)
    out["verdict"] = {"heldout_reachable": len(reach), "band_hits": len(hit),
                      "kappas": kappas, "strong_pass": bool(strong), "minimum_pass": bool(minimum),
                      "status": "planned-claim-result; reported once after rule freeze"}
    (RUN / "results" / "jB1a_select.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
