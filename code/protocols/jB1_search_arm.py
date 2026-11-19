#!/usr/bin/env python3
"""LD-SNN-style search-based per-layer allocation arm (prereg: audits/jB1-search-arm-prereg.md).
Validation-only coordinate descent over per-layer windows; one-shot test evaluation. Prior-art
family attribution: LD-SNN (ISLPED 2022); graph instantiation measured here."""
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
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("jB1e", RUN / "protocols" / "jB1_engine.py")
jB1e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jB1e)
convert_staged = jB1e.convert_staged

BAND = 0.015
GRID = [1, 2, 3, 4, 6, 8, 12, 16]


def val_acc(data, model, ts):
    r = convert_staged(data, model, ts)
    # staged engine reports test acc; recompute val acc from its semantics via core.convert on val
    # -- convert_staged returns only test acc, so evaluate val here directly
    X, adj = data["X"], data["adj"]
    Ms = model["Ms"]
    L1 = model["n_layers"] - 1
    thetas = [1.0] * L1
    const = core._const_terms(data, model)
    rate = np.asarray(X, dtype=np.float32)
    for l in range(L1):
        T_l = max(1, int(ts[l]))
        beta = thetas[l] / (2.0 * T_l)
        if l == 0:
            inst = (np.asarray(adj @ (rate @ Ms[0]).astype(np.float32))
                    + model["bs_layers"][0]).astype(np.float32) + np.float32(beta)
        else:
            inst = (np.asarray(adj @ (rate @ Ms[l]).astype(np.float32))
                    + const[l]).astype(np.float32) + np.float32(beta)
        u = np.zeros((rate.shape[0], Ms[l].shape[1]), np.float32)
        cnt = np.zeros((rate.shape[0], Ms[l].shape[1]), np.int32)
        for _ in range(T_l):
            u += inst
            s = (u >= thetas[l]).astype(np.int8)
            u -= s * thetas[l]
            cnt += s
        rate = (cnt / float(T_l)).astype(np.float32)
    logits = np.asarray(adj @ (rate @ Ms[-1]).astype(np.float32)) + model["b_out"]
    return float((logits.argmax(1)[data["val"]] == data["y"][data["val"]]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--L", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tag", default="pilot")
    a = ap.parse_args()
    out = {"stage": "jB1-search-arm", "prereg": "audits/jB1-search-arm-prereg.md",
           "tag": a.tag, "cells": []}
    for name in a.datasets:
        data = core.load(name, root=str(PRIOR / "results" / ".data"))
        for n_layers in a.L:
            L1 = n_layers - 1
            for sd in a.seeds:
                m = core.train_ann(data, hidden=16, n_layers=n_layers, seed=sd)
                # ANN val accuracy from the model's own logits (the search target)
                B2 = np.asarray(data["adj"] @ (m["Hs"][-1] @ m["Ms"][-1]).astype(np.float32)) + m["b_out"]
                ann_val = float((B2.argmax(1)[data["val"]] == data["y"][data["val"]]).mean())
                T_star = None
                for T in range(1, 65):
                    if val_acc(data, m, [T] * L1) >= ann_val - BAND:
                        T_star = T
                        break
                T_star = T_star or 64
                # coordinate descent on VAL at fixed budget
                ts = [T_star] * L1
                best_val = val_acc(data, m, ts)
                for _ in range(2):
                    for l in range(L1):
                        for cand in GRID:
                            if cand == ts[l]:
                                continue
                            trial = ts.copy(); trial[l] = cand
                            v = val_acc(data, m, trial)
                            if v >= best_val - 1e-12 and sum(trial) < sum(ts):
                                ts, best_val = trial, v
                r_search = convert_staged(data, m, ts)
                r_unif = convert_staged(data, m, [T_star] * L1)
                # random-search falsifier: 8 random allocations, best val kept
                rng = np.random.default_rng(sd)
                rts, rb = ts, -1.0
                for _ in range(8):
                    trial = [int(rng.choice(GRID)) for _ in range(L1)]
                    v = val_acc(data, m, trial)
                    if v > rb:
                        rts, rb = trial, v
                r_rand = convert_staged(data, m, rts)
                saving = 1.0 - r_search["layer_updates"] / r_unif["layer_updates"]
                cell = {"dataset": name, "L": n_layers, "seed": sd, "T_star": T_star,
                        "T_l_search": ts, "T_l_random": rts,
                        "acc_search": r_search["acc"], "acc_unif": r_unif["acc"],
                        "acc_random": r_rand["acc"],
                        "updates_search": r_search["layer_updates"],
                        "updates_unif": r_unif["layer_updates"],
                        "saving": saving,
                        "band_ok": r_search["acc"] >= r_unif["acc"] - BAND,
                        "beats_random": r_search["acc"] > r_rand["acc"] + 1e-12 or
                                        (abs(r_search["acc"] - r_rand["acc"]) <= 1e-12 and
                                         r_search["layer_updates"] < r_rand["layer_updates"])}
                out["cells"].append(cell)
                print(f"[{name}] L={n_layers} sd={sd} T*={T_star} T_l={ts} "
                      f"save={100*saving:.1f}% acc {r_search['acc']:.4f} vs {r_unif['acc']:.4f} "
                      f"(rand {r_rand['acc']:.4f})", flush=True)
    cells = out["cells"]
    win = [c for c in cells if c["band_ok"] and c["saving"] >= 0.20]
    out["verdict"] = {"n": len(cells), "band_and_20pct": len(win),
                      "fraction": len(win) / max(len(cells), 1),
                      "arm_success": len(win) >= (len(cells) * 2 + 2) // 3,
                      "arm_partial": len(win) * 2 >= len(cells),
                      "random_falsifier_ok": sum(1 for c in cells if c["beats_random"]) >= (len(cells) + 1) // 2}
    (RUN / "results" / f"jB1_search_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
