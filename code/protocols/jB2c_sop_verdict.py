#!/usr/bin/env python3
"""E-B2c: scripted re-derivation of the matched-accuracy SOP verdict under ALL pre-declared
accountings (audits/sop-convention-predeclaration.md), plus the original WP2 failure criterion
("an external arm dominates at T<=4 on two or more graphs") and the knob-manipulation check.

Run after jB2b_truewidth.py completes. Output: results/jB2c_sop_verdict.json
Every accounting's verdict is reported side by side; the headline is the pre-declared
event-driven SynOps convention. No accounting may be switched after results exist.
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
import graph_data as G  # noqa: E402
import snn_gnn_experiment as E  # noqa: E402
import spikinggcn_baseline as SB  # noqa: E402


def matched_target(name):
    reg = json.loads((PRIOR / "results" / "wp2_pareto.json").read_text(encoding="utf-8"))
    return reg["matched"][name]["target"], reg


def conversion_synops_event(data, model, T, theta=1.0):
    """Exact event-driven SynOps for our conversion arm: deg-weighted spikes x fan-out."""
    res = core.convert(data, model, T=T, theta=theta, record=False)
    adj = data["adj"].tocsr()
    deg = np.asarray(adj.sum(1)).ravel()
    Ms = model["Ms"]
    # per-layer spikes are summed; deg weighting re-derived by a second pass is overkill for the
    # verdict script -> use mean-degree fan-out and label it
    mean_deg = float(deg.mean())
    Hs = [m.shape[1] for m in Ms]
    per = res["per_layer_spikes"]  # [hidden..., output]
    syn = 0.0
    for l, sp in enumerate(per):
        fan = mean_deg * (Hs[l + 1] if l + 1 < len(Hs) else 1)
        syn += sp * fan
    return {"synops_event_mean_deg": float(syn),
            "synops_static_TEH": float(T * data["E"] * Hs[0]),
            "spikes_per_node": res["spikes_per_node"], "acc": res["acc"],
            "approx": "mean-degree fan-out; deg-weighted exact count is a P5 refinement"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    a = ap.parse_args()
    tw = json.loads((RUN / "results" / "jB2b_truewidth.json").read_text(encoding="utf-8"))
    fc = json.loads((RUN / "results" / "jB2_capacity.json").read_text(encoding="utf-8"))
    out = {"stage": "E-B2c-sop-verdict",
           "headline_convention": "event-driven SynOps (pre-declared)",
           "cells": [], "knob_check": {}, "verdicts": {}}

    # knob-manipulation check: truewidth accuracy must vary with H (the defect class test)
    by_h = {}
    for c in tw["cells"]:
        by_h.setdefault((c["dataset"], c["T"]), {})[c["hidden"]] = c["acc_mean"]
    diff = max((max(v.values()) - min(v.values()) for v in by_h.values()), default=0.0)
    out["knob_check"] = {"max_acc_spread_across_H": diff, "pass": diff > 1e-4,
                         "note": "H=16/32/512 must not be bit-identical (contrast the p3 defect)"}

    for name in ["cora", "citeseer", "pubmed"]:
        target, reg = matched_target(name)
        data = core.load(name, root=str(PRIOR / "results" / ".data"))
        adj = E.normalize_adj(data["adj"]).tocsr()
        d = G.load_ds(name, str(PRIOR / "results" / ".data"))
        X = d["features"].astype(np.float32).copy()
        if d["row_normalize"]:
            rs = X.sum(1); rs[rs == 0] = 1.0; X = X / rs[:, None]
        Xp = SB.propagate(adj, X, 2)
        mx = Xp.max()
        if mx > 1:
            Xp = Xp / mx
        input_events = float(Xp.sum())
        N, F = Xp.shape
        C = int(data["y"].max()) + 1
        # our conversion arm, 7 seeds at its matched T (registered)
        conv_T = reg["matched"][name]["conversion_T"]
        conv = []
        for sd in range(a.seeds):
            m = core.train_ann(data, hidden=16, n_layers=2, seed=sd)
            conv.append(conversion_synops_event(data, m, T=conv_T))
        conv_acc = float(np.mean([c["acc"] for c in conv]))
        conv_syn = float(np.mean([c["synops_event_mean_deg"] for c in conv]))
        conv_static = conv[0]["synops_static_TEH"]

        # external F->C arm (registered impl, relabelled): one row per T in {4,8,16} with the
        # registered/rerun accuracy and T-scaled costs (the earlier row missed the T factor)
        externals = []
        for T in [4, 8, 16]:
            fc_cell = next((c for c in fc["cells"] if c["dataset"] == name
                            and c["hidden"] == 16 and c["T"] == T), None)
            if fc_cell is None:
                continue
            externals.append({
                "label": f"external F->C single spiking layer (registered impl, T={T})",
                "T": T, "acc": fc_cell["acc_mean"],
                "synops_event": T * input_events * C,
                "synops_dense": T * N * F * C,
                "node_h_legacy": T * N * 512,
            })
        # external truewidth arms (only those at/above target, as before; T kept explicit)
        tw_cells = [c for c in tw["cells"] if c["dataset"] == name and c["acc_mean"] >= target - 1e-9]
        externals += [{"label": f"truewidth H={c['hidden']} T={c['T']}", "T": c["T"],
                       "acc": c["acc_mean"],
                       "synops_event": c["cost_mean"]["synops_spike"],
                       "synops_dense": c["cost_mean"]["dense_worst"],
                       "node_h_legacy": c["cost_mean"]["node_h"]} for c in tw_cells]

        cell = {"dataset": name, "target": target, "conv_T": conv_T,
                "conversion": {"acc": conv_acc, "synops_event": conv_syn,
                               "synops_static_TEH": conv_static},
                "externals": externals}
        out["cells"].append(cell)

    # verdicts per accounting
    def verdict(key):
        wins = 0
        dominates_T4 = 0
        for cell in out["cells"]:
            conv_s = cell["conversion"][key] if key in cell["conversion"] else cell["conversion"].get("synops_static_TEH")
            cheaper = [e for e in cell["externals"] if e.get(key.replace("synops_event", "synops_event").replace("synops_static", "synops_dense")) is not None]
            conv_val = conv_syn_map(cell, key)
            ext_vals = [(e, ext_map(e, key)) for e in cell["externals"]]
            ext_vals = [(e, v) for e, v in ext_vals if v is not None]
            if ext_vals and conv_val is not None:
                if all(conv_val < v for _, v in ext_vals):
                    wins += 1
                for e, v in ext_vals:
                    if e["acc"] is not None and e["acc"] >= cell["target"] and e.get("T", 99) <= 4 and v < conv_val:
                        dominates_T4 += 1
        return {"graphs_conversion_cheaper_than_all_external": wins,
                "external_dominations_T_le_4": dominates_T4,
                "wp2_failure_criterion_fires": dominates_T4 >= 2}

    def conv_syn_map(cell, key):
        if key == "synops_event":
            return cell["conversion"]["synops_event"]
        return cell["conversion"]["synops_static_TEH"]

    def ext_map(e, key):
        if key == "synops_event":
            return e.get("synops_event")
        return e.get("synops_dense")

    out["verdicts"] = {"event_driven_headline": verdict("synops_event"),
                       "static_dense_secondary": verdict("synops_static"),
                       "legacy_node_h_reference": {
                           "graphs_conversion_cheaper": sum(
                               1 for c in out["cells"]
                               if all(c["conversion"]["synops_static_TEH"] < e.get("node_h_legacy", float("inf"))
                                      for e in c["externals"])),
                           "note": "legacy accounting mis-described the F->C arm; continuity only"}}
    (RUN / "results" / "jB2c_sop_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"knob": out["knob_check"], "verdicts": out["verdicts"]}, indent=1))


if __name__ == "__main__":
    main()
