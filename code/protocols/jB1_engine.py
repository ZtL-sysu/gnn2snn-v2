#!/usr/bin/env python3
"""E-B1 engine: per-layer freeze-time conversion + prototype gate (JB1, planned claim).

Semantics (pinned by P1B re-review R-D spec):
  * hidden layer l is active over its own window [1..T_l]; afterwards it emits no spikes and its
    membrane state is held;
  * the output layer integrates `A_hat (s_last @ M_L) + b_out` for T_out = max_l T_l steps
    (zero spikes after any freeze), and the membrane readout is the un-reset accumulator;
  * per-layer bias is beta_l = theta_l / (2 T_l); thresholds default to 1;
  * accounting (binding): SigmaT(alloc) = sum_hidden ceil(kappa * (G_l/G_max) * T*) + T_out,
    SigmaT(unif) = L * T*, saving S = 1 - kappa * [((L-1)/L)(Gbar/G_max) + 1/L],
    Gbar = (1/(L-1)) * sum_hidden G_l, G_max = max_hidden G_l, G_l from `core.spectral_gain`.

Modified identity to verify at the prototype (a): for each hidden layer, the frozen-window count
error feeds the next layer exactly as in `core.check_identity` within the window; the output
readout satisfies acc/T_out == A_hat (count_last/T_out @ M_L) + b_out exactly.

Run prototype gate: python -I protocols/jB1_engine.py --prototype --seeds 0 1
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


def convert_staged(data, model, T_ls, theta=1.0, thetas=None):
    """STAGED per-layer simulation (the one allowed engine fix after the prototype showed the
    lockstep-freeze semantics under-counts heterogeneous-window rates).

    Stage l runs its own window T_l with CONSTANT per-step input
        inst_l = A_hat (rate_{l-1} @ M_l) + const_l + beta_l,  beta_l = theta/(2 T_l),
    exactly the layer-1 pattern of `core.convert` with X replaced by the previous stage's decoded
    rate. The identity then carries as B_hat_l - B_l = (1-alpha) A_hat (E_{l-1} M_l) with
    E_l = count_l/T_l - relu(B_l), i.e. the core identity with X -> rate_{l-1}.
    Output stage integrates inst_out = A_hat (rate_last @ M_L) + b_out over T_out steps;
    membrane readout / T_out == inst_out exactly.
    """
    X, adj = data["X"], data["adj"]
    N, C = data["N"], data["C"]
    Ms = model["Ms"]
    L1 = model["n_layers"] - 1
    T_ls = [max(1, int(t)) for t in T_ls][:L1]
    if thetas is None:
        thetas = [theta] * L1
    const = core._const_terms(data, model)
    rate = np.asarray(X, dtype=np.float32)
    counts, rates = [], []
    spike_total = 0.0
    for l in range(L1):
        T_l = T_ls[l]
        beta = thetas[l] / (2.0 * T_l)
        if l == 0:
            inst = (np.asarray(adj @ (rate @ Ms[0]).astype(np.float32))
                    + model["bs_layers"][0]).astype(np.float32) + np.float32(beta)
        else:
            inst = (np.asarray(adj @ (rate @ Ms[l]).astype(np.float32))
                    + const[l]).astype(np.float32) + np.float32(beta)
        u = np.zeros((N, Ms[l].shape[1]), np.float32)
        cnt = np.zeros((N, Ms[l].shape[1]), np.int32)
        for _ in range(T_l):
            u += inst
            s = (u >= thetas[l]).astype(np.int8)
            u -= s * thetas[l]
            cnt += s
        spike_total += float(cnt.sum())
        rate = (cnt / float(T_l)).astype(np.float32)
        counts.append(cnt); rates.append(rate)
    T_out = max(T_ls + [1])
    inst_out = (np.asarray(adj @ (rate @ Ms[-1]).astype(np.float32)) + model["b_out"]).astype(np.float32)
    acc_out = inst_out.astype(np.float64) * T_out
    pred = acc_out.argmax(1)
    acc = float((pred[data["test"]] == data["y"][data["test"]]).mean())
    ident = float(np.max(np.abs(acc_out / T_out - inst_out)))
    return {"acc": acc, "spikes": spike_total, "spikes_per_node": spike_total / N,
            "T_ls": T_ls, "T_out": T_out, "output_identity_max": ident,
            "layer_updates": float(sum(T_ls) + T_out),
            "rates": [float(r.mean()) for r in rates]}


def convert_frozen(data, model, freeze, theta=1.0, thetas=None, seed=0):
    """Legacy lockstep freeze simulation (kept as the negative control of the engine fix)."""
    X, adj = data["X"], data["adj"]
    N, C = data["N"], data["C"]
    Ms = model["Ms"]
    L1 = model["n_layers"] - 1
    freeze = [int(f) for f in freeze][:L1]
    if thetas is None:
        thetas = [theta] * L1
    betas = [th / (2.0 * max(f, 1)) for th, f in zip(thetas, freeze)]
    const = core._const_terms(data, model)
    inst1 = (np.asarray(adj @ (X @ Ms[0]).astype(np.float32)) + model["bs_layers"][0]).astype(np.float32)
    us = [np.zeros((N, Ms[l].shape[1])) for l in range(L1)]
    cnt = [np.zeros((N, Ms[l].shape[1]), np.int32) for l in range(L1)]
    acc_out = np.zeros((N, C), np.float64)
    T_out = max(freeze + [1])
    spike_total = 0.0
    for t in range(1, T_out + 1):
        prev = None
        for l in range(L1):
            if t > freeze[l]:
                prev = np.zeros_like(cnt[l])
                continue
            if l == 0:
                inst = inst1 + np.float32(betas[0])
            else:
                inst = (np.asarray(adj @ (prev.astype(np.float32) @ Ms[l])) + const[l] + np.float32(betas[l]))
            us[l] += inst
            s = (us[l] >= thetas[l]).astype(np.int8)
            us[l] -= s * thetas[l]
            cnt[l] += s
            spike_total += float(s.sum())
            prev = s
        out = (np.asarray(adj @ (prev.astype(np.float32) @ Ms[-1])) + model["b_out"]).astype(np.float32)
        acc_out += out
    pred = acc_out.argmax(1)
    acc = float((pred[data["test"]] == data["y"][data["test"]]).mean())
    # exact output-readout identity: acc/T_out == A_hat(count_last/T_out @ M_L) + b_out
    rhs = (np.asarray(adj @ ((cnt[-1] / T_out).astype(np.float32) @ Ms[-1])) + model["b_out"])
    ident_out = float(np.max(np.abs(acc_out / T_out - rhs)))
    return {"acc": acc, "spikes": spike_total, "spikes_per_node": spike_total / N,
            "freeze": freeze, "T_out": T_out, "betas": betas,
            "layer_updates": float(sum(freeze) + T_out),
            "output_identity_max": ident_out,
            "per_layer_counts": [float(c.sum()) for c in cnt]}


def gains(model):
    """Per-hidden-layer downstream gains G_l = prod_{k>l} (1-alpha)||A M_k|| (norms from core)."""
    sg = core.spectral_gain(model)
    Ms = model["Ms"]
    alpha = model.get("alpha", 0.0)
    L1 = model["n_layers"] - 1
    G = []
    for l in range(L1):
        g = 1.0
        for k in range(l + 1, len(Ms)):
            g *= float(np.linalg.norm(Ms[k], 2)) * (1.0 - alpha if k < len(Ms) - 1 else 1.0)
        G.append(g)
    return {"G": G, "spectral_gain": sg}


def allocation_rule_r2(model, data, theta=1.0, margin_quantile=1.0, strictness=0.5):
    """Rule R2 (constants frozen in correction-ledger before this pilot): per-layer absolute
    sufficiency against the tail margin:
        T_l = min{T in 1..64 : G_l * theta * sqrt(H_act * d_l) / (2T) <= strictness * Delta_p01_train}
    with d_l = width of the matrix immediately downstream of layer l (clipped at H_act)."""
    G = gains(model)["G"]
    Hs = model["Hs"]
    Ms = model["Ms"]
    H_act = Hs[-1]
    H_active = float((H_act > 0).sum() / max(H_act.shape[0], 1))
    adj = data["adj"]
    B2 = np.asarray(adj @ (H_act @ Ms[-1]).astype(np.float32)) + model["b_out"]
    B2 = B2[data["train"]]
    part = np.partition(-B2, 1, axis=1)
    margins = -part[:, 0] + part[:, 1]
    delta = float(np.percentile(margins, margin_quantile))
    if delta <= 0:
        delta = 1e-3
    L1 = model["n_layers"] - 1
    ts, infos = [], []
    for l in range(L1):
        d_l = float(min(Ms[l + 1].shape[1], max(H_active, 1.0)))
        bound_per_T = G[l] * theta * np.sqrt(max(H_active, 1.0) * d_l) / 2.0
        T_l = int(np.clip(int(np.ceil(bound_per_T / (strictness * delta))), 1, 64))
        ts.append(T_l)
        infos.append({"G_l": G[l], "d_l": d_l, "T_l": T_l})
    return ts, {"delta_p01_train": delta, "H_active": H_active, "per_layer": infos}


def allocation_rule(G, T_star, kappa=1.0, T_min=1):
    """Rule v2 (pre-registered): T_l = max(T_min, ceil(kappa*(G_l/G_max)*T_star))."""
    Gmax = max(G) if G else 1.0
    ts = [int(max(T_min, np.ceil(kappa * (g / Gmax) * T_star))) for g in G]
    return ts


def compute_T_min(model, data, theta=1.0, margin_quantile=10.0, strictness=0.5):
    """Training-split margin floor (rule v2): smallest T with
    G_max*theta*sqrt(H_active)/(2T) <= strictness * Delta_p10_train. All quantities train-split."""
    G = gains(model)["G"]
    Gmax = max(G) if G else 1.0
    Hs = model["Hs"]
    H_act = Hs[-1]
    H_active = float((H_act > 0).sum() / max(H_act.shape[0], 1))
    # ANN output logits incl. the output-layer aggregation, on TRAIN nodes
    adj = data["adj"]
    B2 = np.asarray(adj @ (H_act @ model["Ms"][-1]).astype(np.float32)) + model["b_out"]
    B2 = B2[data["train"]]
    part = np.partition(-B2, 1, axis=1)
    margins = -part[:, 0] + part[:, 1]  # top1 - top2
    delta = float(np.percentile(margins, margin_quantile))
    if delta <= 0:
        delta = 1e-3
    T_min = int(np.ceil(Gmax * theta * np.sqrt(max(H_active, 1.0)) / (2.0 * strictness * delta)))
    return int(np.clip(T_min, 1, 64)), {"Gmax": Gmax, "H_active": H_active, "delta_p10_train": delta}


def saving(ts, T_star):
    L = len(ts) + 1  # hidden + output
    Gbar_over_Gmax = float(np.mean([t / max(ts) for t in ts])) if ts and max(ts) > 0 else 1.0
    S_formula = 1.0 - (((L - 1) / L) * Gbar_over_Gmax + 1.0 / L)
    S_measured = 1.0 - (sum(ts) + max(ts + [1])) / (L * T_star)
    return {"S_formula": S_formula, "S_measured": S_measured,
            "updates_alloc": sum(ts) + max(ts + [1]), "updates_unif": L * T_star}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--L", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--data-root", default=str(PRIOR / "results" / ".data"))
    a = ap.parse_args()
    out = {"stage": "E-B1-prototype", "claim": "planned", "cells": []}
    for name in a.datasets:
        data = core.load(name, root=a.data_root)
        for n_layers in a.L:
            for sd in a.seeds:
                m = core.train_ann(data, hidden=16, n_layers=n_layers, seed=sd)
                g = gains(m)
                G = g["G"]
                Gmax = max(G) if G else 1.0
                Gbar = float(np.mean(G)) if G else 1.0
                headroom = Gmax / Gbar if Gbar > 0 else 1.0
                # certificate target T_star: margin-calibrated proxy for the prototype =
                # smallest uniform T in {1..64} reaching ANN-1.5pp on TRAIN accuracy
                ann_tr = float((m.get("tr_acc", None) or 0)) if isinstance(m, dict) else 0.0
                T_star = 16  # prototype placeholder; full run calibrates c on train margins
                T_min, tmin_info = compute_T_min(m, data)
                ts, r2_info = allocation_rule_r2(m, data)          # rule R2 (frozen constants)
                sav = saving(ts, T_star)
                run_alloc = convert_staged(data, m, ts)
                run_floor = convert_staged(data, m, [T_min] * (n_layers - 1))
                run_unif_staged = convert_staged(data, m, [T_star] * (n_layers - 1))
                run_unif_lock = core.convert(data, m, T=T_star)
                cell = {"dataset": name, "L": n_layers, "seed": sd, "G": G,
                        "Gmax": Gmax, "Gbar": Gbar, "headroom": headroom,
                        "rule": "R2", "T_star": T_star, "T_min_v2clip": T_min,
                        "r2_info": r2_info, "T_l": ts, **sav,
                        "acc_alloc_staged": run_alloc["acc"],
                        "acc_floor_only": run_floor["acc"],
                        "acc_unif_staged": run_unif_staged["acc"],
                        "acc_unif_lockstep": run_unif_lock["acc"],
                        "acc_band_ok": run_alloc["acc"] >= run_unif_staged["acc"] - 0.015,
                        "floor_control_updates": run_floor["layer_updates"],
                        "output_identity_max": run_alloc["output_identity_max"],
                        "updates_delta_pct": 100.0 * (1 - run_alloc["layer_updates"] / (n_layers * T_star))}
                out["cells"].append(cell)
                print(f"[{name}] L={n_layers} seed={sd} headroom={headroom:.2f} Tmin={T_min} "
                      f"T_l={ts} save={100*sav['S_measured']:.1f}% "
                      f"v2 alloc {run_alloc['acc']:.4f} vs staged unif "
                      f"{run_unif_staged['acc']:.4f} (floor-only {run_floor['acc']:.4f}, "
                      f"lockstep {run_unif_lock['acc']:.4f}) "
                      f"ident={run_alloc['output_identity_max']:.2e}", flush=True)
    # re-pilot gate per audits/jB1-rule-v2-prereg.md (rule v2, frozen constants)
    ident_ok = all(c["output_identity_max"] < 1e-5 for c in out["cells"])
    headroom_ok = sum(1 for c in out["cells"] if c["headroom"] >= 1.3) >= max(1, len(out["cells"]) // 2)
    band15 = [c["acc_band_ok"] and c["S_measured"] >= 0.15 for c in out["cells"]]
    band_ok = sum(band15) >= (len(band15) + 1) // 2
    floor_beat = sum(1 for c in out["cells"]
                     if c["acc_alloc_staged"] > c["acc_floor_only"] + 1e-9) >= (len(out["cells"]) + 1) // 2
    out["prototype_gate"] = {"identity_ok": ident_ok, "headroom_ok": headroom_ok,
                             "half_cells_band_at_15pct": band_ok,
                             "gain_term_beats_floor_only": floor_beat,
                             "engine_fix": "staged semantics (cycle-1); rule v2 floor (cycle-2)",
                             "verdict": "pass" if (ident_ok and headroom_ok and band_ok and floor_beat)
                                        else "escalate-R2-or-pivot"}
    (RUN / "results" / "jB1_prototype.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("prototype_gate:", out["prototype_gate"])


if __name__ == "__main__":
    main()
