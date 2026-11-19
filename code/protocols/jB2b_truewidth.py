#!/usr/bin/env python3
"""E-B2b: faithful external-formulation arm with a REAL hidden spiking layer (repair of the
`hidden`-unused defect found at P3).

Architecture, as the manuscript describes it: SGC-K feature propagation -> ONE fully-connected
spiking hidden layer of width H -> linear rate readout -> class logits. Surrogate-gradient trained
end to end, same protocol family as the registered arm (epochs/lr/dropout recorded in output).

Cost accountings recorded per cell (the R3 repair):
  * synops_spike   : T * (input spikes * H + hidden spikes * C)   [event-driven synaptic ops]
  * dense_worst    : T * N * (F*H + H*C)                          [dense matmul upper bound]
  * node_h         : T * N * H                                    [the old accounting, kept for
                                                                    continuity with tab14 rows]

Neutral framing (JB3): tests whether the external lead is a capacity effect. Any outcome reportable.
Run at P4. Output: results/jB2b_truewidth.json
"""
import argparse
import json
import math
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
from strengthen import SpikeFn  # noqa: E402
import spikinggcn_baseline as SB  # noqa: E402


def train_truewidth(Xp, labels, tr, va, te, C, T, seed, hidden=512, epochs=300, lr=0.01,
                    wd=5e-4, early_stop=30, theta=1.0, dev="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    N, F = Xp.shape
    dev_t = torch.device(dev)
    W1 = torch.nn.Parameter(torch.FloatTensor(F, hidden).uniform_(-1 / math.sqrt(F), 1 / math.sqrt(F)).to(dev_t))
    b1 = torch.nn.Parameter(torch.zeros(hidden, device=dev_t))
    W2 = torch.nn.Parameter(torch.FloatTensor(hidden, C).uniform_(-1 / math.sqrt(hidden), 1 / math.sqrt(hidden)).to(dev_t))
    b2 = torch.nn.Parameter(torch.zeros(C, device=dev_t))
    params = [W1, b1, W2, b2]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    Y = torch.tensor(labels, dtype=torch.long, device=dev)
    Xt = torch.tensor(Xp, dtype=torch.float32, device=dev)
    tr_, va_ = tr.to(dev), va.to(dev)

    def forward(train=True):
        u = torch.zeros(N, hidden, device=dev)
        mem_out = torch.zeros(N, C, device=dev)
        cnt_hidden = torch.zeros(N, hidden, device=dev)
        in_spikes = 0.0
        for _ in range(T):
            x_t = torch.bernoulli(Xt.clamp(0, 1)) if train else Xt
            in_spikes += float(x_t.sum())
            cur = x_t @ W1 + b1
            if train:
                cur = torch.dropout(cur, 0.5, train=True)
            u = u + cur
            s = SpikeFn.apply(u, theta, 1.0)
            u = u - s * theta
            cnt_hidden = cnt_hidden + s
            mem_out = mem_out + s @ W2 + b2
        return mem_out, cnt_hidden, in_spikes

    best, state, pat = -1.0, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        mem, _, _ = forward(True)
        torch.nn.functional.cross_entropy(mem[tr_], Y[tr_]).backward()
        opt.step()
        with torch.no_grad():
            mem, _, _ = forward(False)
            v = float((mem[va_].argmax(1) == Y[va_]).float().mean())
        if v > best:
            best, pat = v, 0
            state = [p.detach().clone() for p in params]
        else:
            pat += 1
            if pat >= early_stop:
                break
    with torch.no_grad():
        for p, s_ in zip(params, state):
            p.data.copy_(s_)
        mem, cnt_hidden, in_spikes = forward(False)
        pred = mem.argmax(1).cpu().numpy()
    acc = float((pred[te] == labels[te]).mean())
    synops_spike = T * (in_spikes * hidden + float(cnt_hidden.sum()) * C)
    dense_worst = T * N * (F * hidden + hidden * C)
    node_h = T * N * hidden
    return acc, best, {"synops_spike": synops_spike, "dense_worst": float(dense_worst),
                       "node_h": float(node_h), "hidden_spikes": float(cnt_hidden.sum()),
                       "input_spikes": in_spikes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--hiddens", type=int, nargs="+", default=[16, 32, 512])
    ap.add_argument("--ts", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--data-root", default=str(PRIOR / "results" / ".data"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"stage": "E-B2b-truewidth", "device": dev, "K": a.K,
           "protocol": "external formulation with real hidden spiking layer; not a published-number reproduction",
           "cells": []}
    for name in a.names:
        d = G.load_ds(name, a.data_root)
        adj = E.normalize_adj(d["adj"]).tocsr()
        X = d["features"].astype(np.float32).copy()
        if d["row_normalize"]:
            rs = X.sum(1); rs[rs == 0] = 1.0; X = X / rs[:, None]
        Xp = SB.propagate(adj, X, a.K)
        mx = Xp.max()
        if mx > 1:
            Xp = Xp / mx
        tr = torch.tensor(np.where(d["train_mask"])[0], dtype=torch.long)
        va = torch.tensor(np.where(d["val_mask"])[0], dtype=torch.long)
        te = np.where(d["test_mask"])[0]
        C = int(d["labels"].max()) + 1
        for hidden in a.hiddens:
            for T in a.ts:
                accs, costs = [], []
                for sd in range(a.seeds):
                    acc, _, cost = train_truewidth(Xp, d["labels"], tr, va, te, C, T, sd,
                                                   hidden=hidden, dev=dev)
                    accs.append(acc); costs.append(cost)
                rec = {"dataset": name, "hidden": hidden, "T": T,
                       "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                       "acc_per_seed": accs,
                       "cost_mean": {k: float(np.mean([c[k] for c in costs])) for k in costs[0]}}
                out["cells"].append(rec)
                print(f"[{name}] truewidth H={hidden} T={T} acc={rec['acc_mean']:.4f}", flush=True)
    (RUN / "results" / "jB2b_truewidth.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", RUN / "results" / "jB2b_truewidth.json")


if __name__ == "__main__":
    main()
