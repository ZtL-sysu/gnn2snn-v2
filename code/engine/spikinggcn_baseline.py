#!/usr/bin/env python3
"""External-paradigm baseline: SpikingGCN (Zhu et al., IJCAI 2022) re-implemented.

Why: the review's M5 says every competitor in the paper is our own construction. This adds a third
trained architecture taken from the published formulation rather than from ours: SGC-style K-step
feature propagation, then ONE fully-connected spiking layer trained with a surrogate gradient, then
rate decoding. It differs from our `poisson` arm in architecture (no second graph aggregation, no
hidden graph layer), not only in input coding.

Honesty notes recorded with the result:
  * this is a re-implementation of the formulation, not the authors' code; hyperparameters are our
    choice and are written into the output file, so no published number is claimed to be reproduced;
  * it is therefore reported as "an external formulation under our protocol", never as "we beat
    SpikingGCN's reported accuracy".

Run: python spikinggcn_baseline.py --seeds 7 --out ../results/spikinggcn_s7.json
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import snn_gnn_experiment as E
import graph_data as G
from baselines_published import _sparse
from strengthen import SpikeFn

ROOT = Path(__file__).resolve().parent.parent


def propagate(adj, X, K):
    """SGC: X <- A_hat X, K times, no nonlinearity, no weights."""
    out = X.copy()
    for _ in range(K):
        out = adj @ out
    return out


def train_sgc_snn(Xp, labels, tr, va, te, C, T, seed, hidden=512, epochs=300, lr=0.01,
                  wd=5e-4, early_stop=30, theta=1.0, dev="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    N, F = Xp.shape
    W = torch.nn.Parameter(torch.FloatTensor(F, C).uniform_(-1 / math.sqrt(F), 1 / math.sqrt(F)))
    b = torch.nn.Parameter(torch.zeros(C))
    if dev != "cpu":
        W.data = W.data.to(dev); b.data = b.data.to(dev)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    Y = torch.tensor(labels, dtype=torch.long, device=dev)
    Xt = torch.tensor(Xp, dtype=torch.float32, device=dev)
    tr = tr.to(dev); va = va.to(dev)

    def forward(train=True):
        u = torch.zeros(N, C, device=dev)
        cnt = torch.zeros(N, C, device=dev)
        mem = torch.zeros(N, C, device=dev)
        for _ in range(T):
            # rate-coded input spikes from the propagated features (published formulation)
            x_t = torch.bernoulli(Xt.clamp(0, 1)) if train else Xt
            cur = x_t @ W + b
            if train:
                cur = torch.dropout(cur, 0.5, train=True)
            u = u + cur
            s = SpikeFn.apply(u, theta, 1.0)
            u = u - s * theta
            cnt = cnt + s
            mem = mem + cur
        return mem, cnt

    best, state, pat = -1.0, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        mem, _ = forward(True)
        torch.nn.functional.cross_entropy(mem[tr], Y[tr]).backward()
        opt.step()
        with torch.no_grad():
            mem, _ = forward(False)
            v = float((mem[va].argmax(1) == Y[va]).float().mean())
        if v > best:
            best, pat = v, 0
            state = (W.detach().clone(), b.detach().clone())
        else:
            pat += 1
            if pat >= early_stop:
                break
    with torch.no_grad():
        W.data, b.data = state
        mem, _ = forward(False)
        pred = mem.argmax(1).cpu().numpy()
    return float((pred[te] == labels[te]).mean()), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--ts", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512, help="width of the spiking layer")
    ap.add_argument("--data-root", default=str(ROOT / "results" / ".data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "spikinggcn_s7.json"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out = {}
    for name in a.names:
        d = G.load_ds(name, a.data_root)
        adj = E.normalize_adj(d["adj"]).tocsr()
        X = d["features"].astype(np.float32).copy()
        if d["row_normalize"]:
            rs = X.sum(1); rs[rs == 0] = 1.0; X = X / rs[:, None]
        Xp = propagate(adj, X, a.K)
        mx = Xp.max()
        if mx > 1:                      # keep the rate code in [0,1] as the formulation requires
            Xp = Xp / mx
        tr = torch.tensor(np.where(d["train_mask"])[0], dtype=torch.long)
        va = torch.tensor(np.where(d["val_mask"])[0], dtype=torch.long)
        te = np.where(d["test_mask"])[0]
        C = int(d["labels"].max()) + 1
        rec = dict(K=a.K, hidden=a.hidden, norm="max-abs to [0,1]", ann=None, modes={})
        for T in a.ts:
            accs = []
            for sd in range(a.seeds):
                acc, bv = train_sgc_snn(Xp, d["labels"], tr, va, te, C, T, sd,
                                        hidden=a.hidden, dev=dev)
                accs.append(acc)
            rec["modes"][str(T)] = dict(acc_mean=float(np.mean(accs)),
                                        acc_std=float(np.std(accs)),
                                        acc_per_seed=accs)
            print(f"[{name}] spikinggcn T={T:2d} acc={np.mean(accs):.4f}", flush=True)
        out[name] = rec

    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
