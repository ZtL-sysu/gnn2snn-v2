#!/usr/bin/env python3
"""M1 repair: a published-paradigm spiking-GNN baseline + paired statistics.

The pre-draft review (M1) objected that the only competitor was our own directly-trained
LIF-GCN. This script adds a second, *published-paradigm* baseline and makes the headline
comparison statistically testable.

Baselines implemented (all under identical splits, seeds, hidden width and T):

  `conv`      our ANN->SNN conversion (proposed). Train ANN once per seed, deploy at any T.
  `direct`    directly-trained LIF-GCN with a CONSTANT input current (the `strengthen.py`
              baseline, reproduced here so per-seed accuracies are available).
  `poisson`   SpikingGCN-style **rate-coded direct training**: node features are Poisson
              (Bernoulli-rate) encoded into input spike trains at every step and the network
              is trained end-to-end with surrogate gradients. This is the formulation used by
              published graph-SNN work (Zhu et al., "Spiking Graph Convolutional Networks",
              IJCAI 2022) and is the natural "same-neuron-model, different-input-coding"
              control: it isolates the effect of *conversion* from the effect of *spiking*.

Why the poisson arm matters scientifically: `direct` and `poisson` differ only in input
coding, so `conv - poisson` vs `conv - direct` tells us whether our advantage comes from the
conversion itself or merely from feeding the network analogue-rate currents.

Per-seed accuracies are stored so the conversion-vs-baseline gap can be tested with a paired
Wilcoxon signed-rank test (same seed => same data order and same init distribution).

Run:
  python baselines_published.py --seeds 7 --out ../results/baselines_s7.json
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import snn_gnn_experiment as E
import graph_data as G
from strengthen import SpikeFn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _sparse(adj_norm):
    coo = adj_norm.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    vals = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, adj_norm.shape).coalesce()


def train_spiking(adj_t, X, labels, tr, va, hidden, C, T, seed, coding="constant",
                  epochs=400, lr=0.01, wd=5e-4, early_stop=30, theta=1.0, width=1.0,
                  dropout=0.5, readout="membrane", mask_mode="perstep"):
    """2-layer spiking GCN, LIF with reset-by-subtraction, surrogate-gradient BPTT.

    coding="constant": every step injects the same analogue pre-activation A_hat@X@W1+b1.
    coding="poisson" : every step injects A_hat@x_t@W1+b1 where x_t ~ Bernoulli(X) is a
                       rate-coded input spike train (SpikingGCN-style). X must be in [0,1].

    mask_mode="perstep": a fresh dropout mask at every timestep (torch.dropout inside the loop).
    mask_mode="shared" : one mask drawn per epoch and reused at every timestep, i.e. the protocol
                       of the earlier locked `direct_s7.json` run. Both are reported because they
                       differ measurably; the per-step protocol is the stronger baseline.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    N, F = X.shape

    def glorot(shape):
        std = 1.0 / math.sqrt(shape[0])
        return torch.FloatTensor(*shape).uniform_(-std, std)

    W1 = torch.nn.Parameter(glorot((F, hidden)))
    W2 = torch.nn.Parameter(glorot((hidden, C)))
    b1 = torch.nn.Parameter(torch.zeros(hidden))
    b2 = torch.nn.Parameter(torch.zeros(C))
    if X.device.type != "cpu":
        W1.data = W1.data.to(X.device); W2.data = W2.data.to(X.device)
        b1.data = b1.data.to(X.device); b2.data = b2.data.to(X.device)
    opt = torch.optim.Adam([W1, W2, b1, b2], lr=lr, weight_decay=wd)
    dev = X.device  # device-agnostic: the CPU is the default, CUDA is used only if X is there
    Y = torch.tensor(labels, dtype=torch.long, device=dev)

    def forward(train=True):
        # NOTE: must be rebuilt inside forward(). A tensor cached across epochs keeps a stale
        # autograd graph whose buffers are freed by the previous backward().
        const_pre = torch.sparse.mm(adj_t, X) @ W1 + b1 if coding == "constant" else None
        u1 = torch.zeros(N, hidden, device=dev)
        u2 = torch.zeros(N, C, device=dev)
        cnt2 = torch.zeros(N, C, device=dev)
        acc2 = torch.zeros(N, C, device=dev)
        spk1 = torch.zeros((), device=dev)
        spk2 = torch.zeros((), device=dev)
        shared_mask = None
        if train and dropout > 0 and mask_mode == "shared":
            shared_mask = (torch.rand(N, hidden, device=dev) >= dropout).to(X.dtype) / (1.0 - dropout)
        for _ in range(T):
            if coding == "constant":
                pre1 = const_pre
            else:
                x_t = torch.bernoulli(X)                 # rate-coded input spikes
                pre1 = torch.sparse.mm(adj_t, x_t) @ W1 + b1
            if train and dropout > 0:
                if mask_mode == "shared":
                    pre1 = pre1 * shared_mask
                else:
                    pre1 = torch.dropout(pre1, dropout, train=True)
            u1 = u1 + pre1
            s1 = SpikeFn.apply(u1, theta, width)
            u1 = u1 - s1 * theta
            cur2 = torch.sparse.mm(adj_t, s1) @ W2 + b2
            u2 = u2 + cur2
            s2 = SpikeFn.apply(u2, theta, width)
            u2 = u2 - s2 * theta
            cnt2 = cnt2 + s2
            acc2 = acc2 + cur2
            spk1 = spk1 + s1.sum()
            spk2 = spk2 + s2.sum()
        out = acc2 if readout == "membrane" else cnt2
        return out, (spk1 + spk2)

    best_val, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        opt.zero_grad()
        logits, _ = forward(True)
        loss = torch.nn.functional.cross_entropy(logits[tr], Y[tr])
        loss.backward(); opt.step()
        with torch.no_grad():
            lg, _ = forward(False)
            v = float((lg[va].argmax(1) == Y[va]).float().mean())
        if v > best_val:
            best_val, patience = v, 0
            best_state = (W1.detach().clone(), W2.detach().clone(),
                          b1.detach().clone(), b2.detach().clone())
        else:
            patience += 1
            if patience >= early_stop:
                break
    with torch.no_grad():
        W1.data.copy_(best_state[0]); W2.data.copy_(best_state[1])
        b1.data.copy_(best_state[2]); b2.data.copy_(best_state[3])
        lg, spk = forward(False)
    return lg.cpu(), float(spk), best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--data-root", default=str(ROOT / "results" / ".data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "baselines_s7.json"))
    ap.add_argument("--mask", default="perstep", choices=["perstep", "shared"],
                    help="dropout protocol for the constant-current arm; 'shared' reproduces the "
                         "protocol of the earlier locked direct_s7.json (one mask per epoch)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="device for the two trained baselines; the conversion arm is NumPy/CPU "
                         "either way, so it is bit-identical regardless of this flag")
    a = ap.parse_args()

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    if a.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = a.device
    print(f"trained baselines on: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""),
          flush=True)

    out = {}
    for name in a.names:
        d = G.load_ds(name, a.data_root)
        adj_norm = E.normalize_adj(d["adj"])
        adj_t = _sparse(adj_norm).to(dev)
        X_np = d["features"].astype(np.float32).copy()
        if d["row_normalize"]:
            rs = X_np.sum(1); rs[rs == 0] = 1.0
            X_np = X_np / rs[:, None]
        X_t = torch.tensor(X_np, device=dev)
        tr = torch.tensor(np.where(d["train_mask"])[0], dtype=torch.long, device=dev)
        va = torch.tensor(np.where(d["val_mask"])[0], dtype=torch.long, device=dev)
        te = np.where(d["test_mask"])[0]
        C = int(d["labels"].max()) + 1

        # ---- ANN trained once per seed; conversion evaluated at every T (deploy-any-T)
        models, ann = [], []
        for sd in range(a.seeds):
            m = E.train_gcn(d["features"], d["labels"], adj_norm, d["train_mask"],
                            d["val_mask"], d["test_mask"], sd, hidden=a.hidden,
                            row_normalize=d["row_normalize"])
            m["seed"] = sd; models.append(m); ann.append(m["acc_test"])
        res = dict(N=int(X_np.shape[0]), ann_test_mean=float(np.mean(ann)), device=dev,
                   ann_per_seed=[float(x) for x in ann], hidden=a.hidden, C=C, results={})

        conv = {}
        for T in a.ts:
            accs, spk = [], []
            for m in models:
                r = E.convert(m, T=T, beta1=E.ftbc_bias(T), mode="naive", l1_mode="if",
                              seed=0, theta=1.0, readout="membrane")
                accs.append(r["acc_test"]); spk.append(r["total_spikes"])
            conv[str(T)] = dict(acc_per_seed=accs, acc_mean=float(np.mean(accs)),
                                acc_std=float(np.std(accs)), spike_mean=float(np.mean(spk)))
        res["results"]["conv"] = conv
        print(f"[{name}] ANN={res['ann_test_mean']:.4f} conv " +
              " ".join(f"T{t}={conv[str(t)]['acc_mean']:.4f}" for t in a.ts), flush=True)

        # ---- baselines: retrained per (T, seed) exactly like the conversion arm is not
        for coding in ("constant", "poisson"):
            label = "direct" if coding == "constant" else "poisson"
            per_T = {}
            for T in a.ts:
                accs, spk = [], []
                for sd in range(a.seeds):
                    lg, spikes, bv = train_spiking(adj_t, X_t, d["labels"], tr, va,
                                                   a.hidden, C, T, sd, coding=coding,
                                                   mask_mode=a.mask)
                    pred = lg.argmax(1).cpu().numpy()
                    accs.append(float((pred[te] == d["labels"][te]).mean()))
                    spk.append(spikes)
                per_T[str(T)] = dict(acc_per_seed=accs, acc_mean=float(np.mean(accs)),
                                     acc_std=float(np.std(accs)), spike_mean=float(np.mean(spk)))
            res["results"][label] = per_T
            print(f"[{name}] {label} " +
                  " ".join(f"T{t}={per_T[str(t)]['acc_mean']:.4f}" for t in a.ts), flush=True)

        # ---- paired statistics on identical seeds
        stats = {}
        for T in a.ts:
            c = np.array(conv[str(T)]["acc_per_seed"])
            row = {}
            for label in ("direct", "poisson"):
                b = np.array(res["results"][label][str(T)]["acc_per_seed"])
                gap = c - b
                p = None
                if wilcoxon is not None and np.any(gap != 0):
                    try:
                        p = float(wilcoxon(c, b).pvalue)
                    except Exception:
                        p = None
                row[label] = dict(mean_gap=float(gap.mean()),
                                  min_gap=float(gap.min()), wins=int((gap > 0).sum()),
                                  wilcoxon_p=p)
            stats[str(T)] = row
        res["paired"] = stats
        out[name] = res
        print(f"[{name}] paired gap vs direct: " + " ".join(
            f"T{t}={stats[str(t)]['direct']['mean_gap']:+.4f}(p={stats[str(t)]['direct']['wilcoxon_p']})"
            for t in a.ts), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
