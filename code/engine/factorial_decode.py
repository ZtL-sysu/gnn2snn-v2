#!/usr/bin/env python3
"""Factorial conversion experiment with threshold-aware decoding and a real oracle.

Crosses:
  hidden threshold  : 1 vs p99(q) of positive training activations
  bias              : 0 vs theta_hidden/(2T)
  decoding          : raw spikes vs threshold-aware spikes (q * s)

For the two-layer GCN, threshold-aware decoding is implemented exactly by folding the
hidden threshold into the output matrix: S_out = A_hat @ ((q*s1) @ W2) + b2 becomes
A_hat @ (s1 @ (q*W2)) + b2. The output bias and output threshold stay at one, so the
ANN decision function is preserved in the real-rate limit.

A real oracle feeds ReLU(B1) through the same output path and asserts that its argmax
equals the ANN argmax. This is not the dormant copy-ANN oracle from gtmc_grid.py.
"""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core
from gtmc_grid import layer_preactivations, _pct


def model_hash(model):
    h = hashlib.sha256()
    for key in ("Ms", "bs_layers", "b_out"):
        for x in model[key]:
            h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()[:16]


def oracle_acc(data, model, decoded, q):
    """Real-rate oracle: feed ReLU(B1)/q_hidden spikes-in-ANN-units."""
    X, adj, y, te = data["X"], data["adj"], data["y"], data["test"]
    M1, M2 = model["Ms"][0], model["Ms"][1]
    b1 = model["bs_layers"][0]
    H = np.maximum(np.asarray(adj @ (X @ M1)) + b1, 0.0)
    if decoded:
        # hidden current in spike units: H/q; output weight carries q
        z = np.asarray(adj @ ((H / q) @ (q * M2))) + model["b_out"]
    else:
        z = np.asarray(adj @ (H @ M2)) + model["b_out"]
    return float((z.argmax(1)[te] == y[te]).mean())


def convert_factorial(data, model, T, q, bias_on, decoded, theta=1.0):
    """Two-layer deterministic IF, fixed constants, with selectable decoding."""
    X, adj, N, C, y, te = data["X"], data["adj"], data["N"], data["C"], data["y"], data["test"]
    M1, M2 = model["Ms"][0], model["Ms"][1]
    b1 = model["bs_layers"][0]
    zeta = model["b_out"]
    th = q
    beta = th / (2.0 * T) if bias_on else 0.0
    inst1 = np.asarray(adj @ (X @ M1)) + b1
    u = np.zeros_like(inst1)
    W2_eff = (q * M2) if decoded else M2
    acc = np.zeros((N, C), np.float64)
    cnt = np.zeros((N, C), np.int32)
    uo = np.zeros((N, C), np.float64)
    spikes = 0.0
    for t in range(T):
        u += inst1 + beta
        s = (u >= th).astype(np.int8)
        u -= s * th
        spikes += float(s.sum())
        out = np.asarray(adj @ (s.astype(np.float32) @ W2_eff)) + zeta
        acc += out
        uo += out
        fired = (uo >= theta)
        cnt += fired
        uo -= fired * theta
        spikes += float(fired.sum())
    pred_mem = acc.argmax(1)
    pred_cnt = cnt.argmax(1)
    return dict(
        acc_mem=float((pred_mem[te] == y[te]).mean()),
        acc_cnt=float((pred_cnt[te] == y[te]).mean()),
        spikes=spikes,
    )


def run(names, seeds, ts, out_path, data_root):
    rows = []
    for name in names:
        data = core.load(name, data_root)
        for sd in seeds:
            model = core.train_ann(data, seed=sd, device="mps")
            ann = float(model["acc_test"])
            acts = layer_preactivations(data, model)
            q = max(1e-3, _pct(acts[0], 99.0))
            h = model_hash(model)
            for T in ts:
                for qtag, qv in (("th1", 1.0), ("thq", q)):
                    for decoded in (False, True):
                        orc = oracle_acc(data, model, decoded, qv)
                        if abs(orc - ann) > 1e-9:
                            raise RuntimeError(f"oracle failed {name} {sd} {T} {qtag} {decoded}: {orc} vs {ann}")
                        for bias in (False, True):
                            r = convert_factorial(data, model, T, qv, bias, decoded)
                            tag = f"{qtag}_{'dec' if decoded else 'raw'}_{'bias' if bias else 'nobias'}"
                            rows.append(dict(dataset=name, seed=sd, T=T, arm=f"fac_{tag}",
                                             acc_mem=r["acc_mem"], acc_cnt=r["acc_cnt"],
                                             spikes=r["spikes"], ann=ann, q=q,
                                             model_hash=h, oracle_acc=orc))
            print(f"[{name}] seed {sd} ann={ann:.4f} q={q:.4f} hash={h}", flush=True)
            Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", out_path, "rows", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.names, list(range(a.seeds)), a.ts, a.out, a.data_root)
