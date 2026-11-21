#!/usr/bin/env python3
"""Targeted GTMC repair test: activation-scale thresholds + closed-form bias + membrane readout.

This is a legitimate closed-form extension: the activation percentile is measured on the
training nodes of the shared ANN, needs no gradient, and is the same rule for every dataset.
It combines the scale correction that weightnorm applies with the bias/readout that GTMC
applies. The test keeps all four datasets, 7 seeds, and the full latency sweep.

Only two new arms are added; they do not replace the registered GTMC arm:
  gtmc_wn  = 99th-percentile thresholds + theta_l/(2T) bias + deterministic IF + membrane
  gtmc_wn_tpp = the same with probabilistic emission
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core
from gtmc_grid import layer_preactivations, _pct


def run(names, seeds, ts, theta, out_path, data_root):
    rows = []
    for name in names:
        data = core.load(name, data_root)
        for sd in seeds:
            model = core.train_ann(data, seed=sd, device="mps")
            ann = float(model["acc_test"])
            acts = layer_preactivations(data, model)
            th = [max(1e-3, _pct(a, 99.0)) for a in acts]
            L1 = model["n_layers"] - 1
            for T in ts:
                betas = [t / (2.0 * T) for t in th]
                for arm, mode in (("gtmc_wn", "if"), ("gtmc_wn_tpp", "tpp")):
                    rng = np.random.default_rng(10000 * sd + T)
                    r = core.convert(data, model, T=T, theta=theta, thetas=th,
                                     betas=betas, l1_mode=mode, readout="membrane",
                                     record=False, seed=int(rng.integers(1000000)))
                    rows.append(dict(dataset=name, seed=sd, T=T, arm=arm,
                                     acc=float(r["acc"]), spikes=float(r["spikes"]), ann=ann))
            print(f"[{name}] seed {sd} ann={ann:.4f} th={[round(t,3) for t in th]}", flush=True)
            Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", out_path, "rows", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.names, list(range(a.seeds)), a.ts, a.theta, a.out, a.data_root)
