#!/usr/bin/env python3
"""Faithful baseline re-run + matched-readout supplement, without editing logged rows.

Three new arms, all closed-form on the same trained ANN:

  weightnorm_faithful
      Classical percentile weight normalisation: divide each affine map by its
      train-node percentile q, so the spiking network uses W_l/q_l and threshold 1,
      preserving the ANN decision function. Uses membrane readout for parity with GTMC.

  matched_readout
      RTS bias with membrane readout (all other settings identical to GTMC except the
      bias). Isolates the contribution of the readout from the contribution of the bias.

  gtmc_gauge
      GTMC applied after gauge normalisation of the ANN weights, giving percentile
      thresholds q_l and bias q_l/(2T). Same closed form on every dataset.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core
from gtmc_grid import layer_preactivations, _pct


def gauge(model, acts):
    """Return a scaled copy of the model: M_l/q_l and b_l/q_l for hidden layers,
    output M_L/q_L and b_L/q_L, where q_l is the p99 train activation. The ANN
    decision function is unchanged; spiking thresholds become one."""
    m = dict(model)
    q = [max(1e-3, _pct(a, 99.0)) for a in acts]
    Ms = [x.copy().astype(np.float32) for x in model["Ms"]]
    bs = [x.copy().astype(np.float32) for x in model["bs_layers"]]
    for l in range(len(Ms) - 1):
        Ms[l] /= q[l]
        bs[l] /= q[l]
    Ms[-1] /= q[len(acts)-1] if False else 1.0
    m["Ms"], m["bs_layers"] = Ms, bs
    return m, q


def run(names, seeds, ts, theta, out_path, data_root):
    rows = []
    for name in names:
        data = core.load(name, data_root)
        for sd in seeds:
            model = core.train_ann(data, seed=sd, device="mps")
            ann = float(model["acc_test"])
            acts = layer_preactivations(data, model)
            q = [max(1e-3, _pct(a, 99.0)) for a in acts]
            L1 = model["n_layers"] - 1
            gm, _ = gauge(model, acts)

            for T in ts:
                # faithful p99 weight normalisation: scale hidden weights + biases,
                # threshold stays 1 at every layer, readout matches GTMC.
                r = core.convert(data, gm, T=T, theta=theta, betas=[0.0]*L1,
                                 l1_mode="if", readout="membrane", record=False)
                rows.append(dict(dataset=name, seed=sd, T=T, arm="weightnorm_faithful",
                                 acc=float(r["acc"]), spikes=float(r["spikes"]), ann=ann))

                # RTS bias + membrane: isolates the readout contribution.
                r = core.convert(data, model, T=T, theta=theta, betas=[theta/(2*T)]*L1,
                                 l1_mode="if", readout="membrane", record=False)
                rows.append(dict(dataset=name, seed=sd, T=T, arm="rts_membrane",
                                 acc=float(r["acc"]), spikes=float(r["spikes"]), ann=ann))

                # Gauge-normalised GTMC.
                r = core.convert(data, gm, T=T, theta=theta,
                                 betas=[theta/(2*T)]*L1, l1_mode="if",
                                 readout="membrane", record=False)
                rows.append(dict(dataset=name, seed=sd, T=T, arm="gtmc_gauge",
                                 acc=float(r["acc"]), spikes=float(r["spikes"]), ann=ann))
            print(f"[{name}] seed {sd} ann={ann:.4f} q={[round(x,3) for x in q]}", flush=True)
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
