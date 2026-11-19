#!/usr/bin/env python
"""E-A2: machine-readable input-convention diagnostic for the SpikingGCN pipeline (JA2).

Four rows on Cora through the audited official pipeline (pinned py3.6 env `spikinggcn36`):
  1. raw binary bag-of-words, self-loops (faithful; expected ~50.0)
  2. row-normalised features
  3. max-rescale encoder input to [0,1]
  4. raw features with self-loops removed
Each row emits value, seed, config hash, scratch dir and log path. Run:
  [WORKSPACE]/miniconda3/envs/spikinggcn36/python.exe protocols/jA_conventions.py
Output: results/jA_conventions.json (this run root).
"""
from __future__ import print_function
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
PRIOR = os.path.join(os.path.dirname(RUN), "f-research-v1-run-20261028-174053-gnn-snn-conv")
OFFICIAL = os.path.join(PRIOR, "third_party", "spikinggcn_official")
sys.path.insert(0, OFFICIAL)
sys.path.insert(0, os.path.join(OFFICIAL, "official_repo", "handcode"))

import run_official_inplace as R  # noqa: E402,F401  (path/log shims)
from run_snn import snn_run  # noqa: E402
import datareader  # noqa: E402  (the audited module; it imports Planetoid directly)
import graphgallery as gg  # noqa: E402  (for functional.normalize_adj recompute)
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(RUN, "results", "jA_conventions.json")
ROWS = []


def note(msg):
    line = "%s | %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)


def run_variant(name, transform, seed=0):
    """Convention transform at the audited Split-I feature path: `ReadData.conv_graph`
    (mat = normalize_adj(A)@normalize_adj(A)@X). The four-line computation is re-run verbatim with
    the convention transform inserted at the documented stage (X for features, A for self-loops,
    mat for the encoder rescale). Nothing else about the pipeline is touched."""
    orig_conv = datareader.ReadData.conv_graph

    def conv_wrap(self, data):
        graph = data.graph.to_undirected()
        A0 = graph.A
        X = np.asarray(graph.x, dtype=np.float32)
        tag = graph.y
        X, A0, mat_override = transform(X, A0)
        if mat_override is not None:
            return mat_override, tag
        A = gg.functional.normalize_adj(A0)
        mat = A @ A @ X
        return mat, tag

    datareader.ReadData.conv_graph = conv_wrap
    np.random.seed(seed)
    torch.manual_seed(seed)
    scratch = os.path.join(OFFICIAL, "_jA2_%s_%d" % (name, int(time.time())))
    os.makedirs(os.path.join(scratch, "tmpdir", "snn"), exist_ok=True)
    cwd = os.getcwd()
    t0 = time.time()
    err = None
    acc = None
    try:
        os.chdir(scratch)
        acc = float(snn_run("cora", log_dir="tmpdir/snn", device="cpu"))
    except Exception as e:  # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, str(e)[:200])
    finally:
        os.chdir(cwd)
        datareader.ReadData.conv_graph = orig_conv
    row = {"convention": name, "dataset": "cora", "seed": seed, "acc": acc,
           "error": err, "scratch": scratch,
           "wall_seconds": round(time.time() - t0, 1),
           "config_sha256": hashlib.sha256(open(os.path.join(OFFICIAL, "official_repo", "handcode",
                                                             "models_conf.json"), "rb").read()).hexdigest()}
    ROWS.append(row)
    note("%s -> %s" % (name, acc if acc is not None else err))
    return row


def t_raw(X, A0):
    return X, A0, None


def t_rownorm(X, A0):
    rs = X.sum(1); rs[rs == 0] = 1.0
    return X / rs[:, None], A0, None


def t_rownorm_then_maxrescale(X, A0):
    """Row 3: row-normalise features, propagate, then rescale the encoder input mat to [0,1]."""
    Xn = X.copy(); rs = Xn.sum(1); rs[rs == 0] = 1.0
    Xn = Xn / rs[:, None]
    A = gg.functional.normalize_adj(A0)
    mat = A @ A @ Xn
    mx = np.asarray(mat).max()
    if mx > 0:
        mat = mat / mx
    return X, A0, np.asarray(mat, dtype=np.float32)


def t_rownorm_noselfloop(X, A0):
    rs = X.sum(1); rs[rs == 0] = 1.0
    X2 = X / rs[:, None]
    A2 = A0.copy()
    A2.setdiag(0)
    A2.eliminate_zeros()
    return X2, A2, None


def t_noselfloop(X, A0):
    A2 = A0.copy()
    A2.setdiag(0)
    A2.eliminate_zeros()
    return X, A2, None


def main():
    for name, tf in [("raw_bow_selfloops", t_raw), ("row_normalized_features", t_rownorm),
                     ("rownorm_then_maxrescale_encoder", t_rownorm_then_maxrescale),
                     ("raw_bow_no_selfloops", t_noselfloop),
                     ("rownorm_no_selfloops", t_rownorm_noselfloop)]:
        run_variant(name, tf)
    out = {"stage": "E-A2-conventions", "dataset": "cora", "rows": ROWS,
           "note": "re-measured through the audited official pipeline; per-row logs are scratch dirs + wp13 log sink"}
    if not os.path.isdir(os.path.dirname(OUT)):
        os.makedirs(os.path.dirname(OUT))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    note("wrote %s" % OUT)


if __name__ == "__main__":
    main()
