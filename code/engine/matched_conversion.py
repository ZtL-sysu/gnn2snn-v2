"""Compare decoded ANN-to-SNN conversions on a shared checkpoint.

The two-layer GCN has constant hidden current and linear output accumulation.
The constant-current count formula gives the same endpoint as binary IF ticks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import core


def positive_percentile(values, percentile):
    positive = values[values > 0]
    return max(1e-3, float(np.percentile(positive, percentile))) if positive.size else 1.0


def rate(current, threshold, steps, shift):
    counts = np.clip(np.floor(steps * current / threshold + shift), 0, steps)
    return (threshold / steps * counts).astype(np.float32)


def calibrate(current, calibration_adj, weight, target, steps, base):
    """Fit per-channel thresholds to unlabeled ANN logits on training nodes."""
    candidates = base * np.array([0.25, 0.4, 0.6, 0.8, 1, 1.25, 1.6, 2, 3, 4])
    thresholds = np.full(current.shape[1], base)
    quantized = rate(current, base, steps, 0.5)
    residual = np.asarray(calibration_adj @ quantized) @ weight - target
    for _ in range(3):
        for channel, receiver in enumerate(weight):
            old = quantized[:, channel]
            alternatives = rate(current[:, channel, None], candidates, steps, 0.5)
            delta = np.asarray(calibration_adj @ (alternatives - old[:, None]))
            losses = np.einsum('ij,ij->j', delta, delta) * (receiver @ receiver)
            losses += 2 * ((residual @ receiver) @ delta)
            best = int(np.argmin(losses))
            if losses[best] < -1e-10:
                residual += delta[:, best, None] * receiver
                quantized[:, channel] = alternatives[:, best]
                thresholds[channel] = candidates[best]
    return thresholds


def run(dataset, seeds, out):
    data = core.load(dataset)
    x, adj, train, test = (data[key] for key in ('X', 'adj', 'train', 'test'))
    labels = data['y']
    for seed in range(seeds):
        model = core.train_ann(data, seed=seed, device='cuda', val_mode='eval')
        hidden_weight, output_weight = model['Ms']
        hidden_bias = model['bs_layers'][0]
        output_bias = model['b_out']
        current = (adj @ x) @ hidden_weight + hidden_bias
        hidden = current.clip(0)
        target = (adj @ hidden) @ output_weight + output_bias
        oracle = float((target.argmax(1)[test] == labels[test]).mean())
        assert abs(oracle - float(model['acc_test'])) < 1e-6
        model_hash = hashlib.sha256(b''.join(a.tobytes() for a in (
            hidden_weight, output_weight, hidden_bias, output_bias))).hexdigest()
        base = positive_percentile(current[train], 99)
        maximum = max(1e-3, float(hidden[train].max()))
        ap = positive_percentile(current[train], 99.9)
        with out.open('a') as sink:
            for steps in (1, 2, 4, 8, 16, 32, 64):
                graph = calibrate(current, adj[train], output_weight,
                                  target[train] - output_bias, steps, base)
                arms = {
                    'naive_if': (1.0, 0), 'max_balance': (maximum, 0),
                    'percentile_norm': (base, 0), 'rts': (1.0, 0.5),
                    'snnc_ap_proxy': (ap, 0.5), 'p99_rounding': (base, 0.5),
                    'graph_calibrated': (graph, 0.5),
                }
                for arm, (threshold, shift) in arms.items():
                    decoded = rate(current, threshold, steps, shift)
                    logits = (adj @ decoded) @ output_weight + output_bias
                    row = dict(dataset=dataset, seed=seed, T=steps, arm=arm,
                               ann=oracle, hash=model_hash,
                               acc=float((logits.argmax(1)[test] == labels[test]).mean()),
                               train_logit_mse=float(np.square(logits[train] - target[train]).mean()),
                               spikes=int(np.rint(decoded / threshold * steps).sum()),
                               threshold=np.asarray(threshold).tolist())
                    sink.write(json.dumps(row) + '\n')
        print(dataset, seed, oracle, model_hash[:12], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--seeds', type=int, default=7)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        run(name, args.seeds, args.out)
