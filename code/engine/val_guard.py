"""Validation-constrained threshold calibration on independent checkpoints.

The training-logit search can change a threshold after quantisation is already
fine. ``guarded`` keeps the training-logit candidate only when validation
accuracy does not fall. ``direct`` selects each channel by validation accuracy.
Class biases are fit on the validation split and are never shown test labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import core
from matched_conversion import calibrate, positive_percentile, rate


DATASETS = ('cora', 'citeseer', 'pubmed')
SEEDS = tuple(range(21, 28))
WINDOWS = (1, 2, 4, 8)
SEARCH_MULTIPLIERS = np.array([
    0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95,
    1, 1.05, 1.1, 1.15, 1.25, 1.4, 1.6, 2, 3, 4,
])


def classify(decoded, adj, weight, bias, labels, val, test):
    logits = (adj @ decoded) @ weight + bias
    prediction = logits.argmax(1)
    return logits, float(np.mean(prediction[val] == labels[val])), float(np.mean(prediction[test] == labels[test]))


def label_bias(logits, labels, val, steps=80):
    """Fit one additive bias per class on validation labels."""
    bias = np.zeros(logits.shape[1], dtype=np.float64)
    observed = logits[val].astype(np.float64)
    one_hot = np.eye(logits.shape[1])[labels[val]]
    for _ in range(steps):
        shifted = observed + bias
        shifted -= shifted.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        bias -= 0.5 * (probability - one_hot).mean(axis=0)
    return bias


def search(current, adj, weight, output_bias, target, labels, train, val, test, steps, base, mode):
    """Coordinate search over channel thresholds.

    ``guarded`` may leave the p99 initialisation only for a training-logit
    improvement that does not lower validation accuracy. ``direct`` keeps the
    current threshold unless another candidate has higher validation accuracy,
    with a training-logit decrease used only to break exact ties.
    """
    candidates = base * SEARCH_MULTIPLIERS
    thresholds = np.full(current.shape[1], base)
    quantized = rate(current, thresholds, steps, 0.5)
    residual = np.asarray(adj[train] @ quantized) @ weight - (target[train] - output_bias)
    for _ in range(3):
        for channel, receiver in enumerate(weight):
            old = quantized[:, channel]
            alternatives = rate(current[:, channel, None], candidates, steps, 0.5)
            delta = np.asarray(adj[train] @ (alternatives - old[:, None]))
            losses = np.einsum('ij,ij->j', delta, delta) * float(receiver @ receiver)
            losses += 2 * ((residual @ receiver) @ delta)
            _, base_val, _ = classify(quantized, adj, weight, output_bias, labels, val, test)
            best = None
            best_val = base_val
            best_loss = 0.0
            order = np.argsort(losses) if mode == 'guarded' else range(len(candidates))
            for index in order:
                loss = float(losses[index])
                if mode == 'guarded' and loss >= -1e-10:
                    break
                trial = quantized.copy()
                trial[:, channel] = alternatives[:, index]
                _, trial_val, _ = classify(trial, adj, weight, output_bias, labels, val, test)
                if mode == 'guarded':
                    if trial_val + 1e-12 >= base_val:
                        best = int(index)
                        break
                    continue
                better = trial_val > best_val + 1e-12
                tied = np.isclose(trial_val, best_val) and loss < best_loss - 1e-10
                if better or tied:
                    best = int(index)
                    best_val = trial_val
                    best_loss = loss
            if best is None:
                continue
            quantized[:, channel] = alternatives[:, best]
            residual += delta[:, best, None] * receiver
            thresholds[channel] = candidates[best]
    return thresholds


def run_checkpoint(data, seed, sink):
    model = core.train_ann(data, seed=seed, device='cuda', val_mode='eval')
    first, second = model['Ms']
    hidden_bias, output_bias = model['bs_layers'][0], model['b_out']
    x, adj, train, val, test = (data[key] for key in ('X', 'adj', 'train', 'val', 'test'))
    labels = data['y']
    current = (adj @ x) @ first + hidden_bias
    target = (adj @ current.clip(0)) @ second + output_bias
    ann = float(np.mean(target[test].argmax(1) == labels[test]))
    assert abs(ann - float(model['acc_test'])) < 1e-6
    digest = hashlib.sha256(b''.join(a.tobytes() for a in (first, second, hidden_bias, output_bias))).hexdigest()
    base = positive_percentile(current[train], 99)
    maximum = max(1e-3, float(current[train].max()))
    ap = positive_percentile(current[train], 99.9)
    for steps in WINDOWS:
        fitted = {
            'graph_calibrated': calibrate(current, adj[train], second, target[train] - output_bias, steps, base),
            'val_guarded': search(current, adj, second, output_bias, target, labels, train, val, test, steps, base, 'guarded'),
            'val_direct': search(current, adj, second, output_bias, target, labels, train, val, test, steps, base, 'direct'),
        }
        specs = {
            'naive_if': (1.0, 0.0), 'max_balance': (maximum, 0.0), 'percentile_norm': (base, 0.0),
            'rts': (1.0, 0.5), 'snnc_ap_proxy': (ap, 0.5), 'p99_rounding': (base, 0.5),
        }
        specs.update({name: (threshold, 0.5) for name, threshold in fitted.items()})
        for name, (threshold, shift) in specs.items():
            decoded = rate(current, threshold, steps, shift)
            logits, val_acc, acc = classify(decoded, adj, second, output_bias, labels, val, test)
            logit_shift = logits + (target[val] - logits[val]).mean(axis=0)
            labelled = logits + label_bias(logits, labels, val)
            sink.write(json.dumps(dict(
                dataset=data['name'], seed=seed, T=steps, arm=name, hash=digest, ann=ann,
                acc=acc, val_acc=val_acc,
                train_mse=float(np.mean((logits[train] - target[train]) ** 2)),
                acc_logit_bias=float(np.mean(logit_shift[test].argmax(1) == labels[test])),
                acc_label_bias=float(np.mean(labelled[test].argmax(1) == labels[test])),
                spikes=int(np.rint(decoded / np.asarray(threshold) * steps).sum()),
            )) + '\n')
    print(data['name'], seed, f'{ann:.4f}', digest[:12], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as sink:
        for name in DATASETS:
            data = core.load(name)
            for seed in SEEDS:
                run_checkpoint(data, seed, sink)
                sink.flush()
