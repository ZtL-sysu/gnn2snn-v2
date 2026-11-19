# SOP Cost Convention — Pre-Declared (before any E-B2b data exists)

Date: 2026-11-18, recorded at the P3A→P4 boundary, before `jB2b_truewidth.py` has been run. This
pre-declaration exists so the headline cost convention cannot be chosen after seeing which
accounting favours our arm (P3A review: "pre-declared headline SOP convention"; the choice is made
on standard-practice grounds only).

## Decision

**Headline cost metric: event-driven synaptic operations (SynOps)** — each presynaptic spike
crossing a synapse counts one operation; measured at evaluation time from actual spike counts and
fan-out, per inference, mean over test nodes, for every arm. This is the standard neuromorphic cost
metric and matches the paper's own spike-count reporting. No arm-specific exceptions.

- Message-passing units: fan-out of a spike at (node n, layer l) = deg(n) × H_{l+1} (all neighbour
  units of the next layer through the weight block).
- Dense layers (external arm's F→H and H→C): fan-out = H (resp. C) per presynaptic spike.
- Input spikes count once at their first synapse.

**Secondary metrics (all reported, none hidden):**

1. static-update proxy: `T × (potential synapses touched per step)` — for message-passing arms
   `T×E×H` (the registered tab14 convention, kept for continuity); for dense layers
   `T×N×F×H` / `T×N×H×C` worst-case dense matmul.
2. the old `T×N×H` node-width accounting for the external arm (continuity with the defective
   record; labelled as such).

## Pre-registered consequence rules (from P3A, before results)

- The matched-accuracy verdict `conversion_fewer_sop_than_all` and the original failure criterion
  ("an external arm dominating at T ≤ 4 on two or more graphs") are re-derived under **all three
  accountings** by one script; each verdict is reported side by side.
- If the failure criterion fires under the headline (event-driven) accounting, the WP2 strong cell
  is marked failed and the guidance failure-signal handling applies (claim removed/narrowed,
  negative result kept) — no accounting switch afterwards.
- If the verdict differs across accountings, the paper reports the sensitivity explicitly and the
  headline claims stay bound to the pre-declared convention.

## Why event-driven (grounds, not outcome)

It is the cost a neuromorphic deployment pays (spike events × synaptic fan-out), it is computable
identically for every arm including ones without graph aggregation, and it is the convention the
paper's own spikes/node reporting already speaks to. The static proxy `T×E×H` charges idle synapses
and is retained only as a secondary continuity row.
