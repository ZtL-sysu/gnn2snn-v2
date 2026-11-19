# F2 Novelty Map — Per-Layer Timestep Allocation (E-B1 Gate)

Date: 2026-11-18. Execution: multi-modal workflow (5 search angles → adversarial verification of
high-risk hits → synthesis), 10 agents, WebSearch/WebFetch + arXiv API + publisher full-text fetches.
Evidence archives synced to `inputs/novelty-evidence/` (Frontiers MT-SNN full text; Lacuna page).
Gate: **closed with mandatory claim re-scope** — the broad claim class is occupied; the narrowed
claim survives the search.

## Collision verdict (binding for JB1 wording)

**JB1 as originally worded ("per-layer step allocation ... new capability") is NOT novel.** Same
decision variable (per-layer timestep budget T_l in ANN→SNN conversion) already published:

1. **PASCAL** (Ramesh & Srinivasan, arXiv:2505.01730, 2025) — adversarially verified against v1+v2
   full text: "a systematic method to configure the quantization step of QCFS activation in a
   layerwise manner, which effectively determines the optimal number of timesteps per layer for the
   converted SNN". Budgets from layerwise quantization-error statistics
   (activation-distribution metric), multi-stage IF neuron for adjacent-layer step mismatch;
   ResNet-18/CIFAR-10. **Decision-variable collision (same_method=true at gate rule).**
2. **Mixed-timestep temporal alignment / MT-SNN** (Frontiers in Neuroscience 2026,
   DOI 10.3389/fnins.2026.1783326) — conversion pipeline mapping layer-wise activation bit-widths
   to layer-wise timesteps, jointly optimized quantization levels + thresholds, Scaled Synaptic
   Current Accumulation for cross-layer temporal mismatch; ResNet-34/ImageNet 73.63% at 4.88 avg
   steps. Explicitly claims "layer-wise temporal flexibility of conversion-based SNNs".
3. **LD-SNN** (Nallathambi et al., ISLPED 2022, DOI 10.1145/3531437.3539708) — per-layer timesteps
   "independently optimized", 1.67–3.84× fewer synaptic updates at preserved accuracy. The
   *headline phenomenon* (per-layer beats uniform on updates) is established prior art.
   (Full text Cloudflare-blocked; T_l rule unverified — citing a search-based characterization.)

**What remains novel (narrowed claim):** closed-form training-free T_l derivation from the trained
weights' downstream spectral gain `G_l = Π(1−α)‖ÂM_k‖` with training-split margin calibration
`T_l = c·G_l·√H_active/Δ_ref`; the per-layer freeze-time realization with `β_l = θ/(2T_l)` inside a
single spiking simulation; the GCN-family **graph** conversion setting; and the specific
≥20%-fewer-total-layer-updates-at-ANN−1.5pp result vs the best uniform-T allocation on held-out
graphs. No verified work does per-layer timestep allocation in graph-network conversion at all.

## Cluster map

- **Collision cluster (per-layer T in conversion):** PASCAL 2025; MT-SNN Frontiers 2026 (+ presumed
  OpenReview preprint, identity unverified).
- **Headline-adjacent:** LD-SNN ISLPED 2022 (per-layer T → 1.67–3.84× fewer synaptic updates).
- **Model-derived per-layer time budgets outside conversion:** TVLSI 2024 LCT (PCA temporal
  redundancy, 3.22× accel.); Spatio-Temporal Pruning for spiking LLMs (arXiv:2508.20122, per-layer
  t^l from activation PCA); STISA (block-level budgets); CN120218133A patent (snippets only).
- **Global/input-dependent adaptive T (per-sample, not per-layer):** SEENN (NeurIPS 2023), DT-SNN
  entropy stopping, Dynamic Confidence (ICCV 2023), BNTT early exit, AdaFire, NeuCODEX, First-to-Spike.
- **Mixed-timestep direct training (no conversion):** Shrinking Your TimeStep (2401.01912), MTT /
  Temporal Flexibility (ICLR 2025), SPARQ.
- **Foundations:** QCFS (ICLR 2022, T=quantization level, θ/2 init), "Is Conventional SNN Really
  Efficient" (timestep=bit analogy, Bit Budget), HAWQ (mixed-precision allocator archetype; venue
  unverified in-session), Bu 2024 (no per-layer budgets).
- **Graph-side:** no verified graph-domain SNN work with per-layer timestep allocation found
  (secondary wiki names — SpikingGCN, SpikeNet, GC-SNN, GA-SNN, Bi-GCN — unconfirmed, must not be
  cited as-is).

## Mandatory wording changes (binding)

- **Forbidden:** any "first per-layer/non-uniform/layer-wise timestep allocation in ANN→SNN
  conversion"; any wording implying per-layer budgets beating uniform T on update counts is itself
  novel.
- **Required framing:** per-layer allocation in conversion is established prior art (PASCAL;
  MT-SNN); LD-SNN already showed the update savings. Our contribution is the closed-form
  gain/margin derivation (vs PASCAL's quantization-error statistics, MT-SNN's learned quantization
  levels, LD-SNN's search-based optimization), the freeze-time/θ/(2T_l) realization, and the
  graph-conversion result. Scoping: "within WebSearch/WebFetch + arXiv + publisher full text,
  2018–2026, we are not aware of any work deriving per-layer timestep budgets from trained-weight
  downstream spectral gain with margin calibration, nor any per-layer timestep allocation in
  graph-network conversion" — with the boundary's known gaps stated (LD-SNN full text blocked;
  patent snippets only; several search-summary-only mechanisms).

## Baseline arms to add (binding)

1. Uniform-T at matched accuracy (existing reference arm).
2. **PASCAL-adapted arm:** layerwise quantization-step allocation (activation-distribution metric)
   ported to the GCN pipeline — the comparator a reviewer will demand.
3. **LD-SNN-style search arm:** grid/greedy search over per-layer T_l under an update budget with
   accuracy feedback — labelled an independent reimplementation of the search idea (their exact
   algorithm unverified).
4. **Isolation arm:** uniform T_l realized via the freeze mechanism (separates derivation value from
   realization value).
5. Optional: global confidence early-exit arm (SEENN-style) bounding the per-input alternative.

## Unverified suspicions (recorded, not cited as fact)

OpenReview preprint ≡ Frontiers MT-SNN (unverified identity); Lacuna/Tiptree "Spending Spikes Where
They Matter" page (archived in `inputs/novelty-evidence/jb1-verify/`) occupies the model-derived
per-layer-budget idea class but is non-archival, machine-generated provenance unclear, prior-art
standing uncertain; LD-SNN T_l rule; CN120218133A mechanism (patent snippets only); STISA/Spatio-
Temporal Pruning formulas from search summaries; HAWQ venue; MT-SNN "all conversion papers fix T
globally" assertion not checked against its cited base papers.

## Gate decision

F2 **closed**: collision confirmed at the decision-variable level → JB1 re-scoped to the narrowed
claim with the mandatory framing and the four baseline arms above. This is a versioned claim change
inside the P1A/P1B repair cycle (P1A not yet re-accepted), not a silent swap.
