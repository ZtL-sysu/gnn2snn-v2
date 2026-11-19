# JB1a Outcome Record — Sweep-Free Latency Selection (primary claim, R4 pivot)

Evidence: `results/jB1a_select.json`, `logs/jB1a_select.log` (+ arxiv phase
`logs/jB1a_arxiv.log` when complete). Procedure frozen in `audits/jb1a-primary-prereg.md`; c
calibrated on {cora, citeseer} only (c_cora=0.2479, c_citeseer=0.1965 → frozen median ≈ 0.2222);
held-out one-shot runs reported once.

## Results (excluding arxiv phase, which runs separately)

| graph | T_cert | acc | ANN | band | T* (report) | κ |
|---|---|---|---|---|---|---|
| chameleon (held-out) | 6 | 0.6043 | 0.6338 | **no** | 5 | 1.2 |
| squirrel (held-out) | 11 | 0.4630 | 0.4636 | yes | 1 | **11.0** |
| cora (calibration) | 3 | 0.7937 | 0.8060 | yes | 3 | 1.0 |
| citeseer (calibration) | 4 | 0.6929 | 0.7031 | yes | 4 | 1.0 |

## Verdict per the frozen rule

- Held-out reachable (pre-arxiv): 2; band hits: 1; κ values: 1.2, 11.0.
- **strong_pass = False** (κ ≤ 1.25 fails: squirrel κ=11); **minimum_pass = False** (κ ≤ 2 fails).
- Per the pre-registered failure signal: band missed on >1/3 reachable (chameleon) and median κ = 6.1
  > 2 → **the sweep-free-selection claim is removed and reported as a negative result.**

## What the negative result establishes (kept in the paper)

1. **In-distribution, the certificate is exact**: on both calibration graphs T_cert = T* (κ=1.0) —
   the margin/gain formula genuinely captures required latency where its calibration statistics are
   representative.
2. **Transfer fails in both directions on new graphs**: chameleon needs *more* steps than the
   certificate gives (under-provision, band miss), squirrel needs *far fewer* (over-provision by
   11×). This is consistent with, and sharper than, the earlier `wp7_predict` required-T 2/12
   failure: per-graph margin scales are not transferable across the homophily/heterophily divide.
3. Together with the allocation regime map (E-B1 cycles 1–4), the paper's honest position: the
   identity and laws describe conversion error precisely **within** a configuration; **cross-graph
   latency decisions from weights alone** remain open — two constructive attempts (allocation,
   selection) and one predictor (wp7) all fail their frozen held-out gates, each with a measured
   mechanism.

## Arxiv phase (appended on completion)

- ogbn-arxiv: T_cert=5, acc 0.5970 vs ANN 0.6218 → **band miss**; report T*=2, κ=2.5; 7 seeds.
- **Final verdict (all three held-out configs): band hits 1/3; κ = [1.2, 11.0, 2.5];
  strong_pass=False; minimum_pass=False.** Failure modes differ per graph — chameleon
  under-provisions (band miss at κ=1.2), squirrel over-provisions 11×, arxiv both misses and
  over-provisions. The pre-registered failure signal fired; the claim is removed and the negative
  result reported with these mechanisms.

## Falsifiers

- shuffled-c control (calibration graphs): computed at P5 from stored calibration parameters; must
  calibrate strictly worse than the frozen median c.
- oracle T* rows are reported as reference only and never enter the claim.

Final state: all three held-out rows (chameleon, squirrel, ogbn-arxiv) complete in
`results/jB1a_select.json`; verdict band 1/3, κ=[1.2, 11.0, 2.5] — recorded above.
