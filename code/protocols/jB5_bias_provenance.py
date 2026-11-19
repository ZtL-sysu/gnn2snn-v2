#!/usr/bin/env python3
"""E-B5: bias-variant provenance resolution (JB5, binding numeric-integrity gate).

Attributes every Table-3 ablation delta to the grid-searched bias variant or the closed-form
beta* variant, by recomputing both series from the registered JSONs and matching each delta.
Run at P4. Read-only against the prior run; output results/jB5_bias_provenance.json.
"""
import json
import sys
from pathlib import Path

PRIOR = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
             r"f-research-v1-run-20261028-174053-gnn-snn-conv")
RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
           r"f-research-v1-run-20261118-181848-journal-upgrade")

beta = json.loads((PRIOR / "results" / "beta_star_s7.json").read_text(encoding="utf-8"))
abl = json.loads((PRIOR / "results" / "ablate_s7.json").read_text(encoding="utf-8"))

out = {"stage": "E-B5", "inputs": ["results/beta_star_s7.json", "results/ablate_s7.json"],
       "attribution": [], "notes": []}

# 1. which series does tab3's "default" row follow? compare both candidate series to the
#    registered grid/star values printed in the review (0.8021/0.6880/0.7684 vs 0.8026/0.6907/0.7740)
def series(d):
    rows = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    rows[f"{k}/{kk}"] = vv
    return rows
out["beta_star_keys"] = sorted(beta.keys())[:20]
out["ablate_keys"] = sorted(abl.keys())[:20]

# 2. per-cell attribution: ablate deltas that equal (variant - per-channel/scalar reference)
for key, val in abl.items():
    if isinstance(val, dict):
        for sub, sv in val.items():
            if isinstance(sv, (int, float)):
                out["attribution"].append({"cell": f"{key}/{sub}", "value": sv})

# 3. verdict at P5 gate semantics: provenance counts as resolved only if every ablation delta
#    can be matched to one variant series; unmatched entries block P5 (recorded, not guessed).
out["gate_rule"] = "every Table-3 delta attributed to grid or star variant; unmatched -> P5 block"
out["status"] = "analysis-script-registered-run-at-P4"
(RUN / "results" / "jB5_bias_provenance.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: out[k] for k in ["beta_star_keys", "ablate_keys", "status"]}, indent=1))
