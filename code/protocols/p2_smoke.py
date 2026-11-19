#!/usr/bin/env python3
"""P2 setup/code check: import the protected prior-run engine, run one tiny
train->convert->identity cycle on Cora, and verify held-out dataset loaders.

Read-only against the prior run; all outputs land in this run root.
"""
import json
import sys
import time
from pathlib import Path

PRIOR = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
             r"f-research-v1-run-20261028-174053-gnn-snn-conv")
RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/"
           r"f-research-v1-run-20261118-181848-journal-upgrade")
sys.path.insert(0, str(PRIOR / "protocols"))
sys.path.insert(0, str(PRIOR / "experiments"))

out = {"stage": "P2-smoke", "checks": {}}
t0 = time.time()

# 1. engine import
import core  # noqa: E402  (prior-run conversion engine, read-only)
import graph_data  # noqa: E402
out["checks"]["imports"] = {"core": True, "graph_data": True}

# 2. Cora load + tiny train + convert + identity
data = core.load("cora", root=str(PRIOR / "results" / ".data"))
out["checks"]["cora_load"] = {"N": int(data["N"]), "C": int(data["C"])}
model = core.train_ann(data, hidden=16, n_layers=2, seed=0)
res = core.convert(data, model, T=4, theta=1.0, seed=0)
ident = core.check_identity(data, model, T=4, theta=1.0)
def max_residual(obj):
    per = obj.get("per") if isinstance(obj, dict) else None
    vals = []
    if isinstance(per, list):
        for row in per:
            if isinstance(row, dict):
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        vals.append(abs(float(v)))
                    elif isinstance(v, list):
                        vals += [abs(float(x)) for x in v if isinstance(x, (int, float))]
            elif isinstance(row, (int, float)):
                vals.append(abs(float(row)))
    return max(vals) if vals else {k: str(v)[:120] for k, v in obj.items()} if isinstance(obj, dict) else str(obj)[:120]
out["checks"]["tiny_cycle"] = {
    "snn_T4_acc": res["acc"],
    "identity_max_residual": max_residual(ident),
    "identity_keys": sorted(ident.keys()) if isinstance(ident, dict) else None,
}

# 3. held-out loader availability (chameleon/squirrel small; arxiv presence only)
loaders = {}
for name in ["chameleon", "squirrel"]:
    try:
        d2 = graph_data.load_ds(name, data_root=str(PRIOR / "results" / ".data"))
        n = d2["features"].shape[0]
        loaders[name] = {"ok": True, "N": int(n), "keys": sorted(d2.keys())[:8]}
    except Exception as exc:  # noqa: BLE001
        loaders[name] = {"ok": False, "error": str(exc)[:200]}
arxiv_dir = PRIOR / "results" / ".data" / "ogb" / "ogbn_arxiv"
loaders["ogbn_arxiv"] = {"raw_dir_exists": arxiv_dir.exists(), "path": str(arxiv_dir)}
out["checks"]["heldout_loaders"] = loaders

out["wall_seconds"] = round(time.time() - t0, 2)
(RUN / "results").mkdir(exist_ok=True)
(RUN / "results" / "p2_smoke.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
