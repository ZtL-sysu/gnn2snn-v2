import json, hashlib
from pathlib import Path

RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade")

def ev(*rels):
    return [{"path": r, "sha256": hashlib.sha256((RUN / r).read_bytes()).hexdigest()} for r in rels]

MODEL_B = "GCN-family graph ANN-to-SNN conversion (Paper B)"
MODEL_A = "Spiking-GNN reproduction audit (Paper A)"

def R(id, statement, metric, value, unit, agg, unc, scope, evid, model, strength="main", **kw):
    f = {"id": id, "kind": "result", "statement": statement, "status": "observed_verified",
         "model": model, "condition": "as specified in the named artifact",
         "metric": metric, "value": value, "unit": unit,
         "aggregation": agg, "uncertainty": unc, "measurement_scope": scope,
         "claim_strength": strength, "evidence": evid}
    f.update(kw)
    return f

# --- Paper A: correct A-transport (P9 M3 source-mixing correction) ---
pa = json.loads((RUN / "writing/paperA-packet.json").read_text(encoding="utf-8"))
for f in pa["facts"]:
    if f["id"] == "A-transport":
        f["statement"] = ("Split transportability vs matching seed noise, per dataset: Cora gap 8.0 pp = "
                          "4.06x SpikingGCN's own seed s.d. (1.97 pp, 5 seeds) = 1.35x the 3sigma bar (marginal); "
                          "CiteSeer 5.1 pp = 4.11x a labelled proxy noise (SpikeGCL 1.24 pp s.d.; matching cell single-seed) "
                          "= 1.37x the bar (marginal); PubMed 11.7 pp = 2.18x its s.d. (5.37 pp) = 0.73x the bar "
                          "(confounded, re-scoped). No dataset-level strong verdict stands.")
        f["value"] = "1.35x/1.37x/0.73x the 3sigma bar"
        f["evidence"] = ev("audits/jA1-outcome.md", "logs/jA1_spikegcl_multiseed.log", "inputs/registered/wp10_audit.json")
pa["facts"].append(R("A-sgcn-seeds", "SpikingGCN's own Cora seed spread is 1.97 pp s.d. (per-seed 48.1-52.6, 5 seeds) and its PubMed spread 5.37 pp s.d. (3 seeds, derived).",
                     "seed_sdp_pp", "1.97 / 5.37", "pp", "5 and 3 seeds", "sample s.d.", "Cora and PubMed", ev("inputs/registered/wp10_audit.json"), MODEL_A, claim_strength="boundary"))
(RUN / "writing/paperA-packet.json").write_text(json.dumps(pa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# --- Paper B: register the previously unmapped measured facts (P9 RX M4-M6) ---
pb = json.loads((RUN / "writing/paperB-packet.json").read_text(encoding="utf-8"))
have = {f["id"] for f in pb["facts"]}
new = [
    R("B-closedform-scoring", "The closed form scores the two-layer conversion against the T-step simulation with one sparse matmul: mean absolute discrepancy 0.08 pp (max 0.40) over 105 cells; at L=3 it fails by 3.5 points (33% of cells off by more), a boundary of the closed form.",
      "mae_pp", "0.08 (0.40 max); L=3 3.5", "pp", "105 cells", "prior-run registered", "cora/citeseer/pubmed, T grid", ev("inputs/paper-B-baseline.pdf"), MODEL_B, claim_strength="main"),
    R("B-ranking", "The predictor ranks candidate models almost perfectly but misses the level: Spearman rho=0.967 (LOCO), gap MAE 4.6 pp.",
      "rho_and_mae", "0.967 / 4.6", "ratio / pp", "12 cells", "prior-run registered", "held-out cells", ev("inputs/paper-B-baseline.pdf"), MODEL_B),
    R("B-ensemble", "Multi-view averaging over threshold and initial potential does not beat the ANN; a nine-ANN model ensemble does (+1.21/+1.06/+1.11 pp over controls) - the gain belongs to model ensembling, not to re-sampling discretisation error.",
      "ensemble_delta_pp", "+1.21/+1.06/+1.11", "pp", "3 graphs", "prior-run registered", "cora/citeseer/pubmed", ev("inputs/paper-B-baseline.pdf"), MODEL_B, claim_strength="negative-with-mechanism"),
    R("B-anchor-external", "External anchors on the public split, context only and not audited rows: SpikeGraphormer released script 74.0 (below the conversion at T=16); SpikeGCL self-supervised 86.6 (above every arm in the protocol).",
      "external_anchor_pp", "74.0 / 86.6", "pp", "2 anchors", "external, labelled", "Cora public split", ev("inputs/registered/wp10_audit.json"), MODEL_B, claim_strength="boundary"),
]
for f in new:
    if f["id"] not in have:
        pb["facts"].append(f)
# threshold fact: cover both held-out graphs
for f in pb["facts"]:
    if f["id"] == "B-threshold":
        f["statement"] = ("theta=1.0 lies within one seed s.d. of the accuracy plateau on all five scanned graphs "
                          "including both completed heterophilic holdouts (chameleon optimum 0.75, 0.12 pp from theta=1; "
                          "Squirrel plateau 0.75-1.25 with theta=1.0 inside one s.d.); the q999 adaptive control loses "
                          "22.9 pp on Squirrel.")
        f["value"] = "plateau 0.75-1.25; q999 delta -22.9"
        f["measurement_scope"] = "chameleon and Squirrel held-out cells"
        f["evidence"] = ev("results/jB3_squirrel_complete.json", "inputs/paper-B-baseline.pdf")
(RUN / "writing/paperB-packet.json").write_text(json.dumps(pb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("packets patched:", len(pa["facts"]), "A facts,", len(pb["facts"]), "B facts")
