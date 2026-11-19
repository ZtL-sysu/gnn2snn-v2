import json, re, subprocess, sys
from pathlib import Path

RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade")
PRIOR = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261028-174053-gnn-snn-conv")
SKILL = Path(r"[WORKSPACE]/.claude/skills/f-research-v1/scripts")

def flatten(main: Path) -> str:
    out = []
    for line in main.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*\\input\{([^}]+)\}", line)
        if m:
            sub = (main.parent / m.group(1)).with_suffix(".tex")
            if sub.exists():
                out.append(flatten(sub))
        else:
            out.append(line)
    return "\n".join(out)

policy = {"schema_version": 1, "min_unique": 36, "projectquota": 36,
          "as_of": "2026-11-18", "recent_window_years": 3,
          "window_rationale": "Frozen 3-year window 2024-2026 for a fast-moving topic; carried from the accepted policy with as_of advanced to the journal revision date.",
          "require_recent_majority": True}
(RUN/"writing/citation-policy.json").write_text(json.dumps(policy, indent=2)+"\n", encoding="utf-8")

newB = [
 {"key": "ramesh2025pascal", "title": "PASCAL: Precise and Efficient ANN-SNN Conversion using Spike Accumulation and Adaptive Layerwise Activation", "year": 2025,
  "source_url": "https://arxiv.org/abs/2505.01730", "verification_status": "verified", "relevance_status": "relevant",
  "support_locations": ["related work, per-layer timesteps paragraph"]},
 {"key": "guo2026mixed", "title": "Mixed-Timestep Spiking Neural Networks: A Temporal Alignment Framework for High-Efficiency Neuromorphic Computing", "year": 2026,
  "source_url": "https://doi.org/10.3389/fnins.2026.1783326", "verification_status": "verified", "relevance_status": "relevant",
  "support_locations": ["related work, per-layer timesteps paragraph"]},
 {"key": "nallathambi2022ldsnn", "title": "Layerwise Disaggregated Evaluation of Spiking Neural Networks", "year": 2022,
  "source_url": "https://doi.org/10.1145/3531437.3539708", "verification_status": "verified", "relevance_status": "relevant",
  "support_locations": ["related work; experiments sec:allocation"]},
]
newA = []

for tag, tree, extra in [("A", RUN/"tex_audit-journal", newA), ("B", RUN/"tex-journal", newB)]:
    flat = flatten(tree/"main.tex")
    (RUN/f"writing/flat-{tag}.tex").write_text(flat, encoding="utf-8")
    inv = json.loads((PRIOR/f"contracts/citation-gate/inventory-{'a' if tag=='A' else 'b'}.json").read_text(encoding="utf-8"))
    by = {r["key"]: r for r in inv}
    for r in extra:
        by[r["key"]] = r
    # li2023bit1 venue-year consistency (ICLR 2024)
    if "li2023bit1" in by:
        by["li2023bit1"]["year"] = 2024
    inv2 = list(by.values())
    (RUN/f"writing/citation-inventory-{tag}.json").write_text(json.dumps(inv2, indent=2)+"\n", encoding="utf-8")
    rpt = RUN/f"audits/citation-gate-{tag}.json"
    res = subprocess.run([sys.executable, "-I", str(SKILL/"citation_requirements.py"),
                          "--bib", str(tree/"refs.bib"), "--manuscript", str(RUN/f"writing/flat-{tag}.tex"),
                          "--inventory", str(RUN/f"writing/citation-inventory-{tag}.json"),
                          "--policy", str(RUN/"writing/citation-policy.json"), "--report", str(rpt)],
                         capture_output=True, text=True)
    print(tag, "exit", res.returncode)
    print(res.stdout[-600:])
    print(res.stderr[-300:])
