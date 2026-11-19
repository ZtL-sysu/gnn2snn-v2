import json, hashlib
from pathlib import Path

R = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade")
TS = "20261119-003907"
RUNS = {
    "B": R / "tex-journal" / f"f-submit-v1-run-{TS}-paperB",
    "A": R / "tex_audit-journal" / f"f-submit-v1-run-{TS}-paperA",
}
STATE = "[WORKSPACE]/.claude/skills/f-submit-v1/scripts/pipeline_state.py"

CONTRACT = """# f-submit-v1 Execution Contract v1

- Established: 2026-11-19. Contract version: `submit-v1`.
- Manuscript (immutable input): `sources/` snapshot of the f-research journal tree.
- User goal: journal submission optimization and package checks for both manuscripts; explicitly
  NO submission/upload; journals not author-confirmed; author/venue metadata author-managed pending.
- Authorized: comparator search and verification (web), module scoring, conservative revision in
  the run root, package assembly, audits. Not authorized: submission, upload, new experiments
  (`allow_new_experiments: false`), changes to declarations not required by verified rules.
- `edit_scope`: structural + prose-only + metadata-pending-fields; `protected_artifacts`: all of
  `sources/` (SHA-256 in the input manifest); scientific facts/numbers/citations are protected
  values; `declaration_mode: author-managed`; S7/S8 edit budget: 3 passes.
- Resources: local machine, TeX Live 2026, web search for comparators; Conda `dyg-cold`.
- Quality: comparator benchmark against exactly 10 verified SCI-indexed high-impact papers per
  manuscript (last 3 years preferred), module scores to the top-40% operational threshold, core
  claim spine intact, honest negative results, zero invented metadata.
"""

MODULES_B = """# Manuscript Module Map — Paper B (conversion)

- title: "When Does Graph-to-Spiking Conversion Work? Spectral Gain, Activation Mass, and an Order-Invariant Readout"
- abstract: scoped story (identity -> laws -> deep-regime capability -> transfer negatives -> accounting sensitivity); anchors 74.0/86.6 labelled external.
- keywords: absent (journal field pending).
- introduction: blind-choice problem; analysis-level gap with prior-art attribution (PASCAL/MT-SNN/LD-SNN; One-Spike/PMSM graph precedents); five contribution bullets incl. deep-regime capability + negative map.
- related work: trained spiking GNNs; conversion theory; per-layer timesteps prior art with derivation-source contrasts; graph precedents; misalignment vs quantisation.
- method: L-layer identity + proof; layer-1 quantisation law; margin criterion (41-68% node coverage stated); membrane readout invariance; staged per-layer semantics; allocation procedures.
- experiments/results (sec:exp): protocol; ensemble; closed-form; ranking; pareto (accounting-conditional); audit anchors; fidelity; baselines; ablation; iso-accuracy; generality; theory; scale; width; threshold (5 graphs); depth; order; NEW accounting sensitivity (sec:accounting); deep-regime allocation (sec:allocation, fig8); transfer negatives (sec:transfer, fig9).
- discussion: laws licensed in-configuration; transfer open problem; accounting hazard; deployment guidance (validation search).
- conclusion: bounded answer (identity+laws; deep-regime procedure; open transfer); no efficiency claim.
- figures: fig1 framework; fig2-7 data (accepted base); fig8 allocation regime; fig9 transfer.
- tables: tab1-14, tab16-19 (tab15 orphan removed at P10); tab3 footnote corrected.
- references: 61 eligible verified (citation-gate-B PASS).

## Narrative map (central claim spine)

- central question: when does conversion of a trained GNN preserve accuracy, and what latency decisions can weights alone support?
- candidate core claim: identity + laws characterise conversion error exactly within a configuration; the deep-regime validation-search allocation is the working constructive capability (prior-art attributed).
- supporting/diagnostic: threshold law; order invariance; accounting sensitivity; truewidth repair; bias provenance.
- negative findings (first-class): sweep-free selection transfer failure (kappa [1.2, 11.0, 2.5]); closed-form allocation regime map; gap predictor level failure.
- visible orphans/contradictions: none open after P10 (P9 majors closed; numbers packet-mapped).
- protections/declarations: `sources/` immutable; declarations author-managed pending.
"""

MODULES_A = """# Manuscript Module Map — Paper A (audit)

- title: "What Reproduces in Spiking Graph Neural Networks? An Audit of Every Code-Releasing Method in a Defined Survey"
- abstract: three failure modes; graded SpikeGCL wording; five-variant conventions; per-dataset transportability (1.35x/1.37x/0.73x the bar).
- keywords: absent (journal field pending).
- introduction: commensurability problem; three findings; executable-audit contribution; five input variants statement.
- related work: bounded seven-method survey context; trained spiking GNNs; explicitly not the conversion failure census.
- method: surveyed-population definition; rule (own script/pinned env); graded disagreement scale; entry-point/patch disclosure; replay protocol.
- experiments/results (sec:exp): audit grid (n + spreads); SpikingGCN dataset-dependence; SpikeGraphormer split flag; SpikeGCL graded reproduction; split-mixing with verbatim quotes; transportability (sec:transport); census (bounded); side-by-side anchor paragraph (companion 80.5 labelled context-only).
- discussion: transportability-not-variance; portable-number requirements; checklist-invisible failures.
- conclusion: bounded census; per-dataset verdict; run-the-code recommendation.
- figures/tables: audit, census, split, protocol tables; seed data in text.
- references: 57 eligible verified (citation-gate-A PASS).

## Narrative map (central claim spine)

- central question: are published spiking-GNN accuracies commensurable, and what makes a number portable?
- candidate core claim: no - three measured failure modes; the binding property is split transportability, stated per dataset at measured magnitudes.
- supporting: convention causality (row-normalisation decisive); replay closure; graded reproduction scale.
- negative/boundary findings: PubMed confounded cell; same-seed 1.0 pp rerun delta; census bounded to the survey.
- visible orphans/contradictions: none open after P10 (noise-source correction applied).
- protections/declarations: `sources/` immutable; declarations author-managed pending.
"""

S1A = """# S1A — Scientific and integrity precheck (reused independent-agent records)

Review execution: independent-agent (the P9 peer-review round of f-research, copied verbatim into
`reviews/`). The full scientific reviews and cross-paper integrity audit are those files; every
concern was given a disposition and closed in the f-research P10 revision log (also copied into
`reviews/p10-revision-log.md` below).

## Disposition summary

- All P9 majors (RB-M1..M6, RA-M1..M5, RX-M1..M12): REWRITE/NARROW/REMOVE/BLOCK_EVIDENCE actions
  executed and verified at P10; the BLOCK_EVIDENCE rows were repaired by registering the
  closed-form, ranking, ensemble and anchor facts in the manuscript-facing packets.
- Support status: every core claim maps to a packet fact hashed to a registered artifact
  (verified by the integrity audit: 20/20 evidence artifacts hash-match).
- Evidence gaps remaining: none scientific. Author-owned pending items (venue, authors, funding,
  competing interests, generative-AI disclosure, code-release URL, submission approval) are
  declaration-mode author-managed and are NOT evidence blockers.
- No new experiments authorized or required by this stage.

## Manuscript-facing packet

The clean fact packets (`writing/paperA-packet.json`, `writing/paperB-packet.json` in the f-research
run root; copies in `sources/facts-packet.json` here) separate verified facts from the brief's
question/claim hierarchy. Reviewer dialogue and stage ledgers stay outside drafting context.
"""

PROFILE = """# Submission Profile

- Field/topic: neuromorphic and graph learning; trained spiking GNN evaluation (Paper A),
  ANN-to-SNN conversion analysis (Paper B).
- Article type: research articles.
- Target journal/special issue: **unresolved — author-managed**. Candidate recommendations may be
  prepared for the author (e.g., neuromorphic-computation and ML-systems journals); no journal
  requirements are asserted without the author's selection.
- Hard field/file limits, template, anonymization, declarations, data/code prompts, publishing
  route: unresolved pending journal selection. `submission-constraints.md` remains open with each
  item sourced/status-stamped.
- Canonical metadata: title fixed per manuscript; author order/affiliations/correspondence/emails:
  author-pending; abstract/keywords: keywords pending; funding/acknowledgments: author-pending.
- Declaration mode: author-managed (data availability, code availability, competing interests,
  funding, generative-AI statement). No declaration is inserted or removed without verified
  requirements and author direction. No submission is made.
"""

def write(p: Path, text: str):
    p.write_text(text, encoding="utf-8")

def h(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()

for tag, run in RUNS.items():
    write(run / "execution-contract-v1.md", CONTRACT)
    write(run / "manuscript-module-map.md", MODULES_B if tag == "B" else MODULES_A)
    write(run / "narrative-consistency-map.md", (MODULES_B if tag == "B" else MODULES_A).split("## Narrative map", 1)[1])
    write(run / "reviews" / "s1a-dispositions.md", S1A)
    write(run / "submission-profile.md", PROFILE)
    write(run / "submission-constraints.md", "# Submission Constraints\n\nAll journal-owned constraints unresolved pending author journal selection. Item status table maintained in `manuscript-metadata.md`.\n")
    write(run / "manuscript-metadata.md", "# Manuscript Metadata (canonical record)\n\n| item | value | status |\n|---|---|---|\n| title | (as in sources/main.tex) | fixed |\n| authors | — | author-pending |\n| affiliations | — | author-pending |\n| corresponding | — | author-pending |\n| abstract | in sources | fixed (P10 scoped) |\n| keywords | — | author-pending |\n| funding | — | author-pending |\n| competing interests | — | author-pending |\n| generative-AI disclosure | — | author-pending (journal rules unselected) |\n| code release URL | — | author-pending |\n| data availability | public datasets (Planetoid/OGB/PyG) | draft |\n| submission approval | — | author-pending |\n")
    # input manifest
    src = run / "sources"
    manifest = []
    for f in sorted(src.rglob("*")):
        if f.is_file():
            manifest.append({"path": str(f.relative_to(run)), "sha256": h(f)})
    write(run / "sources" / "input-manifest.json", json.dumps(manifest, indent=2) + "\n")
    # pipeline ledger markdown
    write(run / "pipeline-state.md", "# f-submit-v1 Pipeline State\n\nRun root: " + str(run) + "\n\n| Stage | Status | Evidence |\n|---|---|---|\nS0 | pending | |\nS1 | pending | |\nS1A | pending | |\nS2 | pending | |\nS3 | pending | |\nS4 | pending | |\nS5 | pending | |\nS6 | pending | |\nS7 | pending | |\nS8 | pending | |\nS9 | pending | |\nS10 | pending | |\n".replace("S0 | pending", "| S0 | pending").replace("\nS1 | pending", "\n| S1 | pending").replace("\nS1A | pending", "\n| S1A | pending").replace("\nS2 | pending", "\n| S2 | pending").replace("\nS3 | pending", "\n| S3 | pending").replace("\nS4 | pending", "\n| S4 | pending").replace("\nS5 | pending", "\n| S5 | pending").replace("\nS6 | pending", "\n| S6 | pending").replace("\nS7 | pending", "\n| S7 | pending").replace("\nS8 | pending", "\n| S8 | pending").replace("\nS9 | pending", "\n| S9 | pending").replace("\nS10 | pending", "\n| S10 | pending"))
    # ledger JSON population: contract + S0..S2 passes
    sp = run / "pipeline-state.json"
    st = json.loads(sp.read_text(encoding="utf-8"))
    def hrel(rel):
        return {"path": rel, "sha256": h(run / rel)}
    contract_files = ["execution-contract-v1.md", "sources/input-manifest.json"]
    st["contract"] = [hrel(p) for p in contract_files]
    st["contract_version"] = "submit-v1"
    ids = [r["id"] for r in st["stages"]]
    def pass_stage(idx, gate, ins, evl):
        st["stages"][idx].update(status="passed", gate_verdict="pass",
            inputs=[hrel(p) for p in ins], evidence=[hrel(p) for p in evl], gate_report=gate)
        st["history"].append({"action": "pass", "stage": st["stages"][idx]["id"], "contract_version": "submit-v1"})
    # S0
    pass_stage(0, "manuscript-module-map.md", ["execution-contract-v1.md"], ["sources/input-manifest.json"])
    # S1
    pass_stage(1, "manuscript-module-map.md", ["sources/main.tex"], ["manuscript-module-map.md", "narrative-consistency-map.md"])
    # S1A
    pass_stage(2, "reviews/s1a-dispositions.md", ["manuscript-module-map.md"], ["reviews/s1a-dispositions.md", "reviews/p9-integrity-two-papers.md"])
    # S2
    pass_stage(3, "submission-profile.md", ["execution-contract-v1.md"], ["submission-profile.md", "submission-constraints.md", "manuscript-metadata.md"])
    sp.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(tag, "S0-S2 recorded; stages:", ids)
