import json, re, shutil, hashlib
from pathlib import Path

RUN = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261118-181848-journal-upgrade")
PRIOR = Path(r"[WORKSPACE]/auto-paper/20261028-171500/Spiking-GNN-Conversion/f-research-v1-run-20261028-174053-gnn-snn-conv")
FABLE = Path(r"[WORKSPACE]/docs/research-plans/claude-fable5.2")
DEST = Path(r"C:/code/gnn2snn-v2")
TS = "20261119-003907"

if DEST.exists():
    shutil.rmtree(DEST)
DEST.mkdir(parents=True)

def copy_tree(src: Path, dst: Path, skip_names=(".git", "__pycache__"), skip_suffix=(".pyc",)):
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.rglob("*")):
        if any(part in skip_names for part in f.parts):
            continue
        if f.suffix in skip_suffix or f.name in skip_names:
            continue
        if f.is_file():
            tgt = dst / f.relative_to(src)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, tgt)

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# 1. Papers (final revised LaTeX sources + PDFs)
for tag, sub, label in [("B", "tex-journal", "paper-b-conversion"), ("A", "tex_audit-journal", "paper-a-audit")]:
    run = RUN / sub / f"f-submit-v1-run-{TS}-paper{tag}" / "sources"
    dest = DEST / label
    copy_tree(run, dest, skip_names=("__pycache__",), skip_suffix=(".pyc", ".aux", ".log", ".out", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".synctex.gz"))
    copy_file(RUN / sub / f"f-submit-v1-run-{TS}-paper{tag}" / "delivery" / "revised-manuscript.pdf", dest / "revised-manuscript.pdf")

# 2. Documents: journal-upgrade audits
docs = DEST / "docs" / "journal-upgrade-audits"
audit_files = [
    "audits/deviation-audit.md", "audits/jA1-outcome.md", "audits/jA2-outcome.md",
    "audits/jB1a-outcome.md", "audits/jB1-search-arm-outcome.md", "audits/jB1-search-arm-prereg.md",
    "audits/jb1a-primary-prereg.md", "audits/ccfa-novelty-map-E-B1.md",
    "audits/correction-loop-state.md", "audits/correction-ledger.md",
    "audits/sop-convention-predeclaration.md", "audits/p5-strong-result-lock.md",
    "audits/p9-concern-ledger.md", "audits/p10-revision-log.md", "audits/p20-p21-delivery.md",
]
for rel in audit_files:
    p = RUN / rel
    if p.exists():
        copy_file(p, docs / Path(rel).name)

# 3. Documents: f-submit artifacts (comparators, scoring, audits)
for tag, sub, label in [("B", "tex-journal", "paper-b"), ("A", "tex_audit-journal", "paper-a")]:
    run = RUN / sub / f"f-submit-v1-run-{TS}-paper{tag}"
    dest = DEST / "docs" / f"f-submit-{label}"
    for name in ["comparator-literature-table.md", "comparator-section-patterns.md",
                 "module-quality-review.md", "s7-revision-log.md", "s8-rescore.md",
                 "s8-rescore-round2.md", "final-submission-audit.md", "manuscript-module-map.md",
                 "narrative-consistency-map.md", "submission-profile.md", "manuscript-metadata.md"]:
        p = run / name
        if p.exists():
            copy_file(p, dest / name)

# 4. Code: journal protocols + writing scripts + engine dependencies
copy_tree(RUN / "protocols", DEST / "code" / "protocols", skip_names=("__pycache__",), skip_suffix=(".pyc",))
copy_tree(RUN / "writing", DEST / "code" / "writing", skip_names=("__pycache__",), skip_suffix=(".pyc",))
engine_dst = DEST / "code" / "engine"
for rel in ["experiments/core.py", "protocols/graph_data.py", "protocols/snn_gnn_experiment.py",
            "protocols/strengthen.py", "protocols/baselines_published.py", "protocols/spikinggcn_baseline.py"]:
    p = PRIOR / rel
    if p.exists():
        copy_file(p, engine_dst / Path(rel).name)

# 5. Results evidence (JSON only - no datasets)
res_dst = DEST / "results"
for f in sorted((RUN / "results").glob("*.json")):
    copy_file(f, res_dst / f.name)

# 6. Fable materials (external planning material; no datasets/attachments over 2 MB)
fdest = DEST / "fable-material"
copy_tree(FABLE / "github-gnn2gsnn", fdest / "github-gnn2gsnn", skip_names=(".git", "__pycache__"), skip_suffix=(".pyc",))
copy_tree(FABLE / "notion-export", fdest / "notion-export", skip_names=(".git", "__pycache__"), skip_suffix=(".pyc",))
for name in ["README.md", "source-manifest.json"]:
    p = FABLE / name
    if p.exists():
        copy_file(p, fdest / name)

# 7. Secret scan on staged copies
secret_pat = re.compile(r"gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}|sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|BEGIN (RSA|OPENSSH|EC|PGP) PRIVATE KEY")
findings = []
for f in DEST.rglob("*"):
    if f.is_file() and f.suffix.lower() not in (".pdf", ".png", ".jpg", ".svg", ".zip", ".npz", ".pt", ".pkl", ".gz", ".ico", ".woff", ".woff2", ".ttf"):
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in secret_pat.finditer(t):
            findings.append((str(f.relative_to(DEST)), m.group(0)[:24]))
print("SECRET FINDINGS:", len(findings))
for x in findings[:20]:
    print(" ", x)

# 8. Path scrub (public copies only): local absolute paths -> neutral tokens
path_pat = re.compile(r"[A-Za-z]:[\\/](?:Users[\\/][^\\/ \n]+|code[\\/][^ \n]*?phd-snn)|[WORKSPACE]|[WORKSPACE]")
scrubbed = 0
for f in DEST.rglob("*"):
    if f.is_file() and f.suffix.lower() not in (".pdf", ".png", ".jpg", ".svg", ".zip", ".npz", ".pt", ".pkl", ".gz", ".ico", ".woff", ".woff2", ".ttf"):
        try:
            t = f.read_text(encoding="utf-8")
        except Exception:
            continue
        t2 = path_pat.sub("[WORKSPACE]", t)
        if t2 != t:
            f.write_text(t2, encoding="utf-8")
            scrubbed += 1
print("path-scrubbed files:", scrubbed)
print("staged file count:", sum(1 for f in DEST.rglob('*') if f.is_file()))
