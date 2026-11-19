# P20–P21 Gate Reports — Journal submission optimization and final delivery

## P20 — journal submission optimization (`f-submit-v1`)

**pass.** The user-requested `f-submit-v1` pipelines were executed to S0–S10 for both manuscripts,
each in its own run root (`tex-journal/f-submit-v1-run-20261119-003907-paperB/`,
`tex_audit-journal/f-submit-v1-run-20261119-003907-paperA/`) with `--complete` PASS. Coverage:
manuscript module maps and narrative spines; independent scientific/integrity precheck (P9 reports
reused with dispositions); 39/29 comparator candidates -> exactly 10 verified SCI-indexed
high-impact papers per manuscript with sourced index/impact evidence and honest access limitations;
section-pattern digests; module scoring to the top-40% operational threshold; three logged S7/S8
revision rounds (defect count decreased every round; final closure verified); final submission
audit with current hash binding. Journal-specific constraints remain unresolved by design (author
journal selection pending) and were recorded as such — no limit, template, or declaration was
invented. No submission or upload occurred.

## P21 — final delivery

**pass.** Delivery bundles per paper (`delivery/`): `revised-manuscript.pdf`, `source.zip`
(clean-extract + compile: 0 errors), `upload-map.md` (file-to-upload map; author-pending fields
named), `delivery-manifest.json` (SHA-256 of every delivery artifact; preserved-original note:
`sources/` is the immutable S0 snapshot). Research-run artifacts complete under this run root:
experiments (`results/`, `logs/`, `protocols/`), traceability (`experiment-traceability.md`),
figure/table audits (P18 report), reference dedup (`audits/citation-gate-*.json`,
`audits/p12-citation-verification.md`), submission constraints/metadata (in the f-submit run
roots, author-pending), final audits, and both f-research `pipeline-state.json`/`.md` ledgers.

Manifest proof: every deliverable and process artifact named in this report resolves under the
guidance directory; source-PDF-package consistency verified by the clean-extract compilation
(`clean-extract/extract-test.log`, 0 errors both).

Author-managed pending items (blocking only the act of submission, not the package): target
journals, authors/affiliations/correspondence, keywords, funding, competing interests,
generative-AI disclosure, code-release URL, publication route, submission approval.
