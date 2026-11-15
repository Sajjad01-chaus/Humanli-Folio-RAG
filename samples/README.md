# Sample document provenance

The three PDFs in `pdfs/` are **fictional operational reports authored for this assignment**.
Northstar is a fictitious organization. These are not Humanli internal documents or real company policies.

Source of truth: [`corpus.json`](corpus.json). Rebuild with `python -m scripts.make_samples`.
The generated PDFs use invariant metadata for deterministic duplicate-ingestion tests.
All sample text is provided under CC0-1.0 so reviewers may redistribute it.

- Operations Handbook 2025: 3 pages; remote work, expenses, access, retention, suppliers.
- Incident Policy 2024: 2 pages; historical response and recovery requirements.
- Incident Policy 2025: 2 pages; superseding requirements and changed recovery objectives.

The corpus deliberately includes policy changes, exceptions, shared facts, and explicit information gaps.
It is small and controlled. It does **not** demonstrate performance on complex financial tables,
long reports, diverse scans, or independently collected questions. Upload your own PDFs to assess those cases.

PDF page numbers shown by Folio are physical, one-based PDF page indices, not printed section/page labels.

## Public report corpus

`public/` contains unmodified reports downloaded from NIST's official publication server:

- NIST AI 100-1, *AI Risk Management Framework 1.0* (January 2023), 48 physical PDF pages.
- NIST AI 600-1, *Generative Artificial Intelligence Profile* (July 2024), 64 physical PDF pages.

`public/manifest.json` records original URLs, sizes, and SHA-256 checksums. These are NIST publications,
not authored sample policies, and their original notices and attribution remain in the files.
Regenerate the download with `python -m scripts.fetch_public_samples`.
The Folio code license does not relicense third-party publications.

Six additional authored regression cases use this corpus in `evaluation/public-cases.json`.
Their exact expected pages are reference locations, not an exhaustive enumeration of all valid evidence
pages in these long documents; page-recall results should be interpreted with that limitation.
