# Assignment coverage

**Selected assignment:** Generative AI — Intelligent Multi-Document RAG System.

- **Upload multiple PDFs:** `POST /api/documents`, multipart uploads, per-file outcomes, catalog and file persistence.
- **Natural-language questions:** `POST /api/sessions/{id}/chat`; Groq GPT-OSS generates structured claims.
- **Retrieve across documents:** local semantic search + BM25 + fusion + cross-encoder reranking.
- **Ground answers in uploaded documents:** bounded evidence context, per-claim sources, reflection and abstention.
- **Relevant source references:** original document identity, physical PDF page, full retrieved excerpt, PDF page link.
- **Unavailable information:** explicit `unavailable` and `partial` outcomes; no web-search fallback.
- **Chat history:** persisted session/turn records, restored by the UI after reload/restart.
- **Conversational memory:** bounded history-aware standalone-question rewrite before retrieval.
- **Sample reports:** three fictional operational PDFs plus two original public NIST technical reports.
- **Clear project structure and run instructions:** root README, pinned dependency lock, sample provenance, architecture, demo walkthrough.

## Reviewer verification

`tests/test_workflows.py` covers the basic assignment behaviors and isolation/persistence failures.
`tests/test_adaptive.py` covers bounded corrective retrieval, evidence reflection, and OCR page routing.
`evaluation/cases.json` includes 30 authored policy cases.
`evaluation/public-cases.json` includes six questions on longer real reports.
Run results preserve full responses and diagnostic metrics for inspection.

## What this submission does not claim

It is not a production multi-tenant service, an independently benchmarked state-of-the-art RAG model,
or an exact implementation of the Self-RAG/CRAG papers. Docker packaging is provided separately from
proof of a successful deployment. OCR routing tests use controlled inputs; actual OCR accuracy depends
on the installed Tesseract engine, language pack, scan quality, and layout.
