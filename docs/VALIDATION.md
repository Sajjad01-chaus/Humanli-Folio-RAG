# Validation record

Validated locally on Windows with Python 3.12 on 15 November 2026.

- **39 integration/behavior tests passed.** Covers ingestion, duplicate suppression, exact physical
  pages, filtering, deletion, session isolation, restart persistence, citation validation, provider
  failure, bounded correction/regeneration, query decomposition, OCR routing, and token pacing.
- Ruff checks passed for `app`, `scripts`, and `tests`.
- Strict mypy passed for all 20 application and utility modules. Dependency injection, structured-log privacy,
  granular error mapping, and cause preservation are covered by the engineering tests.
- The post-refactor live compound-question check completed successfully with GPT-OSS 20B and emitted
  structured stage/provider logs; both requested facts were answered in that run.
- JavaScript syntax check passed for `app/static/app.js`.
- Installed dependency compatibility check passed.
- `docker compose config --quiet` passed. The Docker engine was not running, so the image build and
  container startup have **not** been validated. The README does not claim a successful deployment.
- Real Groq requests completed for GPT-OSS 120B and GPT-OSS 20B. The selected model is recorded per run.
- Cached local sentence embeddings and the cross-encoder ran successfully.
- Sample PDF layout was rendered and visually inspected.
- The browser workflow was checked for sample loading, answer/source display, and restoring history.
- The final GPT-OSS 20B browser walkthrough used all five PDFs (119 physical pages). It answered the
  2024/2025 policy comparison and a contextual follow-up with the correct cited policy pages, in
  approximately 19.0 and 12.5 seconds respectively. Screenshots are in `docs/screenshots/`.
- The interface was inspected at the narrow default layout and desktop/source-panel breakpoints.

## Model quality evidence

- `evaluation/policy-baseline-results.json`: initial 30-case live 120B run; 30 completed, no request errors,
  24/26 expected-term checks, all four unavailable questions abstained. Includes documented failures.
- `evaluation/public-results.json`: initial six-case 120B test on real NIST reports; completed without
  request errors; exposed a compound-question retrieval failure.
- `evaluation/final-regression-results.json`: eight targeted cases with 20B after bounded-revision work;
  completed without errors; all included expected-term checks, citation-page checks, and abstention check passed.
- `evaluation/compound-final-results.json`: final targeted compound-question test with 20B; both
  requested facts answered. It is a development regression, not an independent benchmark.

Read `evaluation/REVIEW.md` for metric definitions, false negatives, failures, and validity limits.

## OCR status

Per-page rendering, routing, failure reporting, and physical-page preservation are covered by tests.
Tesseract is not installed on the development host, so actual scanned-text recognition was **not**
executed there. The optional integration requires Tesseract and its English language pack; the Docker
recipe installs them. No OCR accuracy score is claimed.
