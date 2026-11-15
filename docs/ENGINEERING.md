# Engineering standards review

The pre-publication review found incomplete typing, constructor-created dependencies, limited application
logging, and overly broad adapter error handling. Those gaps were addressed before publication.

## Architecture and responsibilities

The target is a reproducible single-user assignment service. The architecture keeps HTTP handling,
PDF ingestion, persistence, retrieval, model integration, and graph orchestration separate. It does not
claim multi-tenant authorization or distributed-worker readiness.

- `container.py`: composition factory; accepts replacement repository, retrieval, model, and OCR dependencies.
- `ports.py`: typed structural contracts for service dependencies.
- `models.py`: validated generation schemas and typed application records.
- `ingest.py`: document ingestion service; no SQL or HTTP concerns.
- `store.py`: SQLite repository and transaction lifecycle; SQL is confined here.
- `retrieval.py`: local ranking and backend validation; embedding and reranking adapters can be injected.
- `llm.py`: model-facing operations; HTTP client factory, tokenizer, and sleep function are injectable.
- `workflow.py`: bounded LangGraph transitions with typed state.
- `main.py`: FastAPI request validation, dependency resolution, and response/error mapping.

## Typing and checks

`mypy app scripts` enforces strict checking across 16 application modules and four utility modules. There are no blanket application
type-ignore directives or `Any`-typed application contracts. Unannotated third-party scientific-library
internals are excluded from import analysis; their outputs cross explicit typed adapter boundaries.
The evaluation/sample utilities are included in the same strict CI gate, with typed ReportLab stubs.

CI runs Ruff, strict mypy, pytest, and JavaScript syntax checks. Python and frontend sources are formatted.
Comments are limited to short explanations of non-obvious intent; architectural detail belongs in docs.

## Error propagation and logging

Storage, retrieval, provider timeout, transport, rate-limit, and invalid-document failures have explicit
exception types. Adapter errors preserve their original causes. Unexpected exceptions are caught only
at the HTTP/CLI entry boundaries, logged, and terminated with failure rather than a fabricated result.
Invalid structured model output and corrupt persisted history are rejected explicitly.

Application logs are JSON with timestamp, event, level, and request ID. HTTP responses include the same
request ID. Logs record safe metadata and stack-frame locations, excluding questions, PDF passages,
provider response bodies, credentials, exception messages, and source-code lines. Secret settings are
excluded from their object representation. These privacy properties are regression-tested.

The evaluation runner stops on a dependency failure, writes the observed error, and exits unsuccessfully.
Unexpected CLI failures terminate with structured diagnostics. Known fallback modes are explicitly
configured and labeled; failed semantic retrieval does not silently downgrade to lexical retrieval.

## Cleanup

Temporary quota/model diagnostic scripts, the redundant smoke evaluation, the duplicated regression
question file, the superseded single-case artifact, and unused screenshot were removed. The retained
evaluation artifacts document the baseline, public-report behavior, and final targeted regressions.
Provider/library compatibility pins are intentional runtime constraints, not unused feature dependencies.

## Remaining verification limits

Actual Tesseract recognition and Docker image startup were not exercised on this host. Their optional
configuration and these limits are documented. Model accuracy evidence remains a small authored
regression suite, not an independent production benchmark.
