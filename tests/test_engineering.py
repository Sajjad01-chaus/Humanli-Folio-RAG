import io
import json
import sqlite3
import sys
from dataclasses import replace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import create_services
from app.errors import PersistenceError, ProviderError, ProviderTimeoutError, RetrievalError
from app.llm import LLM
from app.main import create_app
from app.observability import configure_logging
from app.retrieval import Retriever
from app.store import Store
from tests.conftest import pdf_bytes


class SmallEncoder:
    def encode(self, text, **kwargs):
        return [0] * len(text.split())


def model_client(settings, handler):
    return LLM(
        replace(settings, api_key="test-only", provider_tpm=0),
        encoder=SmallEncoder(),
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )


def test_dependencies_are_injectable_without_network(settings):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "answered",
                                    "claims": [{"text": "Three remote days.", "sources": [1]}],
                                }
                            )
                        }
                    }
                ]
            },
        )

    llm = model_client(settings, handler)
    services = create_services(settings, llm=llm)
    with TestClient(create_app(services=services)) as client:
        client.post(
            "/api/documents",
            files={
                "files": (
                    "policy.pdf",
                    pdf_bytes("Remote work is permitted three days each week."),
                    "application/pdf",
                )
            },
        )
        session = client.post("/api/sessions").json()["id"]
        response = client.post(f"/api/sessions/{session}/chat", json={"question": "How many remote days?"})
        assert response.status_code == 200 and response.json()["answer"].startswith("Three remote days.")


def test_provider_timeout_preserves_cause(settings):
    def handler(request):
        raise httpx.ReadTimeout("sensitive upstream details", request=request)

    llm = model_client(settings, handler)
    with pytest.raises(ProviderTimeoutError) as error:
        llm.complete([{"role": "user", "content": "Hello"}])
    assert isinstance(error.value.__cause__, httpx.ReadTimeout)
    assert "sensitive" not in str(error.value)


def test_invalid_provider_json_is_explicit(settings):
    llm = model_client(
        settings, lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})
    )
    with pytest.raises(ProviderError, match="valid response"):
        llm.complete([{"role": "user", "content": "Hello"}])


def test_invalid_query_plan_is_not_silently_ignored(settings, monkeypatch):
    llm = LLM(settings)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"queries": "invalid"})
    with pytest.raises(ProviderError, match="invalid structured output"):
        llm.plan_queries("Compare these policies")


def test_storage_errors_are_translated_and_chained(settings):
    store = Store(settings.data_dir)
    with pytest.raises(PersistenceError) as error:
        with store.connect() as db:
            db.execute("INSERT INTO nonexistent_table VALUES (1)")
    assert isinstance(error.value.__cause__, sqlite3.Error)
    assert store.documents() == []


def test_corrupt_history_fails_explicitly(client, uploaded):
    session = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{session}/chat", json={"question": "Remote work policy?"})
    with client.app.state.store.connect() as db:
        db.execute("UPDATE turns SET response='{}' WHERE session_id=?", (session,))
    response = client.get(f"/api/sessions/{session}")
    assert response.status_code == 503 and response.json()["code"] == "storage_unavailable"


def test_retrieval_model_failure_is_not_silently_downgraded(client, uploaded, settings):
    class FailedEncoder:
        def encode(self, *args, **kwargs):
            raise OSError("private details")

    retriever = Retriever(
        client.app.state.store, replace(settings, embeddings="sentence-transformers"), encoder=FailedEncoder()
    )
    with pytest.raises(RetrievalError, match="Retrieval is unavailable"):
        retriever.search("remote work")


def test_invalid_reranker_scores_are_rejected(client, uploaded, settings):
    class InvalidReranker:
        def predict(self, *args, **kwargs):
            return np.array([])

    retriever = Retriever(client.app.state.store, replace(settings, rerank=True), reranker=InvalidReranker())
    with pytest.raises(RetrievalError, match="invalid scores"):
        retriever.search("remote work")


def test_request_logs_are_correlated_and_do_not_contain_sensitive_payloads(client, uploaded, monkeypatch):
    stream = io.StringIO()
    sensitive = "PRIVATE-DOCUMENT-CONTENT-AND-KEY"

    def fail(*args, **kwargs):
        raise RuntimeError(sensitive)

    monkeypatch.setattr(client.app.state.engine, "ask", fail)
    configure_logging(stream)
    try:
        session = client.post("/api/sessions").json()["id"]
        response = client.post(f"/api/sessions/{session}/chat", json={"question": sensitive})
        lines = [json.loads(line) for line in stream.getvalue().splitlines()]
        request_id = response.headers["X-Request-ID"]
        assert response.status_code == 500 and response.json()["request_id"] == request_id
        assert any(
            row["event"] == "request.unhandled_failure" and row["request_id"] == request_id for row in lines
        )
        assert sensitive not in stream.getvalue() and sensitive not in response.text
    finally:
        configure_logging(sys.stdout)


def test_credentials_are_excluded_from_settings_repr_and_env_is_read_per_instance(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "private-key-not-for-logs")
    settings = Settings()
    assert settings.api_key == "private-key-not-for-logs"
    assert settings.api_key not in repr(settings)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="EMBEDDING_BACKEND"):
        Settings(embeddings="misspelled-backend")


def test_missing_stored_pdf_returns_an_actionable_storage_error(client, uploaded):
    path = client.app.state.store.root / "uploads" / f"{uploaded['id']}.pdf"
    path.unlink()
    response = client.get(f"/api/documents/{uploaded['id']}/pdf")
    assert response.status_code == 503
    assert response.json()["code"] == "storage_unavailable"
    assert "upload it again" in response.json()["detail"]
