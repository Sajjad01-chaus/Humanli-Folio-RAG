from dataclasses import replace

from fastapi.testclient import TestClient

from app.errors import ProviderError
from app.ingest import chunk_page
from app.llm import LLM
from app.main import create_app
from app.models import Claim, GeneratedAnswer
from tests.conftest import pdf_bytes


def start(client):
    return client.post("/api/sessions").json()["id"]


def test_pdf_pages_are_real_not_total(client, uploaded):
    sid = start(client)
    response = client.post(
        f"/api/sessions/{sid}/chat", json={"question": "How long are support tickets retained?"}
    )
    assert response.status_code == 200
    assert response.json()["sources"][0]["page"] == 2
    assert response.json()["status"] == "evidence_only"
    assert client.get(f"/api/documents/{uploaded['id']}/pdf").content.startswith(b"%PDF-")


def test_duplicate_upload_is_idempotent(client, uploaded):
    data = pdf_bytes(
        "Remote work is permitted three days per week.",
        "Customer support tickets are retained for 24 months after closure.",
    )
    # Use original file: reportlab timestamps may vary across generation.
    data = client.get(f"/api/documents/{uploaded['id']}/pdf").content
    result = client.post("/api/documents", files={"files": ("copy.pdf", data, "application/pdf")}).json()
    assert result["documents"][0]["duplicate"] is True
    assert len(client.get("/api/documents").json()) == 1


def test_sessions_persist_across_restart_and_stay_separate(client, uploaded, settings):
    a, b = start(client), start(client)
    client.post(f"/api/sessions/{a}/chat", json={"question": "What is the remote work policy?"})
    assert len(client.get(f"/api/sessions/{a}").json()["turns"]) == 1
    assert client.get(f"/api/sessions/{b}").json()["turns"] == []
    with TestClient(create_app(settings)) as restarted:
        assert len(restarted.get(f"/api/sessions/{a}").json()["turns"]) == 1
        assert len(restarted.get("/api/documents").json()) == 1
        result = restarted.post(
            f"/api/sessions/{b}/chat", json={"question": "What is the remote work policy?"}
        ).json()
        assert result["session_id"] == b
        assert len(restarted.get(f"/api/sessions/{b}").json()["turns"]) == 1


def test_deletion_removes_file_and_retrievable_content(client, uploaded, settings):
    sid = start(client)
    client.post(f"/api/sessions/{sid}/chat", json={"question": "Remote work policy?"})
    client.delete(f"/api/documents/{uploaded['id']}")
    assert not (settings.data_dir / "uploads" / f"{uploaded['id']}.pdf").exists()
    result = client.post(f"/api/sessions/{sid}/chat", json={"question": "Remote work policy?"}).json()
    assert result["status"] == "unavailable"
    assert result["sources"] == []


def test_document_filter_cannot_retrieve_other_document(client, uploaded):
    second = client.post(
        "/api/documents",
        files={
            "files": (
                "incident.pdf",
                pdf_bytes("Production recovery time objective is one hour."),
                "application/pdf",
            )
        },
    ).json()["documents"][0]
    sid = start(client)
    result = client.post(
        f"/api/sessions/{sid}/chat",
        json={"question": "What is the recovery time objective?", "document_ids": [second["id"]]},
    ).json()
    assert result["sources"] and all(s["document_id"] == second["id"] for s in result["sources"])
    result = client.post(
        f"/api/sessions/{sid}/chat", json={"question": "Remote work?", "document_ids": []}
    ).json()
    assert result["sources"] == []


def test_reject_invalid_and_unreadable_uploads(client):
    result = client.post(
        "/api/documents", files={"files": ("bad.pdf", b"<script>x</script>", "application/pdf")}
    )
    assert result.json()["errors"]
    result = client.post("/api/documents", files={"files": ("blank.pdf", pdf_bytes(""), "application/pdf")})
    assert "No readable text" in result.json()["errors"][0]["error"]
    assert client.get("/api/documents").json() == []


def test_missing_session_and_stale_filter(client):
    assert client.post("/api/sessions/unknown/chat", json={"question": "Hello?"}).status_code == 404
    sid = start(client)
    assert (
        client.post(
            f"/api/sessions/{sid}/chat", json={"question": "Hello?", "document_ids": ["unknown"]}
        ).status_code
        == 404
    )
    assert client.post(f"/api/sessions/{sid}/chat", json={"question": "   "}).status_code == 422


def test_model_history_and_rewrite_are_session_scoped(client, uploaded, monkeypatch):
    engine = client.app.state.engine
    observed = []

    class FakeLLM:
        enabled = True

        def rewrite(self, question, history):
            observed.append(history)
            return "remote work policy" if history else question

        def answer(self, question, rewritten, evidence, history):
            return GeneratedAnswer(
                status="answered", claims=[Claim(text="Three days per week.", sources=[1])]
            )

    engine.llm = FakeLLM()
    a, b = start(client), start(client)
    client.post(f"/api/sessions/{a}/chat", json={"question": "What is the remote work policy?"})
    client.post(f"/api/sessions/{a}/chat", json={"question": "Who approves it?"})
    client.post(f"/api/sessions/{b}/chat", json={"question": "Who approves it?"})
    assert len(observed[1]) == 1
    assert observed[2] == []


def test_invalid_citations_are_rejected(settings, monkeypatch):
    llm = LLM(replace(settings, api_key="test"))
    monkeypatch.setattr(
        llm,
        "complete",
        lambda *a, **k: {"status": "answered", "claims": [{"text": "Made up", "sources": [99]}]},
    )
    import pytest

    with pytest.raises(ProviderError, match="citation validation"):
        llm.answer("Question", "Question", [{"name": "one.pdf", "page": 1, "text": "Evidence"}], [])


def test_provider_failure_does_not_save_turn(client, uploaded, monkeypatch):
    engine = client.app.state.engine
    monkeypatch.setattr(
        engine.llm, "rewrite", lambda *args: (_ for _ in ()).throw(ProviderError("Unavailable"))
    )
    sid = start(client)
    response = client.post(f"/api/sessions/{sid}/chat", json={"question": "Remote work policy?"})
    assert response.status_code == 502
    assert client.get(f"/api/sessions/{sid}").json()["turns"] == []


def test_token_protects_api_and_pdf(settings):
    with TestClient(create_app(replace(settings, access_token="private"))) as protected:
        assert protected.get("/api/documents").status_code == 401
        assert protected.get("/api/documents", headers={"Authorization": "Bearer private"}).status_code == 200
        assert protected.get("/").status_code == 200


def test_chunk_windows_cover_end_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = list(chunk_page(text))
    assert chunks[-1].endswith("word399")
    assert set(chunks[0].split()) & set(chunks[1].split())
    assert all(len(c.split()) <= 180 for c in chunks)


def test_uploaded_name_is_plain_data(client):
    result = client.post(
        "/api/documents",
        files={
            "files": (
                "../<script>alert(1)</script>.pdf",
                pdf_bytes("A valid policy document with readable text."),
                "application/pdf",
            )
        },
    ).json()
    assert result["documents"][0]["name"] == "script>.pdf"


def test_delete_session_cascades_turns(client, uploaded):
    sid = start(client)
    client.post(f"/api/sessions/{sid}/chat", json={"question": "Remote work policy?"})
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    with client.app.state.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0
