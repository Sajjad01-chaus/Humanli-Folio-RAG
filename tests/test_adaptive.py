from dataclasses import replace

from app.errors import ProviderError
from app.llm import LLM
from app.models import Claim, GeneratedAnswer
from app.workflow import build_workflow


def evidence(identifier):
    return {
        "id": identifier,
        "document_id": "doc",
        "name": "policy.pdf",
        "page": 1,
        "text": "The recovery objective is one hour.",
        "rank_score": 0.02,
        "semantic_score": 0.8,
    }


def test_corrective_graph_retries_once_and_reflects(settings):
    calls = []

    class Search:
        def search(self, query, document_ids, mode):
            calls.append(query)
            return [evidence(query)]

    class Model:
        enabled = True

        def assess_evidence(self, question, chunks):
            return False, "improved recovery query"

        def answer(self, question, standalone, chunks, history):
            assert len(chunks) == 2
            return GeneratedAnswer(status="answered", claims=[Claim(text="One hour.", sources=[1])])

        def verify_claims(self, question, generated, chunks):
            return generated, 0

    graph = build_workflow(Search(), Model(), replace(settings, adaptive=True, verify=True))
    result = graph.invoke(
        {"question": "RTO?", "standalone": "RTO?", "document_ids": None, "mode": "hybrid", "history": []}
    )
    assert calls == ["RTO?", "improved recovery query"]
    assert result["retrieval_attempts"] == 2
    assert result["trace"][-1]["stage"] == "grounding_check"


def test_sufficient_evidence_skips_corrective_search(settings):
    class Search:
        calls = 0

        def search(self, *args, **kwargs):
            self.calls += 1
            return [evidence("one")]

    class Model:
        enabled = True

        def assess_evidence(self, *args):
            return True, "different query"

        def answer(self, *args):
            return GeneratedAnswer(status="unavailable")

    search = Search()
    graph = build_workflow(search, Model(), replace(settings, adaptive=True))
    graph.invoke(
        {"question": "RTO?", "standalone": "RTO?", "document_ids": None, "mode": "hybrid", "history": []}
    )
    assert search.calls == 1


def test_unsupported_claims_are_removed(settings, monkeypatch):
    llm = LLM(settings)
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: {"supported_claims": [0]})
    answer = GeneratedAnswer(
        status="answered",
        claims=[Claim(text="One hour.", sources=[1]), Claim(text="Insurance costs a million.", sources=[1])],
    )
    verified, removed = llm.verify_claims("RTO?", answer, [evidence("one")])
    assert removed == 1 and len(verified.claims) == 1
    assert verified.status == "partial"


def test_all_unsupported_claims_produce_abstention(settings, monkeypatch):
    llm = LLM(settings)
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: {"supported_claims": []})
    answer = GeneratedAnswer(status="answered", claims=[Claim(text="A million.", sources=[1])])
    verified, removed = llm.verify_claims("Cost?", answer, [evidence("one")])
    assert removed == 1 and verified.status == "unavailable" and not verified.claims


def test_invalid_reflection_indices_fail_closed(settings, monkeypatch):
    import pytest

    llm = LLM(settings)
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: {"supported_claims": [99]})
    with pytest.raises(ProviderError):
        llm.verify_claims(
            "RTO?",
            GeneratedAnswer(status="answered", claims=[Claim(text="One hour.", sources=[1])]),
            [evidence("one")],
        )


def test_ocr_fallback_preserves_physical_page(settings, monkeypatch):
    from app.ingest import ingest_pdf
    from app.store import Store
    from tests.conftest import pdf_bytes

    called = []

    def fake_ocr(content, index, command):
        called.append(index)
        return "This scanned page specifies a recovery objective of one hour."

    monkeypatch.setattr("app.ocr.read_scanned_page", fake_ocr)
    store = Store(settings.data_dir)
    result = ingest_pdf(
        store,
        "scan.pdf",
        pdf_bytes("Embedded text page with enough readable content.", ""),
        replace(settings, ocr=True),
    )
    assert called == [1] and result["ocr_pages"] == [2]
    assert any(row["page"] == 2 and "scanned page" in row["text"] for row in store.chunks())


def test_missing_tesseract_is_actionable(settings):
    import pytest

    from app.ocr import read_scanned_page
    from tests.conftest import pdf_bytes

    with pytest.raises(ValueError, match="TESSERACT_CMD"):
        read_scanned_page(pdf_bytes("Text."), 0, "nonexistent-tesseract-executable")


def test_reflection_regeneration_has_hard_limit(settings):
    class Search:
        def search(self, *args, **kwargs):
            return [evidence("one")]

    class Model:
        enabled = True
        generations = 0
        reflections = 0

        def answer(self, *args, **kwargs):
            self.generations += 1
            return GeneratedAnswer(status="answered", claims=[Claim(text="Unsupported.", sources=[1])])

        def verify_claims(self, question, answer, chunks):
            self.reflections += 1
            return GeneratedAnswer(status="unavailable", note="Not supported."), 1

    model = Model()
    graph = build_workflow(Search(), model, replace(settings, adaptive=False, verify=True))
    result = graph.invoke(
        {"question": "RTO?", "standalone": "RTO?", "document_ids": None, "mode": "hybrid", "history": []},
        {"recursion_limit": 16},
    )
    assert model.generations == 2 and model.reflections == 2
    assert result["regeneration_count"] == 1 and result["generated"].status == "unavailable"


def test_compound_question_retrieves_each_part(settings):
    class Search:
        queries = []

        def search(self, query, *args, **kwargs):
            self.queries.append(query)
            return [evidence(query)]

    class Model:
        enabled = True

        def plan_queries(self, *args):
            return ["framework purpose", "confabulation definition"]

        def assess_evidence(self, *args):
            return True, ""

        def answer(self, *args):
            return GeneratedAnswer(status="unavailable")

    search = Search()
    graph = build_workflow(search, Model(), replace(settings, adaptive=True))
    result = graph.invoke(
        {
            "question": "What is the purpose and how is confabulation defined?",
            "standalone": "What is the purpose and how is confabulation defined?",
            "document_ids": None,
            "mode": "hybrid",
            "history": [],
        }
    )
    assert search.queries[-2:] == ["framework purpose", "confabulation definition"]
    assert result["trace"][1]["stage"] == "decompose_query"
    assert len(result["evidence"]) == 3


def test_optional_note_sentinel_is_not_user_visible(settings, monkeypatch):
    answer = GeneratedAnswer(status="answered", claims=[Claim(text="One hour.", sources=[1])], note="None")
    assert answer.note == ""
    assert GeneratedAnswer(status="unavailable", note=None).note == ""
    llm = LLM(settings)
    monkeypatch.setattr(
        llm,
        "complete",
        lambda *a, **k: {
            "status": "answered",
            "claims": [{"text": "One hour.", "sources": [1]}],
            "note": "An extra uncited assertion.",
        },
    )
    assert llm.answer("RTO?", "RTO?", [evidence("one")], []).note == ""
