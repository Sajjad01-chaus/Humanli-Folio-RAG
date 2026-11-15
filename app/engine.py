import threading
import time
from collections import defaultdict

from app.config import Settings
from app.llm import LLM
from app.models import AnswerStatus, ChatResponse, ClaimRecord, HistoryEntry, RetrievalMode, SourceRecord
from app.observability import event
from app.ports import LLMPort, Repository, RetrievalPort
from app.retrieval import Retriever
from app.workflow import build_workflow


class Engine:
    def __init__(
        self,
        store: Repository,
        settings: Settings,
        *,
        retriever: RetrievalPort | None = None,
        llm: LLMPort | None = None,
    ) -> None:
        self.store, self.settings = store, settings
        self.retriever: RetrievalPort = retriever if retriever is not None else Retriever(store, settings)
        self.llm: LLMPort = llm if llm is not None else LLM(settings)
        self.locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    def ask(
        self,
        session_id: str,
        question: str,
        document_ids: list[str] | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> ChatResponse:
        with self.locks[session_id]:
            start = time.perf_counter()
            history: list[HistoryEntry] = [
                {"question": turn["question"][:800], "answer": turn["response"]["answer"][:1600]}
                for turn in self.store.history(session_id)[-4:]
            ]
            rewritten = self.llm.rewrite(question, history)
            rewrite_ms = round((time.perf_counter() - start) * 1000)
            graph = build_workflow(self.retriever, self.llm, self.settings)
            final = graph.invoke(
                {
                    "question": question,
                    "standalone": rewritten,
                    "history": history,
                    "document_ids": document_ids,
                    "mode": mode,
                },
                {"recursion_limit": 16},
            )
            evidence, generated = final["evidence"], final["generated"]
            claims: list[ClaimRecord] = []
            note = ""
            status: AnswerStatus
            if not evidence:
                status, answer = (
                    "unavailable",
                    "I couldnâ€™t find supporting information in the selected documents.",
                )
            elif generated is None:
                status = "evidence_only"
                answer = "Evidence preview â€” configure an LLM API key to generate an answer."
                note = "These are retrieved passages, not a generated answer. Conversational rewriting requires an LLM."
            else:
                status, note = generated.status, generated.note
                claims = [ClaimRecord(text=claim.text, sources=claim.sources) for claim in generated.claims]
                answer = "\n\n".join(
                    claim["text"] + " " + " ".join(f"[{i}]" for i in claim["sources"]) for claim in claims
                )
                if note:
                    answer += ("\n\n" if answer else "") + note
                if not answer:
                    answer = "The documents do not contain enough information to answer this question."
            cited = {i for claim in claims for i in claim["sources"]}
            sources: list[SourceRecord] = [
                {
                    "number": i + 1,
                    "document_id": row["document_id"],
                    "name": row["name"],
                    "page": row["page"],
                    "text": row["text"],
                    "cited": i + 1 in cited,
                    "rank_score": row["rank_score"],
                    "semantic_score": row["semantic_score"],
                    "rerank_score": row.get("rerank_score"),
                }
                for i, row in enumerate(evidence)
            ]
            result: ChatResponse = {
                "status": status,
                "answer": answer,
                "claims": claims,
                "note": note,
                "sources": sources,
                "standalone_question": rewritten,
                "session_id": session_id,
                "document_ids": document_ids,
                "model": self.settings.model if self.llm.enabled else "not configured",
                "retrieval": mode,
                "embedding_backend": self.settings.embeddings,
                "trace": final["trace"],
                "timings": {
                    "rewrite_ms": rewrite_ms,
                    **final["timings"],
                    "total_ms": round((time.perf_counter() - start) * 1000),
                },
            }
            self.store.add_turn(session_id, question, result)
            event(
                "chat.completed",
                status=status,
                sources=len(sources),
                duration_ms=result["timings"]["total_ms"],
            )
            return result

    def delete_session(self, session_id: str) -> None:
        with self.locks[session_id]:
            self.store.delete_session(session_id)
