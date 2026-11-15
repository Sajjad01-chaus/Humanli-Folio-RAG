"""Bounded corrective and reflective retrieval workflow."""

import re
import time
from collections.abc import Callable
from typing import Protocol, TypedDict, cast

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.errors import ApplicationError
from app.models import Evidence, GeneratedAnswer, HistoryEntry, RetrievalMode, TraceEvent
from app.observability import event
from app.ports import LLMPort, RetrievalPort


class RAGState(TypedDict, total=False):
    question: str
    standalone: str
    document_ids: list[str] | None
    mode: RetrievalMode
    history: list[HistoryEntry]
    evidence: list[Evidence]
    sufficient: bool
    correction_query: str
    retrieval_attempts: int
    generated: GeneratedAnswer | None
    removed_claims: int
    regeneration_count: int
    trace: list[TraceEvent]
    timings: dict[str, int]


class Workflow(Protocol):
    def invoke(self, input: RAGState, config: dict[str, int]) -> RAGState: ...


def build_workflow(retriever: RetrievalPort, llm: LLMPort, settings: Settings) -> Workflow:
    def retrieve(state: RAGState) -> RAGState:
        start = time.perf_counter()
        evidence = retriever.search(state["standalone"], state["document_ids"], mode=state["mode"])
        trace: list[TraceEvent] = [{"stage": "retrieve", "passages": len(evidence)}]
        if (
            llm.enabled
            and settings.adaptive
            and re.search(
                r"\band\s+(?:how|what|why|which|when|where|does|is|are)\b|;\s*\w", state["standalone"], re.I
            )
        ):
            queries = llm.plan_queries(state["standalone"])
            seen: set[str] = set()
            combined: list[Evidence] = []
            for query in queries:
                for row in retriever.search(query, state["document_ids"], mode=state["mode"], top_k=3):
                    if row["id"] not in seen:
                        seen.add(row["id"])
                        combined.append(row)
            for row in evidence:
                if row["id"] not in seen:
                    seen.add(row["id"])
                    combined.append(row)
            evidence = combined[:8]
            trace.append({"stage": "decompose_query", "queries": queries, "passages": len(evidence)})
        return {
            "evidence": evidence,
            "retrieval_attempts": 1,
            "timings": {"retrieval_ms": round((time.perf_counter() - start) * 1000)},
            "trace": trace,
        }

    def assess(state: RAGState) -> RAGState:
        start = time.perf_counter()
        sufficient, query = llm.assess_evidence(state["standalone"], state["evidence"])
        return {
            "sufficient": sufficient,
            "correction_query": query,
            "timings": {**state["timings"], "assessment_ms": round((time.perf_counter() - start) * 1000)},
            "trace": [*state["trace"], {"stage": "assess_evidence", "sufficient": sufficient}],
        }

    def route_after_retrieve(state: RAGState) -> str:
        return "assess" if llm.enabled and settings.adaptive and state["evidence"] else "generate"

    def route_after_assess(state: RAGState) -> str:
        if (
            not state["sufficient"]
            and state["correction_query"].strip()
            and state["correction_query"] != state["standalone"]
            and state["retrieval_attempts"] < 2
        ):
            return "correct"
        return "generate"

    def correct(state: RAGState) -> RAGState:
        start = time.perf_counter()
        extra = retriever.search(state["correction_query"], state["document_ids"], mode=state["mode"])
        seen: set[str] = set()
        evidence: list[Evidence] = []
        for row in [*extra[:2], *state["evidence"][:6], *extra[2:], *state["evidence"][6:]]:
            if row["id"] not in seen:
                seen.add(row["id"])
                evidence.append(row)
        return {
            "evidence": evidence[:8],
            "retrieval_attempts": state["retrieval_attempts"] + 1,
            "timings": {**state["timings"], "correction_ms": round((time.perf_counter() - start) * 1000)},
            "trace": [
                *state["trace"],
                {
                    "stage": "corrective_search",
                    "query": state["correction_query"],
                    "passages": len(evidence[:8]),
                    "attempt": 2,
                },
            ],
        }

    def generate(state: RAGState) -> RAGState:
        start = time.perf_counter()
        generated = (
            llm.answer(state["question"], state["standalone"], state["evidence"], state["history"])
            if llm.enabled and state["evidence"]
            else None
        )
        return {
            "generated": generated,
            "regeneration_count": 0,
            "timings": {**state["timings"], "generation_ms": round((time.perf_counter() - start) * 1000)},
        }

    def route_after_generate(state: RAGState) -> str:
        draft = state["generated"]
        return "reflect" if settings.verify and draft is not None and draft.claims else END

    def reflect(state: RAGState) -> RAGState:
        start = time.perf_counter()
        draft = state["generated"]
        if draft is None:
            raise ApplicationError("Reflection was invoked without an answer draft.")
        generated, removed = llm.verify_claims(state["standalone"], draft, state["evidence"])
        return {
            "generated": generated,
            "removed_claims": removed,
            "timings": {**state["timings"], "verification_ms": round((time.perf_counter() - start) * 1000)},
            "trace": [*state["trace"], {"stage": "grounding_check", "removed_claims": removed}],
        }

    def route_after_reflection(state: RAGState) -> str:
        return "regenerate" if state["removed_claims"] and state["regeneration_count"] < 1 else END

    def regenerate(state: RAGState) -> RAGState:
        start = time.perf_counter()
        answer = llm.answer(
            state["question"],
            state["standalone"],
            state["evidence"],
            state["history"],
            feedback="A previous draft included an unsupported assertion. Produce a shorter answer. "
            "Use only directly stated facts, preserve exceptions, and cite the precise passages. "
            "If the question is supported, answer it; otherwise explicitly abstain.",
        )
        return {
            "generated": answer,
            "regeneration_count": state["regeneration_count"] + 1,
            "timings": {**state["timings"], "regeneration_ms": round((time.perf_counter() - start) * 1000)},
            "trace": [*state["trace"], {"stage": "regenerate", "attempt": 1}],
        }

    def instrument(name: str, node: Callable[[RAGState], RAGState]) -> Callable[[RAGState], RAGState]:
        def run(state: RAGState) -> RAGState:
            start = time.perf_counter()
            result = node(state)
            event(
                "workflow.stage_completed",
                stage=name,
                duration_ms=round((time.perf_counter() - start) * 1000),
            )
            return result

        return run

    graph = StateGraph(RAGState)
    for name, function in [
        ("retrieve", retrieve),
        ("assess", assess),
        ("correct", correct),
        ("generate", generate),
        ("reflect", reflect),
        ("regenerate", regenerate),
    ]:
        graph.add_node(name, RunnableLambda[RAGState, RAGState](instrument(name, function)))
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"assess": "assess", "generate": "generate"}
    )
    graph.add_conditional_edges("assess", route_after_assess, {"correct": "correct", "generate": "generate"})
    graph.add_edge("correct", "generate")
    graph.add_conditional_edges("generate", route_after_generate, {"reflect": "reflect", END: END})
    graph.add_conditional_edges("reflect", route_after_reflection, {"regenerate": "regenerate", END: END})
    graph.add_conditional_edges("regenerate", route_after_generate, {"reflect": "reflect", END: END})
    return cast(Workflow, graph.compile())
