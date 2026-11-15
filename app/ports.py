from collections.abc import Collection
from pathlib import Path
from typing import Literal, Protocol

from app.models import (
    ChatResponse,
    ChunkRecord,
    DocumentRecord,
    Evidence,
    GeneratedAnswer,
    HistoryEntry,
    RetrievalMode,
    SessionRecord,
    TurnRecord,
)


class Repository(Protocol):
    @property
    def root(self) -> Path: ...
    def documents(self) -> list[DocumentRecord]: ...
    def find_document(self, sha: str) -> DocumentRecord | None: ...
    def add_document(self, document: DocumentRecord, chunks: list[tuple[int, str]]) -> None: ...
    def chunks(self, document_ids: list[str] | None = None) -> list[ChunkRecord]: ...
    def delete_document(self, document_id: str) -> None: ...
    def session(self, session_id: str) -> SessionRecord | None: ...
    def create_session(self) -> SessionRecord: ...
    def sessions(self) -> list[SessionRecord]: ...
    def history(self, session_id: str) -> list[TurnRecord]: ...
    def add_turn(self, session_id: str, question: str, response: ChatResponse) -> None: ...
    def delete_session(self, session_id: str) -> None: ...


class RetrievalPort(Protocol):
    def search(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 6,
        mode: RetrievalMode = "hybrid",
    ) -> list[Evidence]: ...


class LLMPort(Protocol):
    @property
    def enabled(self) -> bool: ...
    def rewrite(self, question: str, history: list[HistoryEntry]) -> str: ...
    def answer(
        self,
        question: str,
        rewritten: str,
        evidence: list[Evidence],
        history: list[HistoryEntry],
        feedback: str | None = None,
    ) -> GeneratedAnswer: ...
    def assess_evidence(self, question: str, evidence: list[Evidence]) -> tuple[bool, str]: ...
    def plan_queries(self, question: str) -> list[str]: ...
    def verify_claims(
        self, question: str, answer: GeneratedAnswer, evidence: list[Evidence]
    ) -> tuple[GeneratedAnswer, int]: ...


class OCRPort(Protocol):
    def __call__(self, pdf_bytes: bytes, page_index: int, command: str = "tesseract") -> str: ...


class TokenEncoder(Protocol):
    def encode(
        self, text: str, *, disallowed_special: Collection[str] | Literal["all"] = "all"
    ) -> list[int]: ...
