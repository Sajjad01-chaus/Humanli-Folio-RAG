from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator
from typing_extensions import NotRequired, TypedDict

RetrievalMode = Literal["bm25", "hybrid"]
AnswerStatus = Literal["answered", "partial", "unavailable", "evidence_only"]


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    sources: list[int] = Field(min_length=1, max_length=6)


class GeneratedAnswer(BaseModel):
    status: Literal["answered", "partial", "unavailable"]
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    note: str = Field(default="", max_length=1200)

    @field_validator("note", mode="before")
    @classmethod
    def clean_optional_note(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and value.strip().lower() in {"none", "null", "n/a"}):
            return ""
        return value


class DocumentRecord(TypedDict):
    id: str
    name: str
    sha: str
    pages: int
    chunks: int
    created: NotRequired[str]


class IngestResult(DocumentRecord):
    duplicate: bool
    ocr_pages: NotRequired[list[int]]
    warnings: NotRequired[list[str]]


class ChunkRecord(TypedDict):
    id: str
    document_id: str
    name: str
    page: int
    text: str


class Evidence(ChunkRecord):
    rank_score: float
    lexical_score: float
    semantic_score: float
    rerank_score: NotRequired[float | None]


class SessionRecord(TypedDict):
    id: str
    title: str
    created: str


class HistoryEntry(TypedDict):
    question: str
    answer: str


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ClaimRecord(TypedDict):
    text: str
    sources: list[int]


class SourceRecord(TypedDict):
    number: int
    document_id: str
    name: str
    page: int
    text: str
    cited: bool
    rank_score: float
    semantic_score: float
    rerank_score: NotRequired[float | None]


class TraceEvent(TypedDict, total=False):
    stage: str
    passages: int
    sufficient: bool
    query: str
    queries: list[str]
    attempt: int
    removed_claims: int


class ChatResponse(TypedDict):
    status: AnswerStatus
    answer: str
    claims: list[ClaimRecord]
    note: str
    sources: list[SourceRecord]
    standalone_question: str
    session_id: str
    document_ids: list[str] | None
    model: str
    retrieval: RetrievalMode
    embedding_backend: str
    trace: list[TraceEvent]
    timings: dict[str, int]


class TurnRecord(TypedDict):
    id: int
    session_id: str
    question: str
    response: ChatResponse
    created: str


class UploadError(TypedDict):
    name: str
    error: str


class UploadResponse(TypedDict):
    documents: list[IngestResult]
    errors: list[UploadError]


class HistoryResponse(TypedDict):
    session: SessionRecord
    turns: list[TurnRecord]


CHAT_RESPONSE_ADAPTER = TypeAdapter(ChatResponse)
