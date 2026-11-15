import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""), repr=False)
    embeddings: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "sentence-transformers"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    local_only: bool = field(
        default_factory=lambda: os.getenv("EMBEDDING_LOCAL_ONLY", "false").lower() == "true"
    )
    timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "60")))
    access_token: str = field(default_factory=lambda: os.getenv("APP_ACCESS_TOKEN", ""), repr=False)
    rerank: bool = field(default_factory=lambda: os.getenv("RERANK_ENABLED", "true").lower() == "true")
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    adaptive: bool = field(default_factory=lambda: os.getenv("ADAPTIVE_RETRIEVAL", "true").lower() == "true")
    verify: bool = field(default_factory=lambda: os.getenv("VERIFY_GROUNDING", "true").lower() == "true")
    ocr: bool = field(default_factory=lambda: os.getenv("OCR_ENABLED", "false").lower() == "true")
    tesseract: str = field(default_factory=lambda: os.getenv("TESSERACT_CMD", "tesseract"))
    provider_tpm: int = field(default_factory=lambda: int(os.getenv("PROVIDER_TOKENS_PER_MINUTE", "8000")))
    max_input_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_INPUT_TOKENS", "5500")))
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pages: int = 250
    max_documents: int = 30

    def __post_init__(self) -> None:
        if self.embeddings not in {"sentence-transformers", "tfidf"}:
            raise ValueError("EMBEDDING_BACKEND must be sentence-transformers or tfidf.")
        if self.timeout <= 0 or self.provider_tpm < 0 or self.max_input_tokens <= 0:
            raise ValueError("Timeout and context budget must be positive; provider TPM must be nonnegative.")
        if min(self.max_upload_bytes, self.max_pages, self.max_documents) <= 0:
            raise ValueError("Upload, page, and document limits must be positive.")
