from dataclasses import dataclass

from app.config import Settings
from app.engine import Engine
from app.ingest import DocumentService
from app.ports import LLMPort, OCRPort, Repository, RetrievalPort
from app.store import Store


@dataclass(frozen=True)
class Services:
    settings: Settings
    repository: Repository
    engine: Engine
    documents: DocumentService


def create_services(
    settings: Settings,
    *,
    repository: Repository | None = None,
    llm: LLMPort | None = None,
    retriever: RetrievalPort | None = None,
    ocr: OCRPort | None = None,
) -> Services:
    store = repository if repository is not None else Store(settings.data_dir)
    return Services(
        settings=settings,
        repository=store,
        engine=Engine(store, settings, llm=llm, retriever=retriever),
        documents=DocumentService(store, settings, ocr=ocr),
    )
