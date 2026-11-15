import hashlib
import io
import re
import threading
import unicodedata
from collections.abc import Iterator
from uuid import uuid4

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings
from app.errors import DocumentError, PersistenceError
from app.models import DocumentRecord, IngestResult
from app.observability import event, failure
from app.ports import OCRPort, Repository


def chunk_page(text: str, size: int = 180, overlap: int = 35) -> Iterator[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size.")
    words = re.sub(r"[ \t]+", " ", text).split()
    for start in range(0, len(words), size - overlap):
        window = words[start : start + size]
        if window:
            yield " ".join(window)
        if start + size >= len(words):
            break


class DocumentService:
    def __init__(self, store: Repository, settings: Settings, ocr: OCRPort | None = None) -> None:
        from app.ocr import read_scanned_page

        self.store, self.settings = store, settings
        self.ocr = ocr if ocr is not None else read_scanned_page
        self.lock = threading.RLock()

    def ingest(self, name: str, content: bytes) -> IngestResult:
        with self.lock:
            result = self._ingest(name, content)
        event(
            "document.ingested",
            document_id=result["id"],
            pages=result["pages"],
            chunks=result["chunks"],
            duplicate=result["duplicate"],
        )
        return result

    def _ingest(self, name: str, content: bytes) -> IngestResult:
        settings = self.settings
        if len(content) > settings.max_upload_bytes:
            raise DocumentError("PDF exceeds the 20 MB limit.")
        if not content.startswith(b"%PDF-"):
            raise DocumentError("This file is not a valid PDF.")
        sha = hashlib.sha256(content).hexdigest()
        duplicate = self.store.find_document(sha)
        if duplicate:
            return {**duplicate, "duplicate": True, "ocr_pages": [], "warnings": []}
        if len(self.store.documents()) >= settings.max_documents:
            raise DocumentError(f"This workspace supports up to {settings.max_documents} PDFs.")
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise DocumentError("Password-protected PDFs are not supported. Upload an unlocked copy.")
            if not 1 <= len(reader.pages) <= settings.max_pages:
                raise DocumentError(f"Upload PDFs containing 1–{settings.max_pages} pages.")
            pages: list[tuple[int, str]] = []
            ocr_pages: list[int] = []
            for i, page in enumerate(reader.pages):
                text = unicodedata.normalize("NFKC", page.extract_text() or "")
                text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
                if len(text.strip()) < 20 and settings.ocr:
                    text = self.ocr(content, i, settings.tesseract)
                    ocr_pages.append(i + 1)
                pages.append((i + 1, text))
        except PdfReadError as exc:
            failure("document.parse_failed", exc)
            raise DocumentError("The PDF could not be read. Check that it is not damaged.") from exc
        chunks = [(page, text) for page, text in pages for text in chunk_page(text) if len(text) >= 20]
        if not chunks:
            raise DocumentError("No readable text found. Scanned PDFs require OCR; use a text-based PDF.")
        document_id = uuid4().hex
        name = re.sub(r"[\x00-\x1f]", "", name.replace("\\", "/").split("/")[-1])[:160] or "document.pdf"
        document = DocumentRecord(id=document_id, name=name, sha=sha, pages=len(pages), chunks=len(chunks))
        target = self.store.root / "uploads" / f"{document_id}.pdf"
        try:
            target.write_bytes(content)
            self.store.add_document(document, chunks)
        except (PersistenceError, OSError) as exc:
            failure("document.persistence_failed", exc)
            try:
                target.unlink(missing_ok=True)
            except OSError as cleanup_error:
                failure("document.cleanup_failed", cleanup_error)
                raise PersistenceError(
                    "Storage failed and the incomplete upload could not be removed."
                ) from exc
            raise PersistenceError("The PDF could not be saved. Check storage and retry.") from exc
        return {
            **document,
            "duplicate": False,
            "ocr_pages": ocr_pages,
            "warnings": [
                f"Page {page} contains no extracted text." for page, text in pages if not text.strip()
            ],
        }


def ingest_pdf(store: Repository, name: str, content: bytes, settings: Settings) -> IngestResult:
    return DocumentService(store, settings).ingest(name, content)
