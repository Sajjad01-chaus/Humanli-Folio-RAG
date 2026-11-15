import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import ValidationError

from app.errors import PersistenceError
from app.models import (
    CHAT_RESPONSE_ADAPTER,
    ChatResponse,
    ChunkRecord,
    DocumentRecord,
    SessionRecord,
    TurnRecord,
)
from app.observability import failure


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "folio.sqlite3"
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / "uploads").mkdir(exist_ok=True)
        except OSError as exc:
            failure("storage.initialization_failed", exc)
            raise PersistenceError("Cannot initialize local storage. Check directory permissions.") from exc
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, sha TEXT UNIQUE NOT NULL,
                    pages INTEGER NOT NULL, chunks INTEGER NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL, text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunk_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    question TEXT NOT NULL, response TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db: sqlite3.Connection | None = None
        try:
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            failure("storage.transaction_failed", exc)
            raise PersistenceError("Local storage is unavailable. Check disk space and permissions.") from exc
        finally:
            if db is not None:
                if db.in_transaction:
                    db.rollback()
                db.close()

    def documents(self) -> list[DocumentRecord]:
        with self.connect() as db:
            return [
                cast(DocumentRecord, dict(r))
                for r in db.execute("SELECT * FROM documents ORDER BY created,id")
            ]

    def find_document(self, sha: str) -> DocumentRecord | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE sha=?", (sha,)).fetchone()
            return cast(DocumentRecord, dict(row)) if row else None

    def add_document(self, document: DocumentRecord, chunks: list[tuple[int, str]]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO documents(id,name,sha,pages,chunks) VALUES (?,?,?,?,?)",
                (document["id"], document["name"], document["sha"], document["pages"], document["chunks"]),
            )
            db.executemany(
                "INSERT INTO chunks(id,document_id,page,text) VALUES (?,?,?,?)",
                [(uuid4().hex, document["id"], page, text) for page, text in chunks],
            )

    def chunks(self, document_ids: list[str] | None = None) -> list[ChunkRecord]:
        sql = "SELECT c.*, d.name FROM chunks c JOIN documents d ON d.id=c.document_id"
        params: list[str] = []
        if document_ids is not None:
            if not document_ids:
                return []
            sql += " WHERE d.id IN (" + ",".join("?" for _ in document_ids) + ")"
            params = document_ids
        with self.connect() as db:
            return [
                cast(ChunkRecord, dict(r))
                for r in db.execute(sql + " ORDER BY c.document_id,c.page,c.id", params)
            ]

    def session(self, session_id: str) -> SessionRecord | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            return cast(SessionRecord, dict(row)) if row else None

    def create_session(self) -> SessionRecord:
        session_id = uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO sessions(id,title) VALUES (?,?)", (session_id, "New conversation"))
        session = self.session(session_id)
        if session is None:
            raise PersistenceError("The conversation could not be created.")
        return session

    def sessions(self) -> list[SessionRecord]:
        with self.connect() as db:
            return [
                cast(SessionRecord, dict(r))
                for r in db.execute("SELECT * FROM sessions ORDER BY created DESC,rowid DESC")
            ]

    def history(self, session_id: str) -> list[TurnRecord]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM turns WHERE session_id=? ORDER BY id", (session_id,))
            try:
                return [
                    TurnRecord(
                        id=r["id"],
                        session_id=r["session_id"],
                        question=r["question"],
                        response=CHAT_RESPONSE_ADAPTER.validate_json(r["response"]),
                        created=r["created"],
                    )
                    for r in rows
                ]
            except ValidationError as exc:
                failure("storage.invalid_history", exc)
                raise PersistenceError(
                    "Stored conversation data is invalid. Start a new conversation."
                ) from exc

    def add_turn(self, session_id: str, question: str, response: ChatResponse) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO turns(session_id,question,response) VALUES (?,?,?)",
                (session_id, question, CHAT_RESPONSE_ADAPTER.dump_json(response).decode()),
            )
            db.execute(
                "UPDATE sessions SET title=? WHERE id=? AND title='New conversation'",
                (question[:72], session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def delete_document(self, document_id: str) -> None:
        try:
            with self.connect() as db:
                db.execute("DELETE FROM documents WHERE id=?", (document_id,))
                (self.root / "uploads" / f"{document_id}.pdf").unlink(missing_ok=True)
        except OSError as exc:
            failure("storage.document_delete_failed", exc)
            raise PersistenceError(
                "Cannot remove the PDF file. Check file access permissions and retry."
            ) from exc
