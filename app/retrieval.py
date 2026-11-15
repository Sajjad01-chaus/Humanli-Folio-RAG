"""Local lexical/semantic retrieval with rank fusion and cross-encoder reranking."""

import hashlib
import math
import re
import threading
from collections import Counter, defaultdict
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.config import Settings
from app.errors import RetrievalError
from app.models import ChunkRecord, Evidence, RetrievalMode
from app.observability import event, failure
from app.ports import Repository


class Encoder(Protocol):
    def encode(
        self, texts: list[str], *, normalize_embeddings: bool, show_progress_bar: bool
    ) -> NDArray[np.float32]: ...


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], *, show_progress_bar: bool) -> NDArray[np.float32]: ...


class SparseMatrix(Protocol):
    @property
    def T(self) -> "SparseMatrix": ...
    def __matmul__(self, other: "SparseMatrix") -> "SparseMatrix": ...
    def toarray(self) -> NDArray[np.float64]: ...


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


class Retriever:
    def __init__(
        self,
        store: Repository,
        settings: Settings,
        *,
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store, self.settings = store, settings
        self.lock = threading.RLock()
        self.encoder = encoder
        self.reranker = reranker
        self.signature: str | None = None
        self.rows: list[ChunkRecord] = []

    def _prepare(self) -> None:
        rows = self.store.chunks()
        signature = hashlib.sha256("".join(r["id"] for r in rows).encode()).hexdigest()
        if signature == self.signature:
            return
        self.rows = rows
        if not rows:
            self.signature = signature
            return
        texts = [r["text"] for r in rows]
        self.word_counts = [Counter(tokens(text)) for text in texts]
        self.lengths = np.array([sum(c.values()) for c in self.word_counts])
        self.avg_length = max(float(self.lengths.mean()), 1)
        frequency = Counter(word for counts in self.word_counts for word in counts)
        self.idf = {
            w: math.log(1 + (len(rows) - count + 0.5) / (count + 0.5)) for w, count in frequency.items()
        }
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)
        self.lexical_vectors = cast(SparseMatrix, self.vectorizer.fit_transform(texts))
        if self.settings.embeddings == "sentence-transformers":
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer

                self.encoder = cast(
                    Encoder,
                    SentenceTransformer(
                        self.settings.embedding_model, local_files_only=self.settings.local_only
                    ),
                )
            cache = self.store.root / "embeddings"
            cache.mkdir(exist_ok=True)
            model_hash = hashlib.sha256(self.settings.embedding_model.encode()).hexdigest()[:12]
            path = cache / f"{model_hash}-{signature}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    self.vectors = cast(NDArray[np.float32], saved["vectors"])
            else:
                self.vectors = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                np.savez_compressed(path, vectors=self.vectors)
            for old in cache.glob(f"{model_hash}-*.npz"):
                if old != path:
                    old.unlink(missing_ok=True)
        self.signature = signature

    def search(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 6,
        mode: RetrievalMode = "hybrid",
    ) -> list[Evidence]:
        try:
            return self._search(query, document_ids, top_k, mode)
        except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
            failure("retrieval.failed", exc)
            raise RetrievalError(
                "Retrieval is unavailable. Check local model files, storage, and dependencies."
            ) from exc

    def _search(
        self, query: str, document_ids: list[str] | None, top_k: int, mode: RetrievalMode
    ) -> list[Evidence]:
        with self.lock:
            self._prepare()
            if not self.rows:
                return []
            allowed = [
                i
                for i, row in enumerate(self.rows)
                if document_ids is None or row["document_id"] in document_ids
            ]
            if not allowed:
                return []
            bm25 = np.zeros(len(self.rows))
            for word in set(tokens(query)):
                tf = np.array([c.get(word, 0) for c in self.word_counts])
                bm25 += (
                    self.idf.get(word, 0)
                    * tf
                    * 2.5
                    / (tf + 1.5 * (0.25 + 0.75 * self.lengths / self.avg_length))
                )
            query_vectors = cast(SparseMatrix, self.vectorizer.transform([query]))
            lexical = (self.lexical_vectors @ query_vectors.T).toarray().ravel()
            dense = None
            if self.settings.embeddings == "sentence-transformers" and mode != "bm25":
                if self.encoder is None:
                    raise RetrievalError("The semantic encoder was not initialized.")
                q = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
                dense = self.vectors @ q
            rankings = [sorted(allowed, key=lambda i: bm25[i], reverse=True)]
            if mode != "bm25":
                rankings.append(
                    sorted(allowed, key=lambda i: (dense if dense is not None else lexical)[i], reverse=True)
                )
            fused: defaultdict[int, float] = defaultdict(float)
            for ranking in rankings:
                for rank, index in enumerate(ranking[:30]):
                    fused[index] += 1 / (60 + rank + 1)
            results: list[Evidence] = []
            for i in sorted(fused, key=lambda index: fused[index], reverse=True):
                semantic = float(dense[i]) if dense is not None else float(lexical[i])
                if bm25[i] <= 0 and semantic < 0.28:
                    continue
                results.append(
                    {
                        **self.rows[i],
                        "rank_score": round(fused[i], 5),
                        "lexical_score": round(float(bm25[i]), 4),
                        "semantic_score": round(semantic, 4),
                        "rerank_score": None,
                    }
                )
            if self.settings.rerank and mode == "hybrid" and results:
                if self.reranker is None:
                    from sentence_transformers import CrossEncoder

                    self.reranker = cast(
                        Reranker,
                        CrossEncoder(self.settings.reranker_model, local_files_only=self.settings.local_only),
                    )
                candidates = results[:20]
                scores = self.reranker.predict(
                    [(query, row["text"]) for row in candidates], show_progress_bar=False
                )
                if len(scores) != len(candidates) or not np.isfinite(scores).all():
                    raise RetrievalError("The reranker returned invalid scores. Check the configured model.")
                for row, score in zip(candidates, scores):
                    row["rerank_score"] = round(float(score), 4)
                results = sorted(candidates, key=lambda row: row["rerank_score"] or 0.0, reverse=True)
            event(
                "retrieval.completed", candidates=len(results), selected=min(top_k, len(results)), mode=mode
            )
            return results[:top_k]
