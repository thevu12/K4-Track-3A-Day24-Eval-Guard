"""BM25 + multilingual dense Qdrant search, fused with reciprocal ranks.

Compatible with qdrant-client 1.9.x / server 1.9.7. Each corpus/model fingerprint
has its own collection. Re-indexing validates and reuses existing points, and
only fills missing points after an interrupted upload; it never deletes a
collection or overwrites unexpected data. ML/network imports are lazy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import unicodedata
import uuid

import config


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


def _tokenize(text: str) -> list[str]:
    """Unicode syllable tokenization keeps Vietnamese accents and term counts."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class HybridSearch:
    def __init__(
        self, *, client=None, encoder=None, bm25_factory=None, qdrant_models=None,
        model_name=None, embedding_dim=None, collection_name=None, batch_size=None,
        device=None, bm25_top_k=None, dense_top_k=None, hybrid_top_k=None, rrf_k=None,
    ):
        self.model_name = config.EMBEDDING_MODEL if model_name is None else model_name
        self.embedding_dim = config.EMBEDDING_DIM if embedding_dim is None else embedding_dim
        self.collection_prefix = config.COLLECTION_NAME if collection_name is None else collection_name
        self.batch_size = getattr(config, "MODEL_BATCH_SIZE", 16) if batch_size is None else batch_size
        self.device = getattr(config, "EMBEDDING_DEVICE", "cpu") if device is None else device
        self.max_length = getattr(config, "MODEL_MAX_LENGTH", 8192)
        self.bm25_top_k = config.BM25_TOP_K if bm25_top_k is None else bm25_top_k
        self.dense_top_k = config.DENSE_TOP_K if dense_top_k is None else dense_top_k
        self.hybrid_top_k = config.HYBRID_TOP_K if hybrid_top_k is None else hybrid_top_k
        self.rrf_k = getattr(config, "RRF_K", 60) if rrf_k is None else rrf_k
        for name in ("embedding_dim", "batch_size", "max_length", "bm25_top_k", "dense_top_k", "hybrid_top_k", "rrf_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", self.collection_prefix):
            raise ValueError("collection_name must be a safe collection prefix (letters/digits/_/-)")
        self.client, self.encoder = client, encoder
        self._bm25_factory, self._models = bm25_factory, qdrant_models
        self.collection_name = None
        self.chunks, self._ids = [], []
        self._ready = False

    def _load_client(self):
        if self._models is None:
            from qdrant_client import models
            self._models = models
        if self.client is None:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=60)

    def _load_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {"device": self.device}
            if config.HF_TOKEN:
                kwargs["token"] = config.HF_TOKEN
            self.encoder = SentenceTransformer(self.model_name, **kwargs)
            self.encoder.max_seq_length = self.max_length

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load_encoder()
        encoded = self.encoder.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        encoded = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        if len(encoded) != len(texts):
            raise ValueError("Embedding model returned the wrong number of vectors")
        vectors = []
        for vector in encoded:
            values = [float(value) for value in vector]
            if len(values) != self.embedding_dim or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Expected finite {self.embedding_dim}-dimensional embeddings from {self.model_name}")
            norm = math.sqrt(sum(value * value for value in values))
            if not norm or not math.isfinite(norm):
                raise ValueError("Embedding model returned a zero/invalid vector")
            vectors.append([value / norm for value in values])
        return vectors

    def _payload(self, ordinal: int) -> dict:
        return {
            "index_schema": 1, "corpus_sha256": self.corpus_fingerprint,
            "embedding_model": self.model_name,
            "text": self.chunks[ordinal]["text"],
            "metadata": self.chunks[ordinal]["metadata"],
        }

    def _existing_indices(self) -> set[int]:
        """Validate ALL existing points before doing any write to a collection."""
        collection = self.client.get_collection(collection_name=self.collection_name)
        vectors = collection.config.params.vectors
        if isinstance(vectors, dict) or vectors.size != self.embedding_dim or vectors.distance != self._models.Distance.COSINE:
            raise ValueError(f"Refusing incompatible existing Qdrant collection {self.collection_name}")
        count = self.client.count(collection_name=self.collection_name, exact=True).count
        if count > len(self.chunks):
            raise ValueError("Existing collection contains unexpected points; nothing will be overwritten")
        existing = set()
        for start in range(0, len(self._ids), self.batch_size):
            records = self.client.retrieve(
                collection_name=self.collection_name, ids=self._ids[start:start + self.batch_size],
                with_payload=True, with_vectors=False,
            )
            for record in records:
                ordinal = self._positions.get(str(record.id))
                if ordinal is None or record.payload != self._payload(ordinal):
                    raise ValueError("Existing collection payload differs; nothing will be overwritten")
                existing.add(ordinal)
        if len(existing) != count:
            raise ValueError("Existing collection contains unexpected IDs; nothing will be overwritten")
        return existing

    def index(self, chunks):
        self._ready = False
        rows = []
        for ordinal, chunk in enumerate(chunks):
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(f"Chunk {ordinal} must have non-empty text")
            rows.append({"text": chunk["text"], "metadata": deepcopy(dict(chunk.get("metadata", {})))})
        if not rows:
            raise ValueError("Cannot index an empty corpus")
        self.chunks = rows
        fingerprint_input = {
            "index_schema": 1, "model": self.model_name, "dimension": self.embedding_dim,
            "max_length": self.max_length, "normalization": "l2", "chunks": rows,
        }
        self.corpus_fingerprint = hashlib.sha256(_json(fingerprint_input).encode("utf-8")).hexdigest()
        self.collection_name = f"{self.collection_prefix}_{self.corpus_fingerprint[:24]}"
        self._ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{self.corpus_fingerprint}:{i}")) for i in range(len(rows))]
        self._positions = {point_id: ordinal for ordinal, point_id in enumerate(self._ids)}
        self._tokens = [_tokenize(row["text"]) for row in rows]
        if not any(self._tokens):
            raise ValueError("Corpus contains no searchable tokens")
        if self._bm25_factory is None:
            from rank_bm25 import BM25Okapi
            self._bm25_factory = BM25Okapi
        self._bm25 = self._bm25_factory(self._tokens)
        self._load_client()
        if self.client.collection_exists(collection_name=self.collection_name):
            existing = self._existing_indices()
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._models.VectorParams(size=self.embedding_dim, distance=self._models.Distance.COSINE),
            )
            existing = set()
        missing = [i for i in range(len(rows)) if i not in existing]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = self._encode([rows[i]["text"] for i in batch])
            points = [
                self._models.PointStruct(id=self._ids[i], vector=vector, payload=self._payload(i))
                for i, vector in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        if self.client.count(collection_name=self.collection_name, exact=True).count != len(rows):
            raise RuntimeError("Qdrant point count does not match the indexed corpus")
        self._ready = True

    def search(self, question: str) -> list[SearchResult]:
        if not self._ready:
            raise RuntimeError("Call index(chunks) successfully before search()")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Search question must be a non-empty string")
        query_tokens = _tokenize(question)
        bm25_scores = [float(value) for value in self._bm25.get_scores(query_tokens)]
        if len(bm25_scores) != len(self.chunks) or not all(math.isfinite(score) for score in bm25_scores):
            raise ValueError("BM25 returned invalid scores")
        query_terms = set(query_tokens)
        lexical = sorted(
            (i for i, tokens in enumerate(self._tokens) if query_terms.intersection(tokens)),
            key=lambda i: (-bm25_scores[i], i),
        )[:self.bm25_top_k]
        dense_hits = self.client.search(
            collection_name=self.collection_name, query_vector=self._encode([question])[0],
            limit=self.dense_top_k, with_payload=False, with_vectors=False,
        )
        dense, dense_scores = [], {}
        for hit in dense_hits:
            ordinal = self._positions.get(str(hit.id))
            score = float(hit.score)
            if ordinal is None or not math.isfinite(score):
                raise ValueError("Qdrant returned an unknown point or invalid score")
            if ordinal not in dense_scores:
                dense.append(ordinal)
                dense_scores[ordinal] = score
        lexical_ranks = {i: rank for rank, i in enumerate(lexical, start=1)}
        dense_ranks = {i: rank for rank, i in enumerate(dense, start=1)}
        fused = {}
        for ranking in (lexical_ranks, dense_ranks):
            for i, rank in ranking.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:self.hybrid_top_k]
        return [SearchResult(
            self.chunks[i]["text"], float(fused[i]),
            {**deepcopy(self.chunks[i]["metadata"]), "retrieval": {
                "fusion": "reciprocal_rank_fusion", "rrf_k": self.rrf_k,
                "bm25_rank": lexical_ranks.get(i), "bm25_score": bm25_scores[i],
                "dense_rank": dense_ranks.get(i), "dense_score": dense_scores.get(i),
                "embedding_model": self.model_name, "collection": self.collection_name,
            }},
        ) for i in order]
