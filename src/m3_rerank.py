"""Real multilingual cross-encoder reranking, with lazy model loading."""
from __future__ import annotations

from copy import deepcopy
import math

import config
from src.m2_search import SearchResult


class CrossEncoderReranker:
    def __init__(self, *, model=None, model_name=None, device=None, batch_size=None, max_length=None):
        self.model = model
        self.model_name = getattr(config, "RERANK_MODEL", "BAAI/bge-reranker-v2-m3") if model_name is None else model_name
        self.device = getattr(config, "RERANK_DEVICE", "cpu") if device is None else device
        self.batch_size = getattr(config, "MODEL_BATCH_SIZE", 16) if batch_size is None else batch_size
        self.max_length = getattr(config, "RERANK_MAX_LENGTH", 1024) if max_length is None else max_length
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (self.batch_size, self.max_length)):
            raise ValueError("batch_size/max_length must be positive integers")

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            # Public BAAI checkpoint; HF cache/auth settings are handled by HF.
            self.model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)

    def rerank(self, question, docs, top_k=3) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        docs = list(docs)
        if not docs or top_k == 0:
            return []
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Rerank question must be a non-empty string")
        if any(not isinstance(doc.get("text"), str) or not doc["text"].strip() for doc in docs):
            raise ValueError("Rerank documents must contain non-empty text")
        self._load_model()
        predicted = self.model.predict(
            [(question, doc["text"]) for doc in docs], batch_size=self.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        predicted = predicted.tolist() if hasattr(predicted, "tolist") else list(predicted)
        if len(predicted) != len(docs):
            raise ValueError("Cross encoder returned the wrong number of scores")
        scores = []
        for score in predicted:
            # Single-output classification heads may return shape (N, 1).
            if isinstance(score, (list, tuple)) and len(score) == 1:
                score = score[0]
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Cross encoder returned a non-finite score")
            scores.append(score)
        ranking = sorted(range(len(docs)), key=lambda i: (-scores[i], i))[:top_k]
        return [SearchResult(
            docs[i]["text"], scores[i],
            {**deepcopy(docs[i].get("metadata", {})), "reranking": {
                "model": self.model_name, "score": scores[i],
                "retrieval_score": float(docs[i].get("score", 0.0)),
                "max_length": self.max_length,
            }},
        ) for i in ranking]
