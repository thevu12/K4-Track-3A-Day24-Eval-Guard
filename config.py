"""Shared configuration for K4 / Track 3A: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

# This repository's explicit local configuration wins over stale shell placeholders.
load_dotenv(override=True)

# --- LLM provider ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "openai").lower()
if LLM_PROVIDER not in {"openai", "gemini"}:
    raise ValueError("LLM_PROVIDER must be 'openai' or 'gemini'.")
LLM_API_KEY = GEMINI_API_KEY if LLM_PROVIDER == "gemini" else OPENAI_API_KEY
LLM_BASE_URL = ("https://generativelanguage.googleapis.com/v1beta/openai/"
                if LLM_PROVIDER == "gemini" else None)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "lab24_production"
NAIVE_COLLECTION = "lab24_naive"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", "16"))
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", "8192"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "1024"))
RRF_K = 60

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIXTURES_DIR = os.path.join(ROOT_DIR, "fixtures")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
TEST_SET_PATH = os.path.join(FIXTURES_DIR, "test_set_50q.json")
ANSWERS_PATH = os.path.join(ARTIFACTS_DIR, "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(FIXTURES_DIR, "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(FIXTURES_DIR, "adversarial_set_20.json")
CLEAN_SET_PATH = os.path.join(FIXTURES_DIR, "clean_set_8.json")
OUTPUT_SET_PATH = os.path.join(FIXTURES_DIR, "output_set_4.json")
GUARDRAILS_CONFIG_DIR = os.path.join(ROOT_DIR, "guardrails")
PAIR_QUESTION_IDS = (1, 5, 12, 21, 41)

# --- LLM Judge ---
_DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite" if LLM_PROVIDER == "gemini" else "gpt-4o-mini"
_DEFAULT_EMBEDDING_MODEL = (EMBEDDING_MODEL if LLM_PROVIDER == "gemini"
                            else "text-embedding-3-small")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", _DEFAULT_LLM_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", _DEFAULT_LLM_MODEL)
EVALUATOR_LLM_MODEL = os.getenv("EVALUATOR_LLM_MODEL", _DEFAULT_LLM_MODEL)
EVALUATOR_EMBEDDING_MODEL = os.getenv("EVALUATOR_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
EVALUATOR_MAX_WORKERS = int(os.getenv("EVALUATOR_MAX_WORKERS", "4"))

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # Full guard target; Task 12 currently measures input guards only.
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
