"""RAGAS 0.1 adapter: measured metrics only, with evaluator provenance."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from copy import deepcopy
import math
from numbers import Real
from types import SimpleNamespace

from config import (EMBEDDING_DEVICE, EVALUATOR_EMBEDDING_MODEL, EVALUATOR_LLM_MODEL,
                    EVALUATOR_MAX_WORKERS, LLM_PROVIDER, ROOT_DIR)
from src.provider_client import langchain_provider_kwargs

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def evaluate_ragas(questions, answers, contexts, ground_truths):
    """Evaluate aligned rows; never mask failed/NaN metrics with made-up scores.

    Generation and judge models are deliberately not used to infer the evaluator.
    Imports are lazy so the offline unit-test environment stays dependency-light.
    """
    if not questions or not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("RAGAS requires non-empty, equally sized input lists.")
    for name, values in (("question", questions), ("answer", answers), ("ground_truth", ground_truths)):
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Every {name} must be non-empty text.")
    if any(not isinstance(row, list) or any(not isinstance(value, str) or not value.strip()
                                          for value in row) for row in contexts):
        raise ValueError("Contexts must be lists of non-empty strings.")
    if EVALUATOR_MAX_WORKERS < 1:
        raise ValueError("EVALUATOR_MAX_WORKERS must be positive.")
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
    except ImportError as error:
        raise RuntimeError("Install full requirements in Python 3.11 before live RAGAS evaluation.") from error

    dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                 "contexts": contexts, "ground_truth": ground_truths})
    provider_kwargs = langchain_provider_kwargs()
    if LLM_PROVIDER == "gemini":
        class RateLimitedGeminiChat(ChatOpenAI):
            async def _agenerate(self, *args, **kwargs):
                # Free-tier Gemini allows 15 requests/minute for this model.
                await asyncio.sleep(4.25)
                return await super()._agenerate(*args, **kwargs)
        evaluator = RateLimitedGeminiChat(
            model=EVALUATOR_LLM_MODEL, temperature=0, max_retries=2,
            **provider_kwargs)
    else:
        evaluator = ChatOpenAI(model=EVALUATOR_LLM_MODEL, temperature=0, max_retries=2,
                               **provider_kwargs)
    if LLM_PROVIDER == "gemini":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name=EVALUATOR_EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        embedding_model = OpenAIEmbeddings(model=EVALUATOR_EMBEDDING_MODEL, max_retries=2,
                                           **provider_kwargs)
    llm = LangchainLLMWrapper(evaluator)
    embeddings = LangchainEmbeddingsWrapper(embedding_model)
    # RAGAS exports mutable singletons. Own each metric to avoid cross-run model leakage.
    metrics = deepcopy([faithfulness, answer_relevancy, context_precision, context_recall])
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
        # Gemini's OpenAI-compatibility endpoint rejects n>1 candidates.
        # AnswerRelevancy normally uses strictness=3 (one request with n=3).
        if LLM_PROVIDER == "gemini" and hasattr(metric, "strictness"):
            metric.strictness = 1
    workers = 1 if LLM_PROVIDER == "gemini" else EVALUATOR_MAX_WORKERS
    def evaluate_dataset(batch):
        result = evaluate(batch, metrics=metrics, llm=llm, embeddings=embeddings,
                          run_config=RunConfig(timeout=300, max_retries=2, max_workers=workers),
                          raise_exceptions=True)
        return result.to_pandas().to_dict(orient="records")

    if LLM_PROVIDER == "gemini":
        from src.experiment_support import write_json
        rows = []
        for index, (question, answer, context, truth) in enumerate(
                zip(questions, answers, contexts, ground_truths)):
            identity = [question, answer, context, truth, EVALUATOR_LLM_MODEL,
                        EVALUATOR_EMBEDDING_MODEL, "ragas-0.1.22-strictness-1"]
            digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
            cache = Path(ROOT_DIR) / ".cache" / "ragas_rows" / f"{digest}.json"
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf-8"))
            else:
                batch = Dataset.from_dict({"question": [question], "answer": [answer],
                                           "contexts": [context], "ground_truth": [truth]})
                row = evaluate_dataset(batch)[0]
                # Persist only measured finite metrics, never NaN or synthetic replacements.
                if all(isinstance(row.get(name), Real) and not isinstance(row[name], bool)
                       and math.isfinite(row[name]) and 0 <= row[name] <= 1 for name in METRICS):
                    write_json(cache, {name: float(row[name]) for name in METRICS})
            rows.append(row)
            print(f"RAGAS checkpoint {index + 1}/{len(questions)}", flush=True)
    else:
        rows = evaluate_dataset(dataset)
    if len(rows) != len(questions):
        raise ValueError("RAGAS returned a different number of rows.")
    measured = []
    for index, row in enumerate(rows):
        if any(name not in row or isinstance(row[name], bool) or not isinstance(row[name], Real)
               for name in METRICS):
            raise ValueError(f"RAGAS row {index + 1} has invalid metrics; expected numeric scores.")
        values = {name: float(row[name]) for name in METRICS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError(f"RAGAS row {index + 1} has invalid metrics; inspect the failed evaluation.")
        measured.append(SimpleNamespace(**values))
    return {"per_question": measured, "metadata": {
        "evaluator_models": sorted({evaluator.model_name, EVALUATOR_EMBEDDING_MODEL})}}
