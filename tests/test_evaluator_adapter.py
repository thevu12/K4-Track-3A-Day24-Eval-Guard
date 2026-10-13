"""Evaluator/generation error paths without network or paid calls."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import m4_eval as phase


@pytest.fixture
def ragas_fake(monkeypatch):
    monkeypatch.setattr(phase, "LLM_PROVIDER", "openai")
    rows = [{name: 0.5 for name in phase.METRICS}]
    result = SimpleNamespace(to_pandas=lambda: SimpleNamespace(to_dict=lambda orient: rows))
    evaluate = Mock(return_value=result)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=SimpleNamespace(from_dict=lambda data: data)))
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(evaluate=evaluate))
    monkeypatch.setitem(sys.modules, "ragas.metrics", SimpleNamespace(**{
        name: SimpleNamespace(llm=None, embeddings=None) for name in phase.METRICS}))
    monkeypatch.setitem(sys.modules, "ragas.llms", SimpleNamespace(LangchainLLMWrapper=lambda client: client))
    monkeypatch.setitem(sys.modules, "ragas.embeddings", SimpleNamespace(LangchainEmbeddingsWrapper=lambda client: client))
    monkeypatch.setitem(sys.modules, "ragas.run_config", SimpleNamespace(RunConfig=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(
        ChatOpenAI=lambda model, **kw: SimpleNamespace(model_name=model),
        OpenAIEmbeddings=lambda model, **kw: SimpleNamespace(model=model)))
    return rows, evaluate


def test_evaluator_reports_explicit_roles_and_raises_failures(ragas_fake):
    _, evaluate = ragas_fake
    result = phase.evaluate_ragas(["Q"], ["A"], [["C"]], ["R"])
    assert result["metadata"]["evaluator_models"] == sorted([
        phase.EVALUATOR_LLM_MODEL, phase.EVALUATOR_EMBEDDING_MODEL])
    assert result["per_question"][0].faithfulness == 0.5
    call = evaluate.call_args.kwargs
    assert call["raise_exceptions"] is True
    assert call["run_config"]["max_workers"] == phase.EVALUATOR_MAX_WORKERS
    assert all(metric.llm is call["llm"] for metric in call["metrics"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5", None])
def test_no_fake_replacement_for_invalid_metrics(ragas_fake, bad):
    ragas_fake[0][0]["faithfulness"] = bad
    with pytest.raises(ValueError, match="invalid metrics"):
        phase.evaluate_ragas(["Q"], ["A"], [["C"]], ["R"])


def test_evaluator_rejects_missing_rows(ragas_fake):
    ragas_fake[0].clear()
    with pytest.raises(ValueError, match="number of rows"):
        phase.evaluate_ragas(["Q"], ["A"], [["C"]], ["R"])


@pytest.mark.parametrize("inputs", [([], [], [], []), (["Q"], [], [], []),
                                    (["Q"], [""], [["C"]], ["R"]),
                                    (["Q"], ["A"], ["not-list"], ["R"])])
def test_evaluator_validates_before_imports(inputs):
    with pytest.raises(ValueError):
        phase.evaluate_ragas(*inputs)


def test_generation_empty_answer_falls_back_without_logging_payload(capsys):
    from scripts.setup_answers import run_query
    search = SimpleNamespace(search=lambda q: [SimpleNamespace(text="context", score=1, metadata={})])
    reranker = SimpleNamespace(rerank=lambda *a, **k: [SimpleNamespace(text="context")])
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    assert run_query("Q", search, reranker, 3, client) == ("context", ["context"], "extractive_fallback", None)
    client.chat.completions.create.side_effect = RuntimeError("secret-never-log")
    run_query("Q", search, reranker, 3, client)
    assert "secret-never-log" not in capsys.readouterr().out


def test_generation_authentication_failure_aborts_without_logging_payload():
    from scripts.setup_answers import run_query
    search = SimpleNamespace(search=lambda q: [SimpleNamespace(text="context", score=1, metadata={})])
    reranker = SimpleNamespace(rerank=lambda *a, **k: [SimpleNamespace(text="context")])
    error = type("AuthenticationError", (Exception,), {"status_code": 401})("secret-never-log")
    client = Mock()
    client.chat.completions.create.side_effect = error

    with pytest.raises(RuntimeError, match="aborting generation") as failure:
        run_query("Q", search, reranker, 3, client)
    assert "secret-never-log" not in str(failure.value)
    client.chat.completions.create.assert_called_once()
