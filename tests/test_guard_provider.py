"""Provider wiring regressions; no live safety claims or network requests."""
import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src import phase_c_guard as guard


@pytest.mark.parametrize("reply", ["Yes", "No", "", "   ", None])
def test_guard_chat_client_rejects_empty_verdict(monkeypatch, reply):
    class FakeChat:
        def __init__(self, **kwargs):
            self.options = kwargs

        async def _agenerate(self, *args, **kwargs):
            return SimpleNamespace(generations=[SimpleNamespace(text=reply)])

    configuration = SimpleNamespace(models=[SimpleNamespace(model="old")])
    make_rails = Mock(side_effect=lambda config, llm: SimpleNamespace(llm=llm))
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChat))
    monkeypatch.setitem(sys.modules, "nemoguardrails", SimpleNamespace(
        RailsConfig=SimpleNamespace(from_path=lambda path: configuration), LLMRails=make_rails))
    monkeypatch.setattr(guard, "LLM_BASE_URL", "https://example.invalid/")
    pause = AsyncMock()
    monkeypatch.setattr(guard.asyncio, "sleep", pause)
    before = dict(os.environ)
    rails = guard.setup_nemo_rails()
    assert dict(os.environ) == before
    assert configuration.models[0].model == guard.JUDGE_MODEL
    assert rails.llm.options["model"] == guard.JUDGE_MODEL
    if reply and reply.strip():
        assert asyncio.run(rails.llm._agenerate()).generations[0].text == reply
    else:
        with pytest.raises(ValueError, match="no safety verdict"):
            asyncio.run(rails.llm._agenerate())
    pause.assert_awaited_once_with(4.25)


def test_self_check_token_budget_is_explicit():
    nemo = pytest.importorskip("nemoguardrails")
    configuration = nemo.RailsConfig.from_path(guard.GUARDRAILS_CONFIG_DIR)
    prompts = {prompt.task: prompt for prompt in configuration.prompts}
    for task in ("self_check_input", "self_check_output"):
        assert prompts[task].max_tokens == 1024
