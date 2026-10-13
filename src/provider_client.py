"""Provider-neutral constructors for the OpenAI-compatible API surface."""
from __future__ import annotations

from config import LLM_API_KEY, LLM_BASE_URL


def openai_client(**kwargs):
    from openai import OpenAI
    options = {"api_key": LLM_API_KEY, **kwargs}
    if LLM_BASE_URL:
        options["base_url"] = LLM_BASE_URL
    return OpenAI(**options)


def langchain_provider_kwargs() -> dict:
    options = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        options["base_url"] = LLM_BASE_URL
    return options
