"""Offline tests for the provider layer. No network, no API keys."""
import json
import os

import pytest

from hiver.llm.base import CacheMiss, ChatRequest, ChatResponse, Message
from hiver.llm.cache import ResponseCache
from hiver.llm.client import LLM
from hiver.llm.providers import Anthropic, Gemini, Groq, OpenAI

MSGS = (Message("system", "You are terse."), Message("user", "Hi"),
        Message("assistant", "Hello"), Message("user", "Bye"))
REQ = ChatRequest(messages=MSGS, model="M", temperature=0.2, max_tokens=64, json_mode=True)


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(k, "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


# --- cache key contract ---------------------------------------------------

def test_cache_key_ignores_api_key(monkeypatch):
    """Rotating a credential must not invalidate a recorded run."""
    before = REQ.cache_key("groq")
    monkeypatch.setenv("GROQ_API_KEY", "a-completely-different-key")
    assert REQ.cache_key("groq") == before


def test_cache_key_separates_providers_and_params():
    base = REQ.cache_key("groq")
    assert REQ.cache_key("anthropic") != base
    for field, value in [("temperature", 0.9), ("max_tokens", 65),
                         ("json_mode", False), ("model", "other")]:
        import dataclasses
        assert dataclasses.replace(REQ, **{field: value}).cache_key("groq") != base


# --- wire formats ---------------------------------------------------------

def test_anthropic_hoists_system_out_of_messages():
    b = Anthropic()._body(REQ, "M")
    assert b["system"] == "You are terse."
    assert all(m["role"] != "system" for m in b["messages"])
    # Prefill is rejected by current Claude models: never end on an assistant turn.
    assert b["messages"][-1]["role"] == "user"


def test_anthropic_headers_carry_version():
    h = Anthropic()._headers()
    assert h["x-api-key"] == "test-key" and h["anthropic-version"]
    assert "Authorization" not in h


def test_gemini_maps_assistant_role_to_model():
    b = Gemini()._body(REQ, "M")
    assert [c["role"] for c in b["contents"]] == ["user", "model", "user"]
    assert b["systemInstruction"]["parts"][0]["text"] == "You are terse."
    assert b["generationConfig"]["responseMimeType"] == "application/json"


def test_openai_compat_json_mode():
    for cls in (Groq, OpenAI):
        assert cls()._body(REQ, "M")["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("cls,payload", [
    (Groq, {"choices": [{"message": {"content": "X"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}}),
    (Anthropic, {"content": [{"type": "text", "text": "X"}],
                 "usage": {"input_tokens": 3, "output_tokens": 1}}),
    (Gemini, {"candidates": [{"content": {"parts": [{"text": "X"}]}}],
              "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1}}),
])
def test_parsers_agree(cls, payload):
    assert cls()._parse(payload) == ("X", 3, 1)


def test_groq_reasoning_content_fallback():
    """Some reasoning models leave `content` empty and fill a sibling field."""
    d = {"choices": [{"message": {"content": None, "reasoning_content": "X"}}], "usage": {}}
    assert Groq()._parse(d)[0] == "X"


# --- cache behaviour ------------------------------------------------------

def test_replay_miss_raises_instead_of_calling_out(tmp_path):
    llm = LLM(cache_path=tmp_path / "c.jsonl", mode="replay", provider="groq", model="M")
    with pytest.raises(CacheMiss):
        llm.ask("never recorded")


def test_replay_needs_no_credentials(tmp_path, monkeypatch):
    cache = ResponseCache(tmp_path / "c.jsonl", mode="auto")
    req = ChatRequest(messages=(Message("user", "q"),), model="M")
    cache.put(req.cache_key("groq"), req,
              ChatResponse(text="recorded", model="M", provider="groq"))
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    llm = LLM(cache_path=tmp_path / "c.jsonl", mode="replay", provider="groq", model="M")
    assert llm.ask("q").text == "recorded"


def test_cache_survives_a_torn_line(tmp_path):
    p = tmp_path / "c.jsonl"
    req = ChatRequest(messages=(Message("user", "q"),), model="M")
    good = {"key": req.cache_key("groq"), "provider": "groq", "model": "M", "text": "ok"}
    p.write_text(json.dumps(good) + "\n" + '{"key": "half-writ')
    assert len(ResponseCache(p, mode="replay")) == 1


def test_provider_autodetect_prefers_configured(monkeypatch):
    from hiver.llm.client import detect_provider
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert detect_provider() == "gemini"
    monkeypatch.delenv("LLM_PROVIDER")
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert detect_provider() == "gemini"  # only key left


# --- response parsing -----------------------------------------------------

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Sure, here you go: {"a": 1}',
])
def test_json_extraction_tolerates_padding(raw):
    assert ChatResponse(text=raw, model="M", provider="p").json() == {"a": 1}
