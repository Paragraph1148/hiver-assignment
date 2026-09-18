"""HTTP adapters for the four provider wire formats.

Grouped by wire format, not by vendor:
  OpenAICompat -> Groq, OpenAI, Together, OpenRouter, DeepSeek, Ollama, vLLM
  Anthropic    -> Claude
  Gemini       -> Google AI Studio

Anthropic note: assistant prefill (the classic trick for forcing JSON) returns
HTTP 400 on current Claude models. We therefore ask for JSON in the system
prompt on every provider and parse defensively, rather than relying on any
provider-specific structured-output switch. Uniform behaviour beats a slightly
higher parse rate on one vendor.
"""
from __future__ import annotations

import os
import random
import re
import time
from typing import Any

import httpx

from .base import ChatRequest, ChatResponse, ProviderError

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 8
# Hard ceiling on time spent retrying ONE request. Not every 429 is a rate
# limit that clears: a daily quota exhaustion returns the same status and never
# clears, and without a deadline the backoff loop will sit on it indefinitely.
# A run once spent 2h45m retrying an exhausted Gemini quota, producing nothing.
RETRY_DEADLINE_S = 300.0

# Providers put the wait in different places. Gemini returns no retry-after
# header but writes "Please retry in 35.7s" into the error body, and its free
# tier is a handful of requests per minute - so honouring that number is the
# difference between progress and a backoff loop that never syncs with the
# refill window.
_RETRY_IN = re.compile(r"retry in ([0-9.]+)s", re.I)
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"([0-9.]+)s"', re.I)


class BaseProvider:
    name: str = "base"
    default_model: str = ""
    env_key: str = ""

    # Minimum seconds between requests. Staying under the published limit beats
    # discovering it: a 429 costs a full retry window, a short wait costs the
    # wait. Set per provider from its free-tier rate.
    min_interval_s: float = 0.0
    _last_call: float = 0.0

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get(self.env_key, "")
        self.model = model or self.default_model
        self.base_url = (base_url or self.default_base_url()).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ProviderError(f"{self.name}: no API key (set {self.env_key})")

    def default_base_url(self) -> str:
        raise NotImplementedError

    # --- subclass contract -------------------------------------------------
    def _endpoint(self, model: str) -> str: raise NotImplementedError
    def _headers(self) -> dict[str, str]: raise NotImplementedError
    def _body(self, req: ChatRequest, model: str) -> dict[str, Any]: raise NotImplementedError
    def _parse(self, data: dict[str, Any]) -> tuple[str, int, int]: raise NotImplementedError

    # --- shared transport --------------------------------------------------
    def complete(self, req: ChatRequest) -> ChatResponse:
        model = req.model or self.model
        url, headers, body = self._endpoint(model), self._headers(), self._body(req, model)
        started = time.perf_counter()
        last: Exception | None = None

        for attempt in range(MAX_ATTEMPTS):
            if self.min_interval_s:
                gap = time.monotonic() - type(self)._last_call
                if gap < self.min_interval_s:
                    time.sleep(self.min_interval_s - gap)
                type(self)._last_call = time.monotonic()
            if time.perf_counter() - started > RETRY_DEADLINE_S:
                raise ProviderError(
                    f"{self.name}: gave up after {RETRY_DEADLINE_S:.0f}s of retries "
                    f"(likely an exhausted quota rather than a passing rate limit): {last}")
            try:
                with httpx.Client(timeout=self.timeout) as c:
                    r = c.post(url, headers=headers, json=body)
                if r.status_code in RETRY_STATUS:
                    last = ProviderError(f"{self.name}: HTTP {r.status_code}: {r.text[:200]}")
                    wait = self._retry_delay(r.headers.get("retry-after"), r.text)
                    if r.status_code == 429 and wait is None and attempt >= 1:
                        # A 429 that names no wait at all, twice running, is an
                        # exhausted allowance rather than a passing rate limit.
                        raise ProviderError(
                            f"{self.name}: quota exhausted with no retry window, "
                            f"not retrying: {r.text[:160]}")
                    self._sleep(attempt, wait)
                    continue
                if r.status_code >= 400:
                    # 400/401/403/404 are our bug or a bad key: retrying cannot help.
                    raise ProviderError(f"{self.name}: HTTP {r.status_code}: {r.text[:400]}")
                text, ptok, ctok = self._parse(r.json())
                return ChatResponse(
                    text=text, model=model, provider=self.name,
                    prompt_tokens=ptok, completion_tokens=ctok,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                self._sleep(attempt, None)

        raise ProviderError(f"{self.name}: failed after {MAX_ATTEMPTS} attempts: {last}")

    @staticmethod
    def _retry_delay(header: str | None, body: str) -> float | None:
        """Seconds the provider asked us to wait, from wherever it put it."""
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        for pat in (_RETRY_DELAY, _RETRY_IN):
            if m := pat.search(body or ""):
                return float(m.group(1))
        return None

    @staticmethod
    def _sleep(attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            # Add a small margin: waking exactly on the boundary tends to race
            # the refill and burn another attempt.
            time.sleep(min(retry_after + 1.0, 90.0))
            return
        # Full jitter, capped at a minute: free-tier buckets refill per minute,
        # so a ceiling below that guarantees the retry lands in the same
        # exhausted window and burns an attempt for nothing.
        time.sleep(random.uniform(0, min(2.0 * (2 ** attempt), 60.0)))


class OpenAICompat(BaseProvider):
    """Any /v1/chat/completions endpoint."""
    name = "openai_compat"
    env_key = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"

    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def _endpoint(self, model: str) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, req: ChatRequest, model: str) -> dict[str, Any]:
        b: dict[str, Any] = {
            "model": model,
            "messages": [m.as_dict() for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.json_mode:
            b["response_format"] = {"type": "json_object"}
        if req.stop:
            b["stop"] = list(req.stop)
        return b

    def _parse(self, d: dict[str, Any]) -> tuple[str, int, int]:
        msg = d["choices"][0]["message"]
        # Some reasoning models put the answer in `content` and traces elsewhere;
        # an empty content with a populated reasoning field is a real case.
        text = msg.get("content") or msg.get("reasoning_content") or ""
        u = d.get("usage") or {}
        return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


class Groq(OpenAICompat):
    name = "groq"
    env_key = "GROQ_API_KEY"
    default_model = "openai/gpt-oss-120b"

    def default_base_url(self) -> str:
        return "https://api.groq.com/openai/v1"


class OpenAI(OpenAICompat):
    name = "openai"
    env_key = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"


class Anthropic(BaseProvider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_model = "claude-sonnet-5"
    api_version = "2023-06-01"

    def default_base_url(self) -> str:
        return "https://api.anthropic.com/v1"

    def _endpoint(self, model: str) -> str:
        return f"{self.base_url}/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }

    def _body(self, req: ChatRequest, model: str) -> dict[str, Any]:
        # Claude takes the system prompt as a top-level field, not a message.
        system = " ".join(m.content for m in req.messages if m.role == "system").strip()
        turns = [m.as_dict() for m in req.messages if m.role != "system"]
        b: dict[str, Any] = {
            "model": model,
            "messages": turns,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if system:
            b["system"] = system
        if req.stop:
            b["stop_sequences"] = list(req.stop)
        return b

    def _parse(self, d: dict[str, Any]) -> tuple[str, int, int]:
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage") or {}
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


class Gemini(BaseProvider):
    name = "gemini"
    env_key = "GEMINI_API_KEY"
    # Free tier allows 5 requests/minute on this model, so pace at 13s.
    min_interval_s = 13.0
    # Pinned, never an alias: `gemini-flash-latest` silently changes model
    # underneath a recorded cache, which would break replay reproducibility.
    # Note ListModels advertises models that generateContent then refuses for
    # new accounts (2.5-flash is one), so this was chosen by probing, not by
    # reading the model list.
    default_model = "gemini-3.6-flash"

    def default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    def _endpoint(self, model: str) -> str:
        return f"{self.base_url}/models/{model}:generateContent"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def _body(self, req: ChatRequest, model: str) -> dict[str, Any]:
        system = " ".join(m.content for m in req.messages if m.role == "system").strip()
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in req.messages if m.role != "system"
        ]
        cfg: dict[str, Any] = {"temperature": req.temperature, "maxOutputTokens": req.max_tokens}
        if req.json_mode:
            cfg["responseMimeType"] = "application/json"
        if req.stop:
            cfg["stopSequences"] = list(req.stop)
        b: dict[str, Any] = {"contents": contents, "generationConfig": cfg}
        if system:
            b["systemInstruction"] = {"parts": [{"text": system}]}
        return b

    def _parse(self, d: dict[str, Any]) -> tuple[str, int, int]:
        cands = d.get("candidates") or []
        text = ""
        if cands:
            text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
        u = d.get("usageMetadata") or {}
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


REGISTRY: dict[str, type[BaseProvider]] = {
    "groq": Groq,
    "openai": OpenAI,
    "anthropic": Anthropic,
    "gemini": Gemini,
    "openai_compat": OpenAICompat,
}
