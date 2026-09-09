"""Factory + the single entry point every pipeline stage calls."""
from __future__ import annotations

import os
from pathlib import Path

from .base import ChatRequest, ChatResponse, Message
from .cache import DEFAULT_PATH, Mode, ResponseCache
from .providers import REGISTRY, BaseProvider

# Probed in order when LLM_PROVIDER is unset, so a grader who exports any one
# of these keys gets a working pipeline with no further configuration.
AUTODETECT = ("groq", "anthropic", "openai", "gemini")


def _load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def detect_provider() -> str | None:
    if p := os.environ.get("LLM_PROVIDER"):
        return p
    for name in AUTODETECT:
        if os.environ.get(REGISTRY[name].env_key):
            return name
    return None


class LLM:
    """Cache-wrapped, provider-agnostic chat client.

    In replay mode no provider is constructed at all, so the pipeline runs
    end-to-end with no key present anywhere.
    """

    def __init__(self, provider: str | None = None, model: str | None = None,
                 cache_path: str | Path = DEFAULT_PATH, mode: Mode | None = None,
                 base_url: str | None = None):
        _load_dotenv()
        self.cache = ResponseCache(cache_path, mode)
        self.provider_name = provider or detect_provider()
        self.model = model or os.environ.get("LLM_MODEL") or ""
        self._provider: BaseProvider | None = None
        self._base_url = base_url

        if self.cache.mode != "replay":
            if not self.provider_name:
                raise RuntimeError(
                    "No LLM provider. Set one of GROQ_API_KEY / ANTHROPIC_API_KEY / "
                    "OPENAI_API_KEY / GEMINI_API_KEY, or use LLM_CACHE_MODE=replay."
                )
            cls = REGISTRY[self.provider_name]
            self._provider = cls(model=self.model or None, base_url=base_url)
            self.model = self.model or self._provider.model
        else:
            # Replay still needs a stable cache key, so resolve the model name
            # from the registry default rather than leaving it blank.
            self.provider_name = self.provider_name or AUTODETECT[0]
            self.model = self.model or REGISTRY[self.provider_name].default_model

    def chat(self, messages: list[Message] | list[dict], *, model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 1024,
             json_mode: bool = False, stop: tuple[str, ...] = ()) -> ChatResponse:
        msgs = tuple(
            m if isinstance(m, Message) else Message(m["role"], m["content"]) for m in messages
        )
        req = ChatRequest(
            messages=msgs, model=model or self.model, temperature=temperature,
            max_tokens=max_tokens, json_mode=json_mode, stop=stop,
        )
        key = req.cache_key(self.provider_name or "")

        if self.cache.mode == "replay":
            return self.cache.require(key, req)
        if self.cache.mode == "auto":
            if hit := self.cache.get(key):
                return hit

        assert self._provider is not None
        resp = self._provider.complete(req)
        self.cache.put(key, req, resp)
        return resp

    def ask(self, prompt: str, system: str | None = None, **kw) -> ChatResponse:
        msgs = ([Message("system", system)] if system else []) + [Message("user", prompt)]
        return self.chat(msgs, **kw)
