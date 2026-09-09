"""Provider-agnostic chat types.

Deliberately no vendor SDKs anywhere in this package. All four providers expose
a plain HTTPS JSON API, and hand-rolling them keeps `pip install` small (which
the 15-minute reproduction budget cares about) and keeps every line explainable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatRequest:
    """Everything that can change an answer, and nothing that cannot.

    The API key is deliberately absent: it must never influence the cache key,
    otherwise rotating a key would silently invalidate a recorded run.
    """
    messages: tuple[Message, ...]
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    # When set, the provider is asked for strict JSON. Each adapter expresses
    # this differently; the request stays uniform.
    json_mode: bool = False
    stop: tuple[str, ...] = ()

    def cache_key(self, provider: str) -> str:
        payload = {
            "provider": provider,
            "model": self.model,
            "messages": [m.as_dict() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "stop": list(self.stop),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def preview(self, n: int = 160) -> str:
        """Short human-readable stub, stored alongside cached responses so the
        cache file stays auditable without storing every full prompt."""
        last = self.messages[-1].content if self.messages else ""
        return " ".join(last.split())[:n]


@dataclass
class ChatResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def json(self) -> Any:
        """Parse the reply as JSON, tolerating fenced blocks and prose padding."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("```", 2)[1]
            if t.lstrip().lower().startswith("json"):
                t = t.lstrip()[4:]
            t = t.rsplit("```", 1)[0] if "```" in t else t
        t = t.strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            # Last resort: the outermost {...} or [...] span.
            for op, cl in (("{", "}"), ("[", "]")):
                i, j = t.find(op), t.rfind(cl)
                if 0 <= i < j:
                    try:
                        return json.loads(t[i : j + 1])
                    except json.JSONDecodeError:
                        continue
            raise


class ProviderError(RuntimeError):
    """Provider call failed after exhausting retries."""


class CacheMiss(KeyError):
    """Replay mode required a cached response that does not exist."""
