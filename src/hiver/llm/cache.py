"""Content-addressed response cache.

This is the mechanism behind `make reproduce`. Every LLM call is keyed by
sha256(provider, model, messages, params) and appended to a JSONL file that is
committed to the repo. A grader with no API key at all replays the recorded run
and gets numerically identical headline results in minutes.

Modes:
  auto   - serve from cache, call the provider on a miss, record the result
  replay - serve from cache; a miss is a hard error (no network, no spend)
  live   - always call the provider and overwrite the recorded entry

`replay` failing loudly on a miss is deliberate. A cache that silently falls
through to the network turns "reproducible" into "reproducible if you happen to
hold a key", and quietly bills whoever runs it.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .base import CacheMiss, ChatRequest, ChatResponse

Mode = Literal["auto", "replay", "live"]
DEFAULT_PATH = Path("data/llm_cache.jsonl")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


class ResponseCache:
    def __init__(self, path: str | Path = DEFAULT_PATH, mode: Mode | None = None):
        self.path = Path(path)
        self.mode: Mode = mode or os.environ.get("LLM_CACHE_MODE", "auto")  # type: ignore[assignment]
        if self.mode not in ("auto", "replay", "live"):
            raise ValueError(f"bad cache mode {self.mode!r}")
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from an interrupted run
                if k := rec.get("key"):
                    self._index[k] = rec  # later entries win

    def __len__(self) -> int:
        return len(self._index)

    def get(self, key: str) -> ChatResponse | None:
        rec = self._index.get(key)
        if rec is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return ChatResponse(
            text=rec["text"], model=rec["model"], provider=rec["provider"],
            prompt_tokens=rec.get("prompt_tokens", 0),
            completion_tokens=rec.get("completion_tokens", 0),
            latency_ms=rec.get("latency_ms", 0.0), cached=True,
        )

    def put(self, key: str, req: ChatRequest, resp: ChatResponse) -> None:
        rec = {
            "key": key,
            "provider": resp.provider,
            "model": resp.model,
            "text": resp.text,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_ms": round(resp.latency_ms, 1),
            # Stub only: keeps the file auditable without archiving every prompt.
            "preview": req.preview(),
        }
        with self._lock:
            self._index[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.stats.writes += 1

    def require(self, key: str, req: ChatRequest) -> ChatResponse:
        resp = self.get(key)
        if resp is None:
            raise CacheMiss(
                f"replay mode: no cached response for {key[:12]}… "
                f"(prompt: {req.preview(80)!r}). Run with LLM_CACHE_MODE=auto and a key to record it."
            )
        return resp
