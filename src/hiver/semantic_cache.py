"""Semantic response cache - a PRODUCTION cost lever, not an eval artefact.

Not to be confused with `hiver.llm.cache`. That one is a recorded transcript of
an eval run, keyed on the exact prompt bytes, and exists so a grader with no API
key reproduces the published numbers. It never runs in the live agent.

This one runs in the live agent and answers a different question: a support
inbox asks the same thing over and over ("why am I still being charged", "songs
won't play"), so a new message that is semantically close to one already
answered can reuse that answer instead of paying for another call.

The tradeoff is real and has to be measured rather than assumed: a lower
similarity threshold serves more traffic from cache and saves more money, but
serves increasingly wrong answers. `simulate` replays a stream of messages and
reports exactly that curve, so the threshold is chosen from evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Entry:
    vector: np.ndarray
    payload: object
    query: str
    hits: int = 0


@dataclass
class SemanticCache:
    """Cosine-nearest lookup over previously answered queries.

    Linear scan: at the few thousand entries a single brand's live cache holds,
    an exact scan is well under a millisecond and avoids an index that has to be
    kept consistent with the store. Swap for a vector index if it ever grows.
    """
    threshold: float = 0.95
    max_entries: int = 5000
    entries: list[Entry] = field(default_factory=list)
    lookups: int = 0
    hits: int = 0

    def _matrix(self) -> np.ndarray:
        return np.vstack([e.vector for e in self.entries])

    def get(self, vector: np.ndarray) -> tuple[object | None, float]:
        self.lookups += 1
        if not self.entries:
            return None, 0.0
        sims = self._matrix() @ vector        # vectors are L2-normalised
        i = int(np.argmax(sims))
        best = float(sims[i])
        if best >= self.threshold:
            self.hits += 1
            self.entries[i].hits += 1
            return self.entries[i].payload, best
        return None, best

    def put(self, vector: np.ndarray, payload: object, query: str = "") -> None:
        if payload is None:
            # get() reports a miss as None, so a None payload would make every
            # subsequent hit on this entry indistinguishable from a miss.
            raise ValueError("payload must not be None; use a sentinel value")
        if len(self.entries) >= self.max_entries:
            # Evict the least-reused entry: a cache of never-hit entries is
            # just memory, and recency alone would evict popular evergreens.
            self.entries.pop(int(np.argmin([e.hits for e in self.entries])))
        self.entries.append(Entry(vector, payload, query))

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


def simulate(vectors: np.ndarray, preds: list, gold: list,
             threshold: float, order: np.ndarray | None = None) -> dict:
    """Replay a message stream through the cache and score what it actually served.

    `preds` is what a live call returns for each message; `gold` is the truth.
    On a MISS we pay for a call, serve preds[i], and store it. On a HIT we serve
    the prediction made for an EARLIER, similar message - so it counts as correct
    only if that borrowed answer happens to match this message's gold. That is
    the honest accounting: the cache's savings and its errors come from the same
    borrowing.
    """
    idx = np.arange(len(vectors)) if order is None else np.asarray(order)
    cache = SemanticCache(threshold=threshold)
    correct, from_cache = [], []
    for i in idx:
        borrowed, _sim = cache.get(vectors[i])
        if borrowed is not None:
            correct.append(float(borrowed == gold[i]))
            from_cache.append(1.0)
        else:
            correct.append(float(preds[i] == gold[i]))
            from_cache.append(0.0)
            cache.put(vectors[i], preds[i])
    live = float(np.mean([p == g for p, g in zip(preds, gold)]))
    return {"threshold": threshold,
            "hit_rate": float(np.mean(from_cache)),
            "accuracy": float(np.mean(correct)),
            "accuracy_no_cache": live,
            "cost_of_cache_pts": (live - float(np.mean(correct))) * 100,
            "calls_saved": int(np.sum(from_cache)),
            "entries": len(cache.entries)}
