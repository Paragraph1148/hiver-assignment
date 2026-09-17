"""Hybrid retrieval over historical customer->reply pairs.

BM25 and dense embeddings fail differently, which is the whole reason to use
both. BM25 nails rare literal tokens a support corpus is full of - "Onkyo",
"unidays", "1.0.65" - that an embedding blurs into a neighbourhood. Dense
retrieval catches paraphrase, which BM25 cannot see at all. Scores are
normalised per query and blended.

BM25 is implemented here rather than imported: it is thirty lines, it removes a
dependency from the reproduction path, and every term in it is inspectable.

Two exclusions matter for honest evaluation:
  * held-out threads never enter the index, or the evaluation retrieves its own
    answers;
  * for REPLY DRAFTING, deflections ("please DM us") are excluded too - grounding
    a draft in them teaches the agent to deflect. They stay available for intent
    retrieval, where the customer's message is what matters and the brand's reply
    is irrelevant.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

TOKEN = re.compile(r"[a-z0-9']+")


def tokenise(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


class BM25:
    """Okapi BM25 over a fixed corpus."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        # Standard BM25 idf, floored: without the floor a term in over half the
        # documents gets a negative weight and actively penalises matches.
        self.idf = {t: max(math.log((self.n - c + 0.5) / (c + 0.5) + 1.0), 0.01)
                    for t, c in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(docs):
            for t in set(d):
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in query:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


@dataclass
class Hit:
    idx: int
    score: float
    customer_text: str
    reply_text: str
    intent_weak: str
    dense: float
    lexical: float


class HybridIndex:
    def __init__(self, frame: pd.DataFrame, vectors: np.ndarray, alpha: float = 0.5):
        """alpha weights dense against lexical; 0.5 is the default, swept in eval."""
        self.df = frame.reset_index(drop=True)
        self.vecs = vectors
        self.alpha = alpha
        self.bm25 = BM25([tokenise(t) for t in self.df.customer_text])

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        """Per-query min-max. The two scorers live on different scales, so a raw
        sum would let BM25's unbounded range swamp cosine similarity."""
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    def search(self, query: str, qvec: np.ndarray, k: int = 5,
               exclude_threads: set[int] | None = None) -> list[Hit]:
        lex = self._norm(self.bm25.scores(tokenise(query)))
        den = self._norm(self.vecs @ qvec)
        blended = self.alpha * den + (1 - self.alpha) * lex
        if exclude_threads:
            mask = self.df.thread_id.isin(exclude_threads).to_numpy()
            blended = np.where(mask, -np.inf, blended)
        top = np.argpartition(-blended, min(k, len(blended) - 1))[:k]
        top = top[np.argsort(-blended[top])]
        return [Hit(int(i), float(blended[i]), self.df.customer_text.iloc[i],
                    self.df.reply_text.iloc[i], self.df.intent_weak.iloc[i],
                    float(den[i]), float(lex[i])) for i in top]


def build_index(purpose: str = "intent", alpha: float = 0.5) -> HybridIndex:
    """purpose 'intent' keeps every reply; 'draft' drops deflections."""
    import json
    from pathlib import Path

    from .embed import embed

    df = pd.read_parquet("data/interim/SpotifyCares_labelled.parquet")
    held = set(json.loads(Path("data/golden/held_out_threads.json").read_text())["thread_ids"])
    df = df[~df.thread_id.isin(held)].reset_index(drop=True)
    if purpose == "draft":
        df = df[~df.is_deflection].reset_index(drop=True)
    vecs = embed(df.customer_text.tolist(), show_progress=False)
    return HybridIndex(df, vecs, alpha=alpha)
