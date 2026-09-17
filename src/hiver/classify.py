"""Intent classifiers: two baselines, one retrieval model, one LLM.

All three trained models learn from WEAK labels (cluster-derived) and are tested
against HUMAN labels. That is distant supervision, not a bug, but it caps what
they can reach: the weak labels disagree with the golden labels wherever the
clusterer was wrong, and the models faithfully learn the clusterer.

The trivial and simple baselines exist to make the LLM earn its place. If a
TF-IDF model gets within a couple of points, the LLM is not buying anything the
latency and cost justify.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .llm import LLM, Message
from .retrieve import HybridIndex
from .schema import INTENTS

INTENT_NAMES = [k for k, _ in INTENTS]


@dataclass
class Prediction:
    intent: str | None
    confidence: float
    source: str = ""
    failed: bool = False       # the call did not complete; NOT a model answer

    @property
    def ok(self) -> bool:
        return not self.failed and self.intent is not None


class MajorityClassifier:
    """Trivial baseline: always the most common training class."""
    name = "trivial: majority class"

    def fit(self, texts: list[str], labels: list[str]):
        self.label, n = Counter(labels).most_common(1)[0]
        self.prior = n / len(labels)
        return self

    def predict(self, texts: list[str], **kw) -> list[Prediction]:
        return [Prediction(self.label, self.prior) for _ in texts]


class TfidfLogReg:
    """Simple baseline: character+word TF-IDF into logistic regression.

    Character n-grams matter here: support text is full of misspellings and
    product strings that word features miss entirely.
    """
    name = "simple: TF-IDF + logistic regression"

    def fit(self, texts: list[str], labels: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        self.pipe = Pipeline([
            ("feats", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         min_df=3, sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ])
        self.pipe.fit(texts, labels)
        return self

    def predict(self, texts: list[str], **kw) -> list[Prediction]:
        proba = self.pipe.predict_proba(texts)
        classes = self.pipe.named_steps["clf"].classes_
        return [Prediction(classes[i.argmax()], float(i.max())) for i in proba]


class KnnClassifier:
    """Simple baseline: vote among the k nearest historical messages."""
    name = "simple: hybrid-retrieval kNN"

    def __init__(self, index: HybridIndex, k: int = 15):
        self.index, self.k = index, k

    def fit(self, *a, **kw):
        return self

    def predict(self, texts: list[str], vectors: np.ndarray | None = None,
                **kw) -> list[Prediction]:
        out = []
        for t, v in zip(texts, vectors):
            hits = self.index.search(t, v, k=self.k)
            # Weight each neighbour by its blended retrieval score, so a close
            # match counts for more than a distant one in the same k.
            tally: dict[str, float] = {}
            for h in hits:
                tally[h.intent_weak] = tally.get(h.intent_weak, 0.0) + max(h.score, 1e-6)
            best = max(tally, key=tally.get)
            out.append(Prediction(best, tally[best] / sum(tally.values())))
        return out


SYS = """You route incoming customer tweets for Spotify support.

Classify the message into exactly one intent:
{tax}

You are also shown similar past messages with the intent our clustering assigned
them. Those cluster labels are noisy - treat them as evidence, not as the answer.

Return JSON only: {{"intent":"<exact name>","confidence":<0.0-1.0>}}
confidence is your probability that this label is correct. Be honest: use low
values when the message is genuinely ambiguous, because a low score routes the
message to a human rather than being auto-sent."""


class LlmClassifier:
    """The system: retrieval-augmented LLM classification with self-reported
    confidence, which the escalation router consumes downstream."""
    name = "system: retrieval-augmented LLM"

    def __init__(self, index: HybridIndex, llm: LLM, k: int = 6,
                 prompt_variant: int = 0):
        self.index, self.llm, self.k = index, llm, k
        self.prompt_variant = prompt_variant

    def fit(self, *a, **kw):
        return self

    def _system(self) -> str:
        tax = "\n".join(f"  {k} - {d}" for k, d in INTENTS)
        base = SYS.format(tax=tax)
        # Paraphrases used only to measure prompt sensitivity, never to pick a
        # winner: the headline is always variant 0.
        if self.prompt_variant == 1:
            base = base.replace("You route incoming customer tweets for Spotify support.",
                                "You are a support triage assistant for Spotify on Twitter.")
        elif self.prompt_variant == 2:
            base = base.replace("Classify the message into exactly one intent:",
                                "Decide which single category best describes what the customer wants:")
        return base

    def predict(self, texts: list[str], vectors: np.ndarray | None = None,
                progress: bool = False, **kw) -> list[Prediction]:
        out = []
        for n, (t, v) in enumerate(zip(texts, vectors)):
            hits = self.index.search(t, v, k=self.k)
            ctx = "\n".join(f"  - [{h.intent_weak}] {h.customer_text[:160]}" for h in hits)
            user = f"SIMILAR PAST MESSAGES:\n{ctx}\n\nMESSAGE TO CLASSIFY:\n{t}"
            try:
                r = self.llm.chat([Message("system", self._system()), Message("user", user)],
                                  json_mode=True, max_tokens=4000)
                d = r.json()
                intent = d.get("intent")
                conf = float(d.get("confidence", 0.5))
            except Exception as e:
                # An infrastructure failure is NOT a model answer. Recording it
                # as one ("other", 0.0) silently converts an outage into a wrong
                # prediction and depresses the headline number - which is exactly
                # what happened on the first run, where 17 rate-limited calls
                # were scored as errors the model never made.
                out.append(Prediction(None, 0.0, source=type(e).__name__, failed=True))
                continue
            if intent not in INTENT_NAMES:
                # A well-formed reply naming an unknown class IS a model answer,
                # and an honest one to score: it fell back to `other` on purpose.
                intent, conf = "other", min(conf, 0.3)
            out.append(Prediction(intent, max(0.0, min(1.0, conf))))
            if progress and (n + 1) % 25 == 0:
                print(f"    {n+1}/{len(texts)}", flush=True)
        return out


def load_training() -> tuple[list[str], list[str]]:
    """Corpus minus held-out threads, with weak labels."""
    import json
    from pathlib import Path
    df = pd.read_parquet("data/interim/SpotifyCares_labelled.parquet")
    held = set(json.loads(Path("data/golden/held_out_threads.json").read_text())["thread_ids"])
    df = df[~df.thread_id.isin(held)]
    return df.customer_text.tolist(), df.intent_weak.tolist()
