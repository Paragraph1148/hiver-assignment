"""Reply drafting: two baselines and a retrieval-grounded generator.

The nearest-reply baseline is the one that matters. Copying the most similar
historical reply verbatim is a genuinely strong strategy on a support corpus,
where the same twenty problems recur for years - and if the LLM cannot clearly
beat it, the LLM is not earning its latency and cost. Reporting that honestly is
more useful than hiding it behind a chart.

The canned baseline is deliberately the real thing rather than a strawman: it is
what a large share of brands in this dataset actually send, so beating it is the
minimum bar for claiming the system does anything at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM, Message
from .retrieve import Hit, HybridIndex

CANNED = ("Hey there! Sorry to hear you're having trouble. Please send us a DM "
          "with a few more details and we'll take a look.")


@dataclass
class Draft:
    text: str
    precedents: list[Hit] = field(default_factory=list)
    failed: bool = False
    source: str = ""

    @property
    def ok(self) -> bool:
        return not self.failed and bool(self.text.strip())


class CannedReply:
    """Trivial baseline: one fixed deflection, sent to everyone."""
    name = "trivial: one canned deflection"

    def draft(self, texts, vectors=None, **kw) -> list[Draft]:
        return [Draft(CANNED) for _ in texts]


class NearestReply:
    """Simple baseline: return the historical reply to the most similar message."""
    name = "simple: copy nearest historical reply"

    def __init__(self, index: HybridIndex):
        self.index = index

    def draft(self, texts, vectors=None, **kw) -> list[Draft]:
        out = []
        for t, v in zip(texts, vectors):
            hits = self.index.search(t, v, k=1)
            out.append(Draft(hits[0].reply_text if hits else CANNED, precedents=hits))
        return out


SYS = """You draft replies for Spotify's Twitter support team.

You are given the customer's message and how Spotify's agents actually handled
similar cases. Ground your reply in those precedents: the tone, the level of
detail and the kind of resolution should match what this team really does.

Rules:
- Never invent a fact about the customer's account. You cannot see their account.
- Never promise a refund, a credit, or a specific timeline.
- If the message lacks the detail needed to help, ask one specific question.
- Match the precedents' voice: warm, brief, plain. One or two sentences.
- No greeting boilerplate beyond a short opener. No sign-off, no agent initials.
- Write only the reply text. No quotes, no labels, no explanation."""


class RagDrafter:
    name = "system: retrieval-grounded LLM"

    def __init__(self, llm: LLM, index: HybridIndex, k: int = 5):
        self.llm, self.index, self.k = llm, index, k

    def draft(self, texts, vectors=None, intents=None, progress=False, **kw) -> list[Draft]:
        out = []
        for n, (t, v) in enumerate(zip(texts, vectors)):
            hits = self.index.search(t, v, k=self.k)
            prec = "\n\n".join(
                f"CUSTOMER: {h.customer_text[:220]}\nSPOTIFY: {h.reply_text[:220]}"
                for h in hits)
            hint = f"\n\nPredicted intent: {intents[n]}" if intents is not None else ""
            user = f"HOW SPOTIFY HANDLED SIMILAR CASES:\n\n{prec}\n\n---\nCUSTOMER MESSAGE:\n{t}{hint}"
            try:
                r = self.llm.chat([Message("system", SYS), Message("user", user)],
                                  max_tokens=4000, temperature=0.3)
                text = r.text.strip().strip('"')
            except Exception as e:
                out.append(Draft("", precedents=hits, failed=True, source=type(e).__name__))
                continue
            out.append(Draft(text, precedents=hits))
            if progress and (n + 1) % 25 == 0:
                print(f"    {n+1}/{len(texts)}", flush=True)
        return out
