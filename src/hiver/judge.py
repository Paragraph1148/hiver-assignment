"""LLM-as-judge for reply quality, with the biases that invalidate it guarded.

Three design choices, each aimed at a known failure of this method:

  * CROSS-FAMILY. The judge must come from a different model family than the
    generator. Models systematically prefer their own outputs, so grading
    gpt-oss with gpt-oss inflates the score for reasons that have nothing to do
    with reply quality. Default: generate on Groq, judge on Gemini.

  * POSITION-SWAPPED PAIRWISE. In a head-to-head, the same pair is judged twice
    with the order reversed. Judges have a real preference for whichever answer
    they see first; disagreement between the two orders measures that bias
    directly, and only order-consistent verdicts are counted as wins.

  * ABSOLUTE SCORES ON A RUBRIC, dimension by dimension. A single 1-10 "quality"
    number collapses groundedness and tone into one figure that cannot be
    debugged. The dimensions here are separable and each one names a distinct
    failure the system can actually have.

None of this makes the judge correct. It is validated against human ratings on a
subset, and that agreement is reported as the ceiling on every judged number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM, Message

DIMENSIONS = [
    ("grounded", "Is every claim supported by the precedents or by common knowledge? "
                 "Inventing account details, refunds or timelines scores 1."),
    ("helpful", "Does it move the customer forward - a real answer, or the right "
                "question when detail is genuinely missing?"),
    ("voice", "Does it sound like this team: warm, brief, plain? Not corporate, "
              "not over-apologetic, not padded."),
    ("safe", "Does it avoid promises the brand cannot keep and handle sensitive "
             "situations without harm?"),
]


@dataclass
class Verdict:
    scores: dict[str, int] = field(default_factory=dict)
    overall: float = 0.0
    note: str = ""
    failed: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and bool(self.scores)


ABS_SYS = """You grade draft replies written by a support agent for Spotify on Twitter.

You see the customer's message, the draft reply, and real replies Spotify's
agents sent to similar customers. The real replies show the house style; they
are NOT a gold answer, and a draft may legitimately be better than them.

Score each dimension 1-5 (1 unacceptable, 3 adequate to send, 5 excellent):
{dims}

Return JSON only:
{{"grounded":<1-5>,"helpful":<1-5>,"voice":<1-5>,"safe":<1-5>,"note":"<max 15 words>"}}
Be strict. Most real replies are a 3. Reserve 5 for genuinely excellent."""

PAIR_SYS = """You compare two draft replies to the same customer message for
Spotify's Twitter support.

Pick the one you would rather send. Judge on: grounded in what support can
actually do, helpful, in the team's voice, and safe. Length is not quality.

Return JSON only: {"winner":"A"|"B"|"tie","why":"<max 12 words>"}"""


class ReplyJudge:
    def __init__(self, llm: LLM):
        self.llm = llm

    def score(self, customer: str, reply: str, precedents: list[str]) -> Verdict:
        dims = "\n".join(f"  {k}: {d}" for k, d in DIMENSIONS)
        prec = "\n".join(f"  - {p[:200]}" for p in precedents[:4])
        user = (f"CUSTOMER MESSAGE:\n{customer}\n\n"
                f"REAL REPLIES SPOTIFY SENT TO SIMILAR CUSTOMERS:\n{prec}\n\n"
                f"DRAFT REPLY TO GRADE:\n{reply}")
        try:
            r = self.llm.chat([Message("system", ABS_SYS.format(dims=dims)),
                               Message("user", user)], json_mode=True, max_tokens=3000)
            d = r.json()
            scores = {k: int(d[k]) for k, _ in DIMENSIONS if k in d}
            if not scores:
                raise ValueError("no dimensions returned")
        except Exception:
            return Verdict(failed=True)
        return Verdict(scores=scores, overall=sum(scores.values()) / len(scores),
                       note=str(d.get("note", ""))[:60])

    def compare(self, customer: str, a: str, b: str) -> tuple[str, bool]:
        """Judge both orders. Returns (winner, order_consistent).

        A verdict that flips when the options swap is position bias, not a
        preference, and is reported as a tie.
        """
        def once(first: str, second: str) -> str:
            user = (f"CUSTOMER MESSAGE:\n{customer}\n\n"
                    f"REPLY A:\n{first}\n\nREPLY B:\n{second}")
            try:
                r = self.llm.chat([Message("system", PAIR_SYS), Message("user", user)],
                                  json_mode=True, max_tokens=2000)
                w = str(r.json().get("winner", "tie")).upper()
                return w if w in ("A", "B") else "TIE"
            except Exception:
                return "TIE"

        first = once(a, b)                       # a shown first
        second = once(b, a)                      # b shown first
        flip = {"A": "B", "B": "A", "TIE": "TIE"}
        second_in_ab = flip[second]              # translate back to a/b terms
        if first == second_in_ab:
            return ("a" if first == "A" else "b" if first == "B" else "tie"), True
        return "tie", False
