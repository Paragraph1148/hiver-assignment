"""Escalation routing: auto-handle or send to a human, with a stated reason.

A constraint shapes everything here: escalation labels exist ONLY in the 249
golden items. There is no corpus-scale supervision, because whether a message
*should* have been escalated was never recorded - only what the brand happened
to do. So:

  * the prior baseline is fitted by leave-one-out on the golden set, never on
    the same row it scores;
  * the LLM router is zero-shot against the written policy, not trained;
  * and both are additionally checked against OBSERVABLE proxies in the corpus
    (the brand deflected to DM, the thread ran long, the customer voiced
    dissatisfaction). Those proxies are not ground truth either, but they are
    not derived from the annotator's opinion, so agreement with them is
    evidence the routing policy tracks something real rather than one person's
    taste.

Routing is framed as a risk score, not a boolean. The product question is not
"is it right" but "how much volume can be auto-handled at an acceptable error
rate", which is a threshold chosen from a cost curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .llm import LLM, Message
from .schema import AUTO_REASONS, ESCALATION_REASONS


@dataclass
class Route:
    escalate: bool
    reason: str
    risk: float          # 0 = safe to auto-send, 1 = must see a human
    rationale: str = ""


class AlwaysAuto:
    """Trivial baseline. Worth stating plainly: 80% of the golden set is
    auto-handled, so 'send everything' is already right four times in five -
    which is exactly why raw accuracy is the wrong metric for this decision."""
    name = "trivial: auto-handle everything"

    def fit(self, *a, **kw):
        return self

    def predict(self, rows) -> list[Route]:
        return [Route(False, "known_answer", 0.0) for _ in rows]


class IntentPriorRouter:
    """Simple baseline: escalate when the predicted intent's historical
    escalation rate clears a threshold. Fitted leave-one-out."""
    name = "simple: intent prior"

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self.prior: dict[str, float] = {}
        self.base = 0.0

    def fit(self, intents, escalated):
        intents, escalated = np.asarray(intents), np.asarray(escalated, dtype=float)
        self.base = float(escalated.mean())
        for c in set(intents.tolist()):
            m = intents == c
            self.prior[c] = float(escalated[m].mean()) if m.sum() else self.base
        return self

    def predict_loo(self, intents, escalated) -> list[Route]:
        """Leave-one-out: each row is scored by a prior fitted without it, so a
        249-row table cannot quietly memorise itself."""
        intents, escalated = np.asarray(intents), np.asarray(escalated, dtype=float)
        out = []
        for i in range(len(intents)):
            m = (intents == intents[i])
            m[i] = False
            p = float(escalated[m].mean()) if m.sum() else self.base
            out.append(Route(p > self.threshold, "needs_account_access" if p > self.threshold
                             else "known_answer", p))
        return out

    def predict(self, intents) -> list[Route]:
        out = []
        for c in np.asarray(intents):
            p = self.prior.get(c, self.base)
            out.append(Route(p > self.threshold, "needs_account_access" if p > self.threshold
                             else "known_answer", p))
        return out


SYS_V1 = """You decide whether an incoming Spotify support tweet can be answered
automatically or must go to a human agent. Decide from the customer's message
alone - that is all a live agent has before replying.

Escalate when a reply cannot be sent safely without a person. Reasons:
{esc}

Auto-handle when a safe, useful reply can be sent now. Reasons:
{auto}

A clarifying question IS a safe automatic reply: not knowing the details is a
reason to ask, not a reason to escalate.

Return JSON only:
{{"escalate":<true|false>,"reason":"<exact name>","risk":<0.0-1.0>,"why":"<max 12 words>"}}
risk is the probability that auto-sending would be wrong or harmful. Use the
full range - it sets the operating point, so a flat 0.5 is useless."""


SYS = """You decide whether an incoming Spotify support tweet can be answered
automatically or must go to a human agent. Decide from the customer's message
alone - that is all a live agent has before replying.

Escalate when a reply cannot be sent safely without a person. Reasons:
{esc}

Auto-handle when a safe, useful reply can be sent now. Reasons:
{auto}

A clarifying question IS a safe automatic reply when the only thing missing is
detail: not knowing which device, which song or which error is a reason to ask,
not a reason to escalate.

But some situations are escalations NO MATTER how much detail is missing, and a
clarifying question does not make them safe. Escalate immediately, without
asking anything first, whenever the message involves:
  - money already taken: a disputed charge, a double charge, a refund
  - account takeover, a stolen account, or a login the customer did not make
  - anyone claiming legal action, press, or a regulator
Asking "which card was it?" when someone has just been overcharged, or "what
happened?" when someone's account has been stolen, is not triage - it is delay
on exactly the cases where delay costs most.

Return JSON only:
{{"escalate":<true|false>,"reason":"<exact name>","risk":<0.0-1.0>,"why":"<max 12 words>"}}
risk is the probability that auto-sending would be wrong or harmful. Use the
full range - it sets the operating point, so a flat 0.5 is useless."""


class LlmRouter:
    """Policy router.

    The prompt carries an explicit carve-out for money, account takeover and
    legal exposure. Version 1 said only that a clarifying question is always a
    safe auto-reply, and the evaluation caught what that cost: 28 of 37 missed
    escalations were routed to `triage_question`, including account takeovers
    and double charges. The model followed the instruction faithfully; the
    instruction was wrong. Both versions are kept so the fix stays measurable.
    """
    name = "system: LLM policy router"

    def __init__(self, llm: LLM, index=None, k: int = 4, prompt_version: int = 2):
        self.llm, self.index, self.k = llm, index, k
        self.prompt_version = prompt_version

    def fit(self, *a, **kw):
        return self

    def _system(self) -> str:
        if self.prompt_version == 1:
            return SYS_V1.format(
                esc="\n".join(f"  {k} - {d}" for k, d in ESCALATION_REASONS),
                auto="\n".join(f"  {k} - {d}" for k, d in AUTO_REASONS))
        return SYS.format(
            esc="\n".join(f"  {k} - {d}" for k, d in ESCALATION_REASONS),
            auto="\n".join(f"  {k} - {d}" for k, d in AUTO_REASONS))

    def predict(self, texts, vectors=None, intents=None, progress=False) -> list[Route]:
        valid_e = {k for k, _ in ESCALATION_REASONS}
        valid_a = {k for k, _ in AUTO_REASONS}
        out = []
        for n, t in enumerate(texts):
            ctx = ""
            if self.index is not None and vectors is not None:
                hits = self.index.search(t, vectors[n], k=self.k)
                ctx = "\nHOW SUPPORT HANDLED SIMILAR MESSAGES:\n" + "\n".join(
                    f"  customer: {h.customer_text[:120]}\n  reply: {h.reply_text[:120]}"
                    for h in hits)
            hint = f"\nPredicted intent: {intents[n]}" if intents is not None else ""
            try:
                r = self.llm.chat([Message("system", self._system()),
                                   Message("user", f"MESSAGE:\n{t}{hint}{ctx}")],
                                  json_mode=True, max_tokens=4000)
                d = r.json()
                esc = bool(d.get("escalate", False))
                reason = str(d.get("reason", ""))
                risk = float(d.get("risk", 0.5))
                why = str(d.get("why", ""))[:80]
            except Exception:
                esc, reason, risk, why = True, "insufficient_info", 1.0, "router error"
            if esc and reason not in valid_e:
                reason = "insufficient_info"
            if not esc and reason not in valid_a:
                reason = "known_answer"
            out.append(Route(esc, reason, max(0.0, min(1.0, risk)), why))
            if progress and (n + 1) % 25 == 0:
                print(f"    {n+1}/{len(texts)}", flush=True)
        return out


def proxy_escalation_signals(golden, pairs) -> np.ndarray:
    """Observable, annotator-independent evidence that a thread needed a human.

    None of these is ground truth. Together they are a second axis: a routing
    policy that tracks them is doing something beyond reproducing one person's
    judgement.
    """
    idx = pairs.set_index("thread_id")
    sig = []
    for t in golden.thread_id:
        if t not in idx.index:
            sig.append(0.0)
            continue
        r = idx.loc[t]
        r = r.iloc[0] if hasattr(r, "iloc") and getattr(r, "ndim", 1) > 1 else r
        score = 0.0
        score += 0.5 if bool(r.is_deflection) else 0.0      # brand moved it off-channel
        score += 0.3 if int(r.n_turns) >= 6 else 0.0        # long back-and-forth
        score += 0.2 if not bool(r.got_thanks) else 0.0     # never resolved visibly
        sig.append(score)
    return np.array(sig)
