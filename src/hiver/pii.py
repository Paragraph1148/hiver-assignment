"""Scrub personal data out of tweets before it reaches an index or a prompt.

TWCS already pseudonymises author handles (@115712), but the tweet *bodies* are
untouched and customers paste real things into them: emails, phone numbers,
order and booking references, card-shaped digits. Two reasons this matters here
and is not compliance theatre:

  1. Retrieved precedents are pasted into a generation prompt. Without scrubbing,
     one customer's order number can surface in a reply drafted for someone else.
  2. Verbatim identifiers are exactly the kind of token a retriever latches onto,
     so scrubbing also removes a spurious match signal.

Everything is replaced with a typed placeholder rather than deleted, so the model
still sees that an order number was mentioned - which is a real escalation cue.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters. URLs, emails and handles are consumed first, so their digits
# cannot be mistaken for phone numbers by the looser numeric rules below.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("url", re.compile(r"https?://\S+|\bwww\.\S+", re.I), "<URL>"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), "<EMAIL>"),
    # TWCS handles are already pseudonymous IDs; normalising them stops the
    # retriever keying on one specific customer's number. Must precede the
    # numeric rules, which would otherwise eat the digits as a phone number.
    ("handle", re.compile(r"@\d{4,}"), "@<USER>"),
    # Card-shaped: 13-16 digits, optionally split by spaces or dashes. Anchored
    # to end on a digit so the match cannot swallow the following separator.
    ("card", re.compile(r"\b\d(?:[ -]?\d){12,15}\b"), "<CARD>"),
    # Phone: optional +CC and area code, then at least two further digit groups.
    # Requiring the second group stops bare years and quantities ("2017", "100")
    # being masked. Runs after <CARD> so long runs are typed as cards.
    ("phone", re.compile(r"(?<![\w<])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?"
                         r"\d{3,5}(?:[ .-]?\d{3,4}){1,2}(?![\w>])"), "<PHONE>"),
    ("order", re.compile(r"\b(?:order|booking|reference|ref|confirmation|conf|case|ticket)"
                         r"\s*(?:no\.?|number|#|id|:)?\s*[:#]?\s*"
                         r"([A-Z0-9]{2,}-?[A-Z0-9-]{3,})\b", re.I), "<ORDER>"),
]

# Spotify agents sign replies "/JP", "^KC", "-DF". We strip these from drafting
# targets: an agent's initials are not something the model should learn to
# imitate, and they leak identity. Extracted separately for the inter-agent
# disagreement analysis before being removed.
# 1-3 uppercase letters after a slash/caret/tilde/dash, at the very end - the
# trailing <URL> placeholder is allowed after it, because Spotify routinely
# appends a help link below the signature. Uppercase-only avoids eating the
# tail of ordinary words ("and/or").
AGENT_SIG = re.compile(r"\s*[/^~\-]\s*([A-Z]{1,3})\s*(?:<URL>\s*)*$")


@dataclass
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    # Categories that are not personal data: a link, an already-pseudonymous
    # handle, and a mention of the brand itself.
    _NOT_PII = ("url", "handle", "brand")

    @property
    def had_pii(self) -> bool:
        return any(v for k, v in self.counts.items() if k not in self._NOT_PII)


def scrub(text: str, brand_aliases: frozenset[str] = frozenset()) -> ScrubResult:
    """Replace identifiers with typed placeholders, reporting what was found.

    `brand_aliases` are the handles that refer to the brand itself. TWCS gives a
    brand both a name and a numeric id (SpotifyCares is also @115888), and the
    numeric form would otherwise be masked as an anonymous customer - losing the
    fact that the message is addressed to support at all.
    """
    counts: dict[str, int] = {}
    out = text or ""
    for alias in brand_aliases:
        pat = re.compile(rf"@{re.escape(alias)}\b", re.I)
        out, n = pat.subn("@BRAND", out)
        if n:
            counts["brand"] = counts.get("brand", 0) + n
    for name, pat, repl in _RULES:
        if name == "order":
            # Keep the cue word, mask only the identifier it introduces.
            out, n = pat.subn(lambda m: m.group(0).replace(m.group(1), "<ORDER>"), out)
        else:
            out, n = pat.subn(repl, out)
        if n:
            counts[name] = n
    return ScrubResult(out, counts)


def extract_agent(text: str) -> tuple[str, str | None]:
    """Split a trailing agent signature off a brand reply."""
    m = AGENT_SIG.search(text or "")
    if not m:
        return text, None
    return AGENT_SIG.sub("", text).rstrip(), m.group(1)


def normalise(text: str) -> str:
    """Whitespace and entity cleanup applied to every tweet."""
    t = (text or "").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = t.replace("&quot;", '"').replace("&#39;", "'")
    return " ".join(t.split()).strip()
