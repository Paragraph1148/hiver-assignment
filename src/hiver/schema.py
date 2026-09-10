"""The label schema. Single source of truth for the pre-filler, the UI and eval.

Escalation is defined as a decision made from the CUSTOMER MESSAGE ALONE, which
is the information a live agent has before replying. Each reason names a
different underlying cause, because the mitigations differ: a message needing
account access can never be auto-handled, whereas one needing clarification can
be auto-handled with a triage question.
"""
from __future__ import annotations

INTENTS: list[tuple[str, str]] = [
    ("billing_and_subscription", "Payment, charges, plan changes, refunds, student/family pricing"),
    ("account_access",           "Login, password, recovery, deletion, linked accounts, security"),
    ("playback_and_app_issues",  "App crashes, songs won't play, sync, device/connectivity problems"),
    ("content_unavailability",   "Missing tracks/albums/artists, removed content, add-to-catalogue requests"),
    ("playlist_and_discovery",   "Playlist tooling, recommendations, podcasts, discovery features"),
    ("regional_and_verification", "Country availability, launch requests, student/identity verification"),
    ("non_actionable_generic",   "No problem described - just 'help' or a bare link"),
    ("ads_issues",               "Ad frequency, ad content, ads despite Premium"),
    ("other",                    "A real request that fits none of the above"),
]

ESCALATION_REASONS: list[tuple[str, str]] = [
    ("needs_account_access", "Cannot be resolved without looking inside the account"),
    ("payment_or_refund",    "Requires moving money or changing a subscription"),
    ("security_or_fraud",    "Suspected compromise, unauthorised access or fraud"),
    ("insufficient_info",    "Not answerable until the customer supplies more detail"),
    ("angry_or_vulnerable",  "Tone or circumstances warrant a human touch"),
    ("policy_or_legal",      "Legal threat, press, formal complaint or policy exception"),
    ("out_of_scope",         "Not a support request Spotify can act on"),
]

AUTO_REASONS: list[tuple[str, str]] = [
    ("known_answer",      "A documented answer or standard fix applies"),
    ("triage_question",   "Correct reply is a clarifying question, safe to send"),
    ("acknowledge_only",  "Feedback or praise; a courteous acknowledgement suffices"),
]

INTENT_NAMES = [k for k, _ in INTENTS]
ESCALATION_REASON_NAMES = [k for k, _ in ESCALATION_REASONS]
AUTO_REASON_NAMES = [k for k, _ in AUTO_REASONS]


def taxonomy_prompt() -> str:
    lines = ["INTENTS:"]
    lines += [f"  {k} - {d}" for k, d in INTENTS]
    lines += ["", "IF ESCALATING, pick one reason:"]
    lines += [f"  {k} - {d}" for k, d in ESCALATION_REASONS]
    lines += ["", "IF AUTO-HANDLING, pick one reason:"]
    lines += [f"  {k} - {d}" for k, d in AUTO_REASONS]
    return "\n".join(lines)
