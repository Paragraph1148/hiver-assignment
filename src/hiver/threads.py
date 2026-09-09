"""Reconstruct conversations from TWCS's reply graph.

The raw file is a flat tweet table; conversations exist only as
`in_response_to_tweet_id` edges. We rebuild threads by walking each tweet up to
its root with path compression, then keep the threads a given brand took part in.

The unit that matters downstream is the FIRST-CONTACT PAIR: a customer's opening
tweet and the brand's first public reply to it. That is exactly the decision our
agent has to make - message arrives, draft a reply, decide whether to auto-send.

`thread_id` is carried on every row because all splitting is done at thread
level. Splitting on individual pairs would put one turn of a conversation in
train and the next in test, leaking the answer into the retrieval index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .pii import extract_agent, normalise, scrub

RAW_DEFAULT = Path("data/raw/twcs.csv")

DEFLECT = re.compile(
    r"\b(?:dm|d\.m\.|direct message|private message|priv(?:ate)? msg|pm us|"
    r"send us a (?:private )?(?:message|dm)|shoot us a|click .{0,15}message|"
    r"message (?:us|button)|follow (?:us )?(?:and|&) (?:dm|message)|"
    r"follow back|inbox us)\b", re.I)

THANKS = re.compile(
    r"\b(?:thank(?:s| you)|thx|ty so much|appreciate it|that worked|it works now|"
    r"fixed it|sorted now|problem solved|you.?re (?:the best|a lifesaver))\b", re.I)


@dataclass
class Corpus:
    """First-contact pairs plus the thread table they came from."""
    pairs: pd.DataFrame
    threads: pd.DataFrame

    def __repr__(self) -> str:
        return f"Corpus(pairs={len(self.pairs):,}, threads={len(self.threads):,})"


def load_twcs(path: str | Path = RAW_DEFAULT) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text",
                 "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "str", "text": "str"},
    )
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")
    df["text"] = df["text"].fillna("")
    return df


def _root_map(df: pd.DataFrame) -> dict[int, int]:
    """tweet_id -> root tweet_id, iterative with path compression.

    Iterative rather than recursive: some TWCS chains are hundreds deep and
    recursion blows the stack. The visited set guards against the cycles that
    a handful of malformed rows genuinely contain.
    """
    parent: dict[int, int] = {}
    for tid, pid in zip(df["tweet_id"], df["in_response_to_tweet_id"]):
        if pd.notna(pid):
            parent[int(tid)] = int(pid)

    root: dict[int, int] = {}
    for tid in df["tweet_id"]:
        tid = int(tid)
        path, cur, seen = [], tid, set()
        while cur in parent and cur not in root and cur not in seen:
            seen.add(cur)
            path.append(cur)
            cur = parent[cur]
        r = root.get(cur, cur)
        for node in path:
            root[node] = r
        root[tid] = r
    return root


def find_brand_aliases(df: pd.DataFrame, brand: str, top_k: int = 2) -> frozenset[str]:
    """Discover the numeric handle(s) customers use for a brand.

    TWCS pseudonymises the brand's own handle inconsistently: customers write
    both @SpotifyCares and @115888. Rather than hardcode the number per brand,
    find it - the id most often mentioned by customers writing into this brand's
    threads is the brand itself.
    """
    root = _root_map(df)
    tid_to_root = df["tweet_id"].map(root)
    brand_threads = set(tid_to_root[df["author_id"] == brand])
    cust = df[df["inbound"] & tid_to_root.isin(brand_threads)]
    counts = cust["text"].str.extractall(r"@(\d{4,})")[0].value_counts()
    return frozenset([brand, *counts.head(top_k).index.tolist()])


def build_corpus(df: pd.DataFrame, brand: str,
                 brand_aliases: frozenset[str] | None = None) -> Corpus:
    """Extract first-contact pairs for one brand, scrubbed and flagged."""
    if brand_aliases is None:
        brand_aliases = find_brand_aliases(df, brand)
    root = _root_map(df)
    df = df.copy()
    df["thread_id"] = df["tweet_id"].map(root).astype("int64")

    # Keep only threads this brand actually participated in.
    brand_threads = set(df.loc[df["author_id"] == brand, "thread_id"])
    conv = df[df["thread_id"].isin(brand_threads)].copy()

    # created_at is a Twitter-format string; needed for a time-ordered split.
    conv["ts"] = pd.to_datetime(conv["created_at"], format="%a %b %d %H:%M:%S %z %Y",
                                errors="coerce")
    conv = conv.sort_values(["thread_id", "ts"], kind="stable")
    conv["turn"] = conv.groupby("thread_id").cumcount()

    thread_len = conv.groupby("thread_id").size().rename("n_turns")

    # A customer tweet that a brand tweet replies to, where the customer tweet
    # opens the thread. That is first contact.
    by_id = conv.set_index("tweet_id")
    replies = conv[(conv["author_id"] == brand) & conv["in_response_to_tweet_id"].notna()].copy()
    replies["parent"] = replies["in_response_to_tweet_id"].astype("int64")
    replies = replies[replies["parent"].isin(by_id.index)]

    parent_rows = by_id.loc[replies["parent"]]
    keep = (parent_rows["inbound"].to_numpy()) & (parent_rows["turn"].to_numpy() == 0)
    replies = replies[keep]
    parent_rows = parent_rows[keep]

    # Resolution proxy: the customer thanks the brand later in the same thread.
    thanked_threads = set(
        conv.loc[conv["inbound"] & conv["text"].str.contains(THANKS, regex=True), "thread_id"]
    )

    cust_raw = parent_rows["text"].to_numpy()
    rep_raw = replies["text"].to_numpy()

    rows = []
    for i in range(len(replies)):
        cust_s = scrub(normalise(str(cust_raw[i])), brand_aliases)
        rep_s = scrub(normalise(str(rep_raw[i])), brand_aliases)
        body, agent = extract_agent(rep_s.text)
        rows.append((cust_s.text, body, agent,
                     sum(cust_s.counts.values()), cust_s.had_pii))

    out = pd.DataFrame(rows, columns=["customer_text", "reply_text", "agent",
                                      "cust_pii_n", "cust_has_pii"])
    out["thread_id"] = replies["thread_id"].to_numpy()
    out["customer_tweet_id"] = parent_rows.index.to_numpy()
    out["reply_tweet_id"] = replies["tweet_id"].to_numpy()
    out["ts"] = parent_rows["ts"].to_numpy()
    out["n_turns"] = out["thread_id"].map(thread_len).astype("int64")
    out["is_deflection"] = out["reply_text"].str.contains(DEFLECT, regex=True)
    out["got_thanks"] = out["thread_id"].isin(thanked_threads)
    out["cust_len"] = out["customer_text"].str.len()
    out["reply_len"] = out["reply_text"].str.len()

    # One pair per thread: the brand's FIRST reply to the opening tweet.
    out = out.sort_values(["thread_id", "reply_tweet_id"], kind="stable")
    out = out.drop_duplicates("thread_id", keep="first").reset_index(drop=True)

    threads = conv[["thread_id", "tweet_id", "author_id", "inbound", "turn", "ts", "text"]]
    return Corpus(pairs=out, threads=threads.reset_index(drop=True))
