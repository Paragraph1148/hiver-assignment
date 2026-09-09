"""Rank brands by whether GROUNDED REPLY DRAFTING is even possible for them.

A brand is only usable if its PUBLIC replies contain real support content.
Three ways a brand can be unusable, all of which look like "a reply" in the raw data:
  1. channel deflection  -> "DM us" / "send us a private message"
  2. link punting        -> "reach out here: <url>" with no actual content
  3. pure acknowledgement-> "Sorry to hear that!" and nothing else

We also measure whether anything ever gets RESOLVED in public, via the customer
thanking the brand after a reply. That is the only observable success signal in
this dataset, and it is what makes a brand's history worth grounding on.
"""
import re
import pandas as pd

RAW = "/home/user/hiver-assignment/data/raw/twcs.csv"
URL = re.compile(r"https?://\S+")

DEFLECT = re.compile(
    r"\b(dm|d\.m\.|direct message|private message|priv(?:ate)? msg|pm us|"
    r"send us a (?:private )?(?:message|dm)|shoot us a|click .{0,15}message|"
    r"message (?:us|button)|follow (?:us )?(?:and|&) (?:dm|message)|"
    r"follow back|inbox us)\b", re.I)

DIAGNOSTIC = re.compile(
    r"(what (?:version|happens|error|device|browser|model)|which (?:version|device|model|browser)|"
    r"have you tried|can you (?:try|confirm|check|tell us more)|are you (?:able|seeing|getting)|"
    r"how long|when did|does (?:this|it) happen|let us know (?:what|if|which))", re.I)

THANKS = re.compile(
    r"\b(thank(?:s| you)|thx|ty so much|appreciate it|that worked|it works now|"
    r"fixed it|sorted now|problem solved|you.?re (?:the best|a lifesaver))\b", re.I)

def main():
    print("loading twcs.csv ...", flush=True)
    df = pd.read_csv(RAW,
        usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "str", "text": "str"})
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")
    df["text"] = df["text"].fillna("")
    print(f"  {len(df):,} tweets", flush=True)

    inbound_ids = set(df.loc[df["inbound"], "tweet_id"].tolist())

    rep = df[(~df["inbound"]) & df["in_response_to_tweet_id"].notna()].copy()
    rep["parent"] = rep["in_response_to_tweet_id"].astype("int64")
    rep = rep[rep["parent"].isin(inbound_ids)]
    print(f"  {len(rep):,} brand replies to customer tweets", flush=True)

    # Strip URLs before measuring content length: a reply that is 90% link is not 90% content.
    body = rep["text"].str.replace(URL, "", regex=True).str.strip()
    rep["body_len"] = body.str.len()
    rep["has_url"] = rep["text"].str.contains(URL)
    rep["is_deflect"] = rep["text"].str.contains(DEFLECT)
    rep["is_diag"] = rep["text"].str.contains(DIAGNOSTIC)
    # Link punt: sends a URL but carries almost no content of its own.
    rep["is_linkpunt"] = rep["has_url"] & (rep["body_len"] < 70)
    # Thin ack: no link, no question, barely any text.
    rep["is_thin"] = (~rep["has_url"]) & (rep["body_len"] < 60)
    rep["groundable"] = ~(rep["is_deflect"] | rep["is_linkpunt"] | rep["is_thin"])

    # Resolution signal: did the customer come back and thank the brand?
    thanks_parents = df[df["inbound"] & df["text"].str.contains(THANKS)
                        & df["in_response_to_tweet_id"].notna()]
    thanked = set(thanks_parents["in_response_to_tweet_id"].astype("int64").tolist())
    rep["got_thanks"] = rep["tweet_id"].isin(thanked)

    g = rep.groupby("author_id")
    s = pd.DataFrame({
        "replies": g.size(),
        "deflect%": g["is_deflect"].mean() * 100,
        "linkpunt%": g["is_linkpunt"].mean() * 100,
        "thin%": g["is_thin"].mean() * 100,
        "diag%": g["is_diag"].mean() * 100,
        "thanked%": g["got_thanks"].mean() * 100,
        "groundable%": g["groundable"].mean() * 100,
    })
    s = s[s["replies"] >= 5000]
    s["groundable_n"] = (s["replies"] * s["groundable%"] / 100).astype(int)
    s = s.sort_values("groundable_n", ascending=False).round(1)

    pd.set_option("display.width", 220)
    print("\n=== BRAND VIABILITY FOR GROUNDED REPLY DRAFTING ===")
    print("(deflect/linkpunt/thin = ways a reply carries no support content)\n")
    print(s.head(22).to_string())
    s.to_csv("/home/user/hiver-assignment/data/brand_survey.csv")
    print("\nwrote data/brand_survey.csv")

if __name__ == "__main__":
    main()
