"""Turn clusters into a CANDIDATE intent taxonomy.

Two passes, because they are different jobs:
  1. describe each cluster from its exemplars (batched, so the model sees
     neighbouring clusters and names them consistently)
  2. propose a merged taxonomy over all cluster descriptions

The output is a proposal for a human to edit, not a taxonomy. Clusters are
geometry; intents are a product decision.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from hiver.cluster import cluster, exemplars  # noqa: E402
from hiver.embed import embed                 # noqa: E402
from hiver.llm import LLM                     # noqa: E402

DESCRIBE_SYS = """You are analysing clusters of real customer-support tweets sent to Spotify.
For each cluster you are given messages closest to its centre.

Return JSON only: {"clusters":[{"id":<int>,"name":"<snake_case>","description":"<one sentence>",
"actionable":<bool>,"why_not_actionable":"<empty string if actionable>"}]}

"actionable" means: a support agent could give a useful reply from THIS MESSAGE ALONE,
without first asking the customer a clarifying question. A message that only says
"help me" or contains only a link is NOT actionable.
Name clusters consistently; use the same vocabulary across clusters."""

MERGE_SYS = """You are designing the intent taxonomy for an automated Spotify support agent.

You are given clusters discovered from real data, with sizes. Merge them into 8-12
intents that are:
  - mutually exclusive and collectively exhaustive over this data
  - operationally distinct: two intents should differ in what support DOES, not just wording
  - sized sensibly; avoid an intent covering under 2% of volume unless it is high-risk

Return JSON only:
{"intents":[{"name":"<snake_case>","description":"<one sentence>",
"merged_from":[<cluster ids>],"est_share_pct":<float>,
"typical_action":"<what support does>","risk":"low|medium|high",
"risk_reason":"<why>"}]}

"risk" is the harm if an automated reply were sent without human review.

Two constraints, both load-bearing:
  * Assign every cluster id to EXACTLY ONE intent. No cluster may appear twice
    and none may be dropped.
  * The unclustered NOISE bucket is a genuine long tail of specific but
    individually rare problems. It is NOT a bucket of vague messages, and it
    must become its own `other` intent rather than being folded into a
    content-bearing one. Do not list cluster ids for it."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cluster-size", type=int, default=100)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--out", default="data/interim/candidate_taxonomy.json")
    args = ap.parse_args()

    pairs = pd.read_parquet("data/interim/SpotifyCares_pairs.parquet")
    texts = pairs.customer_text.tolist()
    vecs = embed(texts, show_progress=False)
    red = np.load("data/interim/umap_seed0.npy")
    res = cluster(vecs, min_cluster_size=args.min_cluster_size, reduced=red)
    sizes = res.sizes()
    print(f"{res.n_clusters} clusters, {res.noise_frac*100:.1f}% noise", flush=True)

    llm = LLM(provider=args.provider)
    print(f"labelling via {llm.provider_name}/{llm.model}", flush=True)

    # Pass 1 - describe, in batches so naming stays consistent across neighbours.
    described: list[dict] = []
    ids = list(sizes)
    for i in range(0, len(ids), 6):
        batch = ids[i:i + 6]
        blocks = []
        for cid in batch:
            ex = exemplars(texts, vecs, res.labels, cid, k=10)
            joined = "\n".join(f"  - {e[:200]}" for e in ex)
            blocks.append(f"CLUSTER {cid} (n={sizes[cid]}):\n{joined}")
        r = llm.ask("\n\n".join(blocks), system=DESCRIBE_SYS,
                    json_mode=True, max_tokens=2000)
        got = r.json()["clusters"]
        described.extend(got)
        print(f"  described {len(described)}/{len(ids)}", flush=True)

    for d in described:
        d["size"] = sizes.get(d["id"], 0)

    # Pass 2 - propose a merged taxonomy over everything at once.
    summary = "\n".join(
        f"cluster {d['id']} (n={d['size']}, {d['size']/len(texts)*100:.1f}%): "
        f"{d['name']} - {d['description']}"
        f"{'' if d.get('actionable') else '  [NOT ACTIONABLE ALONE]'}"
        for d in sorted(described, key=lambda x: -x["size"])
    )
    noise_n = int((res.labels < 0).sum())
    summary += f"\n\nUNCLUSTERED NOISE: n={noise_n} ({noise_n/len(texts)*100:.1f}%)"

    r = llm.ask(summary, system=MERGE_SYS, json_mode=True, max_tokens=16000)
    taxonomy = r.json()["intents"]

    # A taxonomy that is not a partition silently corrupts every downstream
    # stage, so this is checked rather than assumed. The first LLM proposal
    # double-assigned four clusters and dropped a fifth.
    seen: dict[int, list[str]] = {}
    for t in taxonomy:
        for cid in t.get("merged_from", []):
            if isinstance(cid, int):
                seen.setdefault(cid, []).append(t["name"])
    dupes = {c: names for c, names in seen.items() if len(names) > 1}
    missing = sorted(set(sizes) - set(seen))
    if dupes or missing:
        print("\n!! taxonomy is not a partition")
        for c, names in dupes.items():
            print(f"   cluster {c} (n={sizes[c]}) claimed by {names}")
        for c in missing:
            print(f"   cluster {c} (n={sizes[c]}) unassigned")
    else:
        print("\npartition check: OK - every cluster assigned exactly once")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"clusters": described, "intents": taxonomy,
         "min_cluster_size": args.min_cluster_size,
         "noise_frac": res.noise_frac, "n_messages": len(texts)}, indent=2))
    print(f"\nwrote {out}")

    print(f"\n{'intent':<30} {'share':>7} {'risk':>7}  action")
    for t in sorted(taxonomy, key=lambda x: -x.get("est_share_pct", 0)):
        print(f"  {t['name']:<28} {t.get('est_share_pct',0):>6.1f}% "
              f"{t.get('risk','?'):>7}  {t.get('typical_action','')[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
