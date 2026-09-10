"""Freeze the taxonomy and attach a WEAK intent label to every message.

These labels come from clustering plus LLM cluster-naming. They are NOT ground
truth - they exist to stratify the sample a human then hand-labels, and to give
the retriever a coarse filter. The golden set is the only ground truth, and it
is built by a human from scratch.

Also records `region` per message - core (densely clustered) vs tail (atypical
phrasing of a known intent) vs other (irreducibly heterogeneous). Every metric
is reported split by region, because a single blended accuracy hides that the
tail is much harder than the core.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from hiver.cluster import cluster         # noqa: E402
from hiver.embed import embed             # noqa: E402

# Small intents folded into their nearest operational parent: each would
# otherwise land ~10 golden examples, where per-class F1 is pure noise.
MERGE = {
    "podcasts": "playlist_and_discovery",
    "event_and_presale": "content_unavailability",
    "artist_account_access": "account_access",
}


def main() -> int:
    tax = json.loads(Path("data/interim/candidate_taxonomy.json").read_text())
    lt = json.loads(Path("data/interim/longtail_mapping.json").read_text())

    pairs = pd.read_parquet("data/interim/SpotifyCares_pairs.parquet")
    texts = pairs.customer_text.tolist()
    vecs = embed(texts, show_progress=False)
    labels = np.load("data/interim/labels_mcs100.npy")
    sub = np.load("data/interim/longtail_idx.npy")
    red = np.load("data/interim/longtail_umap.npy")

    def canon(name: str) -> str:
        return MERGE.get(name, name)

    # Main pass: cluster id -> intent.
    cid_to_intent: dict[int, str] = {}
    for i in tax["intents"]:
        for cid in i.get("merged_from", []):
            if isinstance(cid, int):
                cid_to_intent[cid] = canon(i["name"])

    intent = np.array(["other"] * len(texts), dtype=object)
    region = np.array(["other"] * len(texts), dtype=object)
    for cid, name in cid_to_intent.items():
        if cid == 15:      # the grab-bag: resolved by the long-tail pass instead
            continue
        m = labels == cid
        intent[m] = name
        region[m] = "core"

    # Long-tail pass: sub-cluster id -> intent.
    res = cluster(vecs[sub], min_cluster_size=50, reduced=red)
    sub_map = {m["id"]: m for m in lt["mapping"]}
    for scid, m in sub_map.items():
        target = m["assign_to"]
        target = canon(target if target != "NEW" else m["name"])
        if target not in {canon(i["name"]) for i in tax["intents"]}:
            target = "account_access" if "artist" in m["name"] else "other"
        idx = sub[res.labels == scid]
        intent[idx] = target
        region[idx] = "tail"

    pairs["intent_weak"] = intent
    pairs["region"] = region
    pairs.to_parquet("data/interim/SpotifyCares_labelled.parquet", index=False)

    N = len(pairs)
    print(f"{'intent':<28}{'n':>7}{'share':>8}   core / tail / other")
    g = pairs.groupby("intent_weak")
    for name, grp in sorted(g, key=lambda kv: -len(kv[1])):
        r = grp.region.value_counts()
        print(f"  {name:<26}{len(grp):>7}{len(grp)/N*100:>7.1f}%   "
              f"{r.get('core',0):>5} / {r.get('tail',0):>5} / {r.get('other',0):>5}")

    print(f"\n{'region':<12}{'n':>8}{'share':>8}")
    for name, grp in sorted(pairs.groupby("region"), key=lambda kv: -len(kv[1])):
        print(f"  {name:<10}{len(grp):>8}{len(grp)/N*100:>7.1f}%")

    Path("data/taxonomy.json").write_text(json.dumps({
        "brand": "SpotifyCares",
        "intents": sorted(set(intent.tolist())),
        "merged": MERGE,
        "n_messages": N,
        "shares": {k: round(v / N * 100, 2)
                   for k, v in pairs.intent_weak.value_counts().items()},
        "region_shares": {k: round(v / N * 100, 2)
                          for k, v in pairs.region.value_counts().items()},
    }, indent=2))
    print("\nwrote data/taxonomy.json and data/interim/SpotifyCares_labelled.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
