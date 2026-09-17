"""Re-map the corpus's weak labels onto the FINAL 10-class taxonomy.

The weak labels were induced before hand-labelling revealed that
`feature_request` is a real class. Leaving them stale would hand every trained
baseline a structural blind spot on 10% of the golden set - it could not emit a
label it had never seen - and the resulting comparison would say more about the
taxonomy's history than about the models.

This asks the LLM to place each already-described cluster into the final
taxonomy. Two calls, cached. The cluster descriptions are reused as-is; nothing
is re-clustered, so this cannot quietly change the corpus structure.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from hiver.cluster import cluster       # noqa: E402
from hiver.embed import embed           # noqa: E402
from hiver.llm import LLM               # noqa: E402
from hiver.schema import INTENTS        # noqa: E402

SYS = """Assign each cluster of Spotify support messages to exactly one intent
from the taxonomy. Use the cluster's own description.

{tax}

Return JSON only: {{"assign":[{{"id":<int>,"intent":"<exact name from the list>"}}]}}
Every cluster id given must appear exactly once."""


def main() -> int:
    tax = "\n".join(f"  {k} - {d}" for k, d in INTENTS)
    cand = json.loads(Path("data/interim/candidate_taxonomy.json").read_text())
    lt = json.loads(Path("data/interim/longtail_mapping.json").read_text())
    llm = LLM(provider="groq")
    valid = {k for k, _ in INTENTS}

    def assign(described, tag):
        block = "\n".join(
            f"cluster {c['id']} (n={c.get('size', 0)}): {c['name']} - {c['description']}"
            for c in described)
        r = llm.ask(block, system=SYS.format(tax=tax), json_mode=True, max_tokens=8000)
        out = {}
        for a in r.json()["assign"]:
            out[int(a["id"])] = a["intent"] if a["intent"] in valid else "other"
        print(f"  {tag}: {len(out)}/{len(described)} clusters assigned")
        return out

    main_map = assign(cand["clusters"], "main clusters")
    sub_src = [{"id": m["id"], "name": m["name"], "size": m["size"],
                "description": m.get("note", m["name"])} for m in lt["mapping"]]
    sub_map = assign(sub_src, "long-tail sub-clusters")

    pairs = pd.read_parquet("data/interim/SpotifyCares_labelled.parquet")
    texts = pairs.customer_text.tolist()
    vecs = embed(texts, show_progress=False)
    labels = np.load("data/interim/labels_mcs100.npy")
    sub = np.load("data/interim/longtail_idx.npy")
    red = np.load("data/interim/longtail_umap.npy")

    new = np.array(["other"] * len(texts), dtype=object)
    for cid, intent in main_map.items():
        if cid != 15:                      # the grab-bag is resolved by the long-tail pass
            new[labels == cid] = intent
    res = cluster(vecs[sub], min_cluster_size=50, reduced=red)
    for scid, intent in sub_map.items():
        new[sub[res.labels == scid]] = intent

    before = pairs.intent_weak.copy()
    pairs["intent_weak"] = new
    pairs.to_parquet("data/interim/SpotifyCares_labelled.parquet", index=False)

    print(f"\n  labels changed: {int((before != new).sum()):,} of {len(new):,} "
          f"({(before != new).mean()*100:.1f}%)")
    print(f"\n{'intent':<28}{'n':>7}{'share':>8}")
    for name, n in pd.Series(new).value_counts().items():
        print(f"  {name:<26}{n:>7}{n/len(new)*100:>7.1f}%")
    assert "feature_request" in set(new), "feature_request still absent from weak labels"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
