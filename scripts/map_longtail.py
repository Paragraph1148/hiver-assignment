"""Ask whether the 38% long tail contains NEW intents, or just atypical
expressions of the ones we already have.

This distinction decides the whole evaluation design. If the long tail is new
intents, the taxonomy is incomplete. If it is the same intents phrased
unprototypically, the taxonomy is fine but accuracy will be much worse there
than on the dense core - and reporting a single blended number would hide that.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

from hiver.cluster import cluster, exemplars   # noqa: E402
from hiver.embed import embed                  # noqa: E402
from hiver.llm import LLM                      # noqa: E402

SYS = """You are auditing an intent taxonomy for a Spotify support agent.

You are given an EXISTING taxonomy, then clusters of messages that the main
clustering pass could not place. For each cluster decide whether it belongs to
an existing intent, or is genuinely a new intent the taxonomy is missing.

Return JSON only:
{"mapping":[{"id":<int>,"name":"<snake_case>","assign_to":"<existing intent name or NEW>",
"confidence":"high|medium|low","note":"<short justification>"}]}

Assign to NEW only if no existing intent would produce the same support action.
Being phrased unusually is NOT grounds for a new intent."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--min-cluster-size", type=int, default=50)
    args = ap.parse_args()

    tax = json.loads(Path("data/interim/candidate_taxonomy.json").read_text())
    existing = [i["name"] for i in tax["intents"] if i["name"] != "other"]

    pairs = pd.read_parquet("data/interim/SpotifyCares_pairs.parquet")
    texts = pairs.customer_text.tolist()
    vecs = embed(texts, show_progress=False)
    sub = np.load("data/interim/longtail_idx.npy")
    red = np.load("data/interim/longtail_umap.npy")
    sub_texts = [texts[i] for i in sub]

    res = cluster(vecs[sub], min_cluster_size=args.min_cluster_size, reduced=red)
    sizes = res.sizes()
    N = len(texts)

    llm = LLM(provider=args.provider)
    header = ("EXISTING INTENTS:\n" +
              "\n".join(f"  - {i['name']}: {i['description']}"
                        for i in tax["intents"] if i["name"] != "other"))

    mapping: list[dict] = []
    ids = list(sizes)
    for i in range(0, len(ids), 8):
        blocks = []
        for cid in ids[i:i + 8]:
            ex = exemplars(sub_texts, vecs[sub], res.labels, cid, k=8)
            joined = "\n".join(f"    - {e[:170]}" for e in ex)
            blocks.append(f"CLUSTER {cid} (n={sizes[cid]}):\n{joined}")
        r = llm.ask(header + "\n\n" + "\n\n".join(blocks), system=SYS,
                    json_mode=True, max_tokens=8000)
        mapping.extend(r.json()["mapping"])
        print(f"  mapped {len(mapping)}/{len(ids)}", flush=True)

    for m in mapping:
        m["size"] = sizes.get(m["id"], 0)

    clustered = sum(sizes.values())
    still_noise = int((res.labels < 0).sum())
    to_new = sum(m["size"] for m in mapping if m["assign_to"] == "NEW")
    to_exist = clustered - to_new

    print(f"\n--- long tail = {len(sub):,} messages ({len(sub)/N*100:.1f}% of corpus)")
    print(f"  sub-clustered            : {clustered:,} ({clustered/N*100:.1f}% of corpus)")
    print(f"  -> existing intents      : {to_exist:,} ({to_exist/N*100:.1f}%)")
    print(f"  -> genuinely new intents : {to_new:,} ({to_new/N*100:.1f}%)")
    print(f"  irreducible noise        : {still_noise:,} ({still_noise/N*100:.1f}%)")

    by_parent: dict[str, int] = {}
    for m in mapping:
        by_parent[m["assign_to"]] = by_parent.get(m["assign_to"], 0) + m["size"]
    print("\n  long tail absorbed by:")
    for k, v in sorted(by_parent.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<30} {v:>6,}  ({v/N*100:.2f}% of corpus)")

    if news := [m for m in mapping if m["assign_to"] == "NEW"]:
        print("\n  proposed NEW intents:")
        for m in sorted(news, key=lambda x: -x["size"]):
            print(f"    {m['name']:<30} n={m['size']:>5} ({m['size']/N*100:.2f}%)  {m['note'][:50]}")

    Path("data/interim/longtail_mapping.json").write_text(json.dumps(
        {"mapping": mapping, "still_noise": still_noise, "n_messages": N}, indent=2))
    print("\nwrote data/interim/longtail_mapping.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
