"""Measure what a semantic cache would actually buy, and what it would cost.

Two populations, deliberately, because one number cannot answer both questions:

  HIT RATE is a property of real inbox traffic, so it is measured on a
  chronological stream of the corpus. The golden set is stratified to make rare
  intents measurable, which spreads it out and would badly understate how
  repetitive a real queue is.

  ACCURACY COST needs true labels, so it is measured on the golden set, where a
  cache hit serves an earlier message's prediction and is scored against this
  message's human label.

Reporting a saving from one population and a quality cost from the other, each
labelled, beats blending them into one flattering figure.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from hiver.embed import embed                      # noqa: E402
from hiver.semantic_cache import SemanticCache, simulate  # noqa: E402

THRESHOLDS = [0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80]


def main() -> int:
    # --- hit rate on realistic traffic ------------------------------------
    corpus = pd.read_parquet("data/interim/SpotifyCares_labelled.parquet")
    held = set(json.loads(Path("data/golden/held_out_threads.json").read_text())["thread_ids"])
    stream = corpus[~corpus.thread_id.isin(held)].sort_values("ts").head(10000)
    svecs = embed(stream.customer_text.tolist(), show_progress=False)
    labels = stream.intent_weak.tolist()

    print("HIT RATE on 10,000 messages in arrival order (realistic queue)\n")
    print(f"  {'cosine':>7}{'hit rate':>11}{'calls saved':>13}{'cache size':>12}"
          f"{'wrong intent':>14}")
    hit_rows = {}
    for t in THRESHOLDS:
        c = SemanticCache(threshold=t, max_entries=20000)
        mismatch = 0
        for n, v in enumerate(svecs):
            # Payload must be non-None: get() signals a miss with None, so
            # storing None makes every hit look like a miss and the cache grows
            # without bound, overstating the hit rate.
            borrowed, _ = c.get(v)
            if borrowed is None:
                c.put(v, n)
            elif labels[borrowed] != labels[n]:
                # The served answer was written for a message of a different
                # intent. Weak labels, so this is a proxy - but it is the error
                # the cache introduces, measured at realistic hit rates rather
                # than on the stratified golden set where hits barely occur.
                mismatch += 1
        hit_rows[t] = {"hit_rate": c.hit_rate, "hits": c.hits,
                       "mismatch": mismatch / c.hits if c.hits else 0.0}
        print(f"  {t:>7.2f}{c.hit_rate*100:>10.1f}%{c.hits:>13,}{len(c.entries):>12,}"
              f"{(mismatch/c.hits*100 if c.hits else 0):>13.1f}%")

    # --- accuracy cost on labelled data -----------------------------------
    g = pd.read_parquet("data/golden/golden.parquet")
    g = g[g.complete].reset_index(drop=True)
    preds = pd.read_parquet("data/results/intent_preds.parquet").set_index("item_id")
    gvecs = embed(g.customer_text.tolist(), show_progress=False)
    pred = [preds.llm.get(i, "other") for i in g.item_id]
    gold = g.intent.tolist()

    print("\nACCURACY COST on the golden set (true labels, stratified so hit rate is low)\n")
    print(f"  {'cosine':>7}{'hit rate':>11}{'accuracy':>11}{'vs live':>10}")
    rows = []
    for t in THRESHOLDS:
        r = simulate(gvecs, pred, gold, threshold=t)
        rows.append(r)
        print(f"  {t:>7.2f}{r['hit_rate']*100:>10.1f}%{r['accuracy']*100:>10.1f}%"
              f"{-r['cost_of_cache_pts']:>+9.1f}")
    live = rows[0]["accuracy_no_cache"]
    print(f"\n  live (no cache): {live*100:.1f}%")

    out = {"hit_rate_on_traffic": hit_rows, "golden_simulation": rows,
           "stream_n": len(stream)}
    Path("data/results/semantic_cache.json").write_text(json.dumps(out, indent=2))
    print("\nwrote data/results/semantic_cache.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
