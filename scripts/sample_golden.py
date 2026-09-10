"""Draw the stratified sample a human will hand-label.

Design decisions that shape what this set can measure:

  * Stratified by intent x region, not uniform. A uniform draw matches
    production but leaves rare intents with ~5 examples, where per-class F1 is
    noise. We over-sample thin cells and store `weight` so every metric can be
    re-weighted back to true frequency. Both numbers get reported.

  * Region is a stratum in its own right. The tail (atypical phrasing) is
    predicted to be much harder than the core; that is only testable if the
    sample deliberately contains enough of it.

  * The brand's actual reply is sampled but NOT shown during labelling. The
    labeller decides intent and escalation from the customer message alone,
    which is the information the live agent would have. Showing the reply would
    anchor the label to whatever the human agent happened to do.

  * thread_ids are recorded so these threads can be excluded from the retrieval
    index. Without that the evaluation retrieves its own answers.

  * A re-label subset is marked for intra-annotator agreement: the same items
    are re-presented later, shuffled, to measure how self-consistent the single
    annotator is. That number bounds every other number in the report.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

MIN_PER_CELL = 8        # floor so no stratum is unmeasurable
RELABEL_N = 40


def allocate(counts: pd.Series, total: int, min_per_cell: int) -> dict:
    """Proportional allocation with a floor, capped by what each cell holds."""
    cells = list(counts.index)
    alloc = {c: min(min_per_cell, int(counts[c])) for c in cells}
    remaining = total - sum(alloc.values())
    if remaining > 0:
        headroom = {c: int(counts[c]) - alloc[c] for c in cells}
        pool = sum(v for v in counts[cells])
        for c in sorted(cells, key=lambda x: -counts[x]):
            if remaining <= 0:
                break
            want = int(round(total * counts[c] / pool)) - alloc[c]
            take = max(0, min(want, headroom[c], remaining))
            alloc[c] += take
            remaining -= take
        # Distribute any rounding leftovers to the largest cells with headroom.
        for c in sorted(cells, key=lambda x: -counts[x]):
            if remaining <= 0:
                break
            take = min(headroom[c] - (alloc[c] - min(min_per_cell, int(counts[c]))), remaining)
            if take > 0:
                alloc[c] += take
                remaining -= take
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out", default="data/golden")
    args = ap.parse_args()

    df = pd.read_parquet("data/interim/SpotifyCares_labelled.parquet")
    df = df.reset_index(drop=True)
    df["cell"] = df.intent_weak + "|" + df.region

    counts = df.cell.value_counts()
    alloc = allocate(counts, args.n, MIN_PER_CELL)

    rng = np.random.default_rng(args.seed)
    picks = []
    for cell, k in alloc.items():
        idx = df.index[df.cell == cell].to_numpy()
        if k <= 0 or len(idx) == 0:
            continue
        picks.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    sel = np.sort(np.concatenate(picks))

    g = df.loc[sel].copy()
    # Sampling weight: how many corpus messages each sampled item stands for.
    cell_n = g.cell.value_counts()
    g["weight"] = g.cell.map(lambda c: counts[c] / cell_n[c])
    g["item_id"] = [f"g{i:04d}" for i in range(len(g))]

    order = rng.permutation(len(g))          # present in random order, not by intent
    g = g.iloc[order].reset_index(drop=True)
    g["relabel"] = False
    g.loc[rng.choice(len(g), size=min(RELABEL_N, len(g)), replace=False), "relabel"] = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["item_id", "thread_id", "customer_text", "reply_text", "agent",
            "intent_weak", "region", "weight", "relabel", "n_turns",
            "got_thanks", "is_deflection", "cust_has_pii"]
    g[cols].to_parquet(out / "sample.parquet", index=False)
    (out / "held_out_threads.json").write_text(
        json.dumps({"thread_ids": sorted(int(t) for t in g.thread_id),
                    "seed": args.seed, "n": len(g)}, indent=2))

    print(f"sampled {len(g)} items over {len(alloc)} cells (seed {args.seed})")
    print(f"  re-label subset: {int(g.relabel.sum())}")
    print(f"  weight range   : {g.weight.min():.2f} .. {g.weight.max():.2f}")
    print(f"\n{'intent':<28}{'core':>6}{'tail':>6}{'other':>7}{'total':>7}{'corpus%':>9}")
    piv = g.pivot_table(index="intent_weak", columns="region",
                        values="item_id", aggfunc="count").fillna(0).astype(int)
    for name in piv.index:
        row = piv.loc[name]
        tot = int(row.sum())
        share = (df.intent_weak == name).mean() * 100
        print(f"  {name:<26}{row.get('core',0):>6}{row.get('tail',0):>6}"
              f"{row.get('other',0):>7}{tot:>7}{share:>8.1f}%")
    print(f"\n  {'TOTAL':<26}{piv.get('core',pd.Series()).sum():>6}"
          f"{piv.get('tail',pd.Series()).sum():>6}{piv.get('other',pd.Series()).sum():>7}{len(g):>7}")
    print(f"\nwrote {out}/sample.parquet and {out}/held_out_threads.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
