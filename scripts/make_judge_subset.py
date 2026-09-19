"""Fix the subset that reply quality is judged on, once, for every stage.

Judging costs four calls per item (three systems plus Spotify's own reply), so
the judged set is the binding constraint on a free tier, not the drafted set.
The brief invites a subsample; what it cannot tolerate is an ad-hoc one, so the
subset is drawn once with a fixed seed, stratified by intent, and written to
disk. Drafting and judging both read it, which keeps them talking about the same
items and makes the number reproducible.

Intent classification and routing are unaffected and still run on all 249.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

N = 120
SEED = 20260919


def main() -> int:
    g = pd.read_parquet("data/golden/golden.parquet")
    g = g[g.complete].reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    # Proportional by intent with a floor of 4, so no class vanishes entirely.
    counts = g.intent.value_counts()
    picks = []
    for intent, n_total in counts.items():
        k = max(4, int(round(N * n_total / len(g))))
        idx = g.index[g.intent == intent].to_numpy()
        picks.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    sel = np.sort(np.concatenate(picks))[:N]

    sub = g.loc[sel]
    Path("data/golden/judge_subset.json").write_text(json.dumps(
        {"item_ids": sub.item_id.tolist(), "n": len(sub), "seed": SEED}, indent=2))

    print(f"judged subset: {len(sub)} of {len(g)} items (seed {SEED})")
    print(f"\n{'intent':<28}{'subset':>8}{'full':>7}{'share':>9}")
    for intent in counts.index:
        s = int((sub.intent == intent).sum())
        print(f"  {intent:<26}{s:>8}{counts[intent]:>7}"
              f"{s/len(sub)*100:>8.1f}%")
    print(f"\n  escalation rate  subset {(sub.route=='escalate').mean()*100:.1f}%"
          f"  vs full {(g.route=='escalate').mean()*100:.1f}%")
    print("wrote data/golden/judge_subset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
