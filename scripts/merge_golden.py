"""Merge the two annotation passes into the final golden set.

Pass 2 supersedes pass 1 for the 83 reviewed items; the other 167 stand as first
given. Provenance is kept per row so the report can say exactly which labels
were revised and why, and so the drift and agreement analyses stay reproducible.

Deliberately NOT done here: any relabelling by the system's author. The golden
set is the annotator's, and a set edited by whoever built the model under test
cannot measure that model.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiver.schema import ESCALATION_REASON_NAMES, INTENT_NAMES  # noqa: E402


def main() -> int:
    v1 = pd.read_parquet("data/golden/labels_raw.parquet").set_index("item_id")
    v2 = pd.read_parquet("data/golden/labels_v2_raw.parquet").set_index("item_id")
    sample = pd.read_parquet("data/golden/sample.parquet").set_index("item_id")

    cols = ["intent", "route", "reason", "note", "flagged"]
    final = v1[cols].copy()
    final["pass"] = 1
    final["revised_why"] = ""

    for iid, row in v2.iterrows():
        if pd.isna(row.get("intent")) or pd.isna(row.get("route")):
            continue                      # an unfinished revision never overwrites a finished label
        for c in cols:
            final.loc[iid, c] = row.get(c)
        final.loc[iid, "pass"] = 2
        final.loc[iid, "revised_why"] = row.get("why", "")

    final = final.join(sample[["customer_text", "reply_text", "agent", "thread_id",
                               "intent_weak", "region", "weight", "relabel"]])
    final["complete"] = final.intent.notna() & final.route.notna()

    bad_i = set(final.intent.dropna()) - set(INTENT_NAMES)
    bad_r = set(final.reason.dropna()) - set(ESCALATION_REASON_NAMES) - \
        {"known_answer", "triage_question", "acknowledge_only"}
    assert not bad_i, f"unknown intents: {bad_i}"
    assert not bad_r, f"unknown reasons: {bad_r}"

    usable = final[final.complete]
    final.reset_index().to_parquet("data/golden/golden.parquet", index=False)
    usable.reset_index()[["item_id", "customer_text", "intent", "route", "reason",
                          "note", "flagged", "region", "weight", "thread_id"]] \
        .to_json("data/golden/golden.jsonl", orient="records", lines=True)

    print(f"golden set: {len(final)} items, {len(usable)} usable "
          f"({len(final) - len(usable)} still incomplete and excluded)")
    print(f"  from pass 1: {int((final['pass'] == 1).sum())}   "
          f"revised in pass 2: {int((final['pass'] == 2).sum())}")
    print(f"\n{'intent':<28}{'n':>5}{'share':>8}   escalate%")
    for name, g in sorted(usable.groupby("intent"), key=lambda kv: -len(kv[1])):
        esc = (g.route == "escalate").mean() * 100
        print(f"  {name:<26}{len(g):>5}{len(g)/len(usable)*100:>7.1f}%{esc:>10.0f}%")
    print(f"\n  route: auto {int((usable.route=='auto').sum())} / "
          f"escalate {int((usable.route=='escalate').sum())} "
          f"({(usable.route=='escalate').mean()*100:.1f}% escalated)")
    print(f"  weighted to corpus frequency: "
          f"{(usable.weight*(usable.route=='escalate')).sum()/usable.weight.sum()*100:.1f}% escalated")
    print("\nwrote data/golden/golden.parquet and golden.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
