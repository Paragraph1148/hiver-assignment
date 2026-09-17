"""Assemble the items needing a second look, and why each one qualifies.

Four disjoint reasons, recorded per item so the report can say exactly what was
relabelled and on what grounds:

  incomplete  - no intent or no route was recorded; not usable as ground truth
  schema_gap  - the annotator's note names a category the schema lacked
                (feature requests), so there was no correct answer available
  overloaded  - labelled angry_or_vulnerable, which absorbed "wants a human"
  relabel     - pre-marked for the intra-annotator agreement check

Items are presented BLIND in the second pass: no prior label is shown. That is
required for the relabel subset to measure anything, and it keeps the corrected
items unanchored too. The annotator's own note IS shown, because it is their
reasoning rather than a label, and it is what they were reaching for.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

FEATURE_NOTE = r"feature|support(?:ing)? apple|removed feature"


def main() -> int:
    d = pd.read_parquet("data/golden/labels_raw.parquet")
    s = pd.read_parquet("data/golden/sample.parquet").set_index("item_id")
    note = d.note.fillna("").str.lower()

    reason, hits = {}, {}

    def mark(ids, why):
        for i in ids:
            reason.setdefault(i, why)   # first reason wins; order sets priority
            hits[i] = hits.get(i, 0) + 1

    mark(d.loc[d.intent.isna() | d.route.isna(), "item_id"], "incomplete")
    mark(d.loc[note.str.contains(FEATURE_NOTE, regex=True), "item_id"], "schema_gap")
    mark(d.loc[d.reason == "angry_or_vulnerable", "item_id"], "overloaded")
    mark(s.index[s.relabel].tolist(), "relabel")

    queue = [{"id": i, "why": w, "note": str(d.loc[d.item_id == i, "note"].iloc[0] or "")}
             for i, w in reason.items()]
    queue = [q if q["note"] != "nan" else {**q, "note": ""} for q in queue]
    queue.sort(key=lambda q: q["id"])

    Path("data/golden/review_queue.json").write_text(json.dumps(queue, indent=2))

    counts = pd.Series([q["why"] for q in queue]).value_counts()
    print(f"review queue: {len(queue)} items of 250")
    for k, v in counts.items():
        print(f"  {k:<12} {v:>3}")
    multi = sum(1 for v in hits.values() if v > 1)
    print(f"\n  {multi} items qualified on more than one ground and are reviewed once")
    print("\nwrote data/golden/review_queue.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
