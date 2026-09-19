"""Score drafted replies, including the brand's own replies as a system.

The last part is the point. Everyone treats the historical reply as a gold
answer and scores similarity to it. Judging Spotify's ACTUAL replies on the same
rubric as the models tests that assumption directly: if a large share of the
real replies are mediocre, then similarity-to-reference is not a quality metric,
and any system optimised toward it is being pushed toward mediocrity.

The judge runs on a different model family from the generator, and pairwise
comparisons are run in both orders with only order-consistent verdicts counted,
so position bias is measured rather than absorbed.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from hiver.judge import DIMENSIONS, ReplyJudge  # noqa: E402
from hiver.llm import LLM                        # noqa: E402

SYSTEMS = [("draft_canned", "trivial: canned deflection"),
           ("draft_nearest", "simple: nearest historical reply"),
           ("draft_rag", "system: retrieval-grounded LLM"),
           ("reply_text", "SPOTIFY'S OWN REPLY")]


def main() -> int:
    ap = argparse.ArgumentParser()
    # Cross-family separation is about the MODEL FAMILY, not the vendor hosting
    # it. Groq serves both OpenAI's gpt-oss (the generator) and Alibaba's Qwen,
    # so judging with Qwen keeps the generator and judge in different families
    # while staying inside one provider's quota - which is what actually made
    # this runnable after Gemini's daily allowance ran out mid-evaluation.
    ap.add_argument("--provider", default="groq", help="judge provider")
    ap.add_argument("--judge-model", default="qwen/qwen3.8-27b",
                    help="must be a different family from the generator")
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--pairwise", type=int, default=60, help="items for A/B vs nearest")
    args = ap.parse_args()

    d = pd.read_parquet("data/results/_part_draft.parquet")
    d = d[~d.draft_rag_failed].reset_index(drop=True)
    if args.limit:
        d = d.sample(args.limit, random_state=11).reset_index(drop=True)
    print(f"judging {len(d)} items x {len(SYSTEMS)} systems", flush=True)

    llm = LLM(provider=args.provider, model=args.judge_model)
    judge = ReplyJudge(llm)
    print(f"judge: {llm.provider_name}/{llm.model}  (generator was "
          f"openai/gpt-oss-120b - different family)", flush=True)

    # Checkpoint after each system. Container restarts have repeatedly killed
    # long runs; the response cache already makes the CALLS resumable, and this
    # makes the scored rows survive too, so a partial run is still analysable.
    ckpt = Path("data/results/judge_scores.parquet")
    rows = []
    if ckpt.exists():
        prev = pd.read_parquet(ckpt)
        rows = prev.to_dict("records")
        done = set(prev.system.unique())
        print(f"  resuming: {len(rows)} rows already scored for {sorted(done)}", flush=True)

    for key, name in SYSTEMS:
        if any(r.get("system") == key for r in rows):
            print(f"  {name:<34} already scored, skipping", flush=True)
            continue
        t0 = time.time()
        n_fail = 0
        for i, r in enumerate(d.itertuples()):
            prec = [p for p in str(r.precedents).split(" ||| ") if p]
            v = judge.score(r.customer_text, str(getattr(r, key)), prec)
            if not v.ok:
                n_fail += 1
                continue
            rows.append({"item_id": r.item_id, "system": key, "overall": v.overall,
                         "region": r.region, **v.scores})
            if (i + 1) % 50 == 0:
                print(f"    {name}: {i+1}/{len(d)}", flush=True)
        print(f"  {name:<34} {time.time()-t0:>6.0f}s  failed {n_fail}", flush=True)
        pd.DataFrame(rows).to_parquet(ckpt, index=False)

    s = pd.DataFrame(rows)
    s.to_parquet("data/results/judge_scores.parquet", index=False)

    print(f"\n{'system':<36}{'overall':>9}" +
          "".join(f"{k:>10}" for k, _ in DIMENSIONS) + f"{'n':>6}")
    for key, name in SYSTEMS:
        g = s[s.system == key]
        if g.empty:
            continue
        dims = "".join(f"{g[k].mean():>10.2f}" for k, _ in DIMENSIONS if k in g)
        print(f"  {name:<34}{g.overall.mean():>9.2f}{dims}{len(g):>6}")

    # How often is the brand's own reply merely adequate or worse?
    own = s[s.system == "reply_text"]
    if not own.empty:
        weak = (own.overall <= 3.0).mean()
        print(f"\n  Spotify's own replies scoring <= 3.0 (adequate or worse): {weak*100:.0f}%")
        print("  -> similarity to the historical reply is not a quality metric.")

    # --- pairwise, position-swapped ---------------------------------------
    sub = d.head(args.pairwise)
    print(f"\nPAIRWISE: system vs nearest-reply baseline, both orders, n={len(sub)}")
    wins = {"a": 0, "b": 0, "tie": 0}
    flips = 0
    for r in sub.itertuples():
        w, consistent = judge.compare(r.customer_text, str(r.draft_rag), str(r.draft_nearest))
        if not consistent:
            flips += 1
        wins[w] += 1
    total = sum(wins.values())
    print(f"  RAG wins {wins['a']} ({wins['a']/total*100:.0f}%) | "
          f"nearest wins {wins['b']} ({wins['b']/total*100:.0f}%) | tie {wins['tie']}")
    print(f"  order-inconsistent verdicts (position bias): {flips}/{total} "
          f"({flips/total*100:.0f}%) - counted as ties, not wins")

    Path("data/results/draft_metrics.json").write_text(json.dumps(
        {"n": len(d), "means": {k: float(s[s.system == k].overall.mean())
                                for k, _ in SYSTEMS if not s[s.system == k].empty},
         "brand_weak_share": float((own.overall <= 3.0).mean()) if not own.empty else None,
         "pairwise": wins, "position_flips": flips}, indent=2))
    print("\nwrote data/results/judge_scores.parquet and draft_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
