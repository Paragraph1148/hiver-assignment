"""Run reply drafting and escalation routing over the golden set."""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiver.draft import CannedReply, NearestReply, RagDrafter  # noqa: E402
from hiver.embed import embed                                  # noqa: E402
from hiver.llm import LLM                                      # noqa: E402
from hiver.retrieve import build_index                         # noqa: E402
from hiver.route import AlwaysAuto, IntentPriorRouter, LlmRouter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--subset", action="store_true",
                    help="restrict to the fixed judged subset (drafting only; "
                         "judging costs 4 calls per item and is the binding "
                         "constraint on a free tier)")
    ap.add_argument("--stage", default="both", choices=["route", "draft", "both"],
                    help="stages run separately so they can use different "
                         "providers concurrently - separate quotas, no contention")
    args = ap.parse_args()

    g = pd.read_parquet("data/golden/golden.parquet")
    g = g[g.complete].reset_index(drop=True)
    if args.subset:
        keep = set(json.loads(Path("data/golden/judge_subset.json").read_text())["item_ids"])
        g = g[g.item_id.isin(keep)].reset_index(drop=True)
        print(f"restricted to the judged subset: {len(g)} items", flush=True)
    texts = g.customer_text.tolist()
    vecs = embed(texts, show_progress=False)

    preds = pd.read_parquet("data/results/intent_preds.parquet").set_index("item_id")
    pred_intent = [preds.llm.get(i, "other") for i in g.item_id]

    draft_ix = build_index("draft")     # deflections removed
    intent_ix = build_index("intent")
    print(f"draft index {len(draft_ix):,} | intent index {len(intent_ix):,}", flush=True)

    part = Path(f"data/results/_part_{args.stage}.parquet")
    out = pd.DataFrame({"item_id": g.item_id, "customer_text": g.customer_text,
                        "reply_text": g.reply_text, "gold_intent": g.intent,
                        "gold_route": g.route, "gold_reason": g.reason,
                        "region": g.region, "weight": g.weight,
                        "pred_intent": pred_intent})

    # --- routing -----------------------------------------------------------
    esc = (g.route == "escalate").to_numpy()
    if args.stage in ("route", "both"):
        out["route_always_auto"] = ["auto"] * len(g)
        prior = IntentPriorRouter().fit(g.intent, esc)
        loo = prior.predict_loo(pred_intent, esc)
        out["route_prior"] = ["escalate" if r.escalate else "auto" for r in loo]
        out["route_prior_risk"] = [r.risk for r in loo]

        llm = LLM(provider=args.provider)
        router = LlmRouter(llm, index=intent_ix)
        print(f"routing {len(texts)} on {llm.provider_name}/{llm.model} ...", flush=True)
        t0 = time.time()
        routes = router.predict(texts, vectors=vecs, intents=pred_intent, progress=True)
        out["route_llm"] = ["escalate" if r.escalate else "auto" for r in routes]
        out["route_llm_reason"] = [r.reason for r in routes]
        out["route_llm_risk"] = [r.risk for r in routes]
        out["route_llm_why"] = [r.rationale for r in routes]
        print(f"  routing done {time.time()-t0:.0f}s | hits {llm.cache.stats.hits} "
              f"misses {llm.cache.stats.misses}", flush=True)

    # --- drafting ----------------------------------------------------------
    if args.stage in ("draft", "both"):
        out["draft_canned"] = [d.text for d in CannedReply().draft(texts, vecs)]
        nearest = NearestReply(draft_ix).draft(texts, vecs)
        out["draft_nearest"] = [d.text for d in nearest]

        llm = LLM(provider=args.provider)
        rag = RagDrafter(llm, draft_ix)
        print(f"drafting {len(texts)} on {llm.provider_name}/{llm.model} ...", flush=True)
        t0 = time.time()
        drafts = rag.draft(texts, vectors=vecs, intents=pred_intent, progress=True)
        out["draft_rag"] = [d.text for d in drafts]
        out["draft_rag_failed"] = [d.failed for d in drafts]
        out["precedents"] = [" ||| ".join(h.reply_text for h in d.precedents[:4])
                             for d in drafts]
        n_fail = sum(d.failed for d in drafts)
        print(f"  drafting done {time.time()-t0:.0f}s | failed {n_fail}", flush=True)

    Path("data/results").mkdir(parents=True, exist_ok=True)
    out.to_parquet(part, index=False)
    print(f"\nwrote {part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
