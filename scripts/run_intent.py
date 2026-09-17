"""Run every intent classifier over the golden set and save raw predictions.

Predictions are saved rather than scored here, so the metrics can be recomputed
without re-running anything - and so the LLM's answers are recorded once in the
response cache and replayed for free thereafter.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiver.classify import (KnnClassifier, LlmClassifier, MajorityClassifier,  # noqa: E402
                            TfidfLogReg, load_training)
from hiver.embed import embed    # noqa: E402
from hiver.llm import LLM        # noqa: E402
from hiver.retrieve import build_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--variants", default="0", help="comma-separated prompt variants")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--out", default="data/results/intent_preds.parquet")
    args = ap.parse_args()

    g = pd.read_parquet("data/golden/golden.parquet")
    g = g[g.complete].reset_index(drop=True)
    texts = g.customer_text.tolist()
    vecs = embed(texts, show_progress=False)
    print(f"golden: {len(g)} usable items", flush=True)

    X, y = load_training()
    index = build_index("intent")
    print(f"index: {len(index):,} pairs | training: {len(X):,} rows", flush=True)

    preds = pd.DataFrame({"item_id": g.item_id, "gold": g.intent,
                          "region": g.region, "weight": g.weight})

    for key, model in [("majority", MajorityClassifier().fit(X, y)),
                       ("tfidf_lr", TfidfLogReg().fit(X, y)),
                       ("knn", KnnClassifier(index).fit())]:
        t0 = time.time()
        p = model.predict(texts, vectors=vecs)
        preds[key] = [x.intent for x in p]
        preds[f"{key}_conf"] = [x.confidence for x in p]
        print(f"  {model.name:<38} {time.time()-t0:>6.1f}s", flush=True)

    if not args.skip_llm:
        for v in [int(x) for x in args.variants.split(",")]:
            llm = LLM(provider=args.provider)
            clf = LlmClassifier(index, llm, prompt_variant=v)
            key = "llm" if v == 0 else f"llm_v{v}"
            print(f"  running {clf.name} (prompt variant {v}) "
                  f"on {llm.provider_name}/{llm.model} ...", flush=True)
            t0 = time.time()
            p = clf.predict(texts, vectors=vecs, progress=True)
            preds[key] = [x.intent for x in p]
            preds[f"{key}_conf"] = [x.confidence for x in p]
            preds[f"{key}_failed"] = [x.failed for x in p]
            n_fail = sum(x.failed for x in p)
            print(f"    done in {time.time()-t0:.0f}s | "
                  f"cache hits {llm.cache.stats.hits}, misses {llm.cache.stats.misses} | "
                  f"failed calls {n_fail}", flush=True)
            if n_fail:
                print(f"    !! {n_fail} calls did not complete. They are marked failed, "
                      f"NOT scored as wrong answers. Re-run to fill them from cache.",
                      flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
