"""Score the intent classifiers, with uncertainty and against the label ceiling.

Three things this reports that a plain accuracy table does not:

  * a bootstrap CI on every figure, and a PAIRED bootstrap for each model
    against the baselines, so "better" means better than sampling noise;
  * a split by region, testing the pre-registered prediction that the long tail
    is materially harder than the dense core;
  * the annotator's own self-consistency (71.9% on intent) drawn as a ceiling.
    Scoring near or above it does not mean the model is good - it means the
    measurement has run out of resolution.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from hiver.metrics import (bootstrap, deferral_curve, expected_calibration_error,  # noqa: E402
                           macro_f1, paired_bootstrap, per_class_f1)

CEILING = 0.719   # intra-annotator agreement on intent, schema change excluded
MODELS = [("majority", "trivial: majority class"),
          ("tfidf_lr", "simple: TF-IDF + log.reg"),
          ("knn", "simple: hybrid kNN"),
          ("llm", "system: retrieval + LLM")]


def main() -> int:
    p = pd.read_parquet("data/results/intent_preds.parquet")

    # A call that never completed is not a prediction. Scoring it as a wrong
    # answer turns provider downtime into an apparent model weakness, so the
    # rows are reported and excluded rather than silently counted.
    fail_cols = [c for c in p.columns if c.endswith("_failed")]
    failed = p[fail_cols].any(axis=1) if fail_cols else pd.Series(False, index=p.index)
    if failed.any():
        print(f"!! {int(failed.sum())} of {len(p)} items had a call that did not "
              f"complete; excluded from every figure below.\n")
        p = p[~failed].reset_index(drop=True)

    gold = p.gold.to_numpy()
    w = p.weight.to_numpy()
    present = [(k, n) for k, n in MODELS if k in p.columns]

    print(f"INTENT CLASSIFICATION   n={len(p)}   10 classes\n")
    print(f"{'model':<28}{'accuracy [95% CI]':<26}{'weighted':>10}{'macro-F1':>10}")
    rows = {}
    for key, name in present:
        pred = p[key].to_numpy()
        acc = bootstrap(gold, pred, seed=1)
        wacc = bootstrap(gold, pred, seed=1, weight=w)
        f1 = bootstrap(gold, pred, fn=lambda g, q: macro_f1(g, q), seed=1)
        rows[key] = acc
        print(f"  {name:<26}{str(acc):<26}{wacc.point*100:>9.1f}%{f1.point*100:>9.1f}%")
    print(f"\n  {'annotator self-consistency':<26}{CEILING*100:>5.1f}%   "
          f"<- the measurement ceiling, not a model")
    if "llm" in rows:
        a = rows["llm"]
        at_ceiling = a.lo <= CEILING <= a.hi
        gap = CEILING - a.point
        print(f"  {'gap to ceiling':<26}{gap*100:>5.1f} pts   "
              + ("the CI CONTAINS the ceiling: the system is statistically "
                 "indistinguishable\n" + " " * 34 + "from the annotator's own "
                 "self-consistency, so this test set\n" + " " * 34 + "can no "
                 "longer tell them apart." if at_ceiling else
                 "significantly below the ceiling"))

    print("\nPAIRED COMPARISON (difference in accuracy, same items resampled)")
    if "llm" in p.columns:
        for base in ["majority", "tfidf_lr", "knn"]:
            if base not in p.columns:
                continue
            d, pv = paired_bootstrap(gold, p["llm"].to_numpy(), p[base].to_numpy(), seed=2)
            verdict = "significant" if pv < 0.05 else "NOT significant"
            print(f"  llm - {base:<10} {d.point*100:>+6.1f} pts "
                  f"[{d.lo*100:+.1f}, {d.hi*100:+.1f}]  p={pv:.3f}  {verdict}")

    print("\nBY REGION (pre-registered: tail should be harder than core)")
    print(f"  {'model':<26}" + "".join(f"{r:>16}" for r in ["core", "tail", "other"]))
    for key, name in present:
        cells = []
        for r in ["core", "tail", "other"]:
            m = (p.region == r).to_numpy()
            if m.sum() < 5:
                cells.append("       n/a      ")
                continue
            a = bootstrap(gold[m], p[key].to_numpy()[m], seed=3)
            cells.append(f"{a.point*100:>9.1f}% (n={m.sum():>3})")
        print(f"  {name:<26}" + "".join(cells))

    best = "llm" if "llm" in p.columns else present[-1][0]
    print(f"\nPER-CLASS ({best})")
    print(f"  {'intent':<28}{'P':>7}{'R':>7}{'F1':>7}{'n':>6}")
    for c, v in sorted(per_class_f1(gold, p[best].to_numpy()).items(),
                       key=lambda kv: -kv[1]["support"]):
        print(f"  {c:<28}{v['precision']*100:>6.0f}%{v['recall']*100:>6.0f}%"
              f"{v['f1']*100:>6.0f}%{v['support']:>6}")

    print("\nCONFUSION (gold -> predicted, top errors)")
    err = p[p[best] != p.gold]
    for (a, b), n in err.groupby(["gold", best]).size().sort_values(ascending=False).head(8).items():
        print(f"  {a:<28} -> {b:<28} {n}")

    out = {"n": len(p), "ceiling": CEILING,
           "accuracy": {k: [rows[k].point, rows[k].lo, rows[k].hi] for k, _ in present}}

    if f"{best}_conf" in p.columns:
        conf = p[f"{best}_conf"].to_numpy()
        correct = (gold == p[best].to_numpy()).astype(float)
        ece = expected_calibration_error(correct, conf)
        print(f"\nCALIBRATION ({best})   ECE = {ece:.3f}")
        print("  coverage   accuracy   threshold")
        for cov, acc, thr in deferral_curve(correct, conf, steps=11):
            bar = "#" * int(acc * 30)
            print(f"   {cov*100:>5.0f}%    {acc*100:>6.1f}%     {thr:.2f}  {bar}")
        out["ece"] = ece
        out["deferral"] = deferral_curve(correct, conf, steps=21)

    Path("data/results").mkdir(parents=True, exist_ok=True)
    Path("data/results/intent_metrics.json").write_text(json.dumps(out, indent=2))
    print("\nwrote data/results/intent_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
