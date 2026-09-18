"""Score the escalation routers on the terms the decision is actually made on.

Accuracy is the wrong headline here and the trivial baseline proves it: 80% of
the golden set is auto-handled, so "auto-handle everything" scores 80% while
being useless. What matters is how much volume can be auto-handled at an
acceptable rate of bad sends, which is a threshold chosen from a cost curve.

Two errors, with very different prices:
  MISSED ESCALATION - a message that needed a human was auto-answered. A wrong
      or unsafe reply reaches a customer.
  NEEDLESS ESCALATION - an answerable message was sent to a human. Wasted
      agent time, no customer harm.

The ratio between those is a business input, not a modelling one, so the curve
is swept over it rather than one value being assumed.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from hiver.metrics import bootstrap, paired_bootstrap  # noqa: E402
from hiver.route import proxy_escalation_signals       # noqa: E402

MODELS = [("route_always_auto", "trivial: auto-handle everything"),
          ("route_prior", "simple: intent prior (leave-one-out)"),
          ("route_llm", "system: LLM policy router")]


def prf(gold_esc, pred_esc):
    tp = int((gold_esc & pred_esc).sum())
    fp = int((~gold_esc & pred_esc).sum())
    fn = int((gold_esc & ~pred_esc).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn


def main() -> int:
    d = pd.read_parquet("data/results/reply_route.parquet")
    gold = (d.gold_route == "escalate").to_numpy()
    print(f"ESCALATION ROUTING   n={len(d)}   {gold.mean()*100:.1f}% escalated\n")

    print(f"{'model':<38}{'acc':>16}{'P':>7}{'R':>7}{'F1':>7}{'missed':>8}{'needless':>10}")
    accs = {}
    for key, name in MODELS:
        if key not in d.columns:
            continue
        pred = (d[key] == "escalate").to_numpy()
        acc = bootstrap(gold.astype(object), pred.astype(object), seed=5)
        accs[key] = (acc, pred)
        p, r, f1, tp, fp, fn = prf(gold, pred)
        print(f"  {name:<36}{str(acc):>16}{p*100:>6.0f}%{r*100:>6.0f}%{f1*100:>6.0f}%"
              f"{fn:>8}{fp:>10}")

    if "route_llm" in accs:
        print("\nPAIRED COMPARISON (accuracy)")
        for base in ["route_always_auto", "route_prior"]:
            if base not in accs:
                continue
            diff, pv = paired_bootstrap(gold.astype(object), accs["route_llm"][1].astype(object),
                                        accs[base][1].astype(object), seed=6)
            tag = "significant" if pv < 0.05 else "NOT significant"
            print(f"  llm - {base.replace('route_',''):<22}{diff.point*100:>+6.1f} pts "
                  f"[{diff.lo*100:+.1f}, {diff.hi*100:+.1f}]  p={pv:.3f}  {tag}")

    # --- cost curve --------------------------------------------------------
    print("\nCOST CURVE (relative total cost; lower is better)")
    print("  a missed escalation costs Nx a needless one\n")
    print(f"  {'N':>5}" + "".join(f"{n.split(':')[0]:>22}" for _, n in MODELS))
    best_by_ratio = {}
    for ratio in [1, 3, 5, 10, 20, 50]:
        cells, costs = [], {}
        for key, _ in MODELS:
            if key not in d.columns:
                continue
            pred = (d[key] == "escalate").to_numpy()
            _, _, _, _, fp, fn = prf(gold, pred)
            cost = ratio * fn + fp
            costs[key] = cost
            cells.append(f"{cost:>22,}")
        winner = min(costs, key=costs.get)
        best_by_ratio[ratio] = winner
        print(f"  {ratio:>5}" + "".join(cells) + f"   best: {winner.replace('route_','')}")

    # --- risk threshold sweep ---------------------------------------------
    if "route_llm_risk" in d.columns:
        risk = d.route_llm_risk.to_numpy()
        print("\nOPERATING POINTS from the router's own risk score")
        print(f"  {'threshold':>10}{'auto %':>9}{'bad sends':>11}{'bad send rate':>15}")
        for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            auto = risk < t
            bad = int((auto & gold).sum())
            print(f"  {t:>10.1f}{auto.mean()*100:>8.1f}%{bad:>11}"
                  f"{(bad/max(auto.sum(),1))*100:>14.1f}%")

    # --- proxy validation --------------------------------------------------
    g = pd.read_parquet("data/golden/golden.parquet")
    g = g[g.complete].set_index("item_id").loc[d.item_id].reset_index()
    pairs = pd.read_parquet("data/interim/SpotifyCares_labelled.parquet")
    proxy = proxy_escalation_signals(g, pairs)
    print("\nPROXY VALIDATION (observable signals, independent of the annotator)")
    print(f"  human label       r = {np.corrcoef(proxy, gold.astype(float))[0,1]:+.3f}")
    for key, name in MODELS:
        if key not in d.columns:
            continue
        pred = (d[key] == "escalate").to_numpy().astype(float)
        if pred.std() == 0:
            print(f"  {name.split(':')[0]:<17} r =  n/a (constant prediction)")
            continue
        print(f"  {name.split(':')[0]:<17} r = {np.corrcoef(proxy, pred)[0,1]:+.3f}")

    Path("data/results").mkdir(parents=True, exist_ok=True)
    Path("data/results/route_metrics.json").write_text(json.dumps(
        {"n": len(d), "escalation_rate": float(gold.mean()),
         "accuracy": {k: [v[0].point, v[0].lo, v[0].hi] for k, v in accs.items()},
         "best_by_cost_ratio": best_by_ratio}, indent=2))
    print("\nwrote data/results/route_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
