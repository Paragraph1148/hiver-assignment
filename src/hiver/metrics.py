"""Metrics with uncertainty attached.

Every headline figure here carries a bootstrap confidence interval, because at
n=249 the 95% CI on accuracy is roughly +/-6 points. Most baseline gaps reported
without one are indistinguishable from noise, and saying so is the point.

Paired bootstrap is used for model-vs-model comparison: resampling the same
items for both models cancels the item-difficulty variance that dominates an
unpaired comparison, so it detects real differences a naive comparison misses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point*100:.1f}% [{self.lo*100:.1f}-{self.hi*100:.1f}]"


def accuracy(gold: np.ndarray, pred: np.ndarray, weight: np.ndarray | None = None) -> float:
    hit = (gold == pred).astype(float)
    if weight is None:
        return float(hit.mean())
    return float((hit * weight).sum() / weight.sum())


def macro_f1(gold: np.ndarray, pred: np.ndarray, labels: list[str] | None = None) -> float:
    labels = labels if labels is not None else sorted(set(gold) | set(pred))
    fs = []
    for c in labels:
        tp = float(((gold == c) & (pred == c)).sum())
        fp = float(((gold != c) & (pred == c)).sum())
        fn = float(((gold == c) & (pred != c)).sum())
        if tp + fn == 0:            # class absent from gold: undefined, not zero
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs)) if fs else 0.0


def per_class_f1(gold: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for c in sorted(set(gold)):
        tp = float(((gold == c) & (pred == c)).sum())
        fp = float(((gold != c) & (pred == c)).sum())
        fn = float(((gold == c) & (pred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "support": int(tp + fn)}
    return out


def bootstrap(gold: np.ndarray, pred: np.ndarray, fn=accuracy, n: int = 2000,
              seed: int = 0, weight: np.ndarray | None = None) -> Interval:
    rng = np.random.default_rng(seed)
    m = len(gold)
    point = fn(gold, pred, weight) if weight is not None else fn(gold, pred)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        vals[i] = (fn(gold[idx], pred[idx], weight[idx]) if weight is not None
                   else fn(gold[idx], pred[idx]))
    return Interval(point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def paired_bootstrap(gold: np.ndarray, a: np.ndarray, b: np.ndarray,
                     fn=accuracy, n: int = 2000, seed: int = 0) -> tuple[Interval, float]:
    """Difference a - b, with a two-sided p-value for the null of no difference."""
    rng = np.random.default_rng(seed)
    m = len(gold)
    point = fn(gold, a) - fn(gold, b)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        diffs[i] = fn(gold[idx], a[idx]) - fn(gold[idx], b[idx])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return (Interval(point, float(np.percentile(diffs, 2.5)),
                     float(np.percentile(diffs, 97.5))), float(min(p, 1.0)))


def expected_calibration_error(correct: np.ndarray, conf: np.ndarray,
                               bins: int = 10) -> float:
    """How far self-reported confidence sits from observed accuracy.

    This is what makes a confidence score usable for routing: a model that says
    0.9 should be right about 90% of the time, or the deferral threshold is
    meaningless.
    """
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def deferral_curve(correct: np.ndarray, conf: np.ndarray,
                   steps: int = 21) -> list[tuple[float, float, float]]:
    """(coverage, accuracy on the covered slice, threshold).

    The operating question is not "how accurate is it" but "how much volume can
    it take at an acceptable error rate", so this is the curve that matters.
    """
    order = np.argsort(-conf)
    c, k = correct[order], conf[order]
    out = []
    for frac in np.linspace(0.05, 1.0, steps):
        n = max(1, int(round(frac * len(c))))
        out.append((n / len(c), float(c[:n].mean()), float(k[n - 1])))
    return out
