import numpy as np
import pytest

from hiver.metrics import (accuracy, bootstrap, deferral_curve,
                           expected_calibration_error, macro_f1, paired_bootstrap,
                           per_class_f1)

G = np.array(["a", "a", "b", "b", "c", "c"])


def test_accuracy_and_weighting():
    p = np.array(["a", "b", "b", "b", "c", "a"])
    assert accuracy(G, p) == pytest.approx(4 / 6)
    # Weighting must actually move the number, not be silently ignored.
    w = np.array([10.0, 10.0, 1, 1, 1, 1])
    assert accuracy(G, p, w) == pytest.approx((10 + 1 + 1 + 1) / 24)


def test_macro_f1_ignores_classes_absent_from_gold():
    """A class the model invents but gold never contains must not be scored as
    a zero-F1 class; that silently punishes every model for one stray guess."""
    g = np.array(["a", "a", "b", "b"])
    p = np.array(["a", "a", "b", "z"])
    assert macro_f1(g, p) == pytest.approx(macro_f1(g, p, labels=["a", "b"]))


def test_macro_f1_differs_from_accuracy_under_imbalance():
    g = np.array(["a"] * 9 + ["b"])
    p = np.array(["a"] * 10)
    assert accuracy(g, p) == pytest.approx(0.9)
    assert macro_f1(g, p) == pytest.approx(0.9 / 2 * 2 / 2, abs=0.3)
    assert macro_f1(g, p) < accuracy(g, p)


def test_per_class_support_sums_to_n():
    p = np.array(["a", "b", "b", "b", "c", "a"])
    assert sum(v["support"] for v in per_class_f1(G, p).values()) == len(G)


def test_bootstrap_brackets_point_and_is_deterministic():
    p = np.array(["a", "a", "b", "b", "c", "a"])
    i1 = bootstrap(G, p, seed=7)
    i2 = bootstrap(G, p, seed=7)
    assert i1.lo <= i1.point <= i1.hi
    assert (i1.lo, i1.hi) == (i2.lo, i2.hi)


def test_paired_bootstrap_finds_no_difference_between_identical_models():
    p = np.array(["a", "a", "b", "b", "c", "a"])
    interval, pval = paired_bootstrap(G, p, p)
    assert interval.point == 0 and pval == pytest.approx(1.0, abs=0.01)


def test_ece_zero_for_perfect_calibration():
    correct = np.array([1.0] * 90 + [0.0] * 10)
    conf = np.array([0.9] * 100
                    )
    assert expected_calibration_error(correct, conf) == pytest.approx(0.0, abs=1e-9)


def test_ece_large_for_overconfidence():
    correct = np.array([0.0] * 100)
    conf = np.array([1.0] * 100)
    assert expected_calibration_error(correct, conf) == pytest.approx(1.0)


def test_deferral_curve_is_monotone_in_coverage():
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    correct = (rng.random(200) < conf).astype(float)   # confidence is informative
    curve = deferral_curve(correct, conf)
    cov = [c for c, _, _ in curve]
    assert cov == sorted(cov) and cov[-1] == pytest.approx(1.0)
    # Accuracy at low coverage should beat accuracy at full coverage.
    assert curve[0][1] > curve[-1][1]
