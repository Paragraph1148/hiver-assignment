"""The semantic cache is a production cost lever; its accounting must be honest."""
import numpy as np
import pytest

from hiver.semantic_cache import SemanticCache, simulate


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_miss_on_empty_then_hit_on_identical():
    c = SemanticCache(threshold=0.95)
    v = unit(1, 0)
    assert c.get(v) == (None, 0.0)
    c.put(v, "billing")
    payload, sim = c.get(v)
    assert payload == "billing" and sim == pytest.approx(1.0)
    assert c.hit_rate == pytest.approx(0.5)      # one miss, one hit


def test_dissimilar_query_misses():
    c = SemanticCache(threshold=0.95)
    c.put(unit(1, 0), "billing")
    payload, sim = c.get(unit(0, 1))
    assert payload is None and sim == pytest.approx(0.0, abs=1e-6)


def test_threshold_controls_the_tradeoff():
    a, b = unit(1, 0), unit(1, 0.5)              # cosine ~0.894
    strict, loose = SemanticCache(threshold=0.95), SemanticCache(threshold=0.85)
    for c in (strict, loose):
        c.put(a, "billing")
    assert strict.get(b)[0] is None
    assert loose.get(b)[0] == "billing"


def test_eviction_drops_the_least_reused_entry():
    c = SemanticCache(threshold=0.99, max_entries=2)
    c.put(unit(1, 0), "popular")
    c.put(unit(0, 1), "never_used")
    c.get(unit(1, 0))                            # give 'popular' a hit
    c.put(unit(0.5, 0.5), "new")
    assert [e.payload for e in c.entries] == ["popular", "new"]


def test_simulate_reports_what_the_cache_actually_cost():
    """A hit serves an EARLIER message's answer, so it is right only when that
    borrowed answer matches this message's gold."""
    vecs = np.vstack([unit(1, 0), unit(1, 0.01)])   # near-identical pair
    # Live model would get both right, but they have different gold labels.
    res = simulate(vecs, preds=["billing", "playback"], gold=["billing", "playback"],
                   threshold=0.95)
    assert res["hit_rate"] == pytest.approx(0.5)     # second borrows the first
    assert res["accuracy_no_cache"] == pytest.approx(1.0)
    assert res["accuracy"] == pytest.approx(0.5)     # borrowed answer was wrong
    assert res["cost_of_cache_pts"] == pytest.approx(50.0)


def test_simulate_with_no_hits_matches_live_accuracy():
    vecs = np.vstack([unit(1, 0), unit(0, 1)])
    res = simulate(vecs, preds=["a", "b"], gold=["a", "x"], threshold=0.95)
    assert res["hit_rate"] == 0.0
    assert res["accuracy"] == pytest.approx(res["accuracy_no_cache"])


def test_none_payload_is_rejected():
    """Regression: storing None made hits indistinguishable from misses, so the
    cache grew without bound and the measured hit rate was overstated."""
    c = SemanticCache()
    with pytest.raises(ValueError):
        c.put(unit(1, 0), None)
