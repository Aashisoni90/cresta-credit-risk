"""
test_monitoring.py
==================
Tests for §14.3. These target the three ways a PSI implementation usually
produces a confident wrong answer, because a monitor that is silently broken
is worse than no monitor — it converts "nobody is looking" into "something is
looking and it says we are fine".
"""
import numpy as np
import pandas as pd
import pytest

from monitoring import (psi, psi_table, feature_drift_trigger,
                        missingness_trigger, performance_decay_trigger,
                        prevalence_shift_trigger, MIN_EVENTS_FOR_AUC)

rng = np.random.default_rng(42)


def test_identical_distributions_score_near_zero():
    a = pd.Series(rng.normal(size=5000))
    b = pd.Series(rng.normal(size=5000))
    assert psi(a, b)["psi"] < 0.05


def test_a_shifted_distribution_is_detected():
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(2, 1, 5000))
    assert psi(a, b)["psi"] > 0.25


def test_a_vanished_bucket_is_flagged_not_smoothed_away():
    """The usual epsilon-pad turns the most informative drift into a mild one.

    Everything above zero disappears in the current window. That is a total
    structural break, and the result must say the number is a lower bound
    rather than report a tidy moderate figure.
    """
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.uniform(-3, -2, 5000))
    r = psi(a, b)
    assert r["floored_buckets"] > 0
    assert "lower bound" in r["detail"]
    assert r["psi"] > 0.25


def test_missingness_alone_registers_as_drift():
    """A feature that stops being reported scores 0.00 under the usual
    non-null-only implementation. It is the drift that matters most in a
    distress panel: firms under stress stop filing."""
    values = rng.normal(size=4000)
    a = pd.Series(values)
    b = pd.Series(values.copy())
    b[: int(0.6 * len(b))] = np.nan       # same values, 60% now missing
    r = psi(a, b)
    assert r["psi"] > 0.25, "missingness must not be invisible to PSI"
    assert r["actual_null_rate"] == pytest.approx(0.6, abs=0.01)


def test_a_constant_reference_becomes_a_large_categorical_psi():
    """A feature that was constant and now varies has unambiguously drifted.

    Quantile bucketing cannot express that — every edge is the same value — so
    an earlier version refused it as degenerate. The categorical path answers
    it instead: one level held 100% of the reference and now holds almost
    none. Refusing here would have been the monitor declining to report the
    clearest break it can see.
    """
    r = psi(pd.Series([1.0] * 500), pd.Series(rng.normal(size=500)))
    assert r["psi"] is not None and r["psi"] > 1.0
    assert r["floored_buckets"] > 0


def test_boolean_flags_are_measured_not_skipped():
    a = pd.Series([False] * 950 + [True] * 50)
    b = pd.Series([False] * 500 + [True] * 500)
    r = psi(a, b)
    assert r["psi"] is not None and r["psi"] > 0.25


def test_thin_evidence_is_insufficient_and_not_a_pass():
    t = performance_decay_trigger(0.95, 0.50, n_events=MIN_EVENTS_FOR_AUC - 1)
    assert t.status == "insufficient"
    assert not t.fired, "insufficient must not be reported as a fired trigger"
    assert "NOT treated as a pass" in t.detail


def test_real_decay_fires():
    t = performance_decay_trigger(0.95, 0.85, n_events=200)
    assert t.status == "fire" and t.fired


def test_prevalence_shift_fires_independently_of_ranking():
    """AUC is prevalence-invariant; the operating threshold is not."""
    t = prevalence_shift_trigger(0.05, 0.12, n_cur=3000)
    assert t.status == "fire"


def test_one_drifted_feature_is_a_watch_not_a_population_change():
    cols = [f"f{i}" for i in range(10)]
    ref = pd.DataFrame({c: rng.normal(size=2000) for c in cols})
    cur = ref.copy()
    cur["f0"] = rng.normal(3, 1, 2000)
    t = feature_drift_trigger(psi_table(ref, cur, cols))
    assert t.status == "watch", "one feature moving is a data-quality question"


def test_a_third_of_features_drifting_fires():
    cols = [f"f{i}" for i in range(9)]
    ref = pd.DataFrame({c: rng.normal(size=2000) for c in cols})
    cur = ref.copy()
    for c in cols[:4]:
        cur[c] = rng.normal(3, 1, 2000)
    t = feature_drift_trigger(psi_table(ref, cur, cols))
    assert t.status == "fire"


def test_missingness_trigger_reads_in_percentage_points():
    cols = ["a", "b"]
    ref = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
    cur = ref.copy()
    cur.loc[: int(0.3 * len(cur)), "a"] = np.nan
    t = missingness_trigger(psi_table(ref, cur, cols))
    assert t.status == "fire" and t.evidence["feature"] == "a"
