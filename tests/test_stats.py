"""Unit tests: each function is checked against a known value or an independent implementation."""
import numpy as np
import pytest
from scipy import stats as sps

from abtest import stats


def test_srm_balanced_split_is_not_flagged():
    _, p = stats.srm_test(50_000, 50_100)
    assert p > 0.5


def test_srm_detects_mismatch():
    _, p = stats.srm_test(50_000, 52_000)
    assert p < 0.001


def test_sample_size_matches_textbook_value():
    # Baseline 10%, detect +2pp, alpha 0.05 two-sided, power 80% -> ~3,841 per group
    assert stats.sample_size_per_group(0.10, 0.02) == pytest.approx(3841, abs=5)


def test_power_and_sample_size_are_consistent():
    n = stats.sample_size_per_group(0.12, 0.01, power=0.8)
    assert stats.power_of_test(0.12, 0.01, n) == pytest.approx(0.8, abs=0.005)


def test_mde_inverts_sample_size():
    n = stats.sample_size_per_group(0.12, 0.01)
    assert stats.minimum_detectable_effect(0.12, n) == pytest.approx(0.01, rel=0.01)


def test_ztest_matches_chi_square():
    # A two-sided two-proportion z-test is equivalent to a 2x2 chi-square test without correction.
    x_c, n_c, x_t, n_t = 1200, 10_000, 1290, 10_000
    res = stats.two_proportion_ztest(x_c, n_c, x_t, n_t)
    chi2, p, _, _ = sps.chi2_contingency([[x_c, n_c - x_c], [x_t, n_t - x_t]], correction=False)
    assert res.z ** 2 == pytest.approx(chi2, rel=1e-9)
    assert res.p_value == pytest.approx(p, rel=1e-9)


def test_ztest_direction_and_interval():
    res = stats.two_proportion_ztest(1000, 10_000, 1100, 10_000)
    assert res.diff_abs == pytest.approx(0.01)
    assert res.lift_rel == pytest.approx(0.10)
    assert res.ci_low < res.diff_abs < res.ci_high
    one_sided = stats.two_proportion_ztest(1000, 10_000, 1100, 10_000, alternative="larger")
    assert one_sided.p_value == pytest.approx(res.p_value / 2)


def test_ci_excludes_zero_iff_significant_for_clear_effects():
    sig = stats.two_proportion_ztest(1000, 10_000, 1200, 10_000)
    assert sig.p_value < 0.05 and sig.ci_low > 0
    null = stats.two_proportion_ztest(1000, 10_000, 1005, 10_000)
    assert null.p_value > 0.05 and null.ci_low < 0 < null.ci_high


def test_holm_adjustment_known_example():
    adjusted = stats.holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_bayes_probability_is_half_for_identical_groups():
    res = stats.bayesian_ab(500, 5000, 500, 5000)
    assert res.prob_treatment_better == pytest.approx(0.5, abs=0.01)
    assert res.lift_ci_low < 0 < res.lift_ci_high


def test_bayes_agrees_with_clear_winner():
    res = stats.bayesian_ab(1000, 10_000, 1200, 10_000)
    assert res.prob_treatment_better > 0.999
    assert res.expected_loss_treatment < res.expected_loss_control


def test_peeking_inflates_false_positives():
    one_look = stats.peeking_false_positive_rate(0.12, 20_000, n_looks=1, n_sims=4000)
    many_looks = stats.peeking_false_positive_rate(0.12, 20_000, n_looks=20, n_sims=4000)
    assert one_look == pytest.approx(0.05, abs=0.015)
    assert many_looks > 2 * one_look
