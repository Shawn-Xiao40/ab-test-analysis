"""Statistics for analysing a two-variant conversion experiment.

Everything is implemented with numpy / scipy only, so each formula is visible
and unit-tested instead of hidden behind a library call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- design checks

def srm_test(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> tuple[float, float]:
    """Sample ratio mismatch check (chi-square goodness of fit).

    Returns (chi2, p_value). A very small p-value (commonly < 0.001) means the
    observed split is unlikely under the intended allocation, which points to a
    bug in assignment or logging; the experiment results should not be trusted.
    """
    total = n_control + n_treatment
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return float(chi2), float(p)


# --------------------------------------------------------------------------- power

def sample_size_per_group(p_base: float, mde_abs: float, alpha: float = 0.05,
                          power: float = 0.8, two_sided: bool = True) -> int:
    """Users needed in EACH group to detect an absolute lift of `mde_abs`."""
    p2 = p_base + mde_abs
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    p_bar = (p_base + p2) / 2
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
          + z_beta * np.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2) / mde_abs ** 2
    return int(np.ceil(n))


def power_of_test(p_base: float, diff_abs: float, n_per_group: int,
                  alpha: float = 0.05, two_sided: bool = True) -> float:
    """Probability of detecting a true absolute difference `diff_abs`."""
    p2 = p_base + diff_abs
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    p_bar = (p_base + p2) / 2
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se1 = np.sqrt((p_base * (1 - p_base) + p2 * (1 - p2)) / n_per_group)
    return float(stats.norm.cdf((abs(diff_abs) - z_alpha * se0) / se1))


def minimum_detectable_effect(p_base: float, n_per_group: int, alpha: float = 0.05,
                              power: float = 0.8) -> float:
    """Smallest absolute lift detectable with the given sample size (bisection)."""
    lo, hi = 1e-6, 1 - p_base - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_of_test(p_base, mid, n_per_group, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- frequentist test

@dataclass(frozen=True)
class ProportionTest:
    rate_control: float
    rate_treatment: float
    diff_abs: float          # treatment - control
    lift_rel: float          # diff_abs / rate_control
    ci_low: float            # confidence interval for diff_abs
    ci_high: float
    z: float
    p_value: float


def two_proportion_ztest(x_control: int, n_control: int, x_treatment: int, n_treatment: int,
                         alpha: float = 0.05, alternative: str = "two-sided") -> ProportionTest:
    """Two-proportion z-test.

    The test statistic uses the pooled standard error (valid under H0: equal rates);
    the confidence interval uses the unpooled standard error.
    alternative: "two-sided", "larger" (treatment > control) or "smaller".
    """
    p_c, p_t = x_control / n_control, x_treatment / n_treatment
    diff = p_t - p_c
    pooled = (x_control + x_treatment) / (n_control + n_treatment)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    z = diff / se_pooled
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "larger":
        p = stats.norm.sf(z)
    elif alternative == "smaller":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'larger' or 'smaller'")
    se = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    half = stats.norm.ppf(1 - alpha / 2) * se
    return ProportionTest(p_c, p_t, diff, diff / p_c, diff - half, diff + half, float(z), float(p))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


# --------------------------------------------------------------------------- Bayesian view

@dataclass(frozen=True)
class BayesResult:
    prob_treatment_better: float
    expected_loss_treatment: float   # expected conversion lost if we ship treatment and it is worse
    expected_loss_control: float
    lift_ci_low: float               # 95% credible interval for the relative lift
    lift_ci_high: float


def bayesian_ab(x_control: int, n_control: int, x_treatment: int, n_treatment: int,
                prior_alpha: float = 1.0, prior_beta: float = 1.0,
                draws: int = 200_000, seed: int = 0) -> BayesResult:
    """Beta-Binomial model with a Beta(prior_alpha, prior_beta) prior on each rate."""
    rng = np.random.default_rng(seed)
    c = rng.beta(prior_alpha + x_control, prior_beta + n_control - x_control, draws)
    t = rng.beta(prior_alpha + x_treatment, prior_beta + n_treatment - x_treatment, draws)
    lift = (t - c) / c
    lo, hi = np.percentile(lift, [2.5, 97.5])
    return BayesResult(
        prob_treatment_better=float((t > c).mean()),
        expected_loss_treatment=float(np.maximum(c - t, 0).mean()),
        expected_loss_control=float(np.maximum(t - c, 0).mean()),
        lift_ci_low=float(lo), lift_ci_high=float(hi),
    )


# --------------------------------------------------------------------------- peeking

def peeking_false_positive_rate(p_true: float, n_per_group: int, n_looks: int,
                                alpha: float = 0.05, n_sims: int = 2000, seed: int = 0) -> float:
    """Simulate A/A tests and stop at the first 'significant' interim look.

    Both groups share the same true rate, so every rejection is a false positive.
    Returns the share of simulations that were (wrongly) declared significant at
    any of `n_looks` equally spaced looks. With one look this is ~alpha.
    """
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_group / n_looks, n_per_group, n_looks).astype(int)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    # Binomial increments between checkpoints are equivalent to cumulative sums of Bernoulli draws.
    steps = np.diff(np.concatenate([[0], checkpoints]))
    a = np.cumsum(rng.binomial(steps, p_true, size=(n_sims, n_looks)), axis=1)
    b = np.cumsum(rng.binomial(steps, p_true, size=(n_sims, n_looks)), axis=1)
    pooled = (a + b) / (2 * checkpoints)
    se = np.sqrt(pooled * (1 - pooled) * 2 / checkpoints)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (b - a) / checkpoints / se, 0.0)
    return float((np.abs(z) > z_crit).any(axis=1).mean())
