"""Sharpe inference — PSR, DSR, minTRL, expected max SR.

Addresses non-normality (skew/kurtosis) and selection bias after
running many trials. Based on Bailey & López de Prado (2012, 2014).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def estimated_sharpe_ratio(returns: pd.Series) -> float:
    """Estimated Sharpe ratio (risk-free = 0).

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    float
        SR_hat = mean / std(ddof=1). NaN if < 2 obs or std = 0.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    sigma = clean.std(ddof=1)
    if abs(sigma) < 1e-14:
        return np.nan
    return float(clean.mean() / sigma)


def ann_estimated_sharpe_ratio(
    returns: pd.Series | None = None,
    periods: int = 252,
    *,
    sr: float | None = None,
) -> float:
    """Annualized estimated Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series | None
        Period returns. Ignored if sr provided.
    periods : int, default 252
        Periods per year.
    sr : float | None
        Pre-computed SR. Computed from returns if None.

    Returns
    -------
    float
        SR_ann = sqrt(A) * SR.
    """
    if sr is None:
        if returns is None:
            return np.nan
        sr = estimated_sharpe_ratio(returns)
    if np.isnan(sr):
        return np.nan
    return float(sr * np.sqrt(periods))


def estimated_sharpe_ratio_stdev(
    returns: pd.Series,
    *,
    sr: float | None = None,
) -> float:
    """Standard deviation of SR estimate (non-normal adjustment).

    Generalizes for skew and kurtosis:
    sigma_SR = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / (n-1))

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    sr : float | None
        Pre-computed SR. Computed from returns if None.

    Returns
    -------
    float
        Standard deviation of SR estimate. NaN if < 3 obs.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 3:
        return np.nan
    if sr is None:
        sr = estimated_sharpe_ratio(clean)
    if np.isnan(sr):
        return np.nan

    skew = float(scipy_stats.skew(clean))
    # fisher=False → non-excess kurtosis (normal = 3)
    kurt = float(scipy_stats.kurtosis(clean, fisher=False))
    if np.isnan(skew) or np.isnan(kurt):
        logger.warning("SR stdev: skew/kurtosis NaN (zero-variance input?)")
        return np.nan

    numerator = 1.0 + 0.5 * sr**2 - skew * sr + ((kurt - 3) / 4) * sr**2
    if numerator < 0:
        logger.warning("SR stdev numerator < 0 (%.4f), returning NaN", numerator)
        return np.nan
    return float(np.sqrt(numerator / (n - 1)))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    *,
    sr: float | None = None,
    sr_std: float | None = None,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    PSR = Phi((SR_hat - SR*) / sigma_SR).
    Probability that true SR exceeds benchmark.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    sr_benchmark : float, default 0.0
        Benchmark SR (SR*).
    sr : float | None
        Pre-computed SR.
    sr_std : float | None
        Pre-computed SR stdev.

    Returns
    -------
    float
        PSR in [0, 1]. NaN if inputs insufficient.
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if np.isnan(sr):
        return np.nan
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)
    if np.isnan(sr_std) or abs(sr_std) < 1e-14:
        return np.nan
    return float(scipy_stats.norm.cdf((sr - sr_benchmark) / sr_std))


def min_track_record_length(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    prob: float = 0.95,
    *,
    sr: float | None = None,
    sr_std: float | None = None,
) -> float:
    """Minimum Track Record Length (minTRL).

    Minimum number of observations needed for PSR >= prob.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    sr_benchmark : float, default 0.0
        Benchmark SR.
    prob : float, default 0.95
        Required confidence.
    sr : float | None
        Pre-computed SR.
    sr_std : float | None
        Pre-computed SR stdev.

    Returns
    -------
    float
        minTRL in observations. inf if SR <= SR*. NaN if inputs insufficient.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 3:
        return np.nan
    if sr is None:
        sr = estimated_sharpe_ratio(clean)
    if np.isnan(sr):
        return np.nan
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(clean, sr=sr)
    if np.isnan(sr_std):
        return np.nan

    denom = sr - sr_benchmark
    if denom <= 0:
        return np.inf

    z = scipy_stats.norm.ppf(prob)
    return float(1 + sr_std**2 * (n - 1) * (z / denom) ** 2)


def num_independent_trials(
    trials_returns: pd.DataFrame | None = None,
    *,
    m: int | None = None,
    avg_corr: float | None = None,
) -> int:
    """Effective number of independent trials.

    N_eff ~ rho_bar + (1 - rho_bar) * m.

    Parameters
    ----------
    trials_returns : pd.DataFrame | None
        Each column is one trial's returns. Used to compute m and avg_corr.
    m : int | None
        Number of trials (columns). Computed from trials_returns if None.
    avg_corr : float | None
        Average pairwise correlation. Computed if None.

    Returns
    -------
    int
        Effective independent trials (rounded up).

    Notes
    -----
    Uses the linear interpolation variant ``rho + (1 - rho) * m`` from the
    reference implementation. The alternative harmonic variant
    ``m / (1 + (m-1) * rho)`` gives smaller N_eff at moderate correlations
    and is more conservative for DSR. Choose based on your use case.
    """
    if m is None:
        if trials_returns is None:
            raise ValueError("Provide trials_returns or m + avg_corr")
        m = trials_returns.shape[1]
    if avg_corr is None:
        if trials_returns is None:
            raise ValueError("Provide trials_returns or avg_corr")
        corr = trials_returns.corr().values
        upper = corr[np.triu_indices_from(corr, k=1)]
        avg_corr = float(upper.mean()) if len(upper) > 0 else 0.0

    if np.isnan(avg_corr):
        logger.warning("num_independent_trials: avg_corr is NaN; defaulting to 0.0")
        avg_corr = 0.0

    n_eff = avg_corr + (1 - avg_corr) * m
    return int(np.ceil(max(1, n_eff)))


def expected_maximum_sr(
    trials_returns: pd.DataFrame | None = None,
    expected_mean_sr: float = 0.0,
    *,
    independent_trials: int | None = None,
    trials_sr_std: float | None = None,
) -> float:
    """Expected maximum Sharpe ratio after search.

    Uses Euler-Mascheroni approximation for order statistics.

    Parameters
    ----------
    trials_returns : pd.DataFrame | None
        Each column is one trial's returns.
    expected_mean_sr : float, default 0.0
        Mean SR across trials.
    independent_trials : int | None
        N_eff. Computed from trials_returns if None.
    trials_sr_std : float | None
        Std of SR across trials. Computed if None.

    Returns
    -------
    float
        E[max SR].
    """
    EMC = 0.5772156649  # Euler-Mascheroni constant

    if independent_trials is None:
        if trials_returns is None:
            raise ValueError("Provide trials_returns or independent_trials")
        independent_trials = num_independent_trials(trials_returns)

    if trials_sr_std is None:
        if trials_returns is None:
            raise ValueError("Provide trials_returns or trials_sr_std")
        srs = trials_returns.apply(estimated_sharpe_ratio)
        n_valid = srs.notna().sum()
        if n_valid < 2:
            logger.warning(
                "expected_maximum_sr: %d valid trial SRs out of %d; returning NaN",
                n_valid, len(srs),
            )
            return np.nan
        trials_sr_std = float(srs.std(ddof=1))

    if np.isnan(trials_sr_std):
        return np.nan

    if independent_trials <= 1:
        return expected_mean_sr

    z_max = (
        (1 - EMC) * scipy_stats.norm.ppf(1 - 1.0 / independent_trials)
        + EMC * scipy_stats.norm.ppf(1 - 1.0 / (independent_trials * np.e))
    )
    return float(expected_mean_sr + trials_sr_std * z_max)


def deflated_sharpe_ratio(
    trials_returns: pd.DataFrame | None = None,
    returns_selected: pd.Series | None = None,
    expected_mean_sr: float = 0.0,
    *,
    expected_max_sr: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (DSR).

    DSR = PSR with benchmark = E[max SR] instead of SR*.
    Accounts for selection bias from running many trials.

    Parameters
    ----------
    trials_returns : pd.DataFrame | None
        All trial returns (for computing expected max SR).
    returns_selected : pd.Series | None
        Returns of the selected (best) strategy.
    expected_mean_sr : float, default 0.0
        Mean SR across all trials.
    expected_max_sr : float | None
        Pre-computed E[max SR]. Computed from trials_returns if None.

    Returns
    -------
    float
        DSR in [0, 1]. NaN if inputs insufficient.
    """
    if returns_selected is None:
        return np.nan
    if expected_max_sr is None:
        if trials_returns is None:
            raise ValueError("Provide trials_returns or expected_max_sr")
        expected_max_sr = expected_maximum_sr(trials_returns, expected_mean_sr)

    return probabilistic_sharpe_ratio(
        returns_selected, sr_benchmark=expected_max_sr
    )
