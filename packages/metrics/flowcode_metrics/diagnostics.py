from __future__ import annotations

"""Diagnostic metrics for signal analysis.

This module provides functions for analyzing signal quality,
including hit rate, autocorrelation, and signal decay.
"""

import numpy as np
import pandas as pd


def hit_rate(
    signals: pd.Series,
    returns: pd.Series,
    lag: int = 1,
) -> float:
    """
    Compute hit rate (accuracy) of signals.

    Hit rate is the fraction of times the signal correctly predicts
    the direction of future returns.

    Parameters
    ----------
    signals : pd.Series
        Signal values (positive = long, negative = short).
    returns : pd.Series
        Actual returns to predict.
    lag : int, default 1
        Number of periods between signal and return.

    Returns
    -------
    float
        Hit rate between 0 and 1.

    Examples
    --------
    >>> signals = pd.Series([1, -1, 1, -1, 1])
    >>> returns = pd.Series([0.01, -0.01, 0.02, -0.01, 0.01])
    >>> hit_rate(signals, returns, lag=0)
    1.0  # Perfect accuracy

    Notes
    -----
    A hit rate of 0.5 indicates no predictive power (random).
    """
    if len(signals) == 0 or len(returns) == 0:
        return np.nan

    # Align series
    if lag > 0:
        aligned_signals = signals.shift(lag)
    else:
        aligned_signals = signals

    combined = pd.concat([aligned_signals, returns], axis=1).dropna()
    if len(combined) == 0:
        return np.nan

    sig = combined.iloc[:, 0]
    ret = combined.iloc[:, 1]

    # Check if signs match
    correct = (np.sign(sig) == np.sign(ret)).sum()
    total = len(combined)

    return float(correct / total)


def autocorrelation(
    series: pd.Series,
    lag: int = 1,
) -> float:
    """
    Compute autocorrelation at a given lag.

    Parameters
    ----------
    series : pd.Series
        Time series.
    lag : int, default 1
        Lag for autocorrelation.

    Returns
    -------
    float
        Autocorrelation coefficient between -1 and 1.

    Examples
    --------
    >>> series = pd.Series([1, 2, 3, 4, 5])
    >>> autocorrelation(series, lag=1)
    0.9  # Approximately (trending series)
    """
    if len(series) <= lag:
        return np.nan

    return float(series.autocorr(lag=lag))


def autocorrelation_profile(
    series: pd.Series,
    max_lag: int = 20,
) -> pd.Series:
    """
    Compute autocorrelation profile up to max_lag.

    Parameters
    ----------
    series : pd.Series
        Time series.
    max_lag : int, default 20
        Maximum lag to compute.

    Returns
    -------
    pd.Series
        Autocorrelation at each lag (1 to max_lag).
    """
    lags = range(1, max_lag + 1)
    acf = [autocorrelation(series, lag=lag) for lag in lags]
    return pd.Series(acf, index=lags, name="autocorrelation")


def signal_decay(
    signal: pd.Series,
    returns: pd.Series,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute signal predictive power decay over horizons.

    Measures how the signal's predictive power changes as the
    forecast horizon increases.

    Parameters
    ----------
    signal : pd.Series
        Signal values.
    returns : pd.Series
        Return series.
    horizons : list[int] | None, optional
        Forecast horizons to test. Default [1, 5, 10, 21].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: horizon, correlation, hit_rate.

    Examples
    --------
    >>> decay = signal_decay(signal, returns)
    >>> decay
       horizon  correlation  hit_rate
    0        1         0.15      0.55
    1        5         0.10      0.52
    2       10         0.05      0.51
    3       21         0.02      0.50
    """
    if horizons is None:
        horizons = [1, 5, 10, 21]

    results = []
    for h in horizons:
        # Forward returns
        fwd_returns = returns.shift(-h)

        # Align
        combined = pd.concat([signal, fwd_returns], axis=1).dropna()
        if len(combined) < 10:
            results.append({
                "horizon": h,
                "correlation": np.nan,
                "hit_rate": np.nan,
            })
            continue

        sig = combined.iloc[:, 0]
        ret = combined.iloc[:, 1]

        corr = sig.corr(ret)
        hr = (np.sign(sig) == np.sign(ret)).mean()

        results.append({
            "horizon": h,
            "correlation": corr,
            "hit_rate": hr,
        })

    return pd.DataFrame(results)


def information_coefficient(
    signal: pd.Series,
    returns: pd.Series,
    lag: int = 1,
) -> float:
    """
    Compute Information Coefficient (IC).

    IC is the correlation between signal and subsequent returns.

    Parameters
    ----------
    signal : pd.Series
        Signal values.
    returns : pd.Series
        Return series.
    lag : int, default 1
        Periods between signal and return.

    Returns
    -------
    float
        Information coefficient between -1 and 1.

    Notes
    -----
    IC > 0.05 is generally considered meaningful.
    IC > 0.10 is considered strong.
    """
    if len(signal) == 0 or len(returns) == 0:
        return np.nan

    # Shift returns back (or signal forward)
    aligned_returns = returns.shift(-lag)

    combined = pd.concat([signal, aligned_returns], axis=1).dropna()
    if len(combined) < 10:
        return np.nan

    return float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))


def turnover(
    positions: pd.Series,
) -> float:
    """
    Compute average turnover from position series.

    Parameters
    ----------
    positions : pd.Series
        Position sizes over time.

    Returns
    -------
    float
        Average absolute position change per period.
    """
    if len(positions) < 2:
        return np.nan

    changes = positions.diff().abs()
    return float(changes.mean())


def holding_period(
    positions: pd.Series,
) -> float:
    """
    Compute average holding period.

    Parameters
    ----------
    positions : pd.Series
        Position sizes over time.

    Returns
    -------
    float
        Average number of periods a position is held.
    """
    if len(positions) < 2:
        return np.nan

    # Detect position changes
    sign_changes = np.sign(positions) != np.sign(positions.shift(1))
    n_trades = sign_changes.sum()

    if n_trades == 0:
        return float(len(positions))

    return float(len(positions) / n_trades)
