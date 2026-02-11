"""Cross-sectional factor quality metrics.

Computed daily across instruments, then averaged over days.
IC*, RankIC*, IR*, R², and t-stat of daily IC.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _daily_ic(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """Compute daily cross-sectional correlation.

    Parameters
    ----------
    signal : pd.DataFrame
        Signal values, indexed by (date, instrument) or pivoted (date x instrument).
    target : pd.DataFrame
        Target values, same shape as signal.
    method : {"pearson", "spearman"}
        Correlation method.

    Returns
    -------
    pd.Series
        Daily IC values indexed by date.
    """
    # If both are (date x instrument) pivoted DataFrames
    if isinstance(signal.index, pd.DatetimeIndex) and signal.ndim == 2:
        dates = signal.index
        ics = []
        for dt in dates:
            s = signal.loc[dt].dropna()
            t = target.loc[dt].dropna()
            common = s.index.intersection(t.index)
            if len(common) < 3:
                logger.debug("IC: date %s has %d instruments (<3), skipping", dt, len(common))
                ics.append(np.nan)
                continue
            if method == "spearman":
                ics.append(float(s[common].rank().corr(t[common].rank())))
            else:
                ics.append(float(s[common].corr(t[common])))
        return pd.Series(ics, index=dates, name="ic")

    # MultiIndex (date, instrument) → pivot first
    if isinstance(signal.index, pd.MultiIndex):
        sig_piv = signal.iloc[:, 0].unstack() if signal.ndim == 2 else signal.unstack()
        tgt_piv = target.iloc[:, 0].unstack() if target.ndim == 2 else target.unstack()
        return _daily_ic(sig_piv, tgt_piv, method=method)

    raise ValueError("signal must be pivoted (date x instrument) or MultiIndex (date, instrument)")


def ic_star(
    signal: pd.DataFrame,
    target: pd.DataFrame,
) -> float:
    """Information Coefficient (IC*) — mean daily Pearson correlation.

    Parameters
    ----------
    signal : pd.DataFrame
        Factor signal, pivoted (date x instrument).
    target : pd.DataFrame
        Next-period outcome, same shape.

    Returns
    -------
    float
        IC* = E_t[IC_t]. NaN if no valid days.
    """
    ics = _daily_ic(signal, target, method="pearson")
    clean = ics.dropna()
    if len(clean) == 0:
        return np.nan
    return float(clean.mean())


def rank_ic_star(
    signal: pd.DataFrame,
    target: pd.DataFrame,
) -> float:
    """Rank IC* — mean daily Spearman correlation.

    Parameters
    ----------
    signal : pd.DataFrame
        Factor signal, pivoted (date x instrument).
    target : pd.DataFrame
        Next-period outcome, same shape.

    Returns
    -------
    float
        RankIC* = E_t[RankIC_t]. NaN if no valid days.
    """
    ics = _daily_ic(signal, target, method="spearman")
    clean = ics.dropna()
    if len(clean) == 0:
        return np.nan
    return float(clean.mean())


def ir_star(
    signal: pd.DataFrame,
    target: pd.DataFrame,
) -> float:
    """IC Information Ratio — IC* / std(IC).

    Parameters
    ----------
    signal : pd.DataFrame
        Factor signal, pivoted (date x instrument).
    target : pd.DataFrame
        Next-period outcome, same shape.

    Returns
    -------
    float
        IR* = mean(IC_t) / std(IC_t). NaN if < 2 days or std=0.
    """
    ics = _daily_ic(signal, target, method="pearson")
    clean = ics.dropna()
    if len(clean) < 2:
        return np.nan
    sigma = clean.std(ddof=1)
    if abs(sigma) < 1e-14:
        return np.nan
    return float(clean.mean() / sigma)


def r_squared(
    predicted: pd.Series,
    actual: pd.Series,
) -> float:
    """Coefficient of determination R².

    Parameters
    ----------
    predicted : pd.Series
        Model predictions.
    actual : pd.Series
        Actual outcomes.

    Returns
    -------
    float
        R² = 1 - SSE/SST. Can be negative for poor models.
    """
    common = predicted.dropna().index.intersection(actual.dropna().index)
    if len(common) < 2:
        return np.nan
    y = actual[common]
    yhat = predicted[common]
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if abs(ss_tot) < 1e-14:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def tstat_ic(ic_series: pd.Series) -> float:
    """t-statistic for H0: mean(IC) = 0.

    Parameters
    ----------
    ic_series : pd.Series
        Daily IC values.

    Returns
    -------
    float
        t = mean(IC) / (std(IC) / sqrt(T_d)). NaN if < 2 days or std=0.
    """
    clean = ic_series.dropna()
    n = len(clean)
    if n < 2:
        return np.nan
    mu = clean.mean()
    sigma = clean.std(ddof=1)
    if abs(sigma) < 1e-14:
        return np.nan
    return float(mu / (sigma / np.sqrt(n)))
