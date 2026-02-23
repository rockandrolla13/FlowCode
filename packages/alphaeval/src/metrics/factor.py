"""Cross-sectional factor quality metrics.

Computed daily across instruments, then averaged over days.
IC*, RankIC*, IR*, R², and t-stat of daily IC.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ic_star", "rank_ic_star", "ir_star", "r_squared", "tstat_ic"]


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
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got '{method}'")

    # If both are (date x instrument) pivoted DataFrames (any index type)
    if not isinstance(signal.index, pd.MultiIndex) and signal.ndim == 2:
        # Transpose: each date becomes a column, instruments are rows
        s_t = signal.T
        t_t = target.T

        # Count common non-NaN observations per date
        n_common = (s_t.notna() & t_t.notna()).sum()

        # Vectorized correlation across all dates
        if method == "spearman":
            ics = s_t.rank().corrwith(t_t.rank())
        else:
            ics = s_t.corrwith(t_t)

        # Mask dates with < 3 common instruments
        ics = ics.where(n_common >= 3)
        ics.name = "ic"

        n_skipped = int((n_common < 3).sum())
        n_dates = len(signal.index)
        if n_skipped > 0 and n_dates > 0:
            pct = 100 * n_skipped / n_dates
            if pct > 20:
                logger.warning(
                    "_daily_ic: %d of %d dates (%.0f%%) skipped (<3 instruments)",
                    n_skipped, n_dates, pct,
                )
        return ics

    # MultiIndex (date, instrument) → pivot first
    if isinstance(signal.index, pd.MultiIndex):
        if signal.ndim == 2 and signal.shape[1] != 1:
            raise ValueError(
                f"MultiIndex signal must have exactly 1 column, got {signal.shape[1]}. "
                "Select the target column before calling _daily_ic."
            )
        if target.ndim == 2 and target.shape[1] != 1:
            raise ValueError(
                f"MultiIndex target must have exactly 1 column, got {target.shape[1]}. "
                "Select the target column before calling _daily_ic."
            )
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
    signal: pd.DataFrame | None = None,
    target: pd.DataFrame | None = None,
    *,
    ics: pd.Series | None = None,
) -> float:
    """IC Information Ratio — IC* / std(IC).

    Parameters
    ----------
    signal : pd.DataFrame | None
        Factor signal, pivoted (date x instrument). Required if ics is None.
    target : pd.DataFrame | None
        Next-period outcome, same shape. Required if ics is None.
    ics : pd.Series | None
        Pre-computed daily IC series. When provided, signal/target are ignored.

    Returns
    -------
    float
        IR* = mean(IC_t) / std(IC_t). NaN if < 2 days or std=0.
    """
    if ics is None:
        if signal is None or target is None:
            raise ValueError("ir_star: provide (signal, target) or ics")
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

    Unlike ic_star/rank_ic_star/ir_star (which take pivoted DataFrames),
    this function operates on flat predicted/actual vectors. Extract the
    relevant column from your panel before calling.

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
