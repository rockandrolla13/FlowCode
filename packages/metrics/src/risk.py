from __future__ import annotations

"""Risk metrics.

This module provides functions for computing risk metrics
like drawdown, VaR, and expected shortfall.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute running drawdown series.

    Drawdown at each point is the decline from the peak cumulative return.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.

    Returns
    -------
    pd.Series
        Drawdown series (negative values, 0 at peaks).

    Examples
    --------
    >>> returns = pd.Series([0.10, 0.05, -0.15, 0.03])
    >>> drawdown_series(returns)
    0    0.000000
    1    0.000000
    2   -0.130435
    3   -0.104478
    dtype: float64

    Notes
    -----
    Drawdown is always <= 0. A value of -0.10 means 10% below peak.
    """
    if len(returns) == 0:
        return pd.Series(dtype=float)

    # Cumulative returns (wealth index)
    cum_returns = (1 + returns).cumprod()

    # Running maximum
    running_max = cum_returns.cummax()

    # Drawdown
    drawdown = (cum_returns - running_max) / running_max

    return drawdown


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown.

    Maximum drawdown is the largest peak-to-trough decline.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.

    Returns
    -------
    float
        Maximum drawdown (negative value).

    Examples
    --------
    >>> returns = pd.Series([0.10, 0.05, -0.15, 0.03])
    >>> max_drawdown(returns)
    -0.130435  # Approximately

    Notes
    -----
    Returns negative value. A return of -0.20 means 20% max drawdown.
    """
    if len(returns) == 0:
        return np.nan

    dd = drawdown_series(returns)
    return float(dd.min())


def drawdown_duration(returns: pd.Series) -> pd.Series:
    """
    Compute drawdown duration (time since last peak).

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.

    Returns
    -------
    pd.Series
        Number of periods since last peak (0 at peaks).
    """
    if len(returns) == 0:
        return pd.Series(dtype=int)

    dd = drawdown_series(returns)
    is_peak = dd == 0

    # Count periods since last peak
    groups = is_peak.cumsum()
    duration = dd.groupby(groups).cumcount()

    return duration


def max_drawdown_duration(returns: pd.Series) -> int:
    """
    Compute maximum drawdown duration.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.

    Returns
    -------
    int
        Maximum number of periods in drawdown.
    """
    if len(returns) == 0:
        return 0

    duration = drawdown_duration(returns)
    return int(duration.max())


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Compute Value at Risk (VaR).

    Spec §4.2: VaR_α = -quantile(returns, α)

    VaR is the maximum expected loss at a given confidence level,
    expressed as a positive loss magnitude.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    confidence : float, default 0.95
        Confidence level (e.g., 0.95 for 95% VaR).
    method : str, default "historical"
        Method for VaR calculation. Currently only "historical" supported.

    Returns
    -------
    float
        VaR as positive loss magnitude.

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000) * 0.01)
    >>> value_at_risk(returns, confidence=0.95)
    0.0165  # Approximately

    Notes
    -----
    95% VaR of 0.02 means: "We expect losses to exceed 2% only 5% of the time."
    """
    if len(returns) == 0:
        return np.nan

    if method == "historical":
        var = -returns.quantile(1 - confidence)
    else:
        raise ValueError(f"Unknown VaR method: {method}")

    return float(var)


def expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Compute Expected Shortfall (Conditional VaR).

    Spec §4.3: ES_α = -mean(returns where returns ≤ -VaR_α)

    ES is the expected loss given that loss exceeds VaR,
    expressed as a positive loss magnitude.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        Expected shortfall as positive loss magnitude (ES >= VaR).

    Notes
    -----
    Also known as CVaR or Average VaR. ES >= VaR.
    """
    if len(returns) == 0:
        return np.nan

    var = value_at_risk(returns, confidence)
    # VaR is positive; tail is returns <= -VaR (the negative quantile)
    tail_returns = returns[returns <= -var]

    if len(tail_returns) == 0:
        logger.warning(
            "expected_shortfall: no returns in tail at confidence=%.2f, "
            "returning VaR as fallback", confidence
        )
        return var

    return float(-tail_returns.mean())


def volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """
    Compute annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    periods_per_year : int, default 252
        Periods per year for annualization.
    ddof : int, default 1
        Degrees of freedom.

    Returns
    -------
    float
        Annualized volatility.
    """
    if len(returns) < 2:
        return np.nan

    return float(returns.std(ddof=ddof) * np.sqrt(periods_per_year))


def downside_volatility(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized downside volatility.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    target : float, default 0.0
        Target return threshold.
    periods_per_year : int, default 252
        Periods per year.

    Returns
    -------
    float
        Annualized downside volatility.
    """
    if len(returns) < 2:
        return np.nan

    downside = returns[returns < target]
    if len(downside) < 2:
        return np.nan

    downside_std = np.sqrt(((downside - target) ** 2).mean())
    return float(downside_std * np.sqrt(periods_per_year))
