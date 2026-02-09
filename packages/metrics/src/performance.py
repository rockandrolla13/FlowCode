from __future__ import annotations

"""Performance metrics.

This module provides functions for computing risk-adjusted
performance metrics like Sharpe, Sortino, and Calmar ratios.

All metrics are annualized by default assuming 252 trading days.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized return from a series of periodic returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns (e.g., daily).
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Annualized return.

    Examples
    --------
    >>> returns = pd.Series([0.001, 0.002, -0.001, 0.003])
    >>> annualized_return(returns)
    0.315  # Approximately
    """
    if len(returns) == 0:
        return np.nan

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if total_return <= -1.0:
        logger.warning(
            "annualized_return: total return <= -100%% (%.4f), returning NaN",
            total_return,
        )
        return np.nan

    if years <= 0:
        logger.warning(
            "annualized_return: zero periods (years=%.4f), returning NaN",
            years,
        )
        return np.nan

    annualized = (1 + total_return) ** (1 / years) - 1
    return float(annualized)


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Sharpe ratio measures excess return per unit of total risk.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns (e.g., daily).
    risk_free : float, default 0.0
        Risk-free rate (same frequency as returns).
    periods_per_year : int, default 252
        Number of periods per year for annualization.

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Examples
    --------
    >>> returns = pd.Series([0.001, 0.002, -0.001, 0.003, 0.001])
    >>> sharpe_ratio(returns)
    2.5  # Approximately

    Notes
    -----
    Sharpe = (mean(r) - rf) / std(r) * sqrt(periods_per_year)
    """
    if len(returns) < 2:
        return np.nan

    excess_returns = returns - risk_free
    mean_excess = excess_returns.mean()
    std = excess_returns.std(ddof=1)

    if std == 0 or np.isnan(std):
        return np.nan

    sharpe = (mean_excess / std) * np.sqrt(periods_per_year)
    return float(sharpe)


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio.

    Sortino ratio uses downside deviation instead of total volatility,
    penalizing only negative returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    risk_free : float, default 0.0
        Risk-free rate (same frequency as returns).
    periods_per_year : int, default 252
        Number of periods per year.
    target_return : float, default 0.0
        Target return for downside calculation.

    Returns
    -------
    float
        Annualized Sortino ratio.

    Notes
    -----
    Sortino = (mean(r) - rf) / downside_deviation * sqrt(periods_per_year)

    Downside deviation is the root-mean-square of deviations below
    target_return (not sample std). Requires >= 2 downside observations.
    """
    if len(returns) < 2:
        return np.nan

    excess_returns = returns - risk_free
    mean_excess = excess_returns.mean()

    # Downside deviation
    downside_returns = returns[returns < target_return]
    if len(downside_returns) < 2:
        return np.nan

    downside_std = np.sqrt(((downside_returns - target_return) ** 2).mean())

    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Calmar ratio.

    Calmar ratio is annualized return divided by maximum drawdown.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Calmar ratio.

    Notes
    -----
    Calmar = annualized_return / |max_drawdown|
    """
    from .risk import max_drawdown as compute_max_drawdown

    ann_ret = annualized_return(returns, periods_per_year)
    max_dd = compute_max_drawdown(returns)

    if max_dd == 0 or np.isnan(max_dd):
        return np.nan

    calmar = ann_ret / abs(max_dd)
    return float(calmar)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Information Ratio.

    IR measures excess return over benchmark per unit of tracking error.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    benchmark_returns : pd.Series
        Benchmark returns.
    periods_per_year : int, default 252
        Number of periods per year.

    Returns
    -------
    float
        Annualized Information Ratio.
    """
    if len(returns) < 2:
        return np.nan

    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan

    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = excess.std(ddof=1)

    if tracking_error == 0 or np.isnan(tracking_error):
        return np.nan

    ir = (excess.mean() / tracking_error) * np.sqrt(periods_per_year)
    return float(ir)
