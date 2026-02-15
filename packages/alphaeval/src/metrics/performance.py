"""Performance metrics — trade-level and return-level.

Implements: profit_factor, win_rate_trades, expectancy, max_runup,
cagr, tstat_returns, sortino_ratio (1/T downside convention).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..transforms.equity import runup_series
from ..transforms.returns import equity_curve

logger = logging.getLogger(__name__)


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss ratio.

    Parameters
    ----------
    trade_pnls : pd.Series
        Individual trade P&L values.

    Returns
    -------
    float
        Profit factor. inf if no losing trades (but gains exist);
        NaN if empty; 0.0 if no winning trades.
    """
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = trade_pnls[trade_pnls < 0].abs().sum()
    if abs(losses) < 1e-14:
        if gains > 0:
            return np.inf
        return np.nan
    return float(gains / losses)


def win_rate_trades(trade_pnls: pd.Series) -> float:
    """Percentage of winning trades.

    Parameters
    ----------
    trade_pnls : pd.Series
        Individual trade P&L values.

    Returns
    -------
    float
        Win rate as percentage [0, 100]. NaN if no trades.
    """
    clean = trade_pnls.dropna()
    n = len(clean)
    if n == 0:
        return np.nan
    winners = (clean > 0).sum()
    return float(winners / n * 100)


def expectancy(trade_pnls: pd.Series) -> float:
    """Expected value per trade.

    Parameters
    ----------
    trade_pnls : pd.Series
        Individual trade P&L values.

    Returns
    -------
    float
        Mean P&L per trade. NaN if no trades.
    """
    clean = trade_pnls.dropna()
    if len(clean) == 0:
        return np.nan
    return float(clean.mean())


def max_runup(returns: pd.Series) -> float:
    """Maximum run-up (trough-to-peak).

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    float
        Max run-up as fraction. 0.0 if monotonically declining.
    """
    if len(returns.dropna()) < 1:
        return np.nan
    ru = runup_series(returns)
    return float(ru.max())


def cagr(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compound Annual Growth Rate.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    periods_per_year : int, default 252
        Trading periods per year.

    Returns
    -------
    float
        CAGR = E_T^(1/Y) - 1 where E_0=1.0 (unit initial equity).
        Returns -1.0 if terminal equity <= 0. NaN if empty.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 1:
        return np.nan
    eq = equity_curve(clean)
    terminal = eq.iloc[-1]
    if terminal <= 0:
        return -1.0
    years = n / periods_per_year
    if abs(years) < 1e-14:
        return np.nan
    return float(terminal ** (1.0 / years) - 1.0)


def tstat_returns(returns: pd.Series) -> float:
    """t-statistic for H0: mean(returns) = 0.

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    float
        t = mean / (std / sqrt(n)). NaN if n < 2 or std = 0.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 2:
        return np.nan
    mu = clean.mean()
    sigma = clean.std(ddof=1)
    if abs(sigma) < 1e-14:
        return np.nan
    return float(mu / (sigma / np.sqrt(n)))


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sortino ratio with population downside deviation (1/T).

    Uses spec convention: downside_std = sqrt(1/T * sum(min(0, r-tau)^2)).
    This differs from packages/metrics which uses ddof=1.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    target : float, default 0.0
        Minimum acceptable return (MAR) per period.
    periods_per_year : int, default 252
        For annualization.

    Returns
    -------
    float
        Annualized Sortino ratio. NaN if insufficient data or zero downside.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 2:
        return np.nan
    excess = clean - target
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt((downside ** 2).sum() / n)  # 1/T (population)
    if abs(downside_std) < 1e-14:
        return np.nan
    mu = excess.mean()
    return float(np.sqrt(periods_per_year) * mu / downside_std)
