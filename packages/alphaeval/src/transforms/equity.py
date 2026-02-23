"""Equity curve analytics — drawdown and run-up series.

Drawdown measures peak-to-current decline; run-up measures
trough-to-current rise.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .returns import equity_curve

logger = logging.getLogger(__name__)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute running drawdown series.

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    pd.Series
        Drawdown DD_t = (P_t - E_t) / P_t where P_t = running peak.
        Values >= 0 for normal equity curves; can exceed 1.0 for leveraged
        returns below -100%. NaN where peak is zero.
    """
    eq = equity_curve(returns)
    peak = eq.cummax()
    dd = (peak - eq) / peak
    n_inf = int(np.isinf(dd).sum())
    if n_inf > 0:
        logger.warning(
            "drawdown_series: %d inf values (zero peak equity?); replacing with NaN",
            n_inf,
        )
    dd = dd.replace([np.inf, -np.inf], np.nan)
    return dd


def runup_series(returns: pd.Series) -> pd.Series:
    """Compute running run-up series.

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    pd.Series
        Run-up RU_t = (E_t - Q_t) / Q_t where Q_t = running trough.
        Values >= 0 for normal equity curves. NaN where trough is zero.
    """
    eq = equity_curve(returns)
    trough = eq.cummin()
    ru = (eq - trough) / trough
    n_inf = int(np.isinf(ru).sum())
    if n_inf > 0:
        logger.warning(
            "runup_series: %d inf values (zero trough equity?); replacing with NaN",
            n_inf,
        )
    ru = ru.replace([np.inf, -np.inf], np.nan)
    return ru
