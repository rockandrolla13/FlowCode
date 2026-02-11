"""Spread transforms for credit instruments.

Converts OAS/Z/ASW spreads to basis-point changes, duration-based
return proxies, and DV01-based P&L.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def delta_spread_bp(spread: pd.Series) -> pd.Series:
    """Compute period-over-period spread change in basis points.

    Parameters
    ----------
    spread : pd.Series
        Spread series in decimal (e.g. 0.0150 = 150bp).

    Returns
    -------
    pd.Series
        Spread change in bp: (s_t - s_{t-1}) * 10_000.
    """
    return spread.diff() * 1e4


def spread_return_proxy(
    spread: pd.Series,
    duration: pd.Series,
) -> pd.Series:
    """Duration-based spread-return proxy.

    Approximates price return from spread move:
    r^spread_t ~ -D_t * delta_s_t (decimal).

    Parameters
    ----------
    spread : pd.Series
        Spread in decimal.
    duration : pd.Series
        Modified/effective duration.

    Returns
    -------
    pd.Series
        Approximate return from spread change.
    """
    delta_s = spread.diff()
    result = -duration * delta_s
    if result.notna().sum() == 0 and len(spread) > 1:
        logger.warning(
            "spread_return_proxy: result is all-NaN; check index alignment "
            "between spread (len=%d) and duration (len=%d)",
            len(spread), len(duration),
        )
    return result


def dv01_pnl(
    delta_bp: pd.Series,
    dv01: pd.Series,
) -> pd.Series:
    """DV01-based spread P&L proxy.

    PnL^spread_t ~ -DV01_t * delta_s^bp_t.

    Parameters
    ----------
    delta_bp : pd.Series
        Spread change in basis points.
    dv01 : pd.Series
        Dollar value of 1bp (DV01).

    Returns
    -------
    pd.Series
        P&L from spread move.
    """
    result = -dv01 * delta_bp
    if result.notna().sum() == 0 and len(delta_bp) > 0:
        logger.warning(
            "dv01_pnl: result is all-NaN; check index alignment "
            "between delta_bp (len=%d) and dv01 (len=%d)",
            len(delta_bp), len(dv01),
        )
    return result
