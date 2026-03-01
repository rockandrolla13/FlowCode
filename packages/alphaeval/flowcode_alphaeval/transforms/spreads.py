"""Spread transforms for credit instruments.

Converts OAS/Z/ASW spreads to basis-point changes, duration-based
return proxies, and DV01-based P&L.
"""
from __future__ import annotations

import logging

import numpy as np
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
    result = spread.diff() * 1e4
    n_inf = int(np.isinf(result).sum())
    if n_inf > 0:
        logger.warning(
            "delta_spread_bp: %d inf values detected; replacing with NaN",
            n_inf,
        )
        result = result.replace([np.inf, -np.inf], np.nan)
    return result


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
    n_total = len(result)
    n_valid = int(result.notna().sum())
    if n_valid == 0 and n_total > 1:
        raise ValueError(
            "spread_return_proxy: result is all-NaN. Check index alignment "
            f"between spread (len={len(spread)}) and duration (len={len(duration)})."
        )
    if n_total > 1 and n_valid < n_total * 0.5:
        logger.warning(
            "spread_return_proxy: %d of %d values are NaN (>50%%); "
            "check index alignment",
            n_total - n_valid, n_total,
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
    n_total = len(result)
    n_valid = int(result.notna().sum())
    if n_valid == 0 and n_total > 0:
        raise ValueError(
            "dv01_pnl: result is all-NaN. Check index alignment "
            f"between delta_bp (len={len(delta_bp)}) and dv01 (len={len(dv01)})."
        )
    if n_total > 1 and n_valid < n_total * 0.5:
        logger.warning(
            "dv01_pnl: %d of %d values are NaN (>50%%); check index alignment",
            n_total - n_valid, n_total,
        )
    return result
