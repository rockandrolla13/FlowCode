"""Credit-specific signal computations.

This module provides credit-specific signals including:
- Credit PnL calculation (Spec §1.1)
- Range Position normalization (Spec §1.2)

All formulas reference bond_orderflow_spec.tex.
"""

import numpy as np
import pandas as pd


def credit_pnl(
    spread_change: pd.Series,
    pvbp: pd.Series,
    mid_price: pd.Series,
) -> pd.Series:
    """
    Compute Credit PnL from spread changes.

    Formula (Spec §1.1):
        Credit_PnL = -(ΔSpread_1w) × (PVBP / MidPrice)

    Parameters
    ----------
    spread_change : pd.Series
        Change in spread (e.g., 1-week spread change in bps).
    pvbp : pd.Series
        Price Value of a Basis Point (DV01).
    mid_price : pd.Series
        Mid price of the bond.

    Returns
    -------
    pd.Series
        Credit PnL series.

    Examples
    --------
    >>> spread_change = pd.Series([-5, 10, -3])  # bps
    >>> pvbp = pd.Series([0.08, 0.08, 0.08])
    >>> mid_price = pd.Series([100, 100, 100])
    >>> credit_pnl(spread_change, pvbp, mid_price)
    0    0.004
    1   -0.008
    2    0.0024
    dtype: float64

    Notes
    -----
    - Negative spread change (tightening) → positive PnL for long positions
    - PVBP is typically expressed per $100 notional
    """
    pnl = -(spread_change) * (pvbp / mid_price)
    pnl = pnl.replace([np.inf, -np.inf], np.nan)
    pnl.name = "credit_pnl"
    return pnl


def range_position(
    spread_current: pd.Series,
    spread_avg: pd.Series,
    spread_max: pd.Series,
    spread_min: pd.Series,
) -> pd.Series:
    """
    Compute Range Position of current spread.

    Normalizes the current spread relative to its historical window.

    Formula (Spec §1.2):
        RangePos = (Spread_curr - Spread_avg) / (Spread_max - Spread_min)

    Parameters
    ----------
    spread_current : pd.Series
        Current spread value.
    spread_avg : pd.Series
        Average spread over lookback window (e.g., 1 month).
    spread_max : pd.Series
        Maximum spread over lookback window.
    spread_min : pd.Series
        Minimum spread over lookback window.

    Returns
    -------
    pd.Series
        Range position. Values near 0 indicate spread is near average.
        Positive values indicate wider than average.
        Negative values indicate tighter than average.

    Examples
    --------
    >>> spread_curr = pd.Series([150])
    >>> spread_avg = pd.Series([140])
    >>> spread_max = pd.Series([160])
    >>> spread_min = pd.Series([120])
    >>> range_position(spread_curr, spread_avg, spread_max, spread_min)
    0    0.25
    dtype: float64

    Notes
    -----
    - Result is unbounded (can exceed [-1, 1] if current spread
      exceeds historical range)
    - NaN returned if spread_max == spread_min (no range)
    """
    spread_range = spread_max - spread_min
    position = (spread_current - spread_avg) / spread_range
    position = position.replace([np.inf, -np.inf], np.nan)
    position.name = "range_position"
    return position


def compute_range_position_rolling(
    spreads: pd.Series,
    window: int = 21,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute rolling Range Position.

    Calculates range position using a rolling window for
    avg, max, min calculations.

    Parameters
    ----------
    spreads : pd.Series
        Time series of spreads.
    window : int, default 21
        Rolling window size (default 21 = ~1 month).
    min_periods : int | None, optional
        Minimum observations required. Defaults to window // 2.

    Returns
    -------
    pd.Series
        Rolling range position series.

    Examples
    --------
    >>> spreads = pd.Series([100, 110, 105, 115, 108, 120, 112])
    >>> compute_range_position_rolling(spreads, window=5)
    """
    if min_periods is None:
        min_periods = window // 2

    rolling = spreads.rolling(window=window, min_periods=min_periods)

    spread_avg = rolling.mean()
    spread_max = rolling.max()
    spread_min = rolling.min()

    return range_position(spreads, spread_avg, spread_max, spread_min)
