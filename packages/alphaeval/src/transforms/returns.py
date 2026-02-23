"""Return and equity curve transforms.

Converts price series to returns (simple or log) and builds
cumulative equity curves from return series.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def price_to_returns(
    price: pd.Series,
    method: str = "simple",
) -> pd.Series:
    """Convert price series to returns.

    Parameters
    ----------
    price : pd.Series
        Price or total-return-index series.
    method : {"simple", "log"}
        Return calculation method.

    Returns
    -------
    pd.Series
        Return series (first value is NaN).

    Raises
    ------
    ValueError
        If method not in {"simple", "log"}.
    """
    if method == "simple":
        result = price / price.shift(1) - 1
    elif method == "log":
        result = np.log(price / price.shift(1))
    else:
        raise ValueError(f"method must be 'simple' or 'log', got '{method}'")
    n_inf = int(np.isinf(result).sum())
    if n_inf > 0:
        logger.warning(
            "price_to_returns: %d inf values (zero prices?); replacing with NaN",
            n_inf,
        )
        result = result.replace([np.inf, -np.inf], np.nan)
    return result


def equity_curve(
    returns: pd.Series,
    initial: float = 1.0,
) -> pd.Series:
    """Build cumulative equity curve from returns.

    Parameters
    ----------
    returns : pd.Series
        Period returns (e.g. daily).
    initial : float, default 1.0
        Starting equity value E_0.

    Returns
    -------
    pd.Series
        Equity curve E_t = E_{t-1} * (1 + r_t). NaN returns treated as 0.
    """
    n_nan = int(returns.isna().sum())
    if n_nan > 0:
        logger.warning(
            "equity_curve: %d of %d returns are NaN (%.1f%%); treating as 0.0",
            n_nan, len(returns), 100 * n_nan / max(len(returns), 1),
        )
    clean = returns.fillna(0.0)
    return initial * (1.0 + clean).cumprod()
