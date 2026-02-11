"""Risk metrics — parametric VaR and VPIN.

Supplements the historical VaR in packages/metrics/src/risk.py.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def var_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Gaussian parametric Value at Risk.

    Assumes losses L_t = -r_t are normally distributed.
    VaR_alpha = mu_L + z_alpha * sigma_L.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    confidence : float, default 0.95
        Confidence level (e.g. 0.95 for 95% VaR).

    Returns
    -------
    float
        VaR in loss space (positive = expected loss). Converts returns to
        losses via L_t = -r_t before fitting. NaN if < 2 observations.
        If all returns identical (zero vol), returns mu_L directly.
    """
    if not 0 < confidence < 1:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}"
        )
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    losses = -clean
    mu_l = losses.mean()
    sigma_l = losses.std(ddof=1)
    if abs(sigma_l) < 1e-14:
        return float(mu_l)
    z = scipy_stats.norm.ppf(confidence)
    return float(mu_l + z * sigma_l)


def vpin(
    volume_buy: pd.Series,
    volume_sell: pd.Series,
    n_buckets: int = 50,
) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading.

    Rolling VPIN over last n_buckets volume buckets.
    VPIN_t = sum(|V_b+ - V_b-|) / sum(V_b+ + V_b-) over window.

    Parameters
    ----------
    volume_buy : pd.Series
        Buy volume per bucket.
    volume_sell : pd.Series
        Sell volume per bucket.
    n_buckets : int, default 50
        Rolling window in buckets.

    Returns
    -------
    pd.Series
        VPIN values. NaN for first n_buckets-1 periods.
    """
    if not volume_buy.index.equals(volume_sell.index):
        logger.warning(
            "vpin: volume_buy and volume_sell have different indices; "
            "misalignment will introduce NaN"
        )
    imbalance = (volume_buy - volume_sell).abs()
    total_vol = volume_buy + volume_sell

    rolling_imb = imbalance.rolling(n_buckets, min_periods=n_buckets).sum()
    rolling_vol = total_vol.rolling(n_buckets, min_periods=n_buckets).sum()

    result = rolling_imb / rolling_vol
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
