"""Risk metrics — parametric VaR and VPIN.

Supplements the historical VaR in packages/metrics/src/risk.py.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

__all__ = ["var_parametric", "vpin"]


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
        VPIN values in [0, 1]. NaN for first n_buckets-1 periods.
        Negative volumes are clamped to 0 with a warning.

    Raises
    ------
    ValueError
        If volume_buy and volume_sell have different indices.
    """
    if not volume_buy.index.equals(volume_sell.index):
        raise ValueError(
            "vpin: volume_buy and volume_sell have different indices "
            f"(len {len(volume_buy)} vs {len(volume_sell)}). "
            "Align indices before calling vpin."
        )
    if (volume_buy < 0).any() or (volume_sell < 0).any():
        n_neg = int((volume_buy < 0).sum() + (volume_sell < 0).sum())
        logger.warning(
            "vpin: %d negative volume entries detected; clamping to 0",
            n_neg,
        )
        volume_buy = volume_buy.clip(lower=0)
        volume_sell = volume_sell.clip(lower=0)
    imbalance = (volume_buy - volume_sell).abs()
    total_vol = volume_buy + volume_sell

    rolling_imb = imbalance.rolling(n_buckets, min_periods=n_buckets).sum()
    rolling_vol = total_vol.rolling(n_buckets, min_periods=n_buckets).sum()

    result = rolling_imb / rolling_vol
    n_inf = int(np.isinf(result).sum())
    if n_inf > 0:
        logger.warning(
            "vpin: %d inf values from zero total volume windows; replacing with NaN",
            n_inf,
        )
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
