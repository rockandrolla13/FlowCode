"""Benchmark-relative metrics — tracking error.

Information Ratio already exists in packages/metrics; this module
exposes tracking error as a standalone metric.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def tracking_error(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualized tracking error (std of active returns).

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    benchmark : pd.Series
        Benchmark returns, same frequency.
    periods_per_year : int, default 252
        For annualization.

    Returns
    -------
    float
        TE = sqrt(A) * std(r_p - r_b). NaN if < 2 obs.
    """
    active = returns - benchmark
    n_expected = min(len(returns.dropna()), len(benchmark.dropna()))
    n_actual = int(active.notna().sum())
    if n_actual < n_expected:
        logger.warning(
            "tracking_error: index misalignment dropped %d of %d observations",
            n_expected - n_actual, n_expected,
        )
    active = active.dropna()
    if len(active) < 2:
        return np.nan
    return float(np.sqrt(periods_per_year) * active.std(ddof=1))
