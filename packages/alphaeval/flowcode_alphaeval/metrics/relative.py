"""Benchmark-relative metrics — tracking error.

Information Ratio already exists in packages/metrics; this module
exposes tracking error as a standalone metric.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["tracking_error"]


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
        TE = sqrt(A) * std(r_p - r_b, ddof=1). NaN if < 2 obs.

    Raises
    ------
    ValueError
        If indices differ and observations are lost from misalignment.
    """
    if not returns.index.equals(benchmark.index):
        n_expected = min(len(returns.dropna()), len(benchmark.dropna()))
        common = returns.index.intersection(benchmark.index)
        n_common = len(common)
        if n_common < n_expected:
            raise ValueError(
                f"tracking_error: index misalignment — {n_expected - n_common} of "
                f"{n_expected} observations lost. Align indices before calling."
            )
    active = (returns - benchmark).dropna()
    if len(active) < 2:
        return np.nan
    return float(np.sqrt(periods_per_year) * active.std(ddof=1))
