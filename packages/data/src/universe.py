from __future__ import annotations

"""Universe filtering utilities.

This module provides functions to filter bonds by rating,
liquidity, and other criteria to define investment universes.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Standard rating order from highest to lowest
RATING_ORDER = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",  # IG/HY boundary
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
    "D",
]

# Investment grade threshold
IG_THRESHOLD = "BBB-"



def filter_ig(
    df: pd.DataFrame,
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Filter to investment-grade bonds only.

    Investment grade is defined as BBB- or better.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a rating column.
    rating_col : str
        Name of the rating column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only IG bonds.

    Examples
    --------
    >>> ig_bonds = filter_ig(df, rating_col="rating")
    >>> ig_bonds["rating"].unique()
    ['AAA', 'AA+', 'AA', ..., 'BBB-']
    """
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in DataFrame")

    threshold_idx = RATING_ORDER.index(IG_THRESHOLD)
    ig_ratings = set(RATING_ORDER[: threshold_idx + 1])

    mask = df[rating_col].isin(ig_ratings)
    result = df[mask].copy()

    logger.info(f"Filtered to {len(result)} IG bonds (from {len(df)})")

    return result


def filter_hy(
    df: pd.DataFrame,
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Filter to high-yield bonds only.

    High yield is defined as BB+ or worse (below BBB-).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a rating column.
    rating_col : str
        Name of the rating column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only HY bonds.

    Examples
    --------
    >>> hy_bonds = filter_hy(df, rating_col="rating")
    >>> hy_bonds["rating"].unique()
    ['BB+', 'BB', 'BB-', ..., 'D']
    """
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in DataFrame")

    threshold_idx = RATING_ORDER.index(IG_THRESHOLD)
    hy_ratings = set(RATING_ORDER[threshold_idx + 1 :])

    mask = df[rating_col].isin(hy_ratings)
    result = df[mask].copy()

    logger.info(f"Filtered to {len(result)} HY bonds (from {len(df)})")

    return result


def filter_by_rating(
    df: pd.DataFrame,
    min_rating: str | None = None,
    max_rating: str | None = None,
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Filter bonds by rating range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a rating column.
    min_rating : str | None
        Minimum rating (inclusive). Higher is better (AAA > D).
    max_rating : str | None
        Maximum rating (inclusive). Lower is worse.
    rating_col : str
        Name of the rating column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Examples
    --------
    >>> # Get A-rated bonds (A+, A, A-)
    >>> a_bonds = filter_by_rating(df, min_rating="A-", max_rating="A+")
    """
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in DataFrame")

    mask = pd.Series(True, index=df.index)

    if min_rating is not None:
        min_idx = RATING_ORDER.index(min_rating)
        valid_ratings = set(RATING_ORDER[: min_idx + 1])
        mask &= df[rating_col].isin(valid_ratings)

    if max_rating is not None:
        max_idx = RATING_ORDER.index(max_rating)
        valid_ratings = set(RATING_ORDER[max_idx:])
        mask &= df[rating_col].isin(valid_ratings)

    result = df[mask].copy()

    logger.info(
        f"Filtered by rating [{max_rating or 'any'} - {min_rating or 'any'}]: "
        f"{len(result)} bonds (from {len(df)})"
    )

    return result


def filter_by_liquidity(
    df: pd.DataFrame,
    min_volume: float | None = None,
    min_trades: int | None = None,
    volume_col: str = "total_volume",
    trades_col: str = "trade_count",
) -> pd.DataFrame:
    """
    Filter bonds by liquidity criteria.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume/trade count columns.
    min_volume : float | None
        Minimum total volume threshold.
    min_trades : int | None
        Minimum number of trades threshold.
    volume_col : str
        Name of the volume column.
    trades_col : str
        Name of the trade count column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame meeting liquidity criteria.
    """
    mask = pd.Series(True, index=df.index)

    if min_volume is not None and volume_col in df.columns:
        mask &= df[volume_col] >= min_volume

    if min_trades is not None and trades_col in df.columns:
        mask &= df[trades_col] >= min_trades

    result = df[mask].copy()

    logger.info(f"Filtered by liquidity: {len(result)} bonds (from {len(df)})")

    return result
