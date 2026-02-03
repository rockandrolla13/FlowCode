"""Bond reference data loading utilities.

This module provides functions to load bond reference data
that maps CUSIPs to their characteristics (issuer, rating, maturity, etc.).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_reference(
    path: str | Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load bond reference data from a parquet or CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the reference data file (parquet or CSV).
    columns : list[str] | None, optional
        Specific columns to load. If None, loads all columns.

    Returns
    -------
    pd.DataFrame
        Reference data with columns:
        - cusip: Bond identifier (str)
        - issuer: Issuer name (str)
        - rating: Credit rating (str, e.g., 'BBB+')
        - maturity: Maturity date (datetime)
        - coupon: Coupon rate (float)
        - issue_date: Issue date (datetime)
        - amount_outstanding: Amount outstanding in millions (float)

        Index is reset (RangeIndex).

    Raises
    ------
    FileNotFoundError
        If the reference data file does not exist.
    ValueError
        If the file format is not supported.

    Examples
    --------
    >>> ref = load_reference("data/bond_reference.parquet")
    >>> ref.columns.tolist()
    ['cusip', 'issuer', 'rating', 'maturity', 'coupon', ...]
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Reference data file not found: {path}")

    # Load based on file extension
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if columns is not None:
            df = pd.read_parquet(path, columns=columns)
        else:
            df = pd.read_parquet(path)
    elif suffix == ".csv":
        if columns is not None:
            df = pd.read_csv(path, usecols=columns)
        else:
            df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Convert date columns
    date_columns = ["maturity", "issue_date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Loaded {len(df)} bond references from {path}")

    return df


def enrich_with_reference(
    trades: pd.DataFrame,
    reference: pd.DataFrame,
    on: str = "cusip",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Enrich trade data with reference data attributes.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data to enrich.
    reference : pd.DataFrame
        Reference data to join.
    on : str
        Column to join on.
    columns : list[str] | None, optional
        Specific reference columns to include.
        If None, includes all columns except the join key.

    Returns
    -------
    pd.DataFrame
        Enriched trade data with reference attributes.

    Examples
    --------
    >>> enriched = enrich_with_reference(trades, ref, columns=["issuer", "rating"])
    >>> "rating" in enriched.columns
    True
    """
    if columns is not None:
        ref_cols = [on] + [c for c in columns if c != on]
        ref_subset = reference[ref_cols].drop_duplicates(subset=[on])
    else:
        ref_subset = reference.drop_duplicates(subset=[on])

    return pd.merge(trades, ref_subset, on=on, how="left")
