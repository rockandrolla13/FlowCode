"""Panel validation for (date, instrument) DataFrames.

Validates MultiIndex structure, required columns, and dtype coercion
for alphaeval inputs.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_panel(
    data: pd.DataFrame,
    required_cols: set[str] | None = None,
    *,
    date_col: str = "date",
    instrument_col: str = "instrument",
) -> pd.DataFrame:
    """Validate and coerce a panel DataFrame for alphaeval.

    Parameters
    ----------
    data : pd.DataFrame
        Input data, either with (date, instrument) MultiIndex or columns.
    required_cols : set[str] | None
        Column names that must be present. None skips column check.
    date_col : str, default "date"
        Name of date level/column.
    instrument_col : str, default "instrument"
        Name of instrument level/column.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with (date, instrument) MultiIndex.

    Raises
    ------
    ValueError
        If required columns missing or date/instrument not found.
    """
    df = data.copy()

    # If MultiIndex, check level names
    if isinstance(df.index, pd.MultiIndex):
        names = [n for n in df.index.names if n is not None]
        if date_col not in names or instrument_col not in names:
            raise ValueError(
                f"MultiIndex must have levels '{date_col}' and "
                f"'{instrument_col}', got {names}"
            )
    else:
        # Columns → MultiIndex
        for col in (date_col, instrument_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        df = df.set_index([date_col, instrument_col])

    # Coerce date level to datetime
    date_level = df.index.get_level_values(date_col)
    if not pd.api.types.is_datetime64_any_dtype(date_level):
        logger.info("Coercing '%s' level to datetime", date_col)
        df.index = df.index.set_levels(
            pd.to_datetime(df.index.levels[df.index.names.index(date_col)]),
            level=date_col,
        )

    # Check required columns
    if required_cols is not None:
        # Exclude index level names from required check
        check_cols = required_cols - {date_col, instrument_col}
        missing = check_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    return df
