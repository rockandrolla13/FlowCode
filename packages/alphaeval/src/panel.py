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
        If required columns missing, date/instrument not found, or
        duplicate (date, instrument) rows detected.
    """
    # If MultiIndex, validate level names without copying
    if isinstance(data.index, pd.MultiIndex):
        names = [n for n in data.index.names if n is not None]
        if date_col not in names or instrument_col not in names:
            raise ValueError(
                f"MultiIndex must have levels '{date_col}' and "
                f"'{instrument_col}', got {names}"
            )
        df = data
    else:
        # Columns → MultiIndex (copy needed for set_index mutation)
        df = data.copy()
        for col in (date_col, instrument_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        df = df.set_index([date_col, instrument_col])

    # Coerce date level to datetime
    date_level = df.index.get_level_values(date_col)
    if not pd.api.types.is_datetime64_any_dtype(date_level):
        logger.info("Coercing '%s' level to datetime", date_col)
        if df is data:  # MultiIndex passthrough — copy to avoid mutating caller
            df = data.copy()
        df.index = df.index.set_levels(
            pd.to_datetime(df.index.levels[df.index.names.index(date_col)]),
            level=date_col,
        )

    # Check for duplicate (date, instrument) keys
    dupes = df.index.duplicated()
    if dupes.any():
        n_dupes = int(dupes.sum())
        raise ValueError(
            f"validate_panel: {n_dupes} duplicate ({date_col}, {instrument_col}) "
            "rows detected. De-duplicate before calling validate_panel."
        )

    # Check required columns
    if required_cols is not None:
        # Exclude index level names from required check
        check_cols = required_cols - {date_col, instrument_col}
        missing = check_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    return df
