from __future__ import annotations

"""Data validation utilities.

This module provides functions to validate data quality
for TRACE trades and bond reference data.
"""

import logging
import re
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, int | float]


def validate_cusip(cusip: str) -> bool:
    """
    Validate CUSIP format.

    A valid CUSIP is 9 characters: 6 alphanumeric (issuer) +
    2 alphanumeric (issue) + 1 check digit.

    Parameters
    ----------
    cusip : str
        CUSIP string to validate.

    Returns
    -------
    bool
        True if valid CUSIP format.
    """
    if not isinstance(cusip, str):
        return False

    if len(cusip) != 9:
        return False

    # Basic alphanumeric check (simplified)
    pattern = r"^[A-Z0-9]{9}$"
    return bool(re.match(pattern, cusip.upper()))


def validate_trace(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> ValidationResult:
    """
    Validate TRACE trade data quality.

    Parameters
    ----------
    df : pd.DataFrame
        TRACE data to validate.
    required_columns : list[str] | None
        Required columns. Defaults to standard TRACE columns.

    Returns
    -------
    ValidationResult
        Validation result with errors, warnings, and stats.

    Examples
    --------
    >>> result = validate_trace(trades)
    >>> result.is_valid
    True
    >>> result.stats["total_rows"]
    10000
    """
    if required_columns is None:
        required_columns = ["date", "cusip", "price", "volume", "side"]

    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, int | float] = {"total_rows": len(df)}

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check for empty DataFrame
    if len(df) == 0:
        warnings.append("DataFrame is empty")
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    # Validate CUSIP format (vectorized)
    if "cusip" in df.columns:
        cusip_str = df["cusip"].astype(str).str.upper()
        valid_cusip_mask = (cusip_str.str.len() == 9) & cusip_str.str.match(r"^[A-Z0-9]{9}$")
        invalid_cusips = df[~valid_cusip_mask.fillna(False)]
        stats["invalid_cusips"] = len(invalid_cusips)
        if len(invalid_cusips) > 0:
            warnings.append(f"Found {len(invalid_cusips)} invalid CUSIPs")

    # Check for null values
    if required_columns:
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isna().sum()
                stats[f"null_{col}"] = null_count
                if null_count > 0:
                    warnings.append(f"Column '{col}' has {null_count} null values")

    # Validate price range
    if "price" in df.columns:
        invalid_prices = df[(df["price"] <= 0) | (df["price"] > 200)]
        stats["invalid_prices"] = len(invalid_prices)
        if len(invalid_prices) > 0:
            warnings.append(f"Found {len(invalid_prices)} suspicious prices")

    # Validate volume
    if "volume" in df.columns:
        negative_volume = df[df["volume"] < 0]
        stats["negative_volume"] = len(negative_volume)
        if len(negative_volume) > 0:
            errors.append(f"Found {len(negative_volume)} negative volumes")

    # Validate side
    if "side" in df.columns:
        valid_sides = {"B", "S", "buy", "sell", "BUY", "SELL"}
        invalid_sides = df[~df["side"].isin(valid_sides)]
        stats["invalid_sides"] = len(invalid_sides)
        if len(invalid_sides) > 0:
            errors.append(f"Found {len(invalid_sides)} invalid sides")

    logger.info(f"Validated {len(df)} TRACE trades: {len(errors)} errors, {len(warnings)} warnings")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_reference(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> ValidationResult:
    """
    Validate bond reference data quality.

    Parameters
    ----------
    df : pd.DataFrame
        Reference data to validate.
    required_columns : list[str] | None
        Required columns. Defaults to standard reference columns.

    Returns
    -------
    ValidationResult
        Validation result with errors, warnings, and stats.
    """
    if required_columns is None:
        required_columns = ["cusip", "issuer", "rating"]

    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, int | float] = {"total_rows": len(df)}

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if len(df) == 0:
        warnings.append("DataFrame is empty")
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    # Check for duplicate CUSIPs
    if "cusip" in df.columns:
        duplicates = df["cusip"].duplicated().sum()
        stats["duplicate_cusips"] = duplicates
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate CUSIPs")

    # Validate CUSIP format (vectorized)
    if "cusip" in df.columns:
        cusip_str = df["cusip"].astype(str).str.upper()
        valid_cusip_mask = (cusip_str.str.len() == 9) & cusip_str.str.match(r"^[A-Z0-9]{9}$")
        invalid_cusips = df[~valid_cusip_mask.fillna(False)]
        stats["invalid_cusips"] = len(invalid_cusips)
        if len(invalid_cusips) > 0:
            warnings.append(f"Found {len(invalid_cusips)} invalid CUSIPs")

    # Check for null values in required columns
    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            stats[f"null_{col}"] = null_count
            if null_count > 0:
                warnings.append(f"Column '{col}' has {null_count} null values")

    logger.info(f"Validated {len(df)} references: {len(errors)} errors, {len(warnings)} warnings")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
