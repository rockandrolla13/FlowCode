"""Tests for validation module."""

import pandas as pd
import pytest

from src.validation import validate_cusip, validate_trace, validate_reference


class TestValidateCusip:
    """Tests for validate_cusip function."""

    def test_valid_cusip(self) -> None:
        """Test valid 9-character CUSIP."""
        assert validate_cusip("037833100") is True
        assert validate_cusip("AAPL00001") is True

    def test_invalid_cusip_length(self) -> None:
        """Test CUSIP with wrong length."""
        assert validate_cusip("12345678") is False  # 8 chars
        assert validate_cusip("1234567890") is False  # 10 chars

    def test_invalid_cusip_type(self) -> None:
        """Test non-string CUSIP."""
        assert validate_cusip(123456789) is False  # type: ignore
        assert validate_cusip(None) is False  # type: ignore


class TestValidateTrace:
    """Tests for validate_trace function."""

    def test_valid_trace_data(self) -> None:
        """Test validation of valid TRACE data."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "cusip": ["037833100", "594918104"],
            "price": [99.5, 100.25],
            "volume": [1000, 2000],
            "side": ["B", "S"],
        })

        result = validate_trace(df)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.stats["total_rows"] == 2

    def test_missing_required_columns(self) -> None:
        """Test validation fails with missing columns."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "cusip": ["037833100"],
        })

        result = validate_trace(df)

        assert result.is_valid is False
        assert any("Missing required columns" in e for e in result.errors)

    def test_negative_volume_is_error(self) -> None:
        """Test that negative volume is flagged as error."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "cusip": ["037833100"],
            "price": [99.5],
            "volume": [-100],
            "side": ["B"],
        })

        result = validate_trace(df)

        assert result.is_valid is False
        assert result.stats["negative_volume"] == 1

    def test_invalid_side_is_error(self) -> None:
        """Test that invalid side is flagged as error."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "cusip": ["037833100"],
            "price": [99.5],
            "volume": [100],
            "side": ["X"],  # Invalid side
        })

        result = validate_trace(df)

        assert result.is_valid is False
        assert result.stats["invalid_sides"] == 1

    def test_empty_dataframe_is_warning(self) -> None:
        """Test that empty DataFrame is a warning, not error."""
        df = pd.DataFrame(columns=["date", "cusip", "price", "volume", "side"])

        result = validate_trace(df)

        assert result.is_valid is True
        assert any("empty" in w.lower() for w in result.warnings)


class TestValidateReference:
    """Tests for validate_reference function."""

    def test_valid_reference_data(self) -> None:
        """Test validation of valid reference data."""
        df = pd.DataFrame({
            "cusip": ["037833100", "594918104"],
            "issuer": ["APPLE INC", "MICROSOFT CORP"],
            "rating": ["AA+", "AAA"],
        })

        result = validate_reference(df)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_duplicate_cusips_is_warning(self) -> None:
        """Test that duplicate CUSIPs are flagged as warning."""
        df = pd.DataFrame({
            "cusip": ["037833100", "037833100"],  # Duplicate
            "issuer": ["APPLE INC", "APPLE INC"],
            "rating": ["AA+", "AA+"],
        })

        result = validate_reference(df)

        assert result.is_valid is True  # Warning, not error
        assert result.stats["duplicate_cusips"] == 1

    def test_missing_required_columns(self) -> None:
        """Test validation fails with missing columns."""
        df = pd.DataFrame({
            "cusip": ["037833100"],
        })

        result = validate_reference(df)

        assert result.is_valid is False
        assert any("Missing required columns" in e for e in result.errors)
