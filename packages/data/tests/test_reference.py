"""Tests for reference data loading module."""

import pandas as pd
import pytest

from src.reference import load_reference, enrich_with_reference


class TestLoadReference:
    """Tests for load_reference function."""

    def test_file_not_found(self, tmp_path) -> None:
        """Test FileNotFoundError on missing file."""
        with pytest.raises(FileNotFoundError):
            load_reference(tmp_path / "nonexistent.parquet")

    def test_loads_parquet(self, tmp_path) -> None:
        """Test loading a valid parquet file."""
        df = pd.DataFrame({
            "cusip": ["ABC123456"],
            "issuer": ["ACME Corp"],
            "rating": ["BBB+"],
        })
        path = tmp_path / "ref.parquet"
        df.to_parquet(path)

        result = load_reference(path)

        assert len(result) == 1
        assert "cusip" in result.columns

    def test_loads_csv(self, tmp_path) -> None:
        """Test loading a valid CSV file."""
        df = pd.DataFrame({
            "cusip": ["ABC123456"],
            "issuer": ["ACME Corp"],
            "rating": ["BBB+"],
        })
        path = tmp_path / "ref.csv"
        df.to_csv(path, index=False)

        result = load_reference(path)

        assert len(result) == 1

    def test_unsupported_format(self, tmp_path) -> None:
        """Test ValueError on unsupported file format."""
        path = tmp_path / "ref.xlsx"
        path.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_reference(path)

    def test_date_columns_converted(self, tmp_path) -> None:
        """Test date columns are converted to datetime."""
        df = pd.DataFrame({
            "cusip": ["ABC"],
            "maturity": ["2030-01-01"],
            "issue_date": ["2020-01-01"],
        })
        path = tmp_path / "ref.parquet"
        df.to_parquet(path)

        result = load_reference(path)

        assert pd.api.types.is_datetime64_any_dtype(result["maturity"])
        assert pd.api.types.is_datetime64_any_dtype(result["issue_date"])

    def test_column_selection(self, tmp_path) -> None:
        """Test loading specific columns."""
        df = pd.DataFrame({
            "cusip": ["ABC"],
            "issuer": ["ACME"],
            "rating": ["BBB"],
        })
        path = tmp_path / "ref.parquet"
        df.to_parquet(path)

        result = load_reference(path, columns=["cusip", "rating"])

        assert set(result.columns) == {"cusip", "rating"}


class TestEnrichWithReference:
    """Tests for enrich_with_reference function."""

    def test_basic_enrichment(self) -> None:
        """Test basic trade enrichment with reference data."""
        trades = pd.DataFrame({
            "cusip": ["ABC", "XYZ"],
            "price": [100, 200],
        })
        reference = pd.DataFrame({
            "cusip": ["ABC", "XYZ"],
            "issuer": ["ACME", "Beta"],
            "rating": ["BBB", "A"],
        })

        result = enrich_with_reference(trades, reference)

        assert "issuer" in result.columns
        assert "rating" in result.columns

    def test_enrichment_with_selected_columns(self) -> None:
        """Test enrichment with specific columns."""
        trades = pd.DataFrame({"cusip": ["ABC"], "price": [100]})
        reference = pd.DataFrame({
            "cusip": ["ABC"],
            "issuer": ["ACME"],
            "rating": ["BBB"],
        })

        result = enrich_with_reference(trades, reference, columns=["rating"])

        assert "rating" in result.columns
        assert "issuer" not in result.columns

    def test_missing_cusip_gets_nan(self) -> None:
        """Test that missing CUSIPs get NaN after left join."""
        trades = pd.DataFrame({"cusip": ["ABC", "MISSING"], "price": [100, 200]})
        reference = pd.DataFrame({"cusip": ["ABC"], "issuer": ["ACME"]})

        result = enrich_with_reference(trades, reference)

        assert pd.isna(result.loc[result["cusip"] == "MISSING", "issuer"].iloc[0])
