"""Tests for trace data loading module."""

import pandas as pd
import pytest
from unittest.mock import patch

from src.trace import load_trace, aggregate_daily_volume


class TestLoadTrace:
    """Tests for load_trace function."""

    def test_file_not_found(self, tmp_path) -> None:
        """Test FileNotFoundError on missing file."""
        with pytest.raises(FileNotFoundError):
            load_trace(tmp_path / "nonexistent.parquet")

    def test_loads_parquet(self, tmp_path) -> None:
        """Test loading a valid parquet file."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "cusip": ["ABC123456", "ABC123456"],
            "price": [100.0, 101.0],
            "volume": [1000, 2000],
            "side": ["B", "S"],
        })
        path = tmp_path / "trace.parquet"
        df.to_parquet(path)

        result = load_trace(path)

        assert len(result) == 2
        assert list(result.columns) == ["date", "cusip", "price", "volume", "side"]

    def test_date_column_converted(self, tmp_path) -> None:
        """Test date column is converted to datetime."""
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "cusip": ["ABC", "ABC"],
        })
        path = tmp_path / "trace.parquet"
        df.to_parquet(path)

        result = load_trace(path)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_date_range_filter(self, tmp_path) -> None:
        """Test date range filtering."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01"]),
            "cusip": ["A", "A", "A"],
        })
        path = tmp_path / "trace.parquet"
        df.to_parquet(path)

        result = load_trace(path, date_range=("2023-01-10", "2023-01-20"))

        assert len(result) == 1
        assert result["date"].iloc[0] == pd.Timestamp("2023-01-15")

    def test_column_selection(self, tmp_path) -> None:
        """Test loading specific columns."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "cusip": ["ABC"],
            "price": [100.0],
            "volume": [1000],
        })
        path = tmp_path / "trace.parquet"
        df.to_parquet(path)

        result = load_trace(path, columns=["cusip", "price"])

        assert set(result.columns) == {"cusip", "price"}

    def test_reset_index(self, tmp_path) -> None:
        """Test that index is reset to RangeIndex."""
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01"]), "cusip": ["ABC"]},
            index=[42],
        )
        path = tmp_path / "trace.parquet"
        df.to_parquet(path)

        result = load_trace(path)

        assert result.index[0] == 0


class TestAggregateDailyVolume:
    """Tests for aggregate_daily_volume function."""

    def test_basic_aggregation(self) -> None:
        """Test basic buy/sell volume aggregation."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"] * 4),
            "cusip": ["ABC"] * 4,
            "volume": [100, 200, 150, 50],
            "side": ["B", "B", "S", "S"],
        })

        result = aggregate_daily_volume(trades)

        assert result["buy_volume"].iloc[0] == 300
        assert result["sell_volume"].iloc[0] == 200
        assert result["total_volume"].iloc[0] == 500

    def test_multiple_cusips(self) -> None:
        """Test aggregation across multiple CUSIPs."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"] * 4),
            "cusip": ["ABC", "ABC", "XYZ", "XYZ"],
            "volume": [100, 200, 300, 400],
            "side": ["B", "S", "B", "S"],
        })

        result = aggregate_daily_volume(trades)

        assert len(result) == 2

    def test_only_buys(self) -> None:
        """Test when only buy trades exist."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "cusip": ["ABC"],
            "volume": [100],
            "side": ["B"],
        })

        result = aggregate_daily_volume(trades)

        assert result["buy_volume"].iloc[0] == 100
        assert result["sell_volume"].iloc[0] == 0
