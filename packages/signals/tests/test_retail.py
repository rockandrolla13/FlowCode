"""Tests for retail signal module."""

import numpy as np
import pandas as pd
import pytest

from src.retail import (
    qmp_classify,
    compute_retail_imbalance,
    compute_imbalance_from_volumes,
)


class TestQmpClassify:
    """Tests for qmp_classify function."""

    def test_buy_above_threshold(self) -> None:
        """Test trade above threshold is classified as buy."""
        result = qmp_classify(price=100.5, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "buy"

    def test_sell_below_threshold(self) -> None:
        """Test trade below threshold is classified as sell."""
        result = qmp_classify(price=99.5, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "sell"

    def test_at_threshold_is_sell(self) -> None:
        """Test trade exactly at threshold is classified as sell."""
        # price = mid + threshold * spread = 100 + 0.1 * 1 = 100.1
        result = qmp_classify(price=100.1, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "sell"

    def test_just_above_threshold_is_buy(self) -> None:
        """Test trade just above threshold is classified as buy."""
        result = qmp_classify(price=100.11, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "buy"

    def test_custom_threshold(self) -> None:
        """Test with custom threshold."""
        result = qmp_classify(price=100.3, mid=100.0, spread=1.0, threshold=0.5)
        assert result == "sell"  # 100.3 < 100 + 0.5 * 1 = 100.5


class TestComputeRetailImbalance:
    """Tests for compute_retail_imbalance function."""

    def test_balanced_volume(self) -> None:
        """Test imbalance is 0 when buy and sell volumes are equal."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "cusip": ["ABC123456", "ABC123456"],
            "volume": [100, 100],
            "side": ["B", "S"],
        })

        result = compute_retail_imbalance(trades)

        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(0.0)

    def test_all_buys(self) -> None:
        """Test imbalance is 1 when all trades are buys."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "cusip": ["ABC123456", "ABC123456"],
            "volume": [100, 200],
            "side": ["B", "B"],
        })

        result = compute_retail_imbalance(trades)

        assert result.iloc[0] == pytest.approx(1.0)

    def test_all_sells(self) -> None:
        """Test imbalance is -1 when all trades are sells."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "cusip": ["ABC123456", "ABC123456"],
            "volume": [100, 200],
            "side": ["S", "S"],
        })

        result = compute_retail_imbalance(trades)

        assert result.iloc[0] == pytest.approx(-1.0)

    def test_positive_imbalance(self) -> None:
        """Test positive imbalance when buys > sells."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "cusip": ["ABC123456", "ABC123456"],
            "volume": [300, 100],  # 300 buy, 100 sell
            "side": ["B", "S"],
        })

        result = compute_retail_imbalance(trades)

        # (300 - 100) / (300 + 100) = 200 / 400 = 0.5
        assert result.iloc[0] == pytest.approx(0.5)

    def test_multiple_cusips(self) -> None:
        """Test imbalance computed separately for each cusip."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"] * 4),
            "cusip": ["ABC123456", "ABC123456", "XYZ789012", "XYZ789012"],
            "volume": [100, 100, 300, 100],
            "side": ["B", "S", "B", "S"],
        })

        result = compute_retail_imbalance(trades)

        assert len(result) == 2
        # ABC: (100-100)/(100+100) = 0
        # XYZ: (300-100)/(300+100) = 0.5

    def test_multiindex_output(self) -> None:
        """Test output has MultiIndex (date, cusip)."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "cusip": ["ABC123456"],
            "volume": [100],
            "side": ["B"],
        })

        result = compute_retail_imbalance(trades)

        assert result.index.names == ["date", "cusip"]

    def test_missing_columns_raises(self) -> None:
        """Test that missing required columns raises ValueError."""
        trades = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "cusip": ["ABC123456"],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_retail_imbalance(trades)


class TestComputeImbalanceFromVolumes:
    """Tests for compute_imbalance_from_volumes function."""

    def test_basic_imbalance(self) -> None:
        """Test basic imbalance calculation from volumes."""
        buy = pd.Series([100, 300], index=["A", "B"])
        sell = pd.Series([100, 100], index=["A", "B"])

        result = compute_imbalance_from_volumes(buy, sell)

        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.5)

    def test_zero_volume_is_nan(self) -> None:
        """Test that zero total volume results in NaN."""
        buy = pd.Series([0])
        sell = pd.Series([0])

        result = compute_imbalance_from_volumes(buy, sell)

        assert pd.isna(result.iloc[0])
