"""Tests for credit signal module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.credit import (
    credit_pnl,
    range_position,
    compute_range_position_rolling,
)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "spec" / "fixtures"


class TestCreditPnL:
    """Tests for credit_pnl function."""

    def test_tightening_positive_pnl(self) -> None:
        """Test spread tightening produces positive PnL."""
        # Spread tightens by 5 bps
        spread_change = pd.Series([-5.0])
        pvbp = pd.Series([0.08])
        mid_price = pd.Series([100.0])

        result = credit_pnl(spread_change, pvbp, mid_price)

        # -(-5) * (0.08 / 100) = 5 * 0.0008 = 0.004
        assert result.iloc[0] == pytest.approx(0.004)

    def test_widening_negative_pnl(self) -> None:
        """Test spread widening produces negative PnL."""
        # Spread widens by 10 bps
        spread_change = pd.Series([10.0])
        pvbp = pd.Series([0.08])
        mid_price = pd.Series([100.0])

        result = credit_pnl(spread_change, pvbp, mid_price)

        # -(10) * (0.08 / 100) = -10 * 0.0008 = -0.008
        assert result.iloc[0] == pytest.approx(-0.008)

    def test_zero_spread_change(self) -> None:
        """Test zero spread change produces zero PnL."""
        spread_change = pd.Series([0.0])
        pvbp = pd.Series([0.08])
        mid_price = pd.Series([100.0])

        result = credit_pnl(spread_change, pvbp, mid_price)

        assert result.iloc[0] == pytest.approx(0.0)

    def test_zero_mid_price_nan(self) -> None:
        """Test zero mid price produces NaN."""
        spread_change = pd.Series([-5.0])
        pvbp = pd.Series([0.08])
        mid_price = pd.Series([0.0])

        result = credit_pnl(spread_change, pvbp, mid_price)

        assert np.isnan(result.iloc[0])

    def test_vectorized(self) -> None:
        """Test function works on multiple values."""
        spread_change = pd.Series([-5.0, 10.0, -3.0])
        pvbp = pd.Series([0.08, 0.08, 0.08])
        mid_price = pd.Series([100.0, 100.0, 100.0])

        result = credit_pnl(spread_change, pvbp, mid_price)

        assert len(result) == 3
        assert result.iloc[0] > 0  # Tightening
        assert result.iloc[1] < 0  # Widening
        assert result.iloc[2] > 0  # Tightening


class TestRangePosition:
    """Tests for range_position function."""

    def test_at_average(self) -> None:
        """Test spread at average returns 0."""
        spread_curr = pd.Series([140.0])
        spread_avg = pd.Series([140.0])
        spread_max = pd.Series([160.0])
        spread_min = pd.Series([120.0])

        result = range_position(spread_curr, spread_avg, spread_max, spread_min)

        assert result.iloc[0] == pytest.approx(0.0)

    def test_wider_than_average(self) -> None:
        """Test spread wider than average returns positive."""
        spread_curr = pd.Series([150.0])
        spread_avg = pd.Series([140.0])
        spread_max = pd.Series([160.0])
        spread_min = pd.Series([120.0])

        result = range_position(spread_curr, spread_avg, spread_max, spread_min)

        # (150 - 140) / (160 - 120) = 10 / 40 = 0.25
        assert result.iloc[0] == pytest.approx(0.25)

    def test_tighter_than_average(self) -> None:
        """Test spread tighter than average returns negative."""
        spread_curr = pd.Series([130.0])
        spread_avg = pd.Series([140.0])
        spread_max = pd.Series([160.0])
        spread_min = pd.Series([120.0])

        result = range_position(spread_curr, spread_avg, spread_max, spread_min)

        # (130 - 140) / (160 - 120) = -10 / 40 = -0.25
        assert result.iloc[0] == pytest.approx(-0.25)

    def test_zero_range_nan(self) -> None:
        """Test zero range (max == min) returns NaN."""
        spread_curr = pd.Series([140.0])
        spread_avg = pd.Series([140.0])
        spread_max = pd.Series([140.0])  # Same as min
        spread_min = pd.Series([140.0])

        result = range_position(spread_curr, spread_avg, spread_max, spread_min)

        assert np.isnan(result.iloc[0])

    def test_beyond_range(self) -> None:
        """Test spread beyond historical range is not clamped."""
        spread_curr = pd.Series([180.0])  # Beyond max of 160
        spread_avg = pd.Series([140.0])
        spread_max = pd.Series([160.0])
        spread_min = pd.Series([120.0])

        result = range_position(spread_curr, spread_avg, spread_max, spread_min)

        # (180 - 140) / (160 - 120) = 40 / 40 = 1.0
        assert result.iloc[0] == pytest.approx(1.0)


class TestComputeRangePositionRolling:
    """Tests for compute_range_position_rolling function."""

    def test_rolling_calculation(self) -> None:
        """Test rolling range position calculation."""
        spreads = pd.Series([100, 110, 120, 130, 140, 150, 160])

        result = compute_range_position_rolling(spreads, window=5, min_periods=3)

        # First 2 values should be NaN (min_periods=3)
        assert result.iloc[:2].isna().all()
        # Rest should have values
        assert result.iloc[2:].notna().all()

    def test_trending_series(self) -> None:
        """Test with trending spreads."""
        # Steadily widening spreads
        spreads = pd.Series([100, 105, 110, 115, 120, 125, 130])

        result = compute_range_position_rolling(spreads, window=5, min_periods=3)

        # Last value should be above average (positive range position)
        # as spread is trending up
        assert result.iloc[-1] > 0

    def test_default_min_periods(self) -> None:
        """Test default min_periods is window // 2."""
        spreads = pd.Series([100, 110, 120, 130, 140])

        result = compute_range_position_rolling(spreads, window=4)

        # min_periods = 4 // 2 = 2, so first value should be NaN
        assert np.isnan(result.iloc[0])
        # Second value should have data
        assert not np.isnan(result.iloc[1])


class TestCreditPnlFixtures:
    """Fixture-backed tests for credit PnL (spec §1.1)."""

    @pytest.fixture
    def pnl_cases(self):
        with open(FIXTURES_DIR / "pnl_cases.json") as f:
            return json.load(f)

    def test_all_pnl_fixture_cases(self, pnl_cases) -> None:
        """Test credit_pnl against golden fixture cases."""
        for case in pnl_cases["cases"]:
            inp = case["input"]
            spread_change = pd.Series([inp["spread_change"]])
            pvbp = pd.Series([inp["pvbp"]])
            mid_price = pd.Series([inp["mid_price"]])

            result = credit_pnl(spread_change, pvbp, mid_price)

            tol = case.get("tolerance", 0.001)
            assert result.iloc[0] == pytest.approx(case["expected"], abs=tol), (
                f"Case '{case['name']}': expected {case['expected']}, got {result.iloc[0]}"
            )


class TestRangePositionFixtures:
    """Fixture-backed tests for range position (spec §1.2)."""

    @pytest.fixture
    def range_position_cases(self):
        with open(FIXTURES_DIR / "range_position_cases.json") as f:
            return json.load(f)

    def test_all_range_position_fixture_cases(self, range_position_cases) -> None:
        """Test range_position against golden fixture cases."""
        for case in range_position_cases["cases"]:
            inp = case["input"]
            spread_curr = pd.Series([inp["spread_curr"]], dtype=float)
            spread_avg = pd.Series([inp["spread_avg"]], dtype=float)
            spread_max = pd.Series([inp["spread_max"]], dtype=float)
            spread_min = pd.Series([inp["spread_min"]], dtype=float)

            result = range_position(spread_curr, spread_avg, spread_max, spread_min)

            if case["expected"] is None:
                assert np.isnan(result.iloc[0]), f"Case '{case['name']}': expected NaN"
            else:
                assert result.iloc[0] == pytest.approx(case["expected"], abs=0.001), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result.iloc[0]}"
                )
