"""Tests for retail signal module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retail import (
    qmp_classify,
    qmp_classify_with_exclusion,
    classify_trades_qmp,
    classify_trades_qmp_with_exclusion,
    compute_retail_imbalance,
    compute_imbalance_from_volumes,
    is_subpenny,
    is_retail_trade,
    classify_retail_trades,
)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "spec" / "fixtures"


class TestQmpClassify:
    """Tests for qmp_classify function (3-state per spec §1.5)."""

    def test_buy_above_upper_threshold(self) -> None:
        """Test trade above upper threshold is classified as buy."""
        result = qmp_classify(price=100.5, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "buy"

    def test_sell_below_lower_threshold(self) -> None:
        """Test trade below lower threshold is classified as sell."""
        result = qmp_classify(price=99.5, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "sell"

    def test_neutral_at_midpoint(self) -> None:
        """Test trade at midpoint is neutral."""
        result = qmp_classify(price=100.0, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "neutral"

    def test_neutral_at_upper_boundary(self) -> None:
        """Test trade exactly at upper threshold is neutral (not strictly above)."""
        # price = mid + threshold * spread = 100 + 0.1 * 1 = 100.1
        result = qmp_classify(price=100.1, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "neutral"

    def test_neutral_at_lower_boundary(self) -> None:
        """Test trade exactly at lower threshold is neutral."""
        # price = mid - threshold * spread = 100 - 0.1 * 1 = 99.9
        result = qmp_classify(price=99.9, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "neutral"

    def test_just_above_upper_threshold_is_buy(self) -> None:
        """Test trade just above upper threshold is classified as buy."""
        result = qmp_classify(price=100.11, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "buy"

    def test_just_below_lower_threshold_is_sell(self) -> None:
        """Test trade just below lower threshold is classified as sell."""
        result = qmp_classify(price=99.89, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "sell"

    def test_neutral_within_band(self) -> None:
        """Test trade within neutral band."""
        result = qmp_classify(price=100.03, mid=100.0, spread=1.0, threshold=0.1)
        assert result == "neutral"

    def test_custom_threshold(self) -> None:
        """Test with custom threshold."""
        # upper = 100 + 0.5*1 = 100.5, lower = 100 - 0.5*1 = 99.5
        result = qmp_classify(price=100.3, mid=100.0, spread=1.0, threshold=0.5)
        assert result == "neutral"  # 100.3 is within [99.5, 100.5]

    def test_negative_spread_returns_neutral(self) -> None:
        """Test that negative spread (crossed market) returns neutral."""
        result = qmp_classify(price=100.5, mid=100.0, spread=-1.0, threshold=0.1)
        assert result == "neutral"

    def test_zero_spread_returns_neutral(self) -> None:
        """Test that zero spread returns neutral."""
        result = qmp_classify(price=100.5, mid=100.0, spread=0.0, threshold=0.1)
        assert result == "neutral"


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
        abc_val = result.xs("ABC123456", level="cusip").iloc[0]
        assert abc_val == pytest.approx(0.0, abs=1e-9)
        # XYZ: (300-100)/(300+100) = 0.5
        xyz_val = result.xs("XYZ789012", level="cusip").iloc[0]
        assert xyz_val == pytest.approx(0.5, abs=1e-9)

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


class TestIsSubpenny:
    """Tests for is_subpenny function."""

    def test_subpenny_price(self) -> None:
        """Test price with subpenny component."""
        assert is_subpenny(100.501) is True
        assert is_subpenny(99.123) is True

    def test_round_cent_price(self) -> None:
        """Test price without subpenny component."""
        assert is_subpenny(100.50) is False
        assert is_subpenny(100.00) is False

    def test_near_boundary(self) -> None:
        """Test prices near cent boundary."""
        assert is_subpenny(100.009) is True
        assert is_subpenny(100.010) is False


class TestIsRetailTrade:
    """Tests for is_retail_trade function."""

    def test_retail_trade(self) -> None:
        """Test trade that meets both criteria is retail."""
        result = is_retail_trade(price=100.501, notional=50_000)
        assert result is True

    def test_no_subpenny(self) -> None:
        """Test trade without subpenny is not retail."""
        result = is_retail_trade(price=100.50, notional=50_000)
        assert result is False

    def test_too_large_notional(self) -> None:
        """Test trade above threshold is not retail."""
        result = is_retail_trade(price=100.501, notional=500_000)
        assert result is False

    def test_custom_threshold(self) -> None:
        """Test with custom notional threshold."""
        result = is_retail_trade(
            price=100.501,
            notional=150_000,
            notional_threshold=100_000
        )
        assert result is False


class TestClassifyRetailTrades:
    """Tests for classify_retail_trades function."""

    def test_vectorized_classification(self) -> None:
        """Test vectorized retail classification."""
        trades = pd.DataFrame({
            "price": [100.501, 100.50, 100.123],
            "notional": [50_000, 50_000, 300_000]
        })

        result = classify_retail_trades(trades)

        assert result.iloc[0] == True   # Subpenny + small
        assert result.iloc[1] == False  # No subpenny
        assert result.iloc[2] == False  # Too large


class TestClassifyTradesQmp:
    """Tests for classify_trades_qmp vectorized 3-state classification."""

    def test_three_state_classification(self) -> None:
        """Test vectorized QMP returns buy/sell/neutral correctly."""
        trades = pd.DataFrame({
            "price": [100.5, 99.5, 100.0, 100.11, 99.89],
            "mid": [100.0, 100.0, 100.0, 100.0, 100.0],
            "spread": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        result = classify_trades_qmp(trades, threshold=0.1)
        assert result.iloc[0] == "buy"      # 100.5 > 100.1
        assert result.iloc[1] == "sell"     # 99.5 < 99.9
        assert result.iloc[2] == "neutral"  # 100.0 in [99.9, 100.1]
        assert result.iloc[3] == "buy"      # 100.11 > 100.1
        assert result.iloc[4] == "sell"     # 99.89 < 99.9

    def test_boundary_values_are_neutral(self) -> None:
        """Test that prices exactly at thresholds are neutral."""
        trades = pd.DataFrame({
            "price": [100.1, 99.9],
            "mid": [100.0, 100.0],
            "spread": [1.0, 1.0],
        })
        result = classify_trades_qmp(trades, threshold=0.1)
        assert result.iloc[0] == "neutral"  # 100.1 == upper, not >
        assert result.iloc[1] == "neutral"  # 99.9 == lower, not <

    def test_matches_scalar_qmp_classify(self) -> None:
        """Test vectorized results match scalar function."""
        cases = [
            (100.5, 100.0, 1.0),
            (99.5, 100.0, 1.0),
            (100.0, 100.0, 1.0),
        ]
        trades = pd.DataFrame(cases, columns=["price", "mid", "spread"])
        vec_result = classify_trades_qmp(trades, threshold=0.1)
        for i, (price, mid, spread) in enumerate(cases):
            scalar_result = qmp_classify(price, mid, spread, threshold=0.1)
            assert vec_result.iloc[i] == scalar_result, (
                f"Mismatch at index {i}: vectorized={vec_result.iloc[i]}, "
                f"scalar={scalar_result}"
            )


class TestQmpClassifyWithExclusion:
    """Tests for qmp_classify_with_exclusion function."""

    def test_buy_above_exclusion(self) -> None:
        """Test price above 60% is classified as buy."""
        result = qmp_classify_with_exclusion(
            price=100.8, bid=100.0, ask=101.0
        )
        assert result == "buy"

    def test_sell_below_exclusion(self) -> None:
        """Test price below 40% is classified as sell."""
        result = qmp_classify_with_exclusion(
            price=100.2, bid=100.0, ask=101.0
        )
        assert result == "sell"

    def test_neutral_in_exclusion(self) -> None:
        """Test price in 40-60% zone is neutral."""
        result = qmp_classify_with_exclusion(
            price=100.5, bid=100.0, ask=101.0
        )
        assert result == "neutral"

    def test_at_boundaries(self) -> None:
        """Test at exact boundaries."""
        # At 40% (on boundary, should be neutral)
        result_40 = qmp_classify_with_exclusion(
            price=100.4, bid=100.0, ask=101.0
        )
        assert result_40 == "neutral"

        # At 60% (on boundary, should be neutral)
        result_60 = qmp_classify_with_exclusion(
            price=100.6, bid=100.0, ask=101.0
        )
        assert result_60 == "neutral"

    def test_zero_spread(self) -> None:
        """Test zero spread returns neutral."""
        result = qmp_classify_with_exclusion(
            price=100.0, bid=100.0, ask=100.0
        )
        assert result == "neutral"


class TestClassifyTradesQmpWithExclusion:
    """Tests for classify_trades_qmp_with_exclusion function."""

    def test_vectorized_classification(self) -> None:
        """Test vectorized QMP classification with exclusion."""
        trades = pd.DataFrame({
            "price": [100.8, 100.2, 100.5, 100.7],
            "bid": [100.0, 100.0, 100.0, 100.0],
            "ask": [101.0, 101.0, 101.0, 101.0]
        })

        result = classify_trades_qmp_with_exclusion(trades)

        assert result.iloc[0] == "buy"     # 80% of spread
        assert result.iloc[1] == "sell"    # 20% of spread
        assert result.iloc[2] == "neutral" # 50% of spread
        assert result.iloc[3] == "buy"     # 70% of spread

    def test_custom_exclusion_zone(self) -> None:
        """Test with custom exclusion zone."""
        trades = pd.DataFrame({
            "price": [100.5],
            "bid": [100.0],
            "ask": [101.0]
        })

        # With 45-55% exclusion, 50% should still be neutral
        result = classify_trades_qmp_with_exclusion(
            trades,
            exclusion_low=0.45,
            exclusion_high=0.55
        )
        assert result.iloc[0] == "neutral"

        # With 60-70% exclusion, 50% should be sell
        result = classify_trades_qmp_with_exclusion(
            trades,
            exclusion_low=0.60,
            exclusion_high=0.70
        )
        assert result.iloc[0] == "sell"


class TestQmpFixtures:
    """Fixture-backed tests for QMP classification (spec §1.5)."""

    @pytest.fixture
    def qmp_cases(self):
        with open(FIXTURES_DIR / "qmp_cases.json") as f:
            return json.load(f)

    def test_all_qmp_fixture_cases(self, qmp_cases) -> None:
        """Test qmp_classify against all golden fixture cases."""
        alpha = qmp_cases["params"]["alpha"]
        for case in qmp_cases["cases"]:
            inp = case["input"]
            mid = (inp["bid"] + inp["ask"]) / 2
            spread = inp["ask"] - inp["bid"]
            result = qmp_classify(
                price=inp["price"], mid=mid, spread=spread, threshold=alpha
            )
            expected = case["expected"].lower()
            assert result == expected, (
                f"Case '{case['name']}': expected {expected}, got {result}"
            )


class TestImbalanceFixtures:
    """Fixture-backed tests for imbalance (spec §1.3)."""

    @pytest.fixture
    def imbalance_cases(self):
        with open(FIXTURES_DIR / "imbalance_cases.json") as f:
            return json.load(f)

    def test_all_imbalance_fixture_cases(self, imbalance_cases) -> None:
        """Test compute_imbalance_from_volumes against golden fixture cases."""
        for case in imbalance_cases["cases"]:
            inp = case["input"]
            buy = pd.Series([inp["buy_volume"]])
            sell = pd.Series([inp["sell_volume"]])
            result = compute_imbalance_from_volumes(buy, sell)

            if case["expected"] is None:
                assert pd.isna(result.iloc[0]), f"Case '{case['name']}': expected NaN"
            else:
                assert result.iloc[0] == pytest.approx(case["expected"]), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result.iloc[0]}"
                )


class TestRetailIdFixtures:
    """Fixture-backed tests for retail identification (spec §1.4)."""

    @pytest.fixture
    def retail_id_cases(self):
        with open(FIXTURES_DIR / "retail_id_cases.json") as f:
            return json.load(f)

    def test_all_retail_id_fixture_cases(self, retail_id_cases) -> None:
        """Test is_retail_trade against golden fixture cases."""
        for case in retail_id_cases["cases"]:
            inp = case["input"]
            result = is_retail_trade(price=inp["price"], notional=inp["notional"])
            assert result == case["expected"], (
                f"Case '{case['name']}': expected {case['expected']}, got {result}"
            )
