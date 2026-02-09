"""Tests for performance metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.performance import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    annualized_return,
)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "spec" / "fixtures"


class TestAnnualizedReturn:
    """Tests for annualized_return function."""

    def test_positive_returns(self) -> None:
        """Test annualized return with positive returns."""
        # 1% daily for 252 days
        returns = pd.Series([0.01] * 252)
        result = annualized_return(returns, periods_per_year=252)
        # Should be approximately (1.01)^252 - 1 ≈ 11.27
        assert result > 1.0  # More than 100% annual return

    def test_zero_returns(self) -> None:
        """Test annualized return with zero returns."""
        returns = pd.Series([0.0] * 100)
        result = annualized_return(returns, periods_per_year=252)
        assert result == pytest.approx(0.0)

    def test_empty_returns(self) -> None:
        """Test annualized return with empty series."""
        returns = pd.Series(dtype=float)
        result = annualized_return(returns)
        assert np.isnan(result)

    def test_total_loss_returns_nan(self) -> None:
        """Test annualized return is NaN for >= 100% loss."""
        returns = pd.Series([-1.0, 0.0])  # total_return = -1.0
        result = annualized_return(returns)
        assert np.isnan(result)


class TestSharpeRatio:
    """Tests for sharpe_ratio function."""

    def test_positive_sharpe(self) -> None:
        """Test positive Sharpe ratio."""
        # Consistent positive returns
        returns = pd.Series([0.001] * 100)
        result = sharpe_ratio(returns, periods_per_year=252)
        assert result > 0

    def test_negative_sharpe(self) -> None:
        """Test negative Sharpe ratio."""
        # Consistent negative returns
        returns = pd.Series([-0.001] * 100)
        result = sharpe_ratio(returns, periods_per_year=252)
        assert result < 0

    def test_zero_volatility(self) -> None:
        """Test Sharpe with zero volatility returns NaN."""
        returns = pd.Series([0.001] * 100)  # All same value
        # Technically this has zero std, but due to floating point...
        # Let's use truly constant
        returns = pd.Series([0.0] * 100)
        result = sharpe_ratio(returns, periods_per_year=252)
        assert np.isnan(result)

    def test_with_risk_free(self) -> None:
        """Test Sharpe with non-zero risk-free rate."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.permutation([0.001] * 100 + [-0.001] * 100))

        sharpe_no_rf = sharpe_ratio(returns, risk_free=0.0)
        sharpe_with_rf = sharpe_ratio(returns, risk_free=0.0001)

        # Higher risk-free should lower Sharpe
        assert sharpe_with_rf < sharpe_no_rf

    def test_insufficient_data(self) -> None:
        """Test Sharpe with insufficient data."""
        returns = pd.Series([0.01])
        result = sharpe_ratio(returns)
        assert np.isnan(result)


class TestSortinoRatio:
    """Tests for sortino_ratio function."""

    def test_all_positive_returns(self) -> None:
        """Test Sortino with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        result = sortino_ratio(returns)
        # No downside, should return NaN (no downside deviation)
        assert np.isnan(result)

    def test_mixed_returns(self) -> None:
        """Test Sortino with mixed returns."""
        returns = pd.Series([0.01, -0.02, 0.01, -0.01, 0.02])
        result = sortino_ratio(returns)
        # Should be finite
        assert not np.isnan(result)

    def test_sortino_vs_sharpe(self) -> None:
        """Test that Sortino differs from Sharpe for asymmetric returns."""
        # Returns with more downside volatility
        returns = pd.Series([0.01, -0.05, 0.01, -0.04, 0.02, -0.03])

        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)

        # Both should be negative (losing strategy)
        # Sortino should differ from Sharpe for asymmetric returns
        assert abs(sharpe - sortino) > 0.001


class TestCalmarRatio:
    """Tests for calmar_ratio function."""

    def test_positive_calmar(self) -> None:
        """Test positive Calmar ratio."""
        # Trending up with small drawdown
        returns = pd.Series([0.01] * 50 + [-0.02] + [0.01] * 50)
        result = calmar_ratio(returns)
        assert result > 0

    def test_handles_zero_drawdown(self) -> None:
        """Test Calmar with zero drawdown returns NaN."""
        # All positive returns = no drawdown
        returns = pd.Series([0.01] * 100)
        result = calmar_ratio(returns)
        # Zero drawdown means division by zero
        assert np.isnan(result)


class TestSharpeFixtures:
    """Fixture-backed tests for Sharpe ratio (spec §3.1)."""

    @pytest.fixture
    def sharpe_cases(self):
        with open(FIXTURES_DIR / "sharpe_cases.json") as f:
            return json.load(f)

    def test_all_sharpe_fixture_cases(self, sharpe_cases) -> None:
        """Test sharpe_ratio against golden fixture cases."""
        for case in sharpe_cases["cases"]:
            if "expected" not in case:
                continue  # Skip cases without expected values (note-only)

            returns = pd.Series(case["input"]["returns"])
            rf = case["input"].get("risk_free_rate", 0.0)

            result = sharpe_ratio(returns, risk_free=rf)

            if case["expected"] is None:
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
            else:
                tol = case.get("tolerance", 0.01)
                assert result == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result}"
                )


class TestSortinoFixtures:
    """Fixture-backed tests for Sortino ratio (spec §3.2)."""

    @pytest.fixture
    def sortino_cases(self):
        with open(FIXTURES_DIR / "sortino_cases.json") as f:
            return json.load(f)

    def test_all_sortino_fixture_cases(self, sortino_cases) -> None:
        """Test sortino_ratio against golden fixture cases."""
        for case in sortino_cases["cases"]:
            if "expected" not in case:
                continue

            returns = pd.Series(case["input"]["returns"])
            rf = case["input"].get("risk_free_rate", 0.0)

            result = sortino_ratio(returns, risk_free=rf)

            if case["expected"] is None:
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
            else:
                tol = case.get("tolerance", 0.01)
                assert result == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result}"
                )


class TestCalmarFixtures:
    """Fixture-backed tests for Calmar ratio (spec §3.3)."""

    @pytest.fixture
    def calmar_cases(self):
        with open(FIXTURES_DIR / "calmar_cases.json") as f:
            return json.load(f)

    def test_all_calmar_fixture_cases(self, calmar_cases) -> None:
        """Test calmar_ratio against golden fixture cases."""
        for case in calmar_cases["cases"]:
            if "expected" not in case:
                continue

            returns = pd.Series(case["input"]["returns"])
            result = calmar_ratio(returns)

            if case["expected"] is None:
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
            else:
                tol = case.get("tolerance", 0.1)
                assert result == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result}"
                )
