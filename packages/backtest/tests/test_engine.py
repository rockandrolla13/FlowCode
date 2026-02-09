"""Tests for backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.engine import run_backtest, compute_returns, compute_metrics
from src.results import BacktestResult


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_basic_returns(self) -> None:
        """Test basic return calculation."""
        dates = pd.date_range("2023-01-01", periods=5)
        positions = pd.DataFrame(
            {"A": [0.0, 1.0, 1.0, 1.0, 1.0]},
            index=dates,
        )
        prices = pd.DataFrame(
            {"A": [100, 101, 102, 101, 103]},
            index=dates,
        )

        returns = compute_returns(positions, prices)

        # Position is lagged, so:
        # Day 0: pos=0 (no previous), ret=0
        # Day 1: pos=0 (prev=0), ret=0
        # Day 2: pos=1 (prev=1), ret=(102-101)/101
        # etc.
        assert len(returns) == 5
        assert returns.iloc[0] == 0 or np.isnan(returns.iloc[0])

    def test_no_lookahead(self) -> None:
        """Test that positions are properly lagged (no lookahead)."""
        dates = pd.date_range("2023-01-01", periods=3)
        # Signal only on day 2
        positions = pd.DataFrame(
            {"A": [0.0, 0.0, 1.0]},
            index=dates,
        )
        prices = pd.DataFrame(
            {"A": [100, 110, 105]},  # Big move day 1, then down
            index=dates,
        )

        returns = compute_returns(positions, prices)

        # Position of 1.0 on day 2 shouldn't capture day 1's return
        # Because position is lagged
        assert returns.iloc[1] == 0  # No position yet

    def test_multiple_assets(self) -> None:
        """Test returns with multiple assets."""
        dates = pd.date_range("2023-01-01", periods=3)
        positions = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]},
            index=dates,
        )
        prices = pd.DataFrame(
            {"A": [100, 102, 104], "B": [100, 101, 102]},
            index=dates,
        )

        returns = compute_returns(positions, prices)

        assert len(returns) == 3


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_positive_returns(self) -> None:
        """Test metrics with positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.005])
        metrics = compute_metrics(returns)

        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert metrics["sharpe_ratio"] > 0
        assert metrics["total_return"] > 0

    def test_negative_returns(self) -> None:
        """Test metrics with negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.005])
        metrics = compute_metrics(returns)

        assert metrics["sharpe_ratio"] < 0
        assert metrics["total_return"] < 0
        assert metrics["max_drawdown"] < 0

    def test_annualized_return_present(self) -> None:
        """Test annualized_return is computed and correct."""
        returns = pd.Series([0.10, -0.05, 0.03])
        metrics = compute_metrics(returns)

        assert "annualized_return" in metrics
        # total_return = (1.10 * 0.95 * 1.03) - 1 = 0.0765
        expected_total = (1.10 * 0.95 * 1.03) - 1
        assert metrics["total_return"] == pytest.approx(expected_total, abs=1e-10)
        # annualized = (1 + total)^(252/3) - 1
        expected_ann = (1 + expected_total) ** (252 / 3) - 1
        assert metrics["annualized_return"] == pytest.approx(expected_ann, rel=1e-6)

    def test_annualized_return_total_loss(self) -> None:
        """Test annualized_return is NaN for >= 100% loss."""
        # total_return = (1-1.0) - 1 = -1.0 → NaN guarded
        returns = pd.Series([-1.0, 0.0])
        metrics = compute_metrics(returns)

        assert np.isnan(metrics["annualized_return"])

    def test_insufficient_data(self) -> None:
        """Test metrics with insufficient data."""
        returns = pd.Series([0.01])
        metrics = compute_metrics(returns)

        assert metrics == {}


class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_basic_backtest(self) -> None:
        """Test basic backtest execution."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=10)
        signal = pd.DataFrame(
            {"A": rng.standard_normal(10), "B": rng.standard_normal(10)},
            index=dates,
        )
        prices = pd.DataFrame(
            {"A": 100 * (1 + rng.standard_normal(10).cumsum() * 0.01),
             "B": 100 * (1 + rng.standard_normal(10).cumsum() * 0.01)},
            index=dates,
        )

        result = run_backtest(signal, prices)

        assert isinstance(result, BacktestResult)
        assert len(result.returns) == 10
        assert len(result.positions) == 10

    def test_backtest_with_transaction_costs(self) -> None:
        """Test backtest applies transaction costs."""
        dates = pd.date_range("2023-01-01", periods=10)
        # Alternating signal to generate turnover
        signal = pd.DataFrame(
            {"A": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]},
            index=dates,
        )
        prices = pd.DataFrame(
            {"A": [100] * 10},  # Flat prices
            index=dates,
        )

        result_no_cost = run_backtest(signal, prices, transaction_cost=0.0)
        result_with_cost = run_backtest(signal, prices, transaction_cost=0.01)

        # With costs, returns should be lower
        assert result_with_cost.returns.sum() < result_no_cost.returns.sum()

    def test_backtest_returns_result(self) -> None:
        """Test backtest returns BacktestResult with all fields."""
        dates = pd.date_range("2023-01-01", periods=5)
        signal = pd.DataFrame({"A": [1, 1, 1, 1, 1]}, index=dates)
        prices = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=dates)

        result = run_backtest(signal, prices)

        assert hasattr(result, "returns")
        assert hasattr(result, "positions")
        assert hasattr(result, "trades")
        assert hasattr(result, "metrics")
        assert hasattr(result, "config")


class TestBacktestResult:
    """Tests for BacktestResult class."""

    def test_total_return(self) -> None:
        """Test total_return property."""
        returns = pd.Series([0.01, 0.02, -0.01])
        result = BacktestResult(
            returns=returns,
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
        )

        expected = (1.01 * 1.02 * 0.99) - 1
        assert result.total_return == pytest.approx(expected)

    def test_summary(self) -> None:
        """Test summary method."""
        returns = pd.Series([0.01, 0.02, -0.01])
        result = BacktestResult(
            returns=returns,
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.5},
        )

        summary = result.summary()
        assert "Total Return" in summary
        assert "Sharpe" in summary
