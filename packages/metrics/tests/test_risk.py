"""Tests for risk metrics."""

import numpy as np
import pandas as pd
import pytest

from src.risk import (
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    value_at_risk,
    expected_shortfall,
    volatility,
)


class TestDrawdownSeries:
    """Tests for drawdown_series function."""

    def test_no_drawdown(self) -> None:
        """Test drawdown is 0 when always increasing."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        result = drawdown_series(returns)
        # All values should be 0 (always at peak)
        assert (result == 0).all()

    def test_simple_drawdown(self) -> None:
        """Test simple drawdown calculation."""
        returns = pd.Series([0.10, -0.10])
        result = drawdown_series(returns)
        # After 10% gain then 10% loss:
        # Wealth: 1.10, then 1.10 * 0.90 = 0.99
        # Drawdown from 1.10: (0.99 - 1.10) / 1.10 ≈ -0.10
        assert result.iloc[0] == 0  # At peak
        assert result.iloc[1] < 0  # In drawdown

    def test_recovery(self) -> None:
        """Test drawdown recovery to 0."""
        returns = pd.Series([0.10, -0.05, 0.10])
        result = drawdown_series(returns)
        # Should recover to 0 at new peak
        assert result.iloc[-1] == 0

    def test_empty_series(self) -> None:
        """Test empty series returns empty."""
        returns = pd.Series(dtype=float)
        result = drawdown_series(returns)
        assert len(result) == 0


class TestMaxDrawdown:
    """Tests for max_drawdown function."""

    def test_basic_max_drawdown(self) -> None:
        """Test basic max drawdown calculation."""
        returns = pd.Series([0.10, -0.20, 0.05])
        result = max_drawdown(returns)
        # Should be negative
        assert result < 0

    def test_max_drawdown_is_minimum(self) -> None:
        """Test max drawdown equals minimum of drawdown series."""
        returns = pd.Series([0.10, -0.15, 0.05, -0.10, 0.20])
        dd = drawdown_series(returns)
        mdd = max_drawdown(returns)
        assert mdd == pytest.approx(dd.min())

    def test_no_drawdown_returns_zero(self) -> None:
        """Test max drawdown is 0 when always positive."""
        returns = pd.Series([0.01, 0.02, 0.03])
        result = max_drawdown(returns)
        assert result == 0

    def test_empty_series(self) -> None:
        """Test empty series returns NaN."""
        returns = pd.Series(dtype=float)
        result = max_drawdown(returns)
        assert np.isnan(result)


class TestMaxDrawdownDuration:
    """Tests for max_drawdown_duration function."""

    def test_basic_duration(self) -> None:
        """Test basic drawdown duration."""
        # Go down, stay down for 3 periods, recover
        returns = pd.Series([0.10, -0.15, -0.01, -0.01, 0.20])
        result = max_drawdown_duration(returns)
        # Should be at least 3 periods in drawdown
        assert result >= 3

    def test_no_drawdown_duration(self) -> None:
        """Test duration is 0 when no drawdown."""
        returns = pd.Series([0.01, 0.02, 0.03])
        result = max_drawdown_duration(returns)
        assert result == 0


class TestValueAtRisk:
    """Tests for value_at_risk function."""

    def test_var_is_percentile(self) -> None:
        """Test VaR equals historical percentile."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var_95 = value_at_risk(returns, confidence=0.95)
        expected = returns.quantile(0.05)

        assert var_95 == pytest.approx(expected)

    def test_var_is_negative(self) -> None:
        """Test VaR is typically negative (represents loss)."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        var = value_at_risk(returns, confidence=0.95)
        assert var < 0

    def test_higher_confidence_more_extreme(self) -> None:
        """Test higher confidence gives more extreme VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var_95 = value_at_risk(returns, confidence=0.95)
        var_99 = value_at_risk(returns, confidence=0.99)

        # 99% VaR should be more negative (extreme)
        assert var_99 < var_95

    def test_empty_series(self) -> None:
        """Test empty series returns NaN."""
        returns = pd.Series(dtype=float)
        result = value_at_risk(returns)
        assert np.isnan(result)


class TestExpectedShortfall:
    """Tests for expected_shortfall function."""

    def test_es_worse_than_var(self) -> None:
        """Test ES is more extreme than VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var = value_at_risk(returns, confidence=0.95)
        es = expected_shortfall(returns, confidence=0.95)

        # ES should be <= VaR (more negative)
        assert es <= var

    def test_es_is_tail_average(self) -> None:
        """Test ES equals average of tail losses."""
        returns = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])

        var = value_at_risk(returns, confidence=0.80)  # 20% tail
        es = expected_shortfall(returns, confidence=0.80)

        # ES should be average of returns <= VaR
        tail = returns[returns <= var]
        assert es == pytest.approx(tail.mean())


class TestVolatility:
    """Tests for volatility function."""

    def test_annualization(self) -> None:
        """Test volatility is properly annualized."""
        returns = pd.Series(np.random.randn(100) * 0.01)

        daily_std = returns.std()
        annualized = volatility(returns, periods_per_year=252)

        assert annualized == pytest.approx(daily_std * np.sqrt(252))

    def test_insufficient_data(self) -> None:
        """Test volatility with insufficient data."""
        returns = pd.Series([0.01])
        result = volatility(returns)
        assert np.isnan(result)
