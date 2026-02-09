"""Tests for risk metrics."""

import json
from pathlib import Path

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
    downside_volatility,
)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "spec" / "fixtures"


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
    """Tests for value_at_risk function (positive loss magnitude per spec §4.2)."""

    def test_var_is_negated_percentile(self) -> None:
        """Test VaR equals negated historical percentile."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(1000) * 0.01)

        var_95 = value_at_risk(returns, confidence=0.95)
        expected = -returns.quantile(0.05)

        assert var_95 == pytest.approx(expected)

    def test_var_is_positive(self) -> None:
        """Test VaR is positive (loss magnitude)."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(1000) * 0.01)
        var = value_at_risk(returns, confidence=0.95)
        assert var > 0

    def test_higher_confidence_larger_var(self) -> None:
        """Test higher confidence gives larger VaR."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(1000) * 0.01)

        var_95 = value_at_risk(returns, confidence=0.95)
        var_99 = value_at_risk(returns, confidence=0.99)

        # 99% VaR should be larger (more extreme loss)
        assert var_99 > var_95

    def test_empty_series(self) -> None:
        """Test empty series returns NaN."""
        returns = pd.Series(dtype=float)
        result = value_at_risk(returns)
        assert np.isnan(result)


class TestExpectedShortfall:
    """Tests for expected_shortfall function (positive loss magnitude per spec §4.3)."""

    def test_es_exceeds_var(self) -> None:
        """Test ES >= VaR (both positive loss magnitudes)."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var = value_at_risk(returns, confidence=0.95)
        es = expected_shortfall(returns, confidence=0.95)

        # ES should be >= VaR (larger loss magnitude)
        assert es >= var

    def test_es_is_negated_tail_average(self) -> None:
        """Test ES equals negated average of tail losses."""
        returns = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])

        var = value_at_risk(returns, confidence=0.80)  # 20% tail
        es = expected_shortfall(returns, confidence=0.80)

        # Tail is returns <= -VaR (the negative quantile)
        tail = returns[returns <= -var]
        assert es == pytest.approx(-tail.mean())


class TestVolatility:
    """Tests for volatility function."""

    def test_annualization(self) -> None:
        """Test volatility is properly annualized."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(100) * 0.01)

        daily_std = returns.std()
        annualized = volatility(returns, periods_per_year=252)

        assert annualized == pytest.approx(daily_std * np.sqrt(252))

    def test_insufficient_data(self) -> None:
        """Test volatility with insufficient data."""
        returns = pd.Series([0.01])
        result = volatility(returns)
        assert np.isnan(result)


class TestDownsideVolatility:
    """Tests for downside_volatility function."""

    def test_returns_nan_when_insufficient_downside(self) -> None:
        """Test NaN when fewer than 2 returns below target."""
        returns = pd.Series([0.01, 0.02, 0.03])  # all positive
        result = downside_volatility(returns, target=0.0)
        assert np.isnan(result)

    def test_returns_nan_for_single_period(self) -> None:
        """Test NaN for fewer than 2 total periods."""
        returns = pd.Series([0.01])
        result = downside_volatility(returns)
        assert np.isnan(result)

    def test_positive_for_downside_data(self) -> None:
        """Test positive result when downside data exists."""
        returns = pd.Series([0.01, -0.02, 0.01, -0.03, 0.02, -0.01])
        result = downside_volatility(returns, target=0.0)
        assert result > 0


class TestExpectedShortfallFallback:
    """Tests for expected_shortfall fallback behavior.

    The tail is never empty with historical VaR (quantile comes from
    the same data), so we mock VaR to exercise the defensive guard.
    """

    def test_fallback_to_var_when_tail_empty(self, monkeypatch) -> None:
        """Test ES returns VaR when no returns fall in tail."""
        import src.risk as risk_module

        # Mock VaR to return a large value so -var is far below any return
        monkeypatch.setattr(risk_module, "value_at_risk", lambda *a, **kw: 100.0)
        returns = pd.Series([0.01, 0.02, 0.03])
        es = expected_shortfall(returns, confidence=0.95)
        assert es == pytest.approx(100.0)


class TestDrawdownFixtures:
    """Fixture-backed tests for drawdown (spec §4.1)."""

    @pytest.fixture
    def drawdown_cases(self):
        with open(FIXTURES_DIR / "drawdown_cases.json") as f:
            return json.load(f)

    def test_all_drawdown_fixture_cases(self, drawdown_cases) -> None:
        """Test max_drawdown against golden fixture cases."""
        for case in drawdown_cases["cases"]:
            returns = pd.Series(case["input"]["returns"])

            if not case["input"]["returns"]:
                result = max_drawdown(returns)
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
                continue

            result = max_drawdown(returns)
            if case["expected"] is None:
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
            else:
                tol = case.get("tolerance", 0.001)
                assert result == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result}"
                )


class TestVarFixtures:
    """Fixture-backed tests for VaR (spec §4.2)."""

    @pytest.fixture
    def var_cases(self):
        with open(FIXTURES_DIR / "var_cases.json") as f:
            return json.load(f)

    def test_all_var_fixture_cases(self, var_cases) -> None:
        """Test value_at_risk against golden fixture cases."""
        for case in var_cases["cases"]:
            returns_data = case["input"]["returns"]
            if not returns_data:
                result = value_at_risk(pd.Series(dtype=float))
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
                continue

            returns = pd.Series(returns_data)

            if "expected" in case:
                confidence = case["input"].get("confidence", 0.95)
                result = value_at_risk(returns, confidence=confidence)
                if case["expected"] is None:
                    assert np.isnan(result), f"Case '{case['name']}': expected NaN"
                else:
                    tol = case.get("tolerance", 0.001)
                    assert result == pytest.approx(case["expected"], abs=tol), (
                        f"Case '{case['name']}': expected {case['expected']}, got {result}"
                    )

            if "expected_low" in case:
                var_low = value_at_risk(returns, confidence=case["input"]["confidence_low"])
                var_high = value_at_risk(returns, confidence=case["input"]["confidence_high"])
                tol = case.get("tolerance", 0.001)
                assert var_low == pytest.approx(case["expected_low"], abs=tol), (
                    f"Case '{case['name']}' low: expected {case['expected_low']}, got {var_low}"
                )
                assert var_high == pytest.approx(case["expected_high"], abs=tol), (
                    f"Case '{case['name']}' high: expected {case['expected_high']}, got {var_high}"
                )
                assert var_high > var_low, (
                    f"Case '{case['name']}': higher confidence VaR should be larger"
                )


class TestEsFixtures:
    """Fixture-backed tests for Expected Shortfall (spec §4.3)."""

    @pytest.fixture
    def es_cases(self):
        with open(FIXTURES_DIR / "es_cases.json") as f:
            return json.load(f)

    def test_all_es_fixture_cases(self, es_cases) -> None:
        """Test expected_shortfall against golden fixture cases."""
        for case in es_cases["cases"]:
            returns_data = case["input"]["returns"]
            if not returns_data:
                result = expected_shortfall(pd.Series(dtype=float))
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
                continue

            returns = pd.Series(returns_data)
            confidence = case["input"].get("confidence", 0.95)
            es = expected_shortfall(returns, confidence=confidence)
            var = value_at_risk(returns, confidence=confidence)

            # ES >= VaR always
            assert es >= var, (
                f"Case '{case['name']}': ES ({es}) should >= VaR ({var})"
            )

            if "expected" in case and case["expected"] is not None:
                tol = case.get("tolerance", 0.001)
                assert es == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {es}"
                )
