"""Tests for performance metrics."""
import numpy as np
import pandas as pd
import pytest

from src.metrics.performance import (
    profit_factor,
    win_rate_trades,
    expectancy,
    max_runup,
    cagr,
    tstat_returns,
    sortino_ratio,
)


class TestProfitFactor:
    def test_basic(self) -> None:
        pnls = pd.Series([100, -50, 200, -30])
        pf = profit_factor(pnls)
        assert pf == pytest.approx(300.0 / 80.0)

    def test_no_losses_inf(self) -> None:
        pnls = pd.Series([100, 200, 50])
        assert profit_factor(pnls) == np.inf

    def test_no_gains_zero(self) -> None:
        pnls = pd.Series([-100, -50])
        assert profit_factor(pnls) == pytest.approx(0.0)

    def test_empty_nan(self) -> None:
        pnls = pd.Series(dtype=float)
        assert np.isnan(profit_factor(pnls))  # no gains, no losses


class TestWinRateTrades:
    def test_basic(self) -> None:
        pnls = pd.Series([100, -50, 200, -30])
        assert win_rate_trades(pnls) == pytest.approx(50.0)

    def test_all_winners(self) -> None:
        pnls = pd.Series([10, 20, 30])
        assert win_rate_trades(pnls) == pytest.approx(100.0)

    def test_empty(self) -> None:
        assert np.isnan(win_rate_trades(pd.Series(dtype=float)))

    def test_nan_excluded(self) -> None:
        pnls = pd.Series([100, np.nan, -50])
        assert win_rate_trades(pnls) == pytest.approx(50.0)  # 1 of 2


class TestExpectancy:
    def test_basic(self) -> None:
        pnls = pd.Series([100, -50, 200, -30])
        assert expectancy(pnls) == pytest.approx(55.0)

    def test_empty(self) -> None:
        assert np.isnan(expectancy(pd.Series(dtype=float)))


class TestMaxRunup:
    def test_basic(self) -> None:
        returns = pd.Series([0.10, -0.05, 0.20])
        ru = max_runup(returns)
        # eq = [1.10, 1.045, 1.254], trough = [1.10, 1.045, 1.045]
        # runup = [0, 0, (1.254-1.045)/1.045]
        expected_ru = (1.10 * 0.95 * 1.20 - 1.10 * 0.95) / (1.10 * 0.95)
        assert ru == pytest.approx(expected_ru)

    def test_monotonic_decline(self) -> None:
        returns = pd.Series([-0.05, -0.05, -0.05])
        assert max_runup(returns) == pytest.approx(0.0)

    def test_empty(self) -> None:
        assert np.isnan(max_runup(pd.Series(dtype=float)))


class TestCagr:
    def test_basic(self) -> None:
        # 252 days of 0.1% daily → ~28.6% annualized
        returns = pd.Series([0.001] * 252)
        result = cagr(returns, periods_per_year=252)
        # (1.001^252)^(1/1) - 1
        expected = (1.001 ** 252) - 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_single_period(self) -> None:
        returns = pd.Series([0.10])
        result = cagr(returns, periods_per_year=252)
        # terminal=1.10, years=1/252, CAGR = 1.10^252 - 1
        expected = 1.10 ** 252 - 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_total_loss(self) -> None:
        returns = pd.Series([-1.0])
        assert cagr(returns) == pytest.approx(-1.0)

    def test_empty(self) -> None:
        assert np.isnan(cagr(pd.Series(dtype=float)))


class TestTstatReturns:
    def test_positive_mean(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
        t = tstat_returns(returns)
        assert t > 0  # positive mean → positive t

    def test_zero_std_nan(self) -> None:
        returns = pd.Series([0.01, 0.01, 0.01])
        assert np.isnan(tstat_returns(returns))

    def test_one_obs_nan(self) -> None:
        assert np.isnan(tstat_returns(pd.Series([0.01])))

    def test_known_value(self) -> None:
        returns = pd.Series([0.02, 0.04, 0.06, 0.08])
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        n = len(returns)
        expected = mu / (sigma / np.sqrt(n))
        assert tstat_returns(returns) == pytest.approx(expected)


class TestSortinoRatio:
    def test_positive_returns_no_downside(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.015])
        # All above target=0 → downside_std = 0 → NaN
        assert np.isnan(sortino_ratio(returns, target=0.0))

    def test_mixed_returns(self) -> None:
        returns = pd.Series([0.05, -0.03, 0.02, -0.01, 0.04])
        result = sortino_ratio(returns, target=0.0)
        # Manual: downside = [0, -0.03, 0, -0.01, 0], down_std = sqrt((0.0009+0.0001)/5)
        down_std = np.sqrt(0.001 / 5)
        expected = np.sqrt(252) * returns.mean() / down_std
        assert result == pytest.approx(expected)

    def test_population_convention(self) -> None:
        """Verify uses 1/T (not 1/(T-1)) for downside deviation."""
        returns = pd.Series([0.05, -0.03, 0.02, -0.01, 0.04])
        n = len(returns)
        excess = returns - 0.0
        downside = np.minimum(excess, 0.0)
        down_std = np.sqrt((downside ** 2).sum() / n)  # 1/T
        expected = np.sqrt(252) * excess.mean() / down_std
        assert sortino_ratio(returns) == pytest.approx(expected, rel=1e-10)

    def test_nonzero_target(self) -> None:
        """Returns between 0 and target count as downside."""
        returns = pd.Series([0.02, 0.005, -0.01, 0.03, 0.001])
        # target=0.01: excess = [0.01, -0.005, -0.02, 0.02, -0.009]
        # downside = [0, -0.005, -0.02, 0, -0.009]
        excess = returns - 0.01
        downside = np.minimum(excess, 0.0)
        down_std = np.sqrt((downside ** 2).sum() / len(returns))
        expected = np.sqrt(252) * excess.mean() / down_std
        assert sortino_ratio(returns, target=0.01) == pytest.approx(expected)

    def test_insufficient_data(self) -> None:
        assert np.isnan(sortino_ratio(pd.Series([0.01])))
