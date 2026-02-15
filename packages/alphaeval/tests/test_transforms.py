"""Tests for transforms — returns, spreads, equity."""
import numpy as np
import pandas as pd
import pytest

from src.transforms.returns import price_to_returns, equity_curve
from src.transforms.spreads import delta_spread_bp, spread_return_proxy, dv01_pnl
from src.transforms.equity import drawdown_series, runup_series


# ── Returns ─────────────────────────────────────────────────────────────────

class TestPriceToReturns:
    def test_simple_returns(self) -> None:
        price = pd.Series([100.0, 110.0, 99.0])
        r = price_to_returns(price, method="simple")
        assert np.isnan(r.iloc[0])
        assert r.iloc[1] == pytest.approx(0.10)
        assert r.iloc[2] == pytest.approx(-0.1, abs=0.001)

    def test_log_returns(self) -> None:
        price = pd.Series([100.0, 110.0])
        r = price_to_returns(price, method="log")
        assert r.iloc[1] == pytest.approx(np.log(1.1))

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            price_to_returns(pd.Series([1.0, 2.0]), method="bad")


class TestEquityCurve:
    def test_basic(self) -> None:
        returns = pd.Series([0.10, -0.05, 0.02])
        eq = equity_curve(returns, initial=1.0)
        assert eq.iloc[0] == pytest.approx(1.10)
        assert eq.iloc[1] == pytest.approx(1.10 * 0.95)
        assert eq.iloc[2] == pytest.approx(1.10 * 0.95 * 1.02)

    def test_nan_treated_as_zero(self) -> None:
        returns = pd.Series([0.10, np.nan, 0.05])
        eq = equity_curve(returns)
        assert eq.iloc[1] == pytest.approx(1.10)  # NaN → 0 return
        assert eq.iloc[2] == pytest.approx(1.10 * 1.05)

    def test_custom_initial(self) -> None:
        returns = pd.Series([0.10])
        eq = equity_curve(returns, initial=1000.0)
        assert eq.iloc[0] == pytest.approx(1100.0)


# ── Spreads ─────────────────────────────────────────────────────────────────

class TestDeltaSpreadBp:
    def test_basic(self) -> None:
        spread = pd.Series([0.0150, 0.0160, 0.0145])  # 150bp, 160bp, 145bp
        delta = delta_spread_bp(spread)
        assert np.isnan(delta.iloc[0])
        assert delta.iloc[1] == pytest.approx(10.0)   # +10bp
        assert delta.iloc[2] == pytest.approx(-15.0)   # -15bp


class TestSpreadReturnProxy:
    def test_basic(self) -> None:
        spread = pd.Series([0.0150, 0.0160])
        duration = pd.Series([5.0, 5.0])
        r = spread_return_proxy(spread, duration)
        # r ~ -5.0 * 0.001 = -0.005
        assert r.iloc[1] == pytest.approx(-0.005)


class TestDv01Pnl:
    def test_basic(self) -> None:
        delta_bp = pd.Series([10.0, -5.0])
        dv01 = pd.Series([500.0, 500.0])
        pnl = dv01_pnl(delta_bp, dv01)
        assert pnl.iloc[0] == pytest.approx(-5000.0)
        assert pnl.iloc[1] == pytest.approx(2500.0)


# ── Equity analytics ────────────────────────────────────────────────────────

class TestDrawdownSeries:
    def test_monotonic_up_zero_dd(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.01])
        dd = drawdown_series(returns)
        assert (dd == 0).all()

    def test_drawdown_after_peak(self) -> None:
        returns = pd.Series([0.10, -0.20, 0.05])
        dd = drawdown_series(returns)
        # eq = [1.10, 0.88, 0.924], peak = [1.10, 1.10, 1.10]
        assert dd.iloc[0] == pytest.approx(0.0)
        assert dd.iloc[1] == pytest.approx((1.10 - 0.88) / 1.10)
        assert dd.iloc[2] == pytest.approx((1.10 - 0.924) / 1.10)

    def test_total_loss_nan(self) -> None:
        returns = pd.Series([-1.0])
        dd = drawdown_series(returns)
        assert np.isnan(dd.iloc[0])  # 0/0 → NaN


class TestRunupSeries:
    def test_monotonic_down_zero_runup(self) -> None:
        returns = pd.Series([-0.05, -0.05, -0.05])
        ru = runup_series(returns)
        assert (ru == 0).all()

    def test_runup_after_trough(self) -> None:
        returns = pd.Series([-0.10, 0.20, 0.05])
        ru = runup_series(returns)
        # eq = [0.90, 1.08, 1.134], trough = [0.90, 0.90, 0.90]
        assert ru.iloc[0] == pytest.approx(0.0)
        assert ru.iloc[1] == pytest.approx((1.08 - 0.90) / 0.90)
        assert ru.iloc[2] == pytest.approx((1.134 - 0.90) / 0.90)

    def test_total_loss_nan(self) -> None:
        returns = pd.Series([-1.0, 0.50])
        ru = runup_series(returns)
        # eq = [0.0, 0.0], trough = [0.0, 0.0], ru = NaN (0/0)
        assert np.isnan(ru.iloc[0])
        assert np.isnan(ru.iloc[1])
