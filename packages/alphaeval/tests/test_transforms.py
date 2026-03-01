"""Tests for transforms — returns, spreads, equity."""
import numpy as np
import pandas as pd
import pytest

from flowcode_alphaeval.transforms.returns import price_to_returns, equity_curve
from flowcode_alphaeval.transforms.spreads import delta_spread_bp, spread_return_proxy, dv01_pnl
from flowcode_alphaeval.transforms.equity import drawdown_series, runup_series


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

    def test_zero_price_inf_replaced(self, caplog) -> None:
        """I6 fix: zero prices produce inf, replaced with NaN + warning."""
        import logging
        price = pd.Series([100.0, 0.0, 50.0])
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.transforms.returns"):
            r = price_to_returns(price, method="simple")
        assert not np.isinf(r).any()
        assert "inf" in caplog.text

    def test_zero_price_log_inf_replaced(self, caplog) -> None:
        """I6 fix: zero prices in log returns also replaced."""
        import logging
        price = pd.Series([100.0, 0.0, 50.0])
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.transforms.returns"):
            r = price_to_returns(price, method="log")
        assert not np.isinf(r).any()


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

    def test_nan_warning_logged(self, caplog) -> None:
        """I11 fix: NaN returns should produce a warning."""
        import logging
        returns = pd.Series([0.10, np.nan, np.nan, 0.05])
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.transforms.returns"):
            equity_curve(returns)
        assert "NaN" in caplog.text
        assert "2 of 4" in caplog.text


# ── Spreads ─────────────────────────────────────────────────────────────────

class TestDeltaSpreadBp:
    def test_basic(self) -> None:
        spread = pd.Series([0.0150, 0.0160, 0.0145])  # 150bp, 160bp, 145bp
        delta = delta_spread_bp(spread)
        assert np.isnan(delta.iloc[0])
        assert delta.iloc[1] == pytest.approx(10.0)   # +10bp
        assert delta.iloc[2] == pytest.approx(-15.0)   # -15bp

    def test_inf_replaced_with_nan(self, caplog) -> None:
        """Finding #6: inf values from extreme spreads replaced with NaN + warning."""
        import logging
        spread = pd.Series([0.0, np.inf, 0.015])
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.transforms.spreads"):
            delta = delta_spread_bp(spread)
        assert not np.isinf(delta).any()
        assert "inf" in caplog.text


class TestSpreadReturnProxy:
    def test_basic(self) -> None:
        spread = pd.Series([0.0150, 0.0160])
        duration = pd.Series([5.0, 5.0])
        r = spread_return_proxy(spread, duration)
        # r ~ -5.0 * 0.001 = -0.005
        assert r.iloc[1] == pytest.approx(-0.005)

    def test_all_nan_raises(self) -> None:
        """I10 fix: all-NaN result must raise ValueError."""
        spread = pd.Series([0.015, 0.016])
        # Misaligned indices → all-NaN product
        duration = pd.Series([5.0, 5.0], index=[10, 11])
        with pytest.raises(ValueError, match="all-NaN"):
            spread_return_proxy(spread, duration)

    def test_high_nan_pct_warns(self, caplog) -> None:
        """I10 fix: >50% NaN produces warning."""
        import logging
        spread = pd.Series([0.015, 0.016, 0.017, 0.018], index=[0, 1, 2, 3])
        duration = pd.Series([np.nan, np.nan, np.nan, 5.0], index=[0, 1, 2, 3])
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.transforms.spreads"):
            spread_return_proxy(spread, duration)
        # diff → NaN at idx 0, duration NaN at 1,2 → only idx 3 valid → >50% NaN
        assert "NaN" in caplog.text


class TestDv01Pnl:
    def test_basic(self) -> None:
        delta_bp = pd.Series([10.0, -5.0])
        dv01 = pd.Series([500.0, 500.0])
        pnl = dv01_pnl(delta_bp, dv01)
        assert pnl.iloc[0] == pytest.approx(-5000.0)
        assert pnl.iloc[1] == pytest.approx(2500.0)

    def test_all_nan_raises(self) -> None:
        """I10 fix: all-NaN result must raise ValueError."""
        delta_bp = pd.Series([10.0, -5.0])
        dv01 = pd.Series([500.0, 500.0], index=[10, 11])  # misaligned
        with pytest.raises(ValueError, match="all-NaN"):
            dv01_pnl(delta_bp, dv01)

    def test_high_nan_pct_warns(self, caplog) -> None:
        """dv01_pnl: >50% NaN produces warning."""
        import logging
        delta_bp = pd.Series([10.0, -5.0, 3.0, -2.0])
        dv01 = pd.Series([np.nan, np.nan, np.nan, 500.0])
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.transforms.spreads"):
            dv01_pnl(delta_bp, dv01)
        assert "NaN" in caplog.text


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

    def test_empty_input(self) -> None:
        """S8: empty returns → empty drawdown series."""
        dd = drawdown_series(pd.Series(dtype=float))
        assert len(dd) == 0


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

    def test_empty_input(self) -> None:
        """S8: empty returns → empty runup series."""
        ru = runup_series(pd.Series(dtype=float))
        assert len(ru) == 0
