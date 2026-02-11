"""Tests for factor quality metrics — IC*, RankIC*, IR*, R², tstat_ic."""
import numpy as np
import pandas as pd
import pytest

from src.metrics.factor import (
    ic_star,
    rank_ic_star,
    ir_star,
    r_squared,
    tstat_ic,
)


def _make_panel(n_dates: int = 20, n_instruments: int = 10, seed: int = 42):
    """Create synthetic signal + target panels for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    instruments = [f"BOND_{i}" for i in range(n_instruments)]
    signal = pd.DataFrame(
        rng.randn(n_dates, n_instruments),
        index=dates,
        columns=instruments,
    )
    # Target is noisy linear function of signal
    target = 0.3 * signal + 0.7 * pd.DataFrame(
        rng.randn(n_dates, n_instruments),
        index=dates,
        columns=instruments,
    )
    return signal, target


class TestIcStar:
    def test_positive_ic(self) -> None:
        signal, target = _make_panel()
        ic = ic_star(signal, target)
        assert ic > 0  # Positive by construction

    def test_random_signal_near_zero(self) -> None:
        rng = np.random.RandomState(99)
        dates = pd.bdate_range("2024-01-01", periods=50)
        instr = [f"B{i}" for i in range(20)]
        signal = pd.DataFrame(rng.randn(50, 20), index=dates, columns=instr)
        target = pd.DataFrame(rng.randn(50, 20), index=dates, columns=instr)
        ic = ic_star(signal, target)
        assert abs(ic) < 0.15  # Random → near zero


class TestIcStarMultiIndex:
    def test_multiindex_matches_pivoted(self) -> None:
        signal, target = _make_panel(n_dates=10, n_instruments=5)
        ic_pivoted = ic_star(signal, target)
        # Convert to MultiIndex (date, instrument) Series
        sig_stacked = signal.stack()
        sig_stacked.index.names = ["date", "instrument"]
        tgt_stacked = target.stack()
        tgt_stacked.index.names = ["date", "instrument"]
        sig_mi = sig_stacked.to_frame("signal")
        tgt_mi = tgt_stacked.to_frame("target")
        ic_mi = ic_star(sig_mi, tgt_mi)
        assert ic_mi == pytest.approx(ic_pivoted, abs=1e-10)


class TestRankIcStar:
    def test_positive_rank_ic(self) -> None:
        signal, target = _make_panel()
        ric = rank_ic_star(signal, target)
        assert ric > 0

    def test_perfect_ranks(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        instruments = ["A", "B", "C", "D", "E"]
        signal = pd.DataFrame(
            [[1, 2, 3, 4, 5]] * 5, index=dates, columns=instruments
        )
        target = pd.DataFrame(
            [[10, 20, 30, 40, 50]] * 5, index=dates, columns=instruments
        )
        ric = rank_ic_star(signal, target)
        assert ric == pytest.approx(1.0)


class TestIrStar:
    def test_positive_ir(self) -> None:
        signal, target = _make_panel()
        ir = ir_star(signal, target)
        assert ir > 0  # Consistent positive IC

    def test_insufficient_days(self) -> None:
        signal, target = _make_panel(n_dates=1)
        assert np.isnan(ir_star(signal, target))

    def test_constant_ic_zero_std_nan(self) -> None:
        """When all daily ICs are identical, IR* should be NaN (0 std)."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        instr = ["A", "B", "C", "D", "E"]
        # Signal = target → perfect IC=1.0 every day → std=0 → NaN
        signal = pd.DataFrame(
            [[1, 2, 3, 4, 5]] * 5, index=dates, columns=instr, dtype=float
        )
        assert np.isnan(ir_star(signal, signal))


class TestRSquared:
    def test_perfect_prediction(self) -> None:
        actual = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert r_squared(actual, actual) == pytest.approx(1.0)

    def test_mean_prediction(self) -> None:
        actual = pd.Series([1.0, 2.0, 3.0, 4.0])
        predicted = pd.Series([2.5, 2.5, 2.5, 2.5])
        assert r_squared(predicted, actual) == pytest.approx(0.0)

    def test_bad_prediction_negative(self) -> None:
        actual = pd.Series([1.0, 2.0, 3.0])
        predicted = pd.Series([3.0, 1.0, 2.0])  # Worse than mean
        assert r_squared(predicted, actual) < 0

    def test_insufficient_data(self) -> None:
        assert np.isnan(r_squared(pd.Series([1.0]), pd.Series([1.0])))


class TestTstatIc:
    def test_positive_ic_series(self) -> None:
        ics = pd.Series([0.05, 0.08, 0.03, 0.06, 0.04])
        t = tstat_ic(ics)
        assert t > 0

    def test_known_value(self) -> None:
        ics = pd.Series([0.10, 0.20, 0.30])
        mu = ics.mean()
        sigma = ics.std(ddof=1)
        n = len(ics)
        expected = mu / (sigma / np.sqrt(n))
        assert tstat_ic(ics) == pytest.approx(expected)

    def test_single_obs_nan(self) -> None:
        assert np.isnan(tstat_ic(pd.Series([0.05])))
