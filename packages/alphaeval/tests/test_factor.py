"""Tests for factor quality metrics — IC*, RankIC*, IR*, R², tstat_ic."""
import numpy as np
import pandas as pd
import pytest

from src.metrics.factor import (
    _daily_ic,
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


class TestDailyIcMultiColumnRejects:
    """C1 fix: MultiIndex signal with >1 column must raise."""

    def test_multi_column_signal_raises(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C"]], names=["date", "instrument"]
        )
        signal = pd.DataFrame(
            np.random.randn(15, 2), index=idx, columns=["sig1", "sig2"]
        )
        target = pd.DataFrame(
            np.random.randn(15, 1), index=idx, columns=["tgt"]
        )
        with pytest.raises(ValueError, match="exactly 1 column"):
            _daily_ic(signal, target)

    def test_multi_column_target_raises(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C"]], names=["date", "instrument"]
        )
        signal = pd.DataFrame(
            np.random.randn(15, 1), index=idx, columns=["sig"]
        )
        target = pd.DataFrame(
            np.random.randn(15, 2), index=idx, columns=["tgt1", "tgt2"]
        )
        with pytest.raises(ValueError, match="exactly 1 column"):
            _daily_ic(signal, target)


class TestDailyIcMethodValidation:
    """method parameter must be 'pearson' or 'spearman'."""

    def test_invalid_method_raises(self) -> None:
        signal, target = _make_panel(n_dates=5, n_instruments=5)
        with pytest.raises(ValueError, match="method must be"):
            _daily_ic(signal, target, method="kendall")

    def test_rangeindex_pivoted_accepted(self) -> None:
        """RangeIndex pivoted DataFrames should work (Finding #1 fix)."""
        rng = np.random.RandomState(42)
        signal = pd.DataFrame(rng.randn(5, 4), columns=["A", "B", "C", "D"])
        target = pd.DataFrame(rng.randn(5, 4), columns=["A", "B", "C", "D"])
        ics = _daily_ic(signal, target)
        assert len(ics) == 5
        # At least some non-NaN (4 instruments >= 3 threshold)
        assert ics.notna().any()

    def test_string_index_pivoted_accepted(self) -> None:
        """String-date index should work (Finding #1 fix)."""
        rng = np.random.RandomState(42)
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        signal = pd.DataFrame(rng.randn(3, 4), index=dates, columns=list("ABCD"))
        target = pd.DataFrame(rng.randn(3, 4), index=dates, columns=list("ABCD"))
        ics = _daily_ic(signal, target)
        assert len(ics) == 3


class TestDailyIcSkipWarning:
    """I12 fix: warn when >20% dates skipped."""

    def test_warns_on_high_skip_pct(self, caplog) -> None:
        """All dates have <3 instruments → >20% skipped → warning."""
        import logging
        dates = pd.bdate_range("2024-01-01", periods=10)
        # Only 2 instruments per date → all dates skipped
        instr = ["A", "B"]
        signal = pd.DataFrame(
            np.random.randn(10, 2), index=dates, columns=instr
        )
        target = pd.DataFrame(
            np.random.randn(10, 2), index=dates, columns=instr
        )
        with caplog.at_level(logging.WARNING, logger="src.metrics.factor"):
            result = _daily_ic(signal, target)
        assert result.isna().all()  # all skipped
        assert "skipped" in caplog.text


class TestIcStarAllNaN:
    """GAP-2: ic_star/rank_ic_star return NaN when all daily ICs are NaN."""

    def test_ic_star_all_nan(self) -> None:
        """All dates <3 instruments → all NaN daily ICs → NaN ic_star."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        # Only 2 instruments per date → <3 → all ICs are NaN
        signal = pd.DataFrame(
            [[1.0, 2.0]] * 5, index=dates, columns=["A", "B"]
        )
        target = pd.DataFrame(
            [[10.0, 20.0]] * 5, index=dates, columns=["A", "B"]
        )
        assert np.isnan(ic_star(signal, target))

    def test_rank_ic_star_all_nan(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        signal = pd.DataFrame(
            [[1.0, 2.0]] * 5, index=dates, columns=["A", "B"]
        )
        target = pd.DataFrame(
            [[10.0, 20.0]] * 5, index=dates, columns=["A", "B"]
        )
        assert np.isnan(rank_ic_star(signal, target))


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

    def test_precomputed_ics(self) -> None:
        """Finding #3: ir_star with pre-computed ICs matches signal/target path."""
        signal, target = _make_panel()
        ir_from_data = ir_star(signal, target)
        ics_series = _daily_ic(signal, target, method="pearson")
        ir_from_ics = ir_star(ics=ics_series)
        assert ir_from_ics == pytest.approx(ir_from_data, abs=1e-10)

    def test_no_args_raises(self) -> None:
        """ir_star with neither signal/target nor ics must raise."""
        with pytest.raises(ValueError, match="provide .* or ics"):
            ir_star()

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

    def test_constant_actual_nan(self) -> None:
        """Zero-variance actual → ss_tot ≈ 0 → NaN."""
        assert np.isnan(r_squared(pd.Series([1.0, 2.0, 3.0]), pd.Series([5.0, 5.0, 5.0])))


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

    def test_constant_ic_zero_std_nan(self) -> None:
        """All ICs identical → std = 0 → NaN."""
        assert np.isnan(tstat_ic(pd.Series([0.05, 0.05, 0.05])))
