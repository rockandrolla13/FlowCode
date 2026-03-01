"""Tests for Sharpe inference — PSR, DSR, minTRL, expected max SR."""
import numpy as np
import pandas as pd
import pytest

from flowcode_alphaeval.metrics.sharpe_inference import (
    estimated_sharpe_ratio,
    ann_estimated_sharpe_ratio,
    estimated_sharpe_ratio_stdev,
    probabilistic_sharpe_ratio,
    min_track_record_length,
    num_independent_trials,
    expected_maximum_sr,
    deflated_sharpe_ratio,
)


def _make_returns(n: int = 500, mu: float = 0.0005, sigma: float = 0.01, seed: int = 42):
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def _make_trials(n_trials: int = 20, n_obs: int = 252, seed: int = 42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        rng.normal(0.0003, 0.01, (n_obs, n_trials)),
        columns=[f"trial_{i}" for i in range(n_trials)],
    )


# ── Estimated SR ────────────────────────────────────────────────────────────

class TestEstimatedSharpeRatio:
    def test_positive(self) -> None:
        r = _make_returns(mu=0.001)
        assert estimated_sharpe_ratio(r) > 0

    def test_known_value(self) -> None:
        r = pd.Series([0.02, 0.04, 0.06, 0.08])
        expected = r.mean() / r.std(ddof=1)
        assert estimated_sharpe_ratio(r) == pytest.approx(expected)

    def test_single_obs_nan(self) -> None:
        assert np.isnan(estimated_sharpe_ratio(pd.Series([0.01])))

    def test_zero_std_nan(self) -> None:
        assert np.isnan(estimated_sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))


class TestAnnEstimatedSR:
    def test_annualization(self) -> None:
        r = _make_returns()
        sr = estimated_sharpe_ratio(r)
        ann = ann_estimated_sharpe_ratio(r, periods=252)
        assert ann == pytest.approx(sr * np.sqrt(252))

    def test_precomputed_sr(self) -> None:
        ann = ann_estimated_sharpe_ratio(sr=0.05, periods=252)
        assert ann == pytest.approx(0.05 * np.sqrt(252))

    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError, match="either 'returns' or 'sr'"):
            ann_estimated_sharpe_ratio()

    def test_zero_std_returns_nan(self) -> None:
        r = pd.Series([0.01, 0.01, 0.01])  # zero std → SR is NaN
        assert np.isnan(ann_estimated_sharpe_ratio(r))


# ── SR stdev ────────────────────────────────────────────────────────────────

class TestSRStdev:
    def test_positive(self) -> None:
        r = _make_returns()
        s = estimated_sharpe_ratio_stdev(r)
        assert s > 0

    def test_insufficient_data(self) -> None:
        assert np.isnan(estimated_sharpe_ratio_stdev(pd.Series([0.01, 0.02])))

    def test_normal_returns_approx(self) -> None:
        """For normal returns, sigma_SR ~ 1/sqrt(n-1)."""
        rng = np.random.RandomState(123)
        r = pd.Series(rng.normal(0, 0.01, 10000))
        s = estimated_sharpe_ratio_stdev(r)
        # For SR near 0 and normal: sigma ~ sqrt(1/(n-1)) ~ 0.01
        assert s == pytest.approx(1.0 / np.sqrt(9999), rel=0.05)


# ── PSR ─────────────────────────────────────────────────────────────────────

class TestSRStdevEdge:
    def test_zero_std_with_explicit_sr_nan(self, caplog) -> None:
        """Zero-variance returns + explicit sr → NaN skew/kurt → NaN + warning."""
        import logging
        r = pd.Series([0.01] * 10)
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.metrics.sharpe_inference"):
            result = estimated_sharpe_ratio_stdev(r, sr=0.5)
        assert np.isnan(result)
        assert "skew/kurtosis NaN" in caplog.text


class TestExpectedMaxSREdge:
    def test_few_valid_srs_nan(self, caplog) -> None:
        """Trials with mostly zero-std columns → few valid SRs → NaN."""
        import logging
        trials = pd.DataFrame({
            "a": [0.01, 0.01, 0.01],  # zero std → SR NaN
            "b": [0.02, 0.02, 0.02],  # zero std → SR NaN
            "c": [0.01, -0.02, 0.03],  # valid
        })
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.metrics.sharpe_inference"):
            result = expected_maximum_sr(trials, independent_trials=2)
        assert np.isnan(result)
        assert "valid trial SRs" in caplog.text


class TestPSR:
    def test_good_strategy_high_psr(self) -> None:
        r = _make_returns(n=500, mu=0.002, sigma=0.01)
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
        assert psr > 0.95

    def test_zero_benchmark(self) -> None:
        r = _make_returns()
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
        assert 0 <= psr <= 1

    def test_monotonic_in_sr(self) -> None:
        """PSR increases as SR increases (same sr_std)."""
        r = _make_returns(n=500)
        sr_std = estimated_sharpe_ratio_stdev(r)
        psr_low = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, sr=0.01, sr_std=sr_std)
        psr_high = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, sr=0.10, sr_std=sr_std)
        assert psr_high > psr_low


# ── minTRL ──────────────────────────────────────────────────────────────────

class TestMinTRL:
    def test_positive(self) -> None:
        r = _make_returns(mu=0.001)
        mtrl = min_track_record_length(r)
        assert mtrl > 0

    def test_increases_with_prob(self) -> None:
        r = _make_returns(mu=0.001)
        mtrl_90 = min_track_record_length(r, prob=0.90)
        mtrl_99 = min_track_record_length(r, prob=0.99)
        assert mtrl_99 > mtrl_90

    def test_sr_below_benchmark_inf(self) -> None:
        r = _make_returns(mu=-0.001)
        sr = estimated_sharpe_ratio(r)
        assert sr < 0, "Seed should produce negative SR for mu=-0.001"
        mtrl = min_track_record_length(r, sr_benchmark=0.0)
        assert mtrl == np.inf

    def test_invalid_prob_raises(self) -> None:
        """I3 fix: prob outside (0,1) must raise ValueError."""
        r = _make_returns()
        with pytest.raises(ValueError, match="prob must be in"):
            min_track_record_length(r, prob=0.0)
        with pytest.raises(ValueError, match="prob must be in"):
            min_track_record_length(r, prob=1.0)
        with pytest.raises(ValueError, match="prob must be in"):
            min_track_record_length(r, prob=-0.5)

    def test_precomputed_sr_logs_debug(self, caplog) -> None:
        """Finding #2: pre-computed sr/sr_std should log debug message."""
        import logging
        r = _make_returns(mu=0.001)
        sr = estimated_sharpe_ratio(r)
        sr_std = estimated_sharpe_ratio_stdev(r, sr=sr)
        with caplog.at_level(logging.DEBUG, logger="flowcode_alphaeval.metrics.sharpe_inference"):
            min_track_record_length(r, sr=sr, sr_std=sr_std)
        assert "pre-computed sr=" in caplog.text
        assert "pre-computed sr_std=" in caplog.text


# ── N_eff ───────────────────────────────────────────────────────────────────

class TestNumIndependentTrials:
    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError):
            num_independent_trials()

    def test_m_without_corr_raises(self) -> None:
        with pytest.raises(ValueError):
            num_independent_trials(m=10)
    def test_uncorrelated_equals_m(self) -> None:
        n = num_independent_trials(m=20, avg_corr=0.0)
        assert n == 20

    def test_perfectly_correlated_equals_one(self) -> None:
        n = num_independent_trials(m=20, avg_corr=1.0)
        assert n == 1

    def test_from_data(self) -> None:
        trials = _make_trials()
        n = num_independent_trials(trials)
        assert 1 <= n <= trials.shape[1]  # clamped to [1, m]

    def test_all_nan_corr_raises(self) -> None:
        """C5 fix: all pairwise correlations NaN must raise ValueError."""
        # Single-row DataFrame → all correlations NaN
        trials = pd.DataFrame({"a": [0.01], "b": [0.02], "c": [0.03]})
        with pytest.raises(ValueError, match="all pairwise correlations are NaN"):
            num_independent_trials(trials)

    def test_partial_nan_corr_warns(self, caplog) -> None:
        """C5 fix: some NaN correlations → warning + uses valid pairs."""
        import logging
        rng = np.random.RandomState(42)
        trials = pd.DataFrame({
            "a": rng.normal(0, 0.01, 100),
            "b": rng.normal(0, 0.01, 100),
            "constant": [0.01] * 100,  # constant → NaN corr with others
        })
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.metrics.sharpe_inference"):
            n = num_independent_trials(trials)
        assert 1 <= n <= 3
        assert "NaN" in caplog.text

    def test_nan_avg_corr_raises(self) -> None:
        """C5 fix: explicit NaN avg_corr must raise."""
        with pytest.raises(ValueError, match="avg_corr is NaN"):
            num_independent_trials(m=10, avg_corr=float("nan"))


# ── Expected Max SR ─────────────────────────────────────────────────────────

class TestExpectedMaxSR:
    def test_increases_with_trials(self) -> None:
        ems_few = expected_maximum_sr(
            independent_trials=5, trials_sr_std=0.3, expected_mean_sr=0.5
        )
        ems_many = expected_maximum_sr(
            independent_trials=100, trials_sr_std=0.3, expected_mean_sr=0.5
        )
        assert ems_many > ems_few

    def test_single_trial_equals_mean(self) -> None:
        ems = expected_maximum_sr(
            independent_trials=1, trials_sr_std=0.3, expected_mean_sr=0.5
        )
        assert ems == pytest.approx(0.5)

    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_maximum_sr()

    def test_insufficient_valid_trials_raises(self) -> None:
        """DataFrame with 1-row trials → all corrs NaN → raises ValueError."""
        trials = pd.DataFrame({"a": [0.01], "b": [0.02], "c": [0.03]})
        with pytest.raises(ValueError, match="all pairwise correlations are NaN"):
            expected_maximum_sr(trials)

    def test_nan_trials_sr_std_nan(self) -> None:
        assert np.isnan(expected_maximum_sr(
            independent_trials=5, trials_sr_std=np.nan, expected_mean_sr=0.5
        ))


# ── DSR ─────────────────────────────────────────────────────────────────────

class TestDSR:
    def test_dsr_leq_psr(self) -> None:
        """DSR <= PSR when accounting for selection bias."""
        trials = _make_trials(n_trials=20)
        # Pick the "best" trial
        srs = trials.apply(lambda c: c.mean() / c.std(ddof=1))
        best_col = srs.idxmax()
        selected = trials[best_col]

        psr = probabilistic_sharpe_ratio(selected, sr_benchmark=0.0)
        dsr = deflated_sharpe_ratio(trials, selected, expected_mean_sr=0.0)
        assert dsr <= psr + 1e-10  # DSR accounts for selection bias

    def test_no_selected_returns_raises(self) -> None:
        with pytest.raises(ValueError, match="returns_selected"):
            deflated_sharpe_ratio(expected_max_sr=1.0)

    def test_no_args_raises(self) -> None:
        r = _make_returns()
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(returns_selected=r)


# ── PSR edge ───────────────────────────────────────────────────────────────

class TestPSREdge:
    def test_sr_std_zero_returns_nan(self) -> None:
        r = _make_returns()
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, sr=0.05, sr_std=0.0)
        assert np.isnan(psr)
