# Alphaeval Iteration 4 — Comprehensive Review
**Date**: 2026-02-15
**Package**: `packages/alphaeval/` (22 files, ~2156 lines)
**Tests**: 109/109 passing
**Agents**: code-reviewer, silent-failure-hunter, comment-analyzer, pr-test-analyzer, type-design-analyzer

---

## Systemic Finding: Logging-as-Error-Handling Anti-Pattern
- **Source**: silent-failure-hunter
- **Scope**: Package-wide
- **Issue**: Nearly every error path writes to `logger.warning()` then returns plausible-but-wrong values. In a library context, Python logging is almost always unconfigured by callers, meaning warnings vanish. The package silently returns wrong answers for the 3 most common real-world problems: duplicate rows, misaligned indices, and missing data.
- **Principle**: Raise on data integrity violations, warn on edge cases, never silently return wrong answers.

---

## Critical Issues (5 found)

### C1. `_daily_ic` silently uses wrong column with multi-column MultiIndex DataFrames
- **Source**: code-reviewer
- **Severity**: :red_circle: Critical
- **File**: `src/metrics/factor.py:56-59`
- **Issue**: `iloc[:, 0]` blindly selects first column when caller passes MultiIndex DataFrame with multiple columns. Reordering columns changes results with no warning.
- **Fix**: Raise `ValueError` if `signal.shape[1] != 1` in MultiIndex branch.

### C2. VPIN can exceed 1.0 with negative volumes — no input validation
- **Source**: code-reviewer
- **Severity**: :red_circle: Critical
- **File**: `src/metrics/risk.py:55-92`
- **Issue**: Negative `volume_buy`/`volume_sell` produce VPIN > 1.0 (e.g., 1.727). VPIN defined on [0, 1].
- **Fix**: Add warning for negative volumes or clamp to zero.

### C3. `validate_panel` silently preserves duplicate (date, instrument) rows
- **Source**: silent-failure-hunter
- **Severity**: :red_circle: Critical
- **File**: `src/panel.py:72-77`
- **Issue**: Duplicates are logged but returned unchanged. Every downstream metric assumes unique keys. Duplicates cause silent double-counting in IC, corrupted equity curves, wrong VPIN. Warning goes to unconfigured logger.
- **Fix**: Raise `ValueError` or add `allow_duplicates=True` opt-in parameter.

### C4. `vpin` index misalignment — logs but continues with outer-joined data
- **Source**: silent-failure-hunter
- **Severity**: :red_circle: Critical
- **File**: `src/metrics/risk.py:79-83`
- **Issue**: When buy/sell volume indices differ, pandas outer-join silently extends the result. Warning only goes to logger. VPIN computed over distorted window.
- **Fix**: Raise `ValueError` on index mismatch.

### C5. `num_independent_trials` NaN correlation defaults to 1.0, making N_eff=1
- **Source**: silent-failure-hunter
- **Severity**: :red_circle: Critical
- **File**: `src/metrics/sharpe_inference.py:247-252`
- **Issue**: One constant-return trial in 20 → NaN correlation matrix → `avg_corr=NaN` → fallback to 1.0 → `N_eff=1` → DSR=PSR (no selection bias correction). Entire DSR framework silently undermined.
- **Fix**: Filter NaN entries from correlation matrix before averaging rather than defaulting.

---

## Important Issues (12 found)

### I1. `profit_factor` missing `.dropna()` unlike siblings
- **Source**: code-reviewer + comment-analyzer
- **File**: `src/metrics/performance.py:19-39`
- **Issue**: `win_rate_trades` and `expectancy` call `.dropna()` but `profit_factor` does not. Coincidentally correct via boolean masks, but inconsistent and fragile.
- **Fix**: Add `clean = trade_pnls.dropna()` at top.

### I2. `min_track_record_length` n-mismatch with pre-computed sr/sr_std
- **Source**: code-reviewer
- **File**: `src/metrics/sharpe_inference.py:154-202`
- **Issue**: Formula uses `n` from `returns.dropna()` but `sr`/`sr_std` can come from different sample size. Confirmed: values diverge (17.55 vs 63.82).
- **Fix**: Document invariant prominently or accept explicit `n_obs` kwarg.

### I3. `min_track_record_length` doesn't validate `prob`
- **Source**: code-reviewer
- **File**: `src/metrics/sharpe_inference.py:154-202`
- **Issue**: `prob=0.0` → z=-inf → minTRL=inf. `prob=1.0` same. No validation unlike `var_parametric`.
- **Fix**: Add `if not 0 < prob < 1: raise ValueError(...)`.

### I4. `EvalConfig` + `MetricResult` + `REQUIRED_*` — all dead types
- **Source**: code-reviewer + type-design-analyzer
- **File**: `src/types.py`
- **Issue**: Defined and exported but never consumed. No metric function accepts `EvalConfig`, none returns `MetricResult`, no code imports `REQUIRED_*`. Type system is aspirational, not operational.
- **Type scores**: EvalConfig 2.5/10, MetricResult 3.5/10, REQUIRED_* 3.3/10.
- **Fix**: Wire into API or document as "convenience grouping" or remove.

### I5. `num_independent_trials` allows `avg_corr` outside bounds
- **Source**: code-reviewer
- **File**: `src/metrics/sharpe_inference.py:205-255`
- **Issue**: `avg_corr=-1.0, m=10` → N_eff=19, exceeding m. Counter-intuitive.
- **Fix**: Clamp `n_eff = max(1, min(m, n_eff))` or warn when n_eff > m.

### I6. `price_to_returns` produces inf with zero prices
- **Source**: code-reviewer
- **File**: `src/transforms/returns.py:39-40`
- **Issue**: Zero price → inf propagates to equity_curve, drawdown, cagr.
- **Fix**: `result.replace([np.inf, -np.inf], np.nan)` (matches drawdown/runup pattern).

### I7. Docstring gaps — 10 inaccuracies identified
- **Source**: comment-analyzer
- **Files**: performance.py (profit_factor, win_rate, max_runup, sortino), factor.py (_daily_ic), equity.py (drawdown), panel.py (duplicates), relative.py (ddof), sharpe_inference.py (expected_max_sr, minTRL), returns.py (pct_change compat)
- **Fix**: Update docstrings per comment-analyzer recommendations.

### I8. `REQUIRED_TRADES` incompatible with `validate_panel`
- **Source**: type-design-analyzer
- **File**: `src/types.py:62`
- **Issue**: `REQUIRED_TRADES` has `trade_id` instead of `date`, but `validate_panel` always requires `date`. Constant is incompatible with the function it's meant to feed.
- **Fix**: Either extend `validate_panel` or restructure constant.

### I9. `tracking_error` index misalignment — logs but returns result from truncated data
- **Source**: silent-failure-hunter
- **File**: `src/metrics/relative.py:37-44`
- **Issue**: Same pattern as VPIN. Outer-join subtraction, warning to unconfigured logger, computation proceeds on intersection which may be much smaller than expected.
- **Fix**: Raise on misalignment or add `align="inner"` opt-in param.

### I10. `spread_return_proxy`/`dv01_pnl` return all-NaN with only log warning
- **Source**: silent-failure-hunter
- **File**: `src/transforms/spreads.py:52-60, 83-90`
- **Issue**: All-NaN warning only fires when EVERY value is NaN. Partial misalignment (99/100 NaN) gets no warning. All-NaN series fed to equity_curve → fills with 0.0 → flat curve → CAGR=0.
- **Fix**: Raise on all-NaN; warn when >50% NaN.

### I11. `equity_curve` fills NaN with 0.0 silently
- **Source**: silent-failure-hunter
- **File**: `src/transforms/returns.py:66-68`
- **Issue**: Documented behavior but dangerous. 20 missing days treated as 20 flat days. Corrupts all downstream: CAGR, drawdown, Sortino.
- **Fix**: Add `logger.warning` with NaN count/percentage when NaN present.

### I12. `_daily_ic` silently skips dates at DEBUG level only
- **Source**: silent-failure-hunter
- **File**: `src/metrics/factor.py:45-48`
- **Issue**: Dates with <3 instruments skipped at DEBUG level. If most dates have <3 instruments, IC is computed from tiny sample with no WARNING.
- **Fix**: Track skip count, warn when significant fraction skipped.

---

## Suggestions (8 found)

### S1. No test for `_daily_ic` with multi-column MultiIndex input
- **File**: `tests/test_factor.py`

### S2. `spread_return_proxy`/`dv01_pnl` all-NaN warning paths not tested
- **File**: `tests/test_transforms.py`

### S3. Missing `__all__` in metrics modules
- **Files**: performance.py, risk.py, factor.py, sharpe_inference.py

### S4. Import ordering violation in equity.py
- **File**: `src/transforms/equity.py:15`

### S5. `validate_panel` copies entire DataFrame unnecessarily
- **File**: `src/panel.py:45`

### S6. `pct_change(fill_method=None)` forward-incompatible with pandas 3.0
- **File**: `src/transforms/returns.py:42`
- **Fix**: Use `price / price.shift(1) - 1` for all-version compatibility.

### S7. Test helpers duplicated across test files
- **Files**: test_sharpe_inference.py, test_factor.py

### S8. No tests for `drawdown_series`/`runup_series` with empty input
- **File**: `tests/test_transforms.py`

---

## Strengths (What's Done Well)

1. **Consistent NaN handling** — uniform `< 2` guards and `abs(sigma) < 1e-14` epsilon checks
2. **Logging over silent failures** — `logger.warning()` for misalignment, NaN skew/kurtosis
3. **Immutability discipline** — copies before mutation, derived Series for analytics
4. **Clear separation of concerns** — transforms/metrics/panel properly layered
5. **Academic citations** — sharpe_inference.py cites Bailey & Lopez de Prado (2012, 2014)
6. **VaR sign convention documented** — "positive = expected loss" stated explicitly
7. **Proactive misalignment warnings** — tracking_error logs exact counts when observations drop
8. **MetricResult aliasing fix** — `meta: dict | None = None` avoids mutable default

---

## 2am Debug Plan

| Failure Mode | First Place to Look | Log/Assert/Test |
|---|---|---|
| `_daily_ic` wrong column (C1) | `factor.py:57` — `iloc[:, 0]` | Add `assert signal.shape[1] == 1` |
| VPIN > 1.0 (C2) | `risk.py:84-91` — negative volumes | Add negative-volume guard + test VPIN in [0,1] |
| Duplicate rows corrupt metrics (C3) | `panel.py:72-77` — logged but preserved | Raise `ValueError` on duplicates |
| VPIN index mismatch (C4) | `risk.py:79-83` — outer join | Raise `ValueError` on mismatch |
| DSR ignores selection bias (C5) | `sharpe_inference.py:247-252` — NaN corr → 1.0 | Filter NaN from corr matrix |
| minTRL inf for prob~0 (I3) | `sharpe_inference.py:201` — `norm.ppf(0.0)` | Add `0 < prob < 1` validation |
