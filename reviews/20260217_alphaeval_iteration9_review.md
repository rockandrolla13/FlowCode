# Code Review Report — alphaeval (Iteration 9)

**Files reviewed:** `src/__init__.py`, `src/types.py`, `src/panel.py`, `src/transforms/{returns,spreads,equity}.py`, `src/metrics/{performance,risk,factor,relative,sharpe_inference}.py`, all 7 test files
**Date:** 2026-02-17
**Overall health:** 🟡 Needs attention
**Tests:** 141 passed, 0 failed, 3 expected warnings

## Executive Summary

The alphaeval package is functionally complete with solid test coverage (141 tests). However, this review identifies **2 critical** correctness issues (DatetimeIndex fragility in `_daily_ic`, silent n-mismatch in `min_track_record_length`), **5 major** issues (API inconsistency, redundant computation, missing validation), and **5 minor** style/consistency items. The dominant pattern is inconsistent defensive programming — some functions have thorough validation while structurally similar siblings have none.

## Findings

### 1. _daily_ic rejects valid non-DatetimeIndex pivoted DataFrames
- **Severity:** 🔴 Critical
- **Pillar:** Correctness / Robustness
- **Location:** `src/metrics/factor.py`, line 43

BEFORE:
```python
if isinstance(signal.index, pd.DatetimeIndex) and signal.ndim == 2:
```

AFTER:
```python
if not isinstance(signal.index, pd.MultiIndex) and signal.ndim == 2:
```

WHY:
PeriodIndex, string-date Index, or integer-date Index all silently fall through to ValueError at line 85. The intent is "pivoted (date × instrument)" — the check should exclude MultiIndex, not require DatetimeIndex specifically. `validate_panel` coerces to datetime, but `_daily_ic` doesn't require panel validation first.

---

### 2. min_track_record_length: pre-computed sr/sr_std can silently corrupt result
- **Severity:** 🔴 Critical
- **Pillar:** Correctness / Silent data corruption
- **Location:** `src/metrics/sharpe_inference.py`, lines 206–224

BEFORE:
```python
clean = returns.dropna()
n = len(clean)
# ...later...
return float(1 + sr_std**2 * (n - 1) * (z / denom) ** 2)
```

AFTER:
```python
# Add runtime check when pre-computed values are passed
if sr is not None or sr_std is not None:
    # Warn if n differs from what sr_std likely assumed
    pass  # see implementation note
```

WHY:
`sr_std` encodes `sqrt(numerator / (n_original - 1))` from the sample it was computed on. The formula `sr_std**2 * (n - 1)` uses the *current* n, producing wrong minTRL when n differs from n_original. The docstring warns, but a runtime check or parameter for `n_original` would prevent silent corruption.

---

### 3. ir_star/tstat_ic API inconsistency — can't share IC computation
- **Severity:** 🟠 Major
- **Pillar:** API design / DRY
- **Location:** `src/metrics/factor.py`, lines 138–163 vs 196–217

BEFORE:
```python
def ir_star(signal: pd.DataFrame, target: pd.DataFrame) -> float:
    ics = _daily_ic(signal, target, method="pearson")  # computes internally
    # ...

def tstat_ic(ic_series: pd.Series) -> float:  # takes pre-computed
```

AFTER:
```python
def ir_star(
    signal: pd.DataFrame | None = None,
    target: pd.DataFrame | None = None,
    *,
    ics: pd.Series | None = None,
) -> float:
```

WHY:
Users calling both `ir_star` and `tstat_ic` must either call `_daily_ic` directly (private API) or redundantly compute ICs twice. Both functions should accept pre-computed ICs.

---

### 4. panel.py computes duplicated() twice
- **Severity:** 🟠 Major
- **Pillar:** Performance
- **Location:** `src/panel.py`, lines 75–76

BEFORE:
```python
if df.index.duplicated().any():
    n_dupes = int(df.index.duplicated().sum())
```

AFTER:
```python
dupes = df.index.duplicated()
if dupes.any():
    n_dupes = int(dupes.sum())
```

WHY:
`duplicated()` is O(n) and called twice on potentially large panels. Cache the result.

---

### 5. profit_factor: redundant abs()
- **Severity:** 🟠 Major
- **Pillar:** Clarity / Dead code
- **Location:** `src/metrics/performance.py`, line 46

BEFORE:
```python
losses = clean[clean < 0].abs().sum()
if abs(losses) < 1e-14:
```

AFTER:
```python
losses = clean[clean < 0].abs().sum()
if losses < 1e-14:
```

WHY:
`losses` is already a non-negative sum of absolute values. The outer `abs()` is dead code that obscures intent.

---

### 6. delta_spread_bp: no validation unlike siblings
- **Severity:** 🟠 Major
- **Pillar:** Consistency / Defensive programming
- **Location:** `src/transforms/spreads.py`, lines 15–28

BEFORE:
```python
def delta_spread_bp(spread: pd.Series) -> pd.Series:
    return spread.diff() * 1e4
```

AFTER:
```python
def delta_spread_bp(spread: pd.Series) -> pd.Series:
    result = spread.diff() * 1e4
    # Add inf guard matching siblings
    return result
```

WHY:
`spread_return_proxy` and `dv01_pnl` in the same file both validate for all-NaN results and warn on >50% NaN. `delta_spread_bp` has zero validation. All three should follow the same defensive pattern. [SUGGEST: add test for inf/NaN edge cases]

---

### 7. _daily_ic loop is O(T×N) Python — not vectorized
- **Severity:** 🟠 Major
- **Pillar:** Performance
- **Location:** `src/metrics/factor.py`, lines 45–67

BEFORE:
```python
for dt in dates:
    s = signal.loc[dt].dropna()
    t = target.loc[dt].dropna()
    common = s.index.intersection(t.index)
    # ...
```

AFTER:
```python
# Use groupby + corrwith for vectorized daily correlation
# (sketch only — handles NaN and <3 instruments per group)
```

WHY:
For a panel with 1000 dates × 500 instruments, the Python loop with per-date `.loc[]` and `.intersection()` is significantly slower than a vectorized `groupby` approach. This is the most compute-intensive function in the package.

---

### 8. drawdown_series docstring claims [0, 1] but doesn't enforce
- **Severity:** 🟡 Minor
- **Pillar:** Documentation accuracy
- **Location:** `src/transforms/equity.py`, line 30

BEFORE:
```python
    Values in [0, 1] for normal equity curves. NaN where peak is zero.
```

AFTER:
```python
    Values >= 0 for normal equity curves; can exceed 1.0 for leveraged
    returns below -100%. NaN where peak is zero.
```

WHY:
A return of -150% (leveraged) produces equity going negative, and drawdown > 1.0. The docstring is misleading for leveraged strategies.

---

### 9. num_independent_trials: negative correlation silently clamped
- **Severity:** 🟡 Minor
- **Pillar:** Documentation / Surprise behavior
- **Location:** `src/metrics/sharpe_inference.py`, line 288

BEFORE:
```python
n_eff = max(1, min(m, n_eff))  # clamp to [1, m]
```

AFTER:
```python
# Document: negative avg_corr would yield n_eff > m; clamped to m.
n_eff = max(1, min(m, n_eff))
```

WHY:
When avg_corr < 0, the formula gives N_eff > m (negatively correlated trials are "more than independent"). The clamp silently discards this, which may surprise users. Add a debug log or docstring note.

---

### 10. metrics/__init__.py is empty — no re-exports
- **Severity:** 🟡 Minor
- **Pillar:** API surface consistency
- **Location:** `src/metrics/__init__.py`

BEFORE:
```python
"""alphaeval metrics — performance, risk, factor, and Sharpe inference."""
```

AFTER:
```python
"""..."""
from .performance import profit_factor, win_rate_trades, ...
from .risk import var_parametric, vpin
# etc.
```

WHY:
`transforms/__init__.py` properly re-exports all 7 functions. `metrics/__init__.py` exports nothing, forcing callers to import from `src.metrics.performance` etc. Inconsistent subpackage API surface.

---

### 11. r_squared takes Series while sibling functions take DataFrame
- **Severity:** 🟡 Minor
- **Pillar:** API consistency
- **Location:** `src/metrics/factor.py`, lines 166–193

WHY:
`ic_star`, `rank_ic_star`, `ir_star` all take `pd.DataFrame` (pivoted or MultiIndex). `r_squared` takes `pd.Series`. Users must extract Series from their panel DataFrames to use it alongside the other factor metrics.

---

### 12. types.py constants not exported
- **Severity:** 🟡 Minor
- **Pillar:** API completeness
- **Location:** `src/types.py`, lines 60–64

WHY:
`REQUIRED_TIMESERIES`, `REQUIRED_SPREAD`, `REQUIRED_FACTOR`, `REQUIRED_TRADES` are module-level constants but not in `__init__.py`'s `__all__`. Users must import from `src.types` directly.

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | 🔴 Critical | Correctness | factor.py:43 | _daily_ic rejects valid non-DatetimeIndex pivoted DataFrames |
| 2 | 🔴 Critical | Silent corruption | sharpe_inference.py:206-224 | minTRL silently wrong with pre-computed sr/sr_std from different n |
| 3 | 🟠 Major | API design | factor.py:138-163 | ir_star/tstat_ic API inconsistency — can't share IC computation |
| 4 | 🟠 Major | Performance | panel.py:75-76 | duplicated() computed twice on large panels |
| 5 | 🟠 Major | Dead code | performance.py:46 | Redundant abs() on already-positive value |
| 6 | 🟠 Major | Consistency | spreads.py:15-28 | delta_spread_bp has zero validation unlike sibling functions |
| 7 | 🟠 Major | Performance | factor.py:45-67 | _daily_ic Python loop is O(T×N) — should be vectorized |
| 8 | 🟡 Minor | Docs accuracy | equity.py:30 | Drawdown docstring claims [0,1] but can exceed 1.0 |
| 9 | 🟡 Minor | Surprise behavior | sharpe_inference.py:288 | Negative correlation silently clamped |
| 10 | 🟡 Minor | API consistency | metrics/__init__.py | Empty init vs transforms' full re-exports |
| 11 | 🟡 Minor | API consistency | factor.py:166 | r_squared takes Series while siblings take DataFrame |
| 12 | 🟡 Minor | API completeness | types.py:60-64 | REQUIRED_* constants not exported |

## Positive Highlights

1. **Comprehensive error-path testing** — 20+ tests specifically for edge cases (empty inputs, NaN, zero-variance, misaligned indices, negative volumes). Error paths are well-covered.
2. **Correct Sortino population convention** — The 1/T downside deviation is clearly documented, tested with `test_population_convention`, and cross-referenced against the packages/metrics ddof=1 variant.
3. **NamedTuple for MetricResult** — Using NamedTuple with `meta: dict | None = None` correctly avoids mutable default aliasing while keeping the type hashable and immutable.
