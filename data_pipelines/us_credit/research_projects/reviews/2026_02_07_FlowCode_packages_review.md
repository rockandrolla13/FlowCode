# Code Review Report

**Files reviewed:** 18 source files + 8 test files across `packages/{data,signals,metrics,backtest}`
**Date:** 2026-02-07
**Overall health:** 🟡 Needs attention

## Executive Summary

The codebase is well-structured with clear package boundaries, consistent docstrings, and good test coverage for core business logic. The dominant patterns requiring attention are: (1) Python 3.10+ type hint syntax (`int | None`, `str | Path`) used throughout, which breaks on Python 3.9 runtimes; (2) `compute_zscore` divides by `rolling_std` without masking zero-variance windows *before* the division, relying on post-hoc `inf` replacement that leaves a correctness gap; (3) several scalar functions (`is_subpenny`, `is_retail_trade`) are called from vectorized contexts via `.apply()` patterns without boundary guards, and `is_subpenny` has a floating-point precision bug. Top priority: fix the division-by-zero masking in `compute_zscore` and the `is_subpenny` floating-point comparison.

## Findings

### 1. `is_subpenny` floating-point precision bug
- **Severity:** 🔴 Critical
- **Pillar:** Semantics & Correctness
- **Location:** `packages/signals/src/retail.py`, lines 59–60

BEFORE:
```python
return (price * 100) % 1 > 0
```

AFTER:
```python
cents = round(price * 100, 6)
frac = cents % 1
return frac > 1e-9
```

WHY:
Floating-point representation means `100.10 * 100 = 10010.000000000002`, so `% 1 > 0` returns `True` for prices that have *no* subpenny component — silently misclassifying institutional trades as retail.

`[SUGGEST: add test for is_subpenny(100.10), is_subpenny(99.99), is_subpenny(50.01) — all should be False]`

---

### 2. `compute_zscore` division by zero not masked before division
- **Severity:** 🔴 Critical
- **Pillar:** Zero-Variance / Division-by-Zero Safety
- **Location:** `packages/signals/src/triggers.py`, lines 71–72

BEFORE:
```python
zscore = (series - rolling_mean) / rolling_std
zscore = zscore.replace([np.inf, -np.inf], np.nan)
```

AFTER:
```python
mask = rolling_std == 0
safe_std = rolling_std.where(~mask, np.nan)
zscore = (series - rolling_mean) / safe_std
```

WHY:
Dividing by zero produces `inf`/`-inf` which is then cleaned up, but between the division and the replace, any downstream code that intercepts `zscore` (e.g. via logging or a subclass override) sees `inf`. Per the standards, zero-variance should yield `NaN` *without* ever producing infinity. The post-hoc replace is a band-aid that can fail if the series contains `inf` from legitimate data.

---

### 3. Python 3.10+ union type syntax used throughout
- **Severity:** 🟠 Major
- **Pillar:** Compatibility & Runtime Safety
- **Location:** Multiple files — `triggers.py:26,82`, `trace.py:21–22`, `reference.py:17`, `credit.py:119`, `universe.py:137,191–192`, `config.py:13,51`, `validation.py:55,152`, `engine.py:167`, `diagnostics.py:128`

BEFORE:
```python
min_periods: int | None = None
path: str | Path
```

AFTER:
```python
from typing import Optional, Union
min_periods: Optional[int] = None
path: Union[str, Path]
```

WHY:
`X | Y` type union syntax requires Python 3.10+ (or `from __future__ import annotations`). If the project targets 3.9 this is a runtime `TypeError`. If 3.10+ is the floor, this should be documented explicitly.

---

### 4. `classify_retail_trades` duplicates scalar `is_subpenny` logic
- **Severity:** 🟠 Major
- **Pillar:** Conciseness / DRY
- **Location:** `packages/signals/src/retail.py`, lines 144–146 vs. 59–60

BEFORE:
```python
# In classify_retail_trades (vectorized):
has_subpenny = (trades[price_col] * 100) % 1 > 0

# In is_subpenny (scalar):
return (price * 100) % 1 > 0
```

AFTER:
```python
# Single vectorized helper; scalar version calls it on a 1-element Series
# or define is_subpenny_mask(prices: pd.Series) -> pd.Series
```

WHY:
The same floating-point bug from Finding #1 is duplicated. If one is fixed, the other must be fixed too. Extracting a shared implementation prevents drift. Per the standards: "Never call scalar-only helpers on Series/arrays. Provide `*_scalar` / `*_vectorized` variants."

---

### 5. `validate_trace` uses `.apply(validate_cusip)` — Python loop on potentially large DataFrame
- **Severity:** 🟠 Major
- **Pillar:** Performance & Allocation (Hot Paths)
- **Location:** `packages/data/src/validation.py`, line 105

BEFORE:
```python
invalid_cusips = df[~df["cusip"].apply(validate_cusip)]
```

AFTER:
```python
valid_mask = df["cusip"].str.match(r"^[A-Z0-9]{9}$", case=False) & (df["cusip"].str.len() == 9)
```

WHY:
`.apply()` on a per-row scalar function is a Python loop in disguise. For large TRACE datasets (millions of rows), this is a performance bottleneck. The regex check can be fully vectorized via `.str.match()`. Same issue exists in `validate_reference` line 200.

---

### 6. `generate_trades` iterates rows with nested Python loop
- **Severity:** 🟠 Major
- **Pillar:** Performance & Allocation (Hot Paths)
- **Location:** `packages/backtest/src/engine.py`, lines 147–158

BEFORE:
```python
for date in position_changes.index[1:]:
    for asset in position_changes.columns:
        change = position_changes.loc[date, asset]
        if change != 0:
            ...
```

AFTER:
```python
# Stack to long format, filter non-zero, vectorized
stacked = position_changes.iloc[1:].stack()
trades_df = stacked[stacked != 0].reset_index()
trades_df.columns = ["date", "asset", "change"]
trades_df["side"] = np.where(trades_df["change"] > 0, "buy", "sell")
trades_df["size"] = trades_df["change"].abs()
```

WHY:
Double Python loop over `dates x assets` creates O(n*m) Python iterations. For a portfolio of 500 assets over 1000 days, this is 500,000 iterations. The stack + filter pattern is fully vectorized.

---

### 7. `equal_weight` and `risk_parity` use per-row Python loops
- **Severity:** 🟠 Major
- **Pillar:** Performance & Allocation (Hot Paths)
- **Location:** `packages/backtest/src/portfolio.py`, lines 63–79 (`equal_weight`) and 140–170 (`risk_parity`)

BEFORE:
```python
for date in signal.index:
    row = directions.loc[date]
    nonzero = row[row != 0]
    ...
    positions.loc[date, nonzero.index] = nonzero * weight
```

AFTER:
```python
# Vectorized equal_weight sketch:
counts = (directions != 0).sum(axis=1).clip(lower=1)
positions = directions.div(counts, axis=0)
```

WHY:
Per-row iteration in portfolio construction is a hot path during backtesting. Both `equal_weight` and `risk_parity` iterate every date, creating per-row Series objects. Vectorized `div` with broadcasting eliminates the loop entirely for the common case (no `max_positions` constraint).

---

### 8. `compute_returns` has dead branch
- **Severity:** 🟡 Minor
- **Pillar:** Conciseness / Dead Code
- **Location:** `packages/backtest/src/engine.py`, lines 53–56

BEFORE:
```python
if isinstance(prices, pd.Series):
    asset_returns = prices.pct_change()
else:
    asset_returns = prices.pct_change()
```

AFTER:
```python
asset_returns = prices.pct_change()
```

WHY:
Both branches execute identical code. The conditional is dead logic.

---

### 9. `compute_metrics` duplicates `metrics.risk.max_drawdown` and `metrics.performance.sharpe_ratio`
- **Severity:** 🟡 Minor
- **Pillar:** DRY / Package Boundaries / Consistency
- **Location:** `packages/backtest/src/engine.py`, lines 93–120

BEFORE:
```python
metrics["volatility"] = float(returns.std() * np.sqrt(252))
metrics["sharpe_ratio"] = float(returns.mean() / returns.std() * np.sqrt(252))
cum_returns = (1 + returns).cumprod()
running_max = cum_returns.cummax()
drawdown = (cum_returns - running_max) / running_max
metrics["max_drawdown"] = float(drawdown.min())
```

AFTER:
```python
from metrics.performance import sharpe_ratio
from metrics.risk import max_drawdown, volatility
metrics["sharpe_ratio"] = sharpe_ratio(returns)
metrics["max_drawdown"] = max_drawdown(returns)
metrics["volatility"] = volatility(returns)
```

WHY:
The `metrics` package already implements these calculations with proper edge-case handling (e.g. `ddof=1`, zero-std guards). The inline recomputation in `engine.py` uses `ddof=0` (pandas default for `.std()`), diverging from `sharpe_ratio()` which uses `ddof=1`. This is a subtle inconsistency that could produce different Sharpe values depending on which code path is used.

---

### 10. `aggregate_daily_volume` uses `.copy()` unnecessarily on filter results
- **Severity:** 🟡 Minor
- **Pillar:** Conciseness / Unnecessary Allocation
- **Location:** `packages/data/src/trace.py`, lines 136–137

BEFORE:
```python
buys = trades[trades[side_col] == "B"].copy()
sells = trades[trades[side_col] == "S"].copy()
```

AFTER:
```python
buys = trades[trades[side_col] == "B"]
sells = trades[trades[side_col] == "S"]
```

WHY:
The filtered DataFrames are immediately grouped and aggregated — never mutated. The `.copy()` allocates memory for no reason.

---

### 11. `filter_by_rating` and `filter_by_liquidity` create `pd.Series([True] * len(df))` mask
- **Severity:** 🟡 Minor
- **Pillar:** Conciseness / Allocation
- **Location:** `packages/data/src/universe.py`, lines 167 and 217

BEFORE:
```python
mask = pd.Series([True] * len(df), index=df.index)
```

AFTER:
```python
mask = pd.Series(True, index=df.index)
```

WHY:
`pd.Series(True, index=df.index)` achieves the same result without constructing a Python list of `len(df)` booleans first.

---

### 12. `metrics/__init__.py` exports don't include `volatility`, `downside_volatility`, `turnover`, `holding_period`
- **Severity:** 🟡 Minor
- **Pillar:** Public API Contract
- **Location:** `packages/metrics/src/__init__.py`, lines 17–37

BEFORE:
```python
from .risk import max_drawdown, drawdown_series, value_at_risk, expected_shortfall
from .diagnostics import hit_rate, autocorrelation, signal_decay, information_coefficient
```

AFTER:
```python
from .risk import max_drawdown, drawdown_series, value_at_risk, expected_shortfall, volatility, downside_volatility
from .diagnostics import hit_rate, autocorrelation, signal_decay, information_coefficient, turnover, holding_period
```

WHY:
Six public functions (`volatility`, `downside_volatility`, `drawdown_duration`, `max_drawdown_duration`, `turnover`, `holding_period`) are defined but not exported from the package `__init__.py`. Users importing `from metrics import volatility` will get `ImportError`.

---

### 13. Test files use unseeded RNG
- **Severity:** 🟡 Minor
- **Pillar:** Testing — Determinism
- **Location:** `packages/backtest/tests/test_engine.py`, lines 109–116; `packages/metrics/tests/test_risk.py`, line 168

BEFORE:
```python
signal = pd.DataFrame(
    {"A": np.random.randn(10), "B": np.random.randn(10)},
    ...
)
```

AFTER:
```python
rng = np.random.default_rng(42)
signal = pd.DataFrame(
    {"A": rng.standard_normal(10), "B": rng.standard_normal(10)},
    ...
)
```

WHY:
Unseeded random tests are non-deterministic and can flake. Per the standards: "Fixed RNG seeds, fixed timestamps, or mocked time sources."

---

### 14. `BacktestResult.__post_init__` checks `if self.returns is None` but dataclass field is typed `pd.Series` (not Optional)
- **Severity:** 🔵 Suggestion
- **Pillar:** Compatibility & Runtime Safety
- **Location:** `packages/backtest/src/results.py`, lines 45–52

BEFORE:
```python
def __post_init__(self) -> None:
    if self.returns is None:
        self.returns = pd.Series(dtype=float)
    if self.positions is None:
        self.positions = pd.DataFrame()
    if self.trades is None:
        self.trades = pd.DataFrame()
```

AFTER:
```python
# Either type fields as Optional[pd.Series] to match the None check,
# or remove the None guards and rely on the type system.
```

WHY:
The type annotations say `pd.Series` (not Optional), so `None` should never arrive. Either the types are wrong or the guards are dead code. Keeping both creates confusion about the contract.

---

### 15. `config.load_config` returns untyped `dict` — no schema validation
- **Severity:** 🔵 Suggestion
- **Pillar:** Configuration Management
- **Location:** `packages/data/src/config.py`, lines 13–48

BEFORE:
```python
def load_config(path: str | Path) -> dict[str, Any]:
    ...
    return config if config is not None else {}
```

AFTER:
```python
# Sketch: typed config with validation
@dataclass
class ZScoreConfig:
    window: int = 252
    threshold: float = 7.0
    min_periods: Optional[int] = None

def load_zscore_config(path: ...) -> ZScoreConfig:
    raw = _load_yaml(path)
    return ZScoreConfig(**raw["zscore"])
```

WHY:
Per the standards: "Single factory — centralize config parsing in one module, never parse raw dicts inline. Typed schema (dataclass/pydantic), validate on load." Currently all config consumers must parse raw dicts and handle missing keys manually.

---

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | 🔴 Critical | Semantics & Correctness | `retail.py:59–60` | `is_subpenny` floating-point precision bug |
| 2 | 🔴 Critical | Division-by-Zero Safety | `triggers.py:71–72` | `compute_zscore` divides by zero before masking |
| 3 | 🟠 Major | Compatibility | 11+ files | Python 3.10+ `X \| Y` union syntax |
| 4 | 🟠 Major | DRY | `retail.py:144` | Duplicated subpenny logic (scalar vs vectorized) |
| 5 | 🟠 Major | Performance | `validation.py:105` | `.apply()` on scalar function for CUSIP validation |
| 6 | 🟠 Major | Performance | `engine.py:147–158` | Nested Python loop in `generate_trades` |
| 7 | 🟠 Major | Performance | `portfolio.py:63–79,140–170` | Per-row loops in position sizing |
| 8 | 🟡 Minor | Dead Code | `engine.py:53–56` | Identical if/else branches in `compute_returns` |
| 9 | 🟡 Minor | DRY / Consistency | `engine.py:93–120` | `compute_metrics` reimplements `metrics` package (with different `ddof`) |
| 10 | 🟡 Minor | Allocation | `trace.py:136–137` | Unnecessary `.copy()` before aggregation |
| 11 | 🟡 Minor | Allocation | `universe.py:167,217` | Verbose boolean mask construction |
| 12 | 🟡 Minor | Public API | `metrics/__init__.py` | 6 public functions not exported |
| 13 | 🟡 Minor | Testing | `test_engine.py`, `test_risk.py` | Unseeded RNG in tests |
| 14 | 🔵 Suggestion | Type Safety | `results.py:45–52` | None guards conflict with non-Optional type hints |
| 15 | 🔵 Suggestion | Config Management | `config.py:13–48` | Raw dict return, no typed schema |

## Positive Highlights

1. **Clean package boundaries** — The `data/` package is the sole file reader, signals receive DataFrames, metrics are pure functions. The dependency graph matches the spec exactly.
2. **Thorough docstrings** — Every public function has NumPy-style docstrings with Parameters, Returns, Examples, and Notes sections. This is exemplary documentation.
3. **Good edge-case awareness in tests** — Tests cover zero-spread, zero-volume, empty DataFrames, insufficient history, and boundary conditions. The test structure (one class per function) is clean and navigable.
