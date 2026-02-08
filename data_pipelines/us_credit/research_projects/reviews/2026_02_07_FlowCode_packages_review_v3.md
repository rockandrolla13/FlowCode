# Code Review Report (v3)

**Files reviewed:** 18 source files across `packages/{data,signals,metrics,backtest}`
**Date:** 2026-02-07
**Previous review:** `2026_02_07_FlowCode_packages_review_v2.md` (v2)
**Overall health:** 🟢 Good

## Executive Summary

The v2 review confirmed all 7 critical/major findings from v1 are resolved. This v3 review focuses on the 9 remaining v2 findings plus newly identified issues. No new critical or major correctness bugs found. The primary concerns are: (1) dead/redundant code that should be cleaned up, (2) a per-row Python loop in `top_n_positions` that breaks the otherwise consistent vectorization pattern, (3) duplicated metrics logic in `compute_metrics` that drifts from the canonical `metrics` package, and (4) missing `__init__.py` exports that break the public API contract.

## Findings

### 1. Dead branch in `compute_returns`

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness
- **Location:** `packages/backtest/src/engine.py`, lines 55–58

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

WHY: Both branches execute identical code. The `isinstance` check adds no value and misleads readers into expecting different behavior.

---

### 2. `compute_metrics` duplicates metrics package with subtle inconsistency risk

- **Severity:** 🟡 Minor
- **Pillar:** Single Responsibility / DRY
- **Location:** `packages/backtest/src/engine.py`, lines 75–123

BEFORE:
```python
def compute_metrics(returns: pd.Series) -> dict[str, float]:
    metrics = {}
    metrics["total_return"] = float((1 + returns).prod() - 1)
    metrics["volatility"] = float(returns.std() * np.sqrt(252))
    if returns.std() > 0:
        metrics["sharpe_ratio"] = float(
            returns.mean() / returns.std() * np.sqrt(252)
        )
    # ... drawdown, win_rate, skew, kurtosis inline
```

AFTER:
```python
def compute_metrics(returns: pd.Series) -> dict[str, float]:
    from metrics.performance import sharpe_ratio, annualized_return
    from metrics.risk import max_drawdown

    if len(returns) < 2:
        return {}

    return {
        "total_return": float((1 + returns).prod() - 1),
        "mean_return": float(returns.mean()),
        "volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "win_rate": float((returns > 0).mean()),
        "loss_rate": float((returns < 0).mean()),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
    }
```

WHY: The inline Sharpe calculation re-implements `metrics.performance.sharpe_ratio()` and loses the explicit `ddof=1` and `risk_free` parameter handling. The drawdown calculation re-implements `metrics.risk.max_drawdown()`. Two sources of truth for the same formula create drift risk. Delegate to the canonical implementations.

---

### 3. Unnecessary `.copy()` in `aggregate_daily_volume`

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness / Performance
- **Location:** `packages/data/src/trace.py`, lines 138–139

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

WHY: These subsets are immediately consumed by `.groupby().sum()` — no mutation occurs. The `.copy()` allocates memory unnecessarily.

---

### 4. Verbose boolean mask initialization in universe filters

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness
- **Location:** `packages/data/src/universe.py`, lines 169, 219

BEFORE:
```python
mask = pd.Series([True] * len(df), index=df.index)
```

AFTER:
```python
mask = pd.Series(True, index=df.index)
```

WHY: Scalar broadcast avoids constructing a temporary Python list of `len(df)` booleans. Functionally identical, less allocation.

---

### 5. Missing `__init__.py` exports in metrics package

- **Severity:** 🟡 Minor
- **Pillar:** Public API Completeness
- **Location:** `packages/metrics/src/__init__.py`

The following public functions are defined but not exported:
- `risk.volatility`
- `risk.downside_volatility`
- `risk.drawdown_duration`
- `risk.max_drawdown_duration`
- `diagnostics.turnover`
- `diagnostics.holding_period`
- `diagnostics.autocorrelation_profile`
- `performance.information_ratio`

BEFORE:
```python
from .performance import sharpe_ratio, sortino_ratio, calmar_ratio, annualized_return
from .risk import max_drawdown, drawdown_series, value_at_risk, expected_shortfall
from .diagnostics import hit_rate, autocorrelation, signal_decay, information_coefficient
```

AFTER:
```python
from .performance import sharpe_ratio, sortino_ratio, calmar_ratio, annualized_return, information_ratio
from .risk import max_drawdown, drawdown_series, value_at_risk, expected_shortfall, volatility, downside_volatility, drawdown_duration, max_drawdown_duration
from .diagnostics import hit_rate, autocorrelation, autocorrelation_profile, signal_decay, information_coefficient, turnover, holding_period
```

WHY: Users importing `from metrics import volatility` get `ImportError`. Either export these or explicitly document them as internal.

---

### 6. `top_n_positions` per-row Python loop

- **Severity:** 🟡 Minor
- **Pillar:** Performance
- **Location:** `packages/backtest/src/portfolio.py`, lines 184–199

BEFORE:
```python
for date in signal.index:
    row = signal.loc[date].dropna()
    if len(row) == 0:
        continue
    if n_long > 0:
        top = row.nlargest(min(n_long, len(row))).index
        positions.loc[date, top] = 1.0 / n_long
    if n_short > 0:
        bottom = row.nsmallest(min(n_short, len(row))).index
        positions.loc[date, bottom] = -1.0 / n_short
```

AFTER (sketch):
```python
ranks_desc = signal.rank(axis=1, ascending=False, method="first", na_option="bottom")
ranks_asc = signal.rank(axis=1, ascending=True, method="first", na_option="bottom")
valid = signal.notna()

long_mask = (ranks_desc <= n_long) & valid & (n_long > 0)
short_mask = (ranks_asc <= n_short) & valid & (n_short > 0)

positions = (long_mask.astype(float) / n_long) - (short_mask.astype(float) / n_short)
```

WHY: Iterating over every date in Python is O(dates) with pandas overhead per row. The rank-based vectorization pattern matches `equal_weight` and `risk_parity` which were already fixed.

---

### 7. `BacktestResult.__post_init__` accepts `None` for non-Optional fields

- **Severity:** 🔵 Suggestion
- **Pillar:** Semantics & Correctness
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
# Remove __post_init__ entirely — callers should pass valid data.
# If None is a valid input, annotate as `returns: pd.Series | None = None`
# and use a factory default.
```

WHY: The type annotations declare non-Optional fields (`returns: pd.Series`). The `None` guards are unreachable by any correctly-typed caller and suggest a contract mismatch. Either add `| None` to the type hints or remove the guards.

---

### 8. Empty `backtest/src/__init__.py`

- **Severity:** 🔵 Suggestion
- **Pillar:** Public API Completeness
- **Location:** `packages/backtest/src/__init__.py`

BEFORE:
```python
# (empty file)
```

AFTER:
```python
"""FlowCode Backtest Package."""

from .engine import run_backtest, compute_returns, compute_metrics
from .portfolio import equal_weight, risk_parity, top_n_positions
from .results import BacktestResult

__all__ = [
    "run_backtest",
    "compute_returns",
    "compute_metrics",
    "equal_weight",
    "risk_parity",
    "top_n_positions",
    "BacktestResult",
]
```

WHY: Unlike `data`, `signals`, and `metrics`, the `backtest` package exports nothing from its `__init__.py`. Users must use `from backtest.src.engine import run_backtest` instead of `from backtest import run_backtest`.

---

### 9. `compute_returns` has unused `price_col` parameter

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness / Clean API
- **Location:** `packages/backtest/src/engine.py`, lines 27–31

BEFORE:
```python
def compute_returns(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    price_col: str = "price",
) -> pd.Series:
```

AFTER:
```python
def compute_returns(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Series:
```

WHY: `price_col` is accepted as a parameter but never used in the function body. It misleads callers into thinking they can configure a price column. Remove it or implement the logic that uses it.

---

### 10. Duplicated CUSIP validation logic in `validate_trace` and `validate_reference`

- **Severity:** 🟡 Minor
- **Pillar:** DRY / Single Responsibility
- **Location:** `packages/data/src/validation.py`, lines 106–112 and lines 203–209

BEFORE:
```python
# In validate_trace (lines 106-112):
if "cusip" in df.columns:
    cusip_str = df["cusip"].astype(str).str.upper()
    valid_cusip_mask = (cusip_str.str.len() == 9) & cusip_str.str.match(r"^[A-Z0-9]{9}$")
    invalid_cusips = df[~valid_cusip_mask.fillna(False)]
    ...

# In validate_reference (lines 203-209):
if "cusip" in df.columns:
    cusip_str = df["cusip"].astype(str).str.upper()
    valid_cusip_mask = (cusip_str.str.len() == 9) & cusip_str.str.match(r"^[A-Z0-9]{9}$")
    invalid_cusips = df[~valid_cusip_mask.fillna(False)]
    ...
```

AFTER:
```python
def _validate_cusip_column(df: pd.DataFrame) -> tuple[int, str | None]:
    """Validate CUSIP column, return (invalid_count, warning_or_none)."""
    cusip_str = df["cusip"].astype(str).str.upper()
    valid_mask = (cusip_str.str.len() == 9) & cusip_str.str.match(r"^[A-Z0-9]{9}$")
    n_invalid = (~valid_mask.fillna(False)).sum()
    warning = f"Found {n_invalid} invalid CUSIPs" if n_invalid > 0 else None
    return n_invalid, warning
```

WHY: The exact same 4-line block is copy-pasted. Extract to a shared helper to maintain one definition of "valid CUSIP format" for vectorized paths.

---

### 11. `re` import unused in vectorized validation path

- **Severity:** 🟡 Minor
- **Pillar:** Conciseness
- **Location:** `packages/data/src/validation.py`, line 10

The module imports `re` at the top (line 10), which is used by the scalar `validate_cusip()` function. However, the vectorized validation in `validate_trace` and `validate_reference` uses `cusip_str.str.match()` (pandas built-in regex). If `validate_cusip()` is still used externally, `re` is needed for it. But note that `validate_cusip()` itself uses `re.match()` while the vectorized path uses `.str.match()` — these are semantically equivalent but maintained separately.

WHY: Not a bug, but the scalar `validate_cusip()` function is now redundant since the vectorized path handles the same check inline. If `validate_cusip()` is still part of the public API, keep both. Otherwise, `re` can be removed along with the scalar function.

---

## Summary Table

| # | Severity | Pillar | Location | Finding |
|---|----------|--------|----------|---------|
| 1 | 🟡 Minor | Conciseness | `engine.py:55–58` | Dead branch in `compute_returns` |
| 2 | 🟡 Minor | DRY/SRP | `engine.py:75–123` | `compute_metrics` duplicates metrics package |
| 3 | 🟡 Minor | Conciseness | `trace.py:138–139` | Unnecessary `.copy()` |
| 4 | 🟡 Minor | Conciseness | `universe.py:169,219` | Verbose boolean mask init |
| 5 | 🟡 Minor | Public API | `metrics/__init__.py` | 8 missing exports |
| 6 | 🟡 Minor | Performance | `portfolio.py:184–199` | `top_n_positions` per-row loop |
| 7 | 🔵 Suggestion | Semantics | `results.py:45–52` | `None` guards for non-Optional fields |
| 8 | 🔵 Suggestion | Public API | `backtest/__init__.py` | Empty exports |
| 9 | 🟡 Minor | Clean API | `engine.py:27–31` | Unused `price_col` parameter |
| 10 | 🟡 Minor | DRY | `validation.py:106–112,203–209` | Duplicated CUSIP validation block |
| 11 | 🟡 Minor | Conciseness | `validation.py:10` | Scalar `validate_cusip` potentially redundant |

## Positive Highlights

1. **All critical and major issues from v1 remain resolved.** The vectorization work, floating-point fix, and zero-std guard are clean and correct.
2. **Consistent `from __future__ import annotations`** across all source files — Python 3.9 compatibility is uniform.
3. **Strong docstring and type hint coverage** — every public function follows NumPy style with examples.
4. **Clean package boundaries** — `data/` is the only package that reads files; signals/metrics are pure computation; backtest orchestrates. No circular dependencies.
5. **Lookahead prevention** is explicit and well-documented in the backtest engine via `positions.shift(1)`.
