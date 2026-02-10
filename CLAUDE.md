# FlowCode Credit Analytics

## Ownership Model

| Domain | Owner | Agent Role |
|--------|-------|------------|
| Problem definition | Human | Clarify ambiguity |
| Architecture decisions | Human | Propose, don't decide |
| Implementation | Agent | Execute within bounds |
| Verification | Human | Agent prepares evidence |

## Comprehension Gates (REQUIRED before merge)

### 1. Sixty-Second Explain
Can you explain the approach, data flow, and control flow **without opening the diff**?

```
Data flow: [source] → [transform] → [sink]
Control flow: [entry point] → [decision points] → [exit]
```

### 2. Change Simulation
Describe ONE plausible follow-up change and identify:
- Which file(s)?
- Which function(s)?
- What would break if done wrong?

### 3. 2am Debug Plan
List TOP 3 failure modes for this change:

| Failure Mode | First Place to Look | Log/Assert/Test |
|--------------|---------------------|-----------------|
| 1. | | |
| 2. | | |
| 3. | | |

---

## Code Standards

### Type Hints Required
All functions must have type hints:
```python
def compute_imbalance(buys: pd.Series, sells: pd.Series) -> pd.Series:
    """All functions must have type hints and docstrings."""
```

### Docstring Format (NumPy Style)
```python
def zscore(x: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling z-score.

    Parameters
    ----------
    x : pd.Series
        Input time series, indexed by date.
    window : int, default 252
        Lookback window in trading days.

    Returns
    -------
    pd.Series
        Z-score series, NaN for insufficient history.

    Notes
    -----
    Uses sample std (ddof=1). First `window-1` values are NaN.
    """
```

### DataFrame Conventions
- Index: Always `DatetimeIndex` named `'date'`
- Columns: `snake_case`, e.g., `'spread_change'`, `'retail_imbalance'`
- MultiIndex for panel: `['date', 'cusip']` or `['date', 'isin']`

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Signals | `snake_case`, verb prefix | `compute_imbalance()` |
| Metrics | `snake_case`, noun | `sharpe_ratio()` |
| Classes | `PascalCase` | `RetailImbalanceSignal` |
| Constants | `UPPER_SNAKE` | `ZSCORE_THRESHOLD = 7.0` |

---

## Configuration

Parameters live in `conf/`:

```
conf/
├── base.yaml           # Paths, shared defaults
└── signals/
    ├── zscore.yaml     # window: 252, threshold: 7.0
    └── streak.yaml     # min_streak: 3
```

Load via:
```python
from data.config import load_config
cfg = load_config("conf/signals/zscore.yaml")
```

---

## Package Boundaries

```
packages/
├── data/           # Data loading — ONLY package that reads files
├── core/           # EXISTING LIBRARY — DO NOT MODIFY
├── signals/        # Thin wrappers over core
├── metrics/        # Performance & risk metrics
└── backtest/       # Strategy harness
```

### Boundary Rules
1. `data/` READS files, returns DataFrames
2. `core/` RECEIVES DataFrames from `data/`, never reads files
3. `signals/` IMPORTS from `core/` and `data/`, never duplicates
4. `metrics/` RECEIVES signal output, computes performance
5. `backtest/` ORCHESTRATES all packages
6. No circular dependencies

### Dependency Graph
```
data (no deps)
  ↑
core (imports data)
  ↑
signals (imports core, data)
  ↑
metrics (imports signals output)
  ↑
backtest (imports signals, metrics, data, core)
```

---

## Data Flow

`packages/data/` is the single source for all data loading.

```
Flow:
1. data.trace.load_trace(path) → DataFrame[date, cusip, price, volume, side]
2. data.reference.load_reference(path) → DataFrame[cusip, issuer, rating, maturity]
3. data.universe.filter_ig(df) → DataFrame (IG bonds only)
4. signals.retail.compute_imbalance(trades) → Series[date, cusip]
5. backtest.engine.run(signal, prices) → BacktestResult
```

**Rule**: Only `packages/data/` reads files. Other packages receive DataFrames.

---

## Spec as Single Source of Truth

All formulas live in `spec/SPEC.md`. Code must match spec. Tests verify parity.

### Formula Registry

| Formula | Spec Section | Code Location | Test Fixture |
|---------|--------------|---------------|--------------|
| Credit PnL | §1.1 | `signals.credit.credit_pnl()` | `fixtures/pnl_cases.json` |
| Range Position | §1.2 | `signals.credit.range_position()` | `fixtures/range_position_cases.json` |
| Imbalance I_t | §1.3 | `signals.retail.compute_retail_imbalance()` | `fixtures/imbalance_cases.json` |
| Retail ID (BJZZ) | §1.4 | `signals.retail.is_retail_trade()` | `fixtures/retail_id_cases.json` |
| QMP Basic | §1.5 | `signals.retail.qmp_classify()` | `fixtures/qmp_cases.json` |
| QMP Exclusion | §1.5 | `signals.retail.qmp_classify_with_exclusion()` | `fixtures/qmp_cases.json` |
| Z-score | §2.1 | `signals.triggers.zscore_trigger()` | `fixtures/zscore_cases.json` |
| Streak | §2.2 | `signals.triggers.streak_trigger()` | `fixtures/streak_cases.json` |
| Sharpe Ratio | §3.1 | `metrics.performance.sharpe_ratio()` | `fixtures/sharpe_cases.json` |
| Sortino Ratio | §3.2 | `metrics.performance.sortino_ratio()` | `fixtures/sortino_cases.json` |
| Calmar Ratio | §3.3 | `metrics.performance.calmar_ratio()` | `fixtures/calmar_cases.json` |
| Max Drawdown | §4.1 | `metrics.risk.max_drawdown()` | `fixtures/drawdown_cases.json` |
| Value at Risk | §4.2 | `metrics.risk.value_at_risk()` | `fixtures/var_cases.json` |
| Expected Shortfall | §4.3 | `metrics.risk.expected_shortfall()` | `fixtures/es_cases.json` |
| Hit Rate | §5.1 | `metrics.diagnostics.hit_rate()` | `fixtures/hit_rate_cases.json` |
| Autocorrelation | §5.2 | `metrics.diagnostics.autocorrelation()` | `fixtures/autocorr_cases.json` |
| Information Coefficient | §5.3 | `metrics.diagnostics.information_coefficient()` | `fixtures/ic_cases.json` |

### Parity Check
```bash
python tools/verify_spec_parity.py  # Fails if code ≠ spec
```

---

## Package: data/

Data loading layer. **Only package that reads files.**

### Public API
- `load_trace(path)` → TRACE trades DataFrame
- `load_reference(path)` → Bond reference DataFrame
- `filter_ig(df)` → Investment-grade filter
- `filter_hy(df)` → High-yield filter
- `load_config(path)` → YAML config dict

### Validation Rules
- TRACE data: Remove cancellations, handle reversals
- Reference data: Validate CUSIP format, rating mapping
- All loaders: Return empty DataFrame on missing file (with warning)

---

## Package: signals/

Signal computations. Contains retail identification, credit metrics, and triggers.

### Public API

**Retail (retail.py):**
- `compute_retail_imbalance(trades)` → Imbalance series
- `is_retail_trade(price, notional)` → bool (BJZZ method)
- `is_subpenny(price)` → bool (subpenny check)
- `qmp_classify(price, mid, spread)` → 'buy', 'sell', or 'neutral'
- `qmp_classify_with_exclusion(price, bid, ask)` → 'buy', 'sell', or 'neutral'

**Credit (credit.py):**
- `credit_pnl(spread_change, pvbp, mid_price)` → Credit P&L series
- `range_position(spread_curr, avg, max, min)` → Range position series

**Triggers (triggers.py):**
- `zscore_trigger(series, window, threshold)` → Ternary signal (1, -1, 0)
- `streak_trigger(series, min_streak)` → Ternary signal (1, -1, 0)

### Rules
- All signals return `pd.Series` indexed by `(date, cusip)`
- No file I/O — receive DataFrames from `data/`
- Parameters from `conf/signals/*.yaml`

---

## Package: metrics/

Performance and risk metrics. **Pure functions only.**

### Public API
- `sharpe_ratio(returns)` → float
- `sortino_ratio(returns)` → float
- `calmar_ratio(returns)` → float
- `max_drawdown(returns)` → float
- `hit_rate(signals, returns)` → float

### Rules
- No side effects
- NaN handling documented per function
- All metrics annualized by default (252 trading days)

---

## Package: backtest/

Strategy harness. Orchestrates all packages.

### Public API
- `run_backtest(signal, prices)` → BacktestResult
- `equal_weight(signal, prices)` → Position DataFrame
- `risk_parity(signal, prices)` → Position DataFrame

### Lookahead Prevention Rules
1. Signal at t can only use data up to t-1
2. Position at t is based on signal at t-1
3. PnL at t is computed using price change from t-1 to t
4. **Never use `.shift(-n)` except in PnL calculation**

---

## Adding New Signals

1. [ ] Add formula to `spec/SPEC.md` with section number
2. [ ] Add fixture to `spec/fixtures/{name}_cases.json`
3. [ ] Implement in `packages/signals/src/`
4. [ ] Add test that loads fixture
5. [ ] Update Formula Registry above
6. [ ] Run `verify_spec_parity.py`

---

## Adding New Data Sources

1. [ ] Add loader to `packages/data/src/`
2. [ ] Add config schema to `conf/`
3. [ ] Add validation rules to `packages/data/src/validation.py`
4. [ ] Add tests with sample data
5. [ ] Document in Package: data/ section above

---

## Failure Mode Catalog

| Component | Failure Mode | Symptom | First Look |
|-----------|--------------|---------|------------|
| Imbalance | Zero denominator | NaN/Inf | `core.imbalance()` div check |
| Z-score | Insufficient history | NaN before warmup | `min_periods` param |
| Streak | Sign flip edge | Off-by-one count | `streak()` loop bounds |
| PnL | Wrong sign convention | Inverted returns | Spec §1.1 sign note |
| Backtest | Lookahead bias | Inflated Sharpe | `.shift(-h)` only in PnL |
| Data | Missing CUSIP | KeyError | Reference data join |
| Config | Missing key | KeyError | YAML structure |

---

## Commands

```bash
# Run all tests
pytest packages/*/tests/ -v

# Run specific package tests
pytest packages/data/tests/ -v
pytest packages/signals/tests/ -v

# Verify spec parity
python tools/verify_spec_parity.py

# Type check
mypy packages/*/src/

# Lint
ruff check packages/

# Install packages (dev mode)
pip install -e packages/data
pip install -e packages/signals
pip install -e packages/metrics
pip install -e packages/backtest
```
