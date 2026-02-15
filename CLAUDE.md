# FlowCode Credit Analytics
In all interactions and commit messages, be extremely concise, careful and technical and sacrifice grammar for the sake of concision.
Make all plans multiphase. Create a `gh` issue for each multiphase plan to track phases, progress, and decisions.
create short PRs of 250 lines or less. 
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

Standalone modules (repo root):
├── intraday_microstructure_analytics.py   # Intraday quote analytics (22 functions, 7 classes)
└── intraday_quote_filters.py              # Intraday filters (16 functions, 5 schemas)
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

## Module: intraday_microstructure_analytics.py

Intraday corporate bond microstructure analytics. Combines 4 composite dealer quote sources (MA, TW, TM, CBBT) with Fincad FV model data on a 5-min (isin, time_bin) grid.

### Schema Layer
- `CompositeSource` enum — MA, TW, TM, CBBT
- `PriceSpace` enum — SPREAD, Z_SPREAD
- `IntradayQuoteSchema` — column names for composite quote fields
- `IntradayFVSchema` — Fincad model output columns + `venue_z_col()` helper
- `JoinedIntradaySchema` — combined quote + FV schema

### Public API (22 functions)

**Session Management:**
- `ResetPolicy` enum — DAILY, WEEKLY, INTRADAY_4H, EVENT_BASED, NONE
- `_make_reset_mask(ts, policy)` → bool mask for session boundaries
- `session_high/low(data, col, policy)` → expanding extremes within sessions

**Carry Analytics:**
- `carry_per_unit_risk(data)` → carry / |cr01| (logged guard on small cr01)
- `carry_breakeven_spread_move(data)` → carry / cr01 in bps
- `fv_deviation_carry_scaled(data, venue, price_space)` → deviation × carry days

**Cross-Source Spreads:**
- `bid_ask_spread(data, venue)` → offer - bid
- `cross_source_vs_composite_fv(data)` → venue spreads - fv_spread (logs missing cols)

**Intraday Dynamics:**
- `quote_staleness(data, venue)` → seconds since last change
- `intraday_spread_range(data, venue, policy)` → (session_low, session_high)
- `cumulative_spread_move(data, venue, policy)` → session-relative cumulative move

**Liquidity & Regime:**
- `liquidity_filtered_universe(data, ...)` → filtered by staleness/ba thresholds
- `bid_ask_regime(data, venue, tight, wide)` → "tight"/"normal"/"wide"/"unknown"

**Cross-Venue:**
- `cross_venue_price_dislocation(data)` → max - min spread across venues
- `cross_venue_dislocation_carry_scaled(data)` → dislocation × carry days
- `multi_venue_confirmation(data, venue_threshold)` → count venues agreeing on direction

**Profiles:**
- `spread_time_profile(data, venue, freq)` → mean spread by time bucket

### Key Patterns
- All functions take a `data: pd.DataFrame` indexed by `(isin, time_bin)`
- NaN guards use `logger.warning()` not silent fallbacks
- `_safe_diff(data, col, isin_col)` prevents cross-ISIN contamination
- `field(default_factory=...)` for mutable dataclass defaults

---

## Module: intraday_quote_filters.py

Intraday quote and TRACE transaction filters. Extends the analytics module with event detection across 5 sections.

### Schema Layer
- `TradeSide` enum — CUSTOMER_BUY, CUSTOMER_SELL, DEALER, UNKNOWN
- `TraceTradeSchema` — raw TRACE trade columns
- `TraceAggSchema` — 5-min aggregated TRACE columns
- `FullIntradaySchema` — combined quote + FV + TRACE aggregate schema
- `SignalType` enum — QUOTE_MOVE, TRACE_PRINT, TRACE_FLOW, TRACE_VS_QUOTE, FV_DEVIATION

### Public API (16 functions)

**Section 1 — Composite Quote Filters:**
- `filter_significant_offer_tightening(data, min_bps, ...)` → tightening events
- `filter_significant_bid_widening(data, min_bps, ...)` → widening events
- `filter_significant_quote_moves(data, min_bps, side, ...)` → directional moves
- `filter_significant_quote_moves_multi_venue(data, min_bps, ...)` → multi-venue confirmation
- `filter_spread_compression_events(data, min_bps, ...)` → bid-ask compression

**Section 2 — TRACE Transaction Filters:**
- `filter_trace_big_prints(data, min_notional, ...)` → large trades
- `filter_trace_block_trades(data, block_threshold, ...)` → block-sized trades
- `filter_trace_volume_surge(data, surge_multiple, ...)` → volume spikes
- `filter_trace_vs_quotes(data, min_bps, ...)` → trades deviating from quotes

**Section 3 — TRACE Aggregated Filters:**
- `aggregate_trace_to_bins(trades, freq)` → bin-level TRACE summary
- `filter_trace_agg_imbalance(data, min_imbalance, direction)` → flow imbalance
- `filter_trace_vwap_vs_composite(data, min_bps, ...)` → VWAP vs composite spread

**Section 4 — FV-Anchored Filters:**
- `filter_rich_cheap_extremes(data, min_deviation, direction)` → FV deviation events
- `filter_rich_cheap_carry_scaled(data, min_carry_days, direction)` → carry-scaled deviations
- `filter_trace_vs_fv(data, min_deviation, direction)` → TRACE execution vs FV

**Section 5 — Unified Interface:**
- `detect_directional_pressure(data, ...)` → combined signal from all filter types

### Key Patterns
- All filters return `.copy()` to prevent SettingWithCopyWarning
- `_safe_diff(data, col, isin_col)` prevents cross-ISIN diff contamination
- `direction` and `side` params validated with `ValueError` on bad input
- Staleness filtering via `quote_staleness()` with configurable `max_stale_sec`
- All NaN/missing-column cases logged, never silently dropped

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
| Intraday diff | Cross-ISIN contamination | False signal at ISIN boundary | `_safe_diff()` isin_col param |
| Intraday staleness | Stale quotes pass filter | Phantom moves from old data | `quote_staleness()` threshold |
| Intraday carry | Division by near-zero cr01 | Inf carry ratio | `carry_per_unit_risk()` guard |
| Intraday regime | Overlapping ba conditions | Double-classified rows | `bid_ask_regime()` np.select order |
| TRACE filters | Missing agg columns | Silent empty results | `filter_trace_vwap_vs_composite()` col check |

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
