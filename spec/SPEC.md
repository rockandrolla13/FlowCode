# FlowCode Credit Analytics Specification

## Overview

This document is the **single source of truth** for all formulas, algorithms, and business rules in FlowCode. Code implementations MUST match these specifications exactly.

---

## 1. Core Computations

### 1.1 Credit PnL (Spread-Based)

**Formula:**
```
Credit_PnL = -(ΔSpread) × (PVBP / MidPrice)
```

Where:
- `ΔSpread` = Spread change (e.g., 1-week, in basis points)
- `PVBP` = Price Value of a Basis Point (DV01)
- `MidPrice` = Mid price of the bond

**Logic:**
- Negative spread change (tightening) → positive PnL for long positions
- PVBP is typically expressed per $100 notional

**Code Location:** `packages/signals/src/credit.py::credit_pnl()`
**Fixture:** `spec/fixtures/pnl_cases.json`

### 1.2 Range Position

**Formula:**
```
RangePos = (Spread_curr - Spread_avg) / (Spread_max - Spread_min)
```

Where:
- `Spread_curr` = Current spread value
- `Spread_avg` = Average spread over lookback window (e.g., 1 month)
- `Spread_max` = Maximum spread over lookback window
- `Spread_min` = Minimum spread over lookback window

**Output:**
- Values near 0 = spread is near average
- Positive values = wider than average (value opportunity)
- Negative values = tighter than average (expensive)
- Result is unbounded (can exceed [-1, 1] if current spread exceeds historical range)

**Edge Cases:**
- If `Spread_max == Spread_min`, return `NaN` (no range)

**Code Location:** `packages/signals/src/credit.py::range_position()`
**Fixture:** `spec/fixtures/range_position_cases.json`

### 1.3 Retail Order Imbalance

**Formula:**
```
I_t = (ΣBuy - ΣSell) / (ΣBuy + ΣSell)
```

Where:
- `ΣBuy` = Sum of buy volume at time t (retail only)
- `ΣSell` = Sum of sell volume at time t (retail only)
- Result: [-1, +1], where +1 = all buys, -1 = all sells

**Edge Cases:**
- If `ΣBuy + ΣSell = 0`, return `NaN`
- If only buys: return `+1`
- If only sells: return `-1`

**Code Location:** `packages/signals/src/retail.py::compute_retail_imbalance()`
**Fixture:** `spec/fixtures/imbalance_cases.json`

### 1.4 Retail Trade Identification (BJZZ)

**Identification Rule:**
A trade is retail if BOTH conditions are met:
```
1. Notional < $200,000
2. mod(Price × 100, 1) > 0  (subpenny pricing)
```

**Logic:**
- Subpenny pricing indicates price improvement from wholesalers
- Price improvement is characteristic of retail order flow
- Large trades (≥$200K) are institutional regardless of pricing

**Code Location:** `packages/signals/src/retail.py::is_retail_trade()`
**Fixture:** `spec/fixtures/retail_id_cases.json`

### 1.5 QMP Classification (Quote Midpoint)

**Basic QMP Rule:**
```
BUY    if Price > Midpoint + (α × Spread)
SELL   if Price < Midpoint - (α × Spread)
NEUTRAL otherwise
```

Where:
- `Midpoint = (Bid + Ask) / 2`
- `Spread = Ask - Bid`
- `α = 0.1` (default threshold)

**Lee-Ready QMP with NBBO Exclusion Zone:**
```
Position = (Price - Bid) / Spread

BUY     if Position > 0.60  (above 60% of spread)
SELL    if Position < 0.40  (below 40% of spread)
NEUTRAL if 0.40 ≤ Position ≤ 0.60  (in exclusion zone)
```

**Logic:**
- Trades near the midpoint (40-60% zone) are ambiguous
- Excluding them reduces noise in direction classification

**Code Locations:**
- Basic: `packages/signals/src/retail.py::qmp_classify()`
- With exclusion: `packages/signals/src/retail.py::qmp_classify_with_exclusion()`
**Fixture:** `spec/fixtures/qmp_cases.json`

---

## 2. Signal Triggers

### 2.1 Z-Score Trigger

**Formula:**
```
Z_t = (X_t - μ) / σ

where:
  μ = rolling mean over window
  σ = rolling std over window
```

**Trigger:**
```
signal = 1  if Z_t > threshold
       = -1 if Z_t < -threshold
       = 0  otherwise
```

**Parameters:**
- `window`: 252 (default, ~1 year trading days)
- `threshold`: 7.0 (default)
- `min_periods`: 126 (minimum observations before computing)

**Code Location:** `packages/signals/src/triggers.py::zscore_trigger()`
**Fixture:** `spec/fixtures/zscore_cases.json`

### 2.2 Streak Trigger

**Formula:**
```
streak_t = consecutive days with same sign(X)
```

**Trigger:**
```
signal = sign(X_t) if |streak_t| >= min_streak
       = 0         otherwise
```

**Parameters:**
- `min_streak`: 3 (default)

**Code Location:** `packages/signals/src/triggers.py::streak_trigger()`
**Fixture:** `spec/fixtures/streak_cases.json`

---

## 3. Performance Metrics

### 3.1 Sharpe Ratio

**Formula:**
```
Sharpe = (μ_r - r_f) / σ_r * √252

where:
  μ_r = mean daily return
  r_f = risk-free rate (default 0)
  σ_r = std of daily returns
  252 = trading days per year
```

**Code Location:** `packages/metrics/src/performance.py::sharpe_ratio()`
**Fixture:** `spec/fixtures/sharpe_cases.json`

### 3.2 Sortino Ratio

**Formula:**
```
Sortino = (μ_r - r_f) / σ_d * √252

where:
  σ_d = downside deviation (std of negative returns only)
```

**Code Location:** `packages/metrics/src/performance.py::sortino_ratio()`
**Fixture:** `spec/fixtures/sortino_cases.json`

### 3.3 Calmar Ratio

**Formula:**
```
Calmar = Annualized_Return / |Max_Drawdown|
```

**Code Location:** `packages/metrics/src/performance.py::calmar_ratio()`
**Fixture:** `spec/fixtures/calmar_cases.json`

---

## 4. Risk Metrics

### 4.1 Maximum Drawdown

**Formula:**
```
Drawdown_t = (Peak_t - Value_t) / Peak_t

where:
  Peak_t = max(Value_0, Value_1, ..., Value_t)

Max_Drawdown = min(Drawdown_t) for all t
```

**Note:** Returns negative value (e.g., -0.15 = 15% drawdown)

**Code Location:** `packages/metrics/src/risk.py::max_drawdown()`
**Fixture:** `spec/fixtures/drawdown_cases.json`

### 4.2 Value at Risk (VaR)

**Formula (Historical):**
```
VaR_α = -quantile(returns, α)

where:
  α = confidence level (default 0.05 for 95% VaR)
```

**Code Location:** `packages/metrics/src/risk.py::value_at_risk()`
**Fixture:** `spec/fixtures/var_cases.json`

### 4.3 Expected Shortfall (CVaR)

**Formula:**
```
ES_α = -mean(returns where returns ≤ -VaR_α)
```

**Code Location:** `packages/metrics/src/risk.py::expected_shortfall()`
**Fixture:** `spec/fixtures/es_cases.json`

---

## 5. Diagnostics

### 5.1 Hit Rate

**Formula:**
```
Hit_Rate = count(returns > 0) / count(returns)
```

**Code Location:** `packages/metrics/src/diagnostics.py::hit_rate()`
**Fixture:** `spec/fixtures/hit_rate_cases.json`

### 5.2 Autocorrelation

**Formula:**
```
AC_k = corr(X_t, X_{t-k})
```

**Code Location:** `packages/metrics/src/diagnostics.py::autocorrelation()`
**Fixture:** `spec/fixtures/autocorr_cases.json`

### 5.3 Information Coefficient

**Formula:**
```
IC = corr(signal_t, return_{t+1})
```

**Note:** Forward return, not concurrent.

**Code Location:** `packages/metrics/src/diagnostics.py::information_coefficient()`
**Fixture:** `spec/fixtures/ic_cases.json`

---

## 6. Backtest Engine

### 6.1 Return Calculation

**Formula:**
```
Portfolio_Return_t = Σ(Position_{t-1,i} * Asset_Return_{t,i})
```

**Lookahead Prevention:**
- Position at t-1 earns return from t-1 to t
- Signal at t determines position at t+1

**Code Location:** `packages/backtest/src/engine.py::compute_returns()`

### 6.2 Transaction Costs

**Formula:**
```
Cost_t = Σ|ΔPosition_{t,i}| * cost_rate
Net_Return_t = Gross_Return_t - Cost_t
```

**Code Location:** `packages/backtest/src/engine.py::run_backtest()`

---

## 7. Data Validation

### 7.1 TRACE Data Requirements

| Field | Type | Constraints |
|-------|------|-------------|
| date | datetime | Not null, valid date |
| cusip | string | 9 characters |
| price | float | > 0 |
| volume | float | ≥ 0 |
| side | string | 'B' or 'S' |

### 7.2 Reference Data Requirements

| Field | Type | Constraints |
|-------|------|-------------|
| cusip | string | 9 characters, unique |
| issuer | string | Not null |
| rating | string | Valid rating (AAA to D) |
| maturity | datetime | > issue_date |

---

## Formula Registry

| Formula | Spec § | Code Location | Fixture |
|---------|--------|---------------|---------|
| Credit PnL | §1.1 | signals.credit.credit_pnl() | pnl_cases.json |
| Range Position | §1.2 | signals.credit.range_position() | range_position_cases.json |
| Imbalance | §1.3 | signals.retail.compute_retail_imbalance() | imbalance_cases.json |
| Retail ID (BJZZ) | §1.4 | signals.retail.is_retail_trade() | retail_id_cases.json |
| QMP Basic | §1.5 | signals.retail.qmp_classify() | qmp_cases.json |
| QMP Exclusion | §1.5 | signals.retail.qmp_classify_with_exclusion() | qmp_cases.json |
| Z-score | §2.1 | signals.triggers.zscore_trigger() | zscore_cases.json |
| Streak | §2.2 | signals.triggers.streak_trigger() | streak_cases.json |
| Sharpe | §3.1 | metrics.performance.sharpe_ratio() | sharpe_cases.json |
| Sortino | §3.2 | metrics.performance.sortino_ratio() | sortino_cases.json |
| Calmar | §3.3 | metrics.performance.calmar_ratio() | calmar_cases.json |
| Max DD | §4.1 | metrics.risk.max_drawdown() | drawdown_cases.json |
| VaR | §4.2 | metrics.risk.value_at_risk() | var_cases.json |
| ES | §4.3 | metrics.risk.expected_shortfall() | es_cases.json |
| Hit Rate | §5.1 | metrics.diagnostics.hit_rate() | hit_rate_cases.json |
| Autocorr | §5.2 | metrics.diagnostics.autocorrelation() | autocorr_cases.json |
| IC | §5.3 | metrics.diagnostics.information_coefficient() | ic_cases.json |
