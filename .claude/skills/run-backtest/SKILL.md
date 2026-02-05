# Skill: Run Backtest

## Purpose

Run a backtest for a signal strategy using the FlowCode backtest engine.

## Prerequisites

- Signal data ready (DataFrame with date index, asset columns)
- Price data ready (same structure)
- Understanding of position sizing method

## Steps

### 1. Prepare Data

```python
import pandas as pd
from packages.data.src.trace import load_trace
from packages.signals.src.retail import compute_retail_imbalance

# Load price data
prices = load_trace("/path/to/prices.parquet")
prices = prices.pivot(index="date", columns="cusip", values="price")

# Generate signal
trades = load_trace("/path/to/trades.parquet")
signal = compute_retail_imbalance(trades)
signal = signal.unstack()  # Convert to DataFrame if MultiIndex Series
```

### 2. Run Backtest

```python
from packages.backtest.src.engine import run_backtest
from packages.backtest.src.portfolio import equal_weight, risk_parity

# Basic backtest with equal weights
result = run_backtest(
    signal=signal,
    prices=prices,
    position_sizer=equal_weight,
    transaction_cost=0.0,
    max_positions=50,
)

# With risk parity
result_rp = run_backtest(
    signal=signal,
    prices=prices,
    position_sizer=risk_parity,
    vol_window=21,
    transaction_cost=0.001,  # 10 bps
)
```

### 3. Analyze Results

```python
# Print summary
print(result.summary())

# Access metrics
print(f"Sharpe: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max DD: {result.metrics['max_drawdown']:.2%}")
print(f"Win Rate: {result.metrics['win_rate']:.2%}")

# Access returns
returns = result.returns
cumulative = (1 + returns).cumprod()

# Access positions
positions = result.positions

# Access trades
trades = result.trades
print(f"Total trades: {len(trades)}")
```

### 4. Visualize (Optional)

```python
import matplotlib.pyplot as plt

# Cumulative returns
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

cumulative.plot(ax=axes[0], title="Cumulative Returns")
axes[0].set_ylabel("Growth of $1")

# Drawdown
from packages.metrics.src.risk import drawdown_series
dd = drawdown_series(returns)
dd.plot(ax=axes[1], title="Drawdown", color="red")
axes[1].set_ylabel("Drawdown")

plt.tight_layout()
plt.savefig("backtest_results.png")
```

### 5. Compare Strategies

```python
# Run multiple strategies
results = {}

for cost in [0.0, 0.001, 0.002]:
    r = run_backtest(signal, prices, transaction_cost=cost)
    results[f"cost_{cost}"] = r

# Compare
for name, r in results.items():
    print(f"{name}: Sharpe={r.metrics['sharpe_ratio']:.2f}, "
          f"Return={r.total_return:.2%}")
```

## Position Sizing Options

### Equal Weight
All positions have equal absolute weight.

```python
from packages.backtest.src.portfolio import equal_weight

result = run_backtest(signal, prices, position_sizer=equal_weight)
```

### Risk Parity
Positions weighted inversely to volatility.

```python
from packages.backtest.src.portfolio import risk_parity

result = run_backtest(
    signal, prices,
    position_sizer=risk_parity,
    vol_window=21,  # 1 month
)
```

### Top N
Take top N long and bottom N short positions.

```python
from packages.backtest.src.portfolio import top_n_positions

result = run_backtest(
    signal, prices,
    position_sizer=top_n_positions,
    n_long=10,
    n_short=10,
)
```

## Lookahead Prevention

The backtest engine automatically prevents lookahead bias:

1. Signal at t determines position at t+1
2. Position at t-1 earns return from t-1 to t
3. Transaction costs are applied at trade time

Never use future data in signal construction.

## Checklist

- [ ] Signal and prices have matching dates/assets
- [ ] No NaN values in critical columns
- [ ] Position sizing method chosen
- [ ] Transaction costs set appropriately
- [ ] Results make financial sense
- [ ] Compared multiple parameterizations
