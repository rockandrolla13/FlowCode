# Input Data Schema

This document defines the expected input data formats for FlowCode.

---

## Quick Start: Column Mapping

**If your column names differ from the defaults**, you have two options:

### Option 1: Rename columns before loading
```python
df = pd.read_parquet("your_data.parquet")
df = df.rename(columns={
    "trade_date": "date",
    "bond_id": "cusip",
    "exec_price": "price",
    "quantity": "volume",
    "direction": "side",
})
```

### Option 2: Pass column names to functions
```python
# Most functions accept column name parameters
imbalance = compute_retail_imbalance(
    trades,
    date_col="trade_date",
    cusip_col="bond_id",
    volume_col="quantity",
    side_col="direction",
    buy_value="BUY",   # your buy indicator
    sell_value="SELL", # your sell indicator
)
```

---

## TRACE Trade Data

**Source**: FINRA TRACE (Trade Reporting and Compliance Engine)

### Default Column Names

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `date` | datetime | Trade date | Yes |
| `cusip` | str | 9-character bond identifier | Yes |
| `price` | float | Execution price (clean price, 0-200 range) | Yes |
| `volume` | float | Trade volume in thousands | Yes |
| `side` | str | Trade direction: `'B'` (buy) or `'S'` (sell) | Yes |

### Optional Columns (for QMP classification)

| Column | Type | Description |
|--------|------|-------------|
| `mid` | float | Quote midpoint price |
| `spread` | float | Bid-ask spread |
| `bid` | float | Best bid price |
| `ask` | float | Best ask price |

### Example Data

```
date,cusip,price,volume,side
2024-01-02,037833100,98.50,100,B
2024-01-02,037833100,98.52,50,S
2024-01-02,594918104,101.25,200,B
```

### Functions That Accept Custom Column Names

| Function | Parameters |
|----------|------------|
| `aggregate_daily_volume()` | `cusip_col`, `date_col`, `volume_col`, `side_col` |
| `compute_retail_imbalance()` | `date_col`, `cusip_col`, `volume_col`, `side_col`, `buy_value`, `sell_value` |
| `classify_trades_qmp()` | `price_col`, `mid_col`, `spread_col` |

---

## Bond Reference Data

**Source**: Your bond master file (Bloomberg, ICE, internal)

### Default Column Names

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `cusip` | str | 9-character bond identifier | Yes |
| `issuer` | str | Issuer name | Yes |
| `rating` | str | Credit rating (e.g., `'BBB+'`, `'BB-'`) | Yes |
| `maturity` | datetime | Maturity date | No |
| `coupon` | float | Coupon rate (e.g., 0.05 for 5%) | No |
| `issue_date` | datetime | Issue date | No |
| `amount_outstanding` | float | Amount outstanding in millions | No |

### Rating Format

Accepted rating formats (case-insensitive):
```
AAA, AA+, AA, AA-, A+, A, A-, BBB+, BBB, BBB-, BB+, BB, BB-, B+, B, B-, CCC+, CCC, CCC-, CC, C, D
```

Investment Grade (IG): `BBB-` and above
High Yield (HY): `BB+` and below

### Example Data

```
cusip,issuer,rating,maturity,coupon
037833100,APPLE INC,AA+,2030-05-15,0.035
594918104,MICROSOFT CORP,AAA,2028-11-01,0.040
```

### Functions That Accept Custom Column Names

| Function | Parameters |
|----------|------------|
| `filter_ig()` | `rating_col` |
| `filter_hy()` | `rating_col` |
| `filter_by_rating()` | `rating_col` |
| `filter_by_liquidity()` | `volume_col`, `trades_col` |
| `enrich_with_reference()` | `on` (join column) |

---

## Validation

### Validate Your Data

```python
from data import validate_trace, validate_reference

# Validate with default columns
result = validate_trace(trades_df)
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")

# Validate with custom required columns
result = validate_trace(
    trades_df,
    required_columns=["trade_date", "bond_id", "exec_price", "quantity", "direction"]
)
```

### Validation Checks

**TRACE Data:**
- Required columns present
- CUSIP format (9 alphanumeric characters)
- Price range (0 < price ≤ 200)
- Volume non-negative
- Side values valid (`B`/`S` or `buy`/`sell`)

**Reference Data:**
- Required columns present
- CUSIP format valid
- No duplicate CUSIPs
- Null value warnings

---

## Common Column Mappings

### Bloomberg Terminal
```python
column_map = {
    "TRADE_DT": "date",
    "ID_CUSIP": "cusip",
    "PX_LAST": "price",
    "VOLUME": "volume",
    "SIDE": "side",
}
```

### ICE Data Services
```python
column_map = {
    "TradeDate": "date",
    "Cusip": "cusip",
    "Price": "price",
    "Quantity": "volume",
    "BuySellIndicator": "side",
}
```

### Internal Systems
```python
# Map your column names here
column_map = {
    "your_date_col": "date",
    "your_cusip_col": "cusip",
    "your_price_col": "price",
    "your_volume_col": "volume",
    "your_side_col": "side",
}

df = df.rename(columns=column_map)
```

---

## Side Value Mapping

Different systems use different buy/sell indicators:

| System | Buy | Sell |
|--------|-----|------|
| FlowCode default | `B` | `S` |
| Alternative | `buy` | `sell` |
| Alternative | `BUY` | `SELL` |
| Alternative | `1` | `-1` |
| Alternative | `P` (purchase) | `S` (sale) |

**To use non-default values:**
```python
imbalance = compute_retail_imbalance(
    trades,
    buy_value="BUY",
    sell_value="SELL",
)
```

---

## File Formats

### Supported Formats

| Format | Extension | Loader |
|--------|-----------|--------|
| Parquet | `.parquet` | `pd.read_parquet()` |
| CSV | `.csv` | `pd.read_csv()` |

### Recommended: Parquet

Parquet is preferred for:
- Faster loading (columnar format)
- Smaller file size (compressed)
- Type preservation (dates stay dates)

```python
# Save as parquet
df.to_parquet("trades.parquet", index=False)

# Load
df = load_trace("trades.parquet")
```

---

## Data Pipeline Example

```python
from data import load_trace, load_reference, validate_trace, filter_ig
from signals import compute_retail_imbalance

# 1. Load and validate
trades = load_trace("trades.parquet")
result = validate_trace(trades)
if not result.is_valid:
    raise ValueError(f"Invalid data: {result.errors}")

# 2. Load reference and filter to IG
reference = load_reference("bonds.parquet")
ig_cusips = filter_ig(reference)["cusip"].unique()
trades = trades[trades["cusip"].isin(ig_cusips)]

# 3. Compute signal
imbalance = compute_retail_imbalance(trades)
```
