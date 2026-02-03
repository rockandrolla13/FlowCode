"""FlowCode Data Package - Data loading layer.

This is the ONLY package that reads files. All other packages
receive DataFrames from this package.

Public API:
- load_config: Load YAML configuration files
- load_trace: Load TRACE trade data from parquet
- load_reference: Load bond reference data
- filter_ig: Filter to investment-grade bonds
- filter_hy: Filter to high-yield bonds
- validate_trace: Validate TRACE data quality
"""

from .config import load_config
from .trace import load_trace
from .reference import load_reference
from .universe import filter_ig, filter_hy, filter_by_rating
from .validation import validate_trace, validate_reference

__all__ = [
    "load_config",
    "load_trace",
    "load_reference",
    "filter_ig",
    "filter_hy",
    "filter_by_rating",
    "validate_trace",
    "validate_reference",
]
