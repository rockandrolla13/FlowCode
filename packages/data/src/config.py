"""Configuration loading utilities.

This module provides functions to load YAML configuration files
used throughout the FlowCode credit analytics system.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML is malformed.

    Examples
    --------
    >>> cfg = load_config("conf/signals/zscore.yaml")
    >>> cfg["zscore"]["window"]
    252
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Get a nested value from a configuration dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    *keys : str
        Sequence of keys to traverse.
    default : Any, optional
        Default value if key path doesn't exist.

    Returns
    -------
    Any
        The value at the nested key path, or default.

    Examples
    --------
    >>> cfg = {"zscore": {"window": 252}}
    >>> get_nested(cfg, "zscore", "window")
    252
    >>> get_nested(cfg, "zscore", "missing", default=100)
    100
    """
    result = config
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
