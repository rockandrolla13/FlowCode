"""Tests for config module."""

from pathlib import Path

import pytest
import yaml

from src.config import load_config, get_nested


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading a valid YAML file."""
        config_data = {"key": "value", "nested": {"inner": 123}}
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = load_config(config_file)

        assert result == config_data

    def test_load_missing_file(self) -> None:
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading an empty YAML file returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = load_config(config_file)

        assert result == {}

    def test_load_string_path(self, tmp_path: Path) -> None:
        """Test loading with string path instead of Path object."""
        config_data = {"test": True}
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = load_config(str(config_file))

        assert result == config_data


class TestGetNested:
    """Tests for get_nested function."""

    def test_get_single_key(self) -> None:
        """Test getting a single-level key."""
        config = {"key": "value"}

        result = get_nested(config, "key")

        assert result == "value"

    def test_get_nested_key(self) -> None:
        """Test getting a nested key."""
        config = {"level1": {"level2": {"level3": 42}}}

        result = get_nested(config, "level1", "level2", "level3")

        assert result == 42

    def test_get_missing_key_returns_default(self) -> None:
        """Test that missing key returns default value."""
        config = {"key": "value"}

        result = get_nested(config, "missing", default="default")

        assert result == "default"

    def test_get_missing_nested_key_returns_default(self) -> None:
        """Test that missing nested key returns default value."""
        config = {"level1": {"level2": "value"}}

        result = get_nested(config, "level1", "missing", "key", default=None)

        assert result is None

    def test_get_default_is_none(self) -> None:
        """Test that default is None when not specified."""
        config = {}

        result = get_nested(config, "missing")

        assert result is None
