"""Tests for TauluConfig Pydantic model and schema generation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from taulu.config import TauluConfig, _parse_value
from taulu.schema import generate_schema
from taulu.split import Split


class TestTauluConfigDefaults:
    """Verify default values match the expected configuration."""

    def test_minimal_creation(self):
        config = TauluConfig(template_path="header.png")
        assert config.template_path == "header.png"


class TestSplitValues:
    """Verify Split values work correctly in TauluConfig."""

    def test_split_template_path(self):
        config = TauluConfig(template_path=Split("left.png", "right.png"))
        assert isinstance(config.template_path, Split)
        assert config.template_path.left == "left.png"
        assert config.template_path.right == "right.png"

    def test_split_numeric_parameter(self):
        config = TauluConfig(
            template_path="x.png",
            search_radius=Split(40, 80),
            binarization_sensitivity=Split(0.1, 0.3),
        )
        assert isinstance(config.search_radius, Split)
        assert config.search_radius.left == 40
        assert config.search_radius.right == 80
        assert isinstance(config.binarization_sensitivity, Split)
        assert config.binarization_sensitivity.left == 0.1
        assert config.binarization_sensitivity.right == 0.3

    def test_scalar_and_split_mixed(self):
        config = TauluConfig(
            template_path=Split("l.png", "r.png"),
            search_radius=60,  # scalar
            line_thickness=Split(8, 12),  # split
        )
        assert config.search_radius == 60
        assert isinstance(config.line_thickness, Split)


class TestParseValue:
    """Verify _parse_value helper correctly converts dicts to Split."""

    def test_dict_with_left_right_becomes_split(self):
        result = _parse_value({"left": "a", "right": "b"})
        assert isinstance(result, Split)
        assert result.left == "a"
        assert result.right == "b"

    def test_scalar_passthrough(self):
        assert _parse_value(42) == 42
        assert _parse_value("hello") == "hello"

    def test_dict_without_left_right_passthrough(self):
        d = {"foo": "bar"}
        assert _parse_value(d) == d


class TestFromToml:
    """Verify TOML loading works correctly."""

    def test_scalar_values(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            'template_path = "header.png"\n'
            "binarization_sensitivity = 0.15\n"
            "search_radius = 80\n"
        )
        config = TauluConfig.from_toml(toml_file)
        assert config.template_path == "header.png"
        assert config.binarization_sensitivity == 0.15
        assert config.search_radius == 80

    def test_split_values(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            "[template_path]\n"
            'left = "left.png"\n'
            'right = "right.png"\n'
            "\n"
            "[search_radius]\n"
            "left = 40\n"
            "right = 80\n"
        )
        config = TauluConfig.from_toml(toml_file)
        assert isinstance(config.template_path, Split)
        assert config.template_path.left == "left.png"
        assert config.template_path.right == "right.png"
        assert isinstance(config.search_radius, Split)
        assert config.search_radius.left == 40
        assert config.search_radius.right == 80

    def test_merge_multiple_files(self, tmp_path: Path):
        base = tmp_path / "base.toml"
        base.write_text(
            'template_path = "base.png"\n'
            "binarization_sensitivity = 0.1\n"
            "search_radius = 40\n"
        )
        override = tmp_path / "override.toml"
        override.write_text('template_path = "override.png"\nsearch_radius = 100\n')
        config = TauluConfig.from_toml(base, override)
        assert config.template_path == "override.png"
        assert config.search_radius == 100
        assert config.binarization_sensitivity == 0.1  # from base

    def test_dollar_keys_filtered(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '"$schema" = "./schema.json"\ntemplate_path = "header.png"\n'
        )
        config = TauluConfig.from_toml(toml_file)
        assert config.template_path == "header.png"


class TestValidation:
    """Verify Pydantic validation catches invalid input."""

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            TauluConfig()  # ty:ignore[missing-argument]

    def test_invalid_feature_detector(self):
        with pytest.raises(ValidationError):
            TauluConfig(template_path="x.png", feature_detector="invalid")  # ty:ignore[invalid-argument-type]


class TestSchema:
    """Verify JSON schema generation."""

    def test_schema_has_all_properties(self):
        schema = generate_schema()
        props = schema["properties"]
        assert "$schema" in props
        assert "template_path" in props
        assert "binarization_sensitivity" in props
        assert "feature_detector" in props
        assert "matching_scale" in props

    def test_schema_has_draft07(self):
        schema = generate_schema()
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_schema_has_title(self):
        schema = generate_schema()
        assert schema["title"] == "TauluConfig"

    def test_schema_defaults_match_config(self):
        schema = generate_schema()
        config = TauluConfig(template_path="x.png")
        props = schema["properties"]
        assert (
            props["binarization_sensitivity"]["default"]
            == config.binarization_sensitivity
        )
        assert props["search_radius"]["default"] == config.search_radius
        assert props["smooth"]["default"] == config.smooth
        assert props["matching_scale"]["default"] == config.matching_scale

    def test_schema_is_valid_json(self):
        schema = generate_schema()
        # Should be JSON-serializable
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert parsed == schema

    def test_schema_property_count(self):
        """All config fields + $schema should be present."""
        schema = generate_schema()
        # 26 config fields + $schema = 27
        assert len(schema["properties"]) == 27
