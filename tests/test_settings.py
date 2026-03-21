"""Tests for pipeline settings loading."""

from simready.config.settings import load_settings


def test_load_defaults():
    settings = load_settings()
    assert settings.up_axis == "Z"
    assert settings.meters_per_unit == 1.0
    assert settings.materials.target_format == "mdl"
    assert settings.validation.strict is False


def test_geometry_lod_levels():
    settings = load_settings()
    assert len(settings.geometry.lod_levels) == 3
    assert settings.geometry.lod_levels[0] == 1.0
