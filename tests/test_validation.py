"""Tests for SimReady validation checks."""

from simready.config.settings import PipelineSettings
from simready.materials.material_map import MDLMaterial
from simready.validation.simready_checks import validate_materials


def _default_settings() -> PipelineSettings:
    return PipelineSettings()


def test_valid_material_passes():
    mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        diffuse_color=(0.5, 0.5, 0.5),
        roughness=0.4,
        metallic=0.8,
        ior=2.0,
        source_material="steel_test",
        confidence=0.75,
    )
    result = validate_materials([mat], _default_settings())
    assert result.passed
    assert len(result.errors) == 0


def test_invalid_roughness_fails():
    mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        roughness=1.5,  # out of range
        source_material="bad_mat",
        confidence=1.0,
    )
    result = validate_materials([mat], _default_settings())
    assert not result.passed
    assert any("roughness" in e for e in result.errors)


def test_low_confidence_errors():
    mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        source_material="guessed_mat",
        confidence=0.1,
    )
    result = validate_materials([mat], _default_settings())
    assert not result.passed
    assert any("confidence" in e for e in result.errors)


def test_missing_source_name_errors():
    mat = MDLMaterial(mdl_name="OmniPBR.mdl", source_material=None, confidence=1.0)
    result = validate_materials([mat], _default_settings())
    assert not result.passed


def test_ior_below_one_errors():
    mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        ior=0.5,
        source_material="impossible_mat",
        confidence=1.0,
    )
    result = validate_materials([mat], _default_settings())
    assert not result.passed
    assert any("IOR" in e for e in result.errors)
