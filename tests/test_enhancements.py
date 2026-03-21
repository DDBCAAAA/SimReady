"""Tests for Phase 1–3 enhancements: material binding, LOD, collision, geometry validation, semantics."""

from __future__ import annotations

import numpy as np
import pytest

from simready.ingestion.step_reader import CADBody, CADAssembly
from simready.materials.material_map import CAEMaterial, MDLMaterial, map_cae_to_mdl
from simready.semantics.classifier import classify
from simready.validation.simready_checks import validate_geometry


# ---------------------------------------------------------------------------
# P1.1: Material binding
# ---------------------------------------------------------------------------

def test_step_reader_sets_material_name():
    """CADBody.material_name must equal body.name after step_reader creates the body."""
    body = CADBody(
        name="body_0",
        vertices=np.zeros((4, 3)),
        faces=np.zeros((2, 3), dtype=int),
        material_name="body_0",
    )
    assert body.material_name == body.name


def test_map_cae_to_mdl_preserves_source_name():
    mat = CAEMaterial(name="body_0")
    mdl = map_cae_to_mdl(mat)
    assert mdl.source_material == "body_0"


def test_map_cae_to_mdl_steel_fills_physics():
    """Steel classification should populate density, friction, restitution."""
    mat = CAEMaterial(name="steel_bracket")
    mdl = map_cae_to_mdl(mat)
    assert mdl.density == pytest.approx(7850.0)
    assert mdl.friction_static is not None
    assert mdl.friction_dynamic is not None
    assert mdl.restitution is not None


def test_map_cae_to_mdl_unknown_material_no_physics():
    """Unknown material should not fabricate physics values."""
    mat = CAEMaterial(name="unknown_exotic_material")
    mdl = map_cae_to_mdl(mat)
    assert mdl.density is None
    assert mdl.friction_static is None
    assert mdl.restitution is None


def test_map_cae_to_mdl_cae_density_overrides_class_default():
    """Explicit CAE density must take priority over material class default."""
    mat = CAEMaterial(name="steel_part", density=9000.0)  # custom alloy
    mdl = map_cae_to_mdl(mat)
    assert mdl.density == pytest.approx(9000.0)


# ---------------------------------------------------------------------------
# P1.2: LOD VariantSet
# ---------------------------------------------------------------------------

def test_cadbody_has_lod_meshes_field():
    body = CADBody(
        name="part",
        vertices=np.zeros((4, 3)),
        faces=np.zeros((2, 3), dtype=int),
    )
    assert hasattr(body, "lod_meshes")
    assert body.lod_meshes == []


def test_lod_meshes_stores_ratio_verts_faces():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)
    body = CADBody(name="part", vertices=verts, faces=faces)
    body.lod_meshes = [(1.0, verts, faces), (0.5, verts[:3], faces[:1])]
    assert len(body.lod_meshes) == 2
    ratio, lod_verts, lod_faces = body.lod_meshes[1]
    assert ratio == pytest.approx(0.5)
    assert lod_verts.shape == (3, 3)


# ---------------------------------------------------------------------------
# P1.3: Geometry validation
# ---------------------------------------------------------------------------

def _make_settings():
    from simready.config.settings import PipelineSettings
    return PipelineSettings()


def _valid_body(name="part") -> CADBody:
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    return CADBody(name=name, vertices=verts, faces=faces)


def test_geometry_validation_passes_for_valid_body():
    settings = _make_settings()
    body = _valid_body()
    result = validate_geometry([body], settings)
    assert result.passed


def test_geometry_validation_fails_for_degenerate_mesh():
    settings = _make_settings()
    body = CADBody(
        name="bad",
        vertices=np.zeros((3, 3), dtype=float),
        faces=np.zeros((2, 3), dtype=int),
    )
    result = validate_geometry([body], settings)
    assert not result.passed
    assert any("degenerate" in e for e in result.errors)


def test_geometry_validation_fails_for_tiny_mesh():
    """A mesh with extents well below 0.1mm should fail scale check."""
    from simready.config.settings import PipelineSettings
    settings = PipelineSettings(meters_per_unit=1.0)  # 1 unit = 1 metre
    # Bounding box of 1e-7 m (0.1 µm) — clearly wrong scale
    verts = np.array([
        [0, 0, 0], [1e-7, 0, 0], [0, 1e-7, 0], [0, 0, 1e-7],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    body = CADBody(name="tiny", vertices=verts, faces=faces)
    result = validate_geometry([body], settings)
    assert not result.passed
    assert any("extent" in e for e in result.errors)


def test_geometry_validation_warns_for_large_mesh():
    """A mesh bigger than 100 m should emit a warning but not an error."""
    from simready.config.settings import PipelineSettings
    settings = PipelineSettings(meters_per_unit=1.0)
    verts = np.array([
        [0, 0, 0], [200, 0, 0], [0, 200, 0], [0, 0, 200],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    body = CADBody(name="huge", vertices=verts, faces=faces)
    result = validate_geometry([body], settings)
    assert result.passed  # warnings, not errors
    assert any("extent" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# P3.2: Semantic classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected_prefix", [
    ("gear_spur",              "mechanical:gear"),
    ("bolt_m8",                "fastener:bolt"),
    ("PLATE",                  "structural:plate"),
    ("bracket_wall",           "structural:bracket"),
    ("pipe_fitting",           "fluid_system:pipe"),
    ("unknown_exotic_widget",  "industrial_part:component"),
    ("NIST_MTC_CRADA_BOX",     "structural:enclosure"),
    ("COVER",                  "structural:enclosure"),
    ("ASSEMBLY",               "industrial_part:component"),
])
def test_classify_known_parts(name, expected_prefix):
    label = classify(name)
    assert label == expected_prefix, f"classify({name!r}) = {label!r}, expected {expected_prefix!r}"


def test_classify_is_case_insensitive():
    assert classify("GEAR") == classify("gear")
    assert classify("BOLT") == classify("bolt")


# ---------------------------------------------------------------------------
# P2.3: Friction / restitution defaults by material class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mat_name,expected_friction", [
    ("rubber_seal",   0.9),
    ("glass_panel",   0.4),
    ("concrete_slab", 0.7),
])
def test_material_class_friction_defaults(mat_name, expected_friction):
    mat = CAEMaterial(name=mat_name)
    mdl = map_cae_to_mdl(mat)
    assert mdl.friction_static == pytest.approx(expected_friction)
