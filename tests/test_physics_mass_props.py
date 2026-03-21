"""Test that PhysicsMassAPI writes absolute mass, CoM, and diagonal inertia.

Uses a steel sphere (radius = 0.01 m) whose theoretical values are:
  mass            = rho * 4/3 * pi * r^3  ≈ 0.03293 kg
  center_of_mass  = (0, 0, 0)             (centred mesh)
  diagonal_inertia = 2/5 * m * r^2 * I   ≈ (1.317e-6, 1.317e-6, 1.317e-6) kg·m²
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from simready.ingestion.step_reader import CADBody


_STEEL_DENSITY = 7850.0   # kg/m³  (matches material_map.py entry for "steel")
_RADIUS = 0.01            # m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_body(radius: float = _RADIUS, name: str = "sphere_part") -> CADBody:
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    return CADBody(
        name=name,
        vertices=mesh.vertices.astype(np.float64),
        faces=mesh.faces.astype(np.int32),
        normals=mesh.vertex_normals.copy(),
    )


def _steel_mdl():
    from simready.materials.material_map import MDLMaterial
    return MDLMaterial(
        mdl_name="OmniPBR.mdl",
        source_material="sphere_part",
        diffuse_color=(0.6, 0.6, 0.6),
        roughness=0.3,
        metallic=0.9,
        density=_STEEL_DENSITY,
    )


# ---------------------------------------------------------------------------
# Fixture: generate the stage once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sphere_usd(tmp_path_factory) -> Path:
    pytest.importorskip("pxr", reason="usd-core not installed")

    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings

    body = _sphere_body()
    mat = _steel_mdl()

    settings = PipelineSettings()
    settings.validation.enabled = False

    out_path = tmp_path_factory.mktemp("mass_props") / "sphere.usda"
    create_stage([body], {"sphere_part": mat}, out_path, settings)

    assert out_path.exists()
    return out_path


# ---------------------------------------------------------------------------
# Tests: USDA text assertions
# ---------------------------------------------------------------------------

def test_physics_mass_attr_present(sphere_usd: Path):
    """physics:mass must appear in the USDA."""
    text = sphere_usd.read_text()
    assert "physics:mass" in text, "physics:mass not written to USD"


def test_physics_center_of_mass_attr_present(sphere_usd: Path):
    """physics:centerOfMass must appear in the USDA."""
    text = sphere_usd.read_text()
    assert "physics:centerOfMass" in text, "physics:centerOfMass not written to USD"


def test_physics_diagonal_inertia_attr_present(sphere_usd: Path):
    """physics:diagonalInertia must appear in the USDA."""
    text = sphere_usd.read_text()
    assert "physics:diagonalInertia" in text, "physics:diagonalInertia not written to USD"


def test_no_density_attr_for_watertight_mesh(sphere_usd: Path):
    """A watertight mesh with known density must use explicit mass, not density hint."""
    text = sphere_usd.read_text()
    assert "physics:density" not in text, (
        "physics:density fallback written for a watertight mesh — "
        "should use explicit physics:mass instead"
    )


# ---------------------------------------------------------------------------
# Tests: numeric correctness via pxr stage read-back
# ---------------------------------------------------------------------------

def test_mass_value_close_to_theory(sphere_usd: Path):
    """Computed mass must be within 2% of the theoretical value."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(sphere_usd))
    body_prim = stage.GetPrimAtPath("/Root/sphere_part")
    assert body_prim.IsValid(), f"Prim /Root/sphere_part not found"

    mass_api = UsdPhysics.MassAPI(body_prim)
    mass = mass_api.GetMassAttr().Get()
    assert mass is not None, "physics:mass attribute has no value"

    # Theoretical mass for a sphere
    r = _RADIUS
    volume_theory = (4 / 3) * math.pi * r ** 3
    mass_theory = volume_theory * _STEEL_DENSITY

    rel_error = abs(mass - mass_theory) / mass_theory
    assert rel_error < 0.02, (
        f"physics:mass={mass:.6f} kg deviates {rel_error*100:.2f}% from "
        f"theoretical {mass_theory:.6f} kg (>2% tolerance)"
    )


def test_center_of_mass_near_origin(sphere_usd: Path):
    """CoM must be within 1e-5 m of the origin (mesh was centred at CoM)."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(sphere_usd))
    body_prim = stage.GetPrimAtPath("/Root/sphere_part")

    mass_api = UsdPhysics.MassAPI(body_prim)
    com = mass_api.GetCenterOfMassAttr().Get()
    assert com is not None, "physics:centerOfMass attribute has no value"

    dist = math.sqrt(com[0] ** 2 + com[1] ** 2 + com[2] ** 2)
    assert dist < 1e-5, (
        f"physics:centerOfMass={tuple(com)} is {dist:.2e} m from origin "
        f"(expected < 1e-5 m for a centred mesh)"
    )


def test_diagonal_inertia_close_to_theory(sphere_usd: Path):
    """Each principal inertia component must be within 2% of 2/5·m·r²."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(sphere_usd))
    body_prim = stage.GetPrimAtPath("/Root/sphere_part")

    mass_api = UsdPhysics.MassAPI(body_prim)
    mass = mass_api.GetMassAttr().Get()
    di = mass_api.GetDiagonalInertiaAttr().Get()
    assert di is not None, "physics:diagonalInertia attribute has no value"

    I_theory = (2 / 5) * mass * _RADIUS ** 2
    for axis, val in enumerate(di):
        rel_error = abs(val - I_theory) / I_theory
        assert rel_error < 0.02, (
            f"diagonalInertia[{axis}]={val:.3e} kg·m² deviates {rel_error*100:.2f}% "
            f"from theoretical {I_theory:.3e} kg·m² (>2% tolerance)"
        )
