"""Tests for link-level mass properties and data provenance in Path A (link-grouped) mode.

When ArticulationTopology.rigid_links is present:
  - PhysicsMassAPI must be applied to the link Xform (not body sub-Xforms).
  - physics:mass must be written to the link Xform.
  - simready:reasoning_step, simready:materialConfidence, simready:physicsComplete
    in link Xform customData.
  - Body sub-Xforms must NOT carry MassAPI or simready provenance.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEEL_DENSITY = 7850.0  # kg/m³


def _make_sphere_body(name: str, radius: float = 0.05):
    """Watertight icosphere body suitable for mass computation tests."""
    from simready.ingestion.step_reader import CADBody
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    return CADBody(
        name=name,
        vertices=mesh.vertices.astype(np.float64),
        faces=mesh.faces.astype(np.int32),
    )


def _make_steel_mat(name: str):
    """Steel MDLMaterial with density, friction, and VLM provenance fields."""
    from simready.materials.material_map import MDLMaterial
    return MDLMaterial(
        mdl_name="OmniPBR.mdl",
        source_material=name,
        density=_STEEL_DENSITY,
        friction_static=0.55,
        friction_dynamic=0.42,
        restitution=0.3,
        confidence=0.9,
        vlm_primary_material="steel",
        vlm_reasoning_step="Steel blade inferred from metallic CAE properties.",
    )


def _single_link_topology(link_name: str, parts: list[str]):
    """Topology: one link group with the given parts, no joints."""
    from simready.articulation_inference import ArticulationTopology, RigidLinkGroup
    return ArticulationTopology(
        base_link_name=link_name,
        rigid_links=[RigidLinkGroup(link_name=link_name, constituent_parts=parts)],
        joints=[],
    )


def _build_stage(tmp_path: Path, bodies, materials, topology):
    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings

    settings = PipelineSettings()
    settings.validation.enabled = False
    out_path = tmp_path / "test_link_mass.usda"
    create_stage(bodies, materials, out_path, settings, topology=topology)
    return out_path


# ---------------------------------------------------------------------------
# test_link_has_mass_api
# ---------------------------------------------------------------------------

def test_link_has_mass_api(tmp_path: Path):
    """PhysicsMassAPI must be applied to the link Xform in Path A."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd, UsdPhysics

    bodies = [_make_sphere_body("part_a"), _make_sphere_body("part_b")]
    materials = {n: _make_steel_mat(n) for n in ["part_a", "part_b"]}
    topology = _single_link_topology("link_X", ["part_a", "part_b"])

    out_path = _build_stage(tmp_path, bodies, materials, topology)
    stage = Usd.Stage.Open(str(out_path))

    link_prim = stage.GetPrimAtPath("/Root/link_X")
    assert link_prim.IsValid(), "link_X Xform not found in stage"
    assert link_prim.HasAPI(UsdPhysics.MassAPI), (
        "PhysicsMassAPI must be applied to the link Xform, not child meshes"
    )


# ---------------------------------------------------------------------------
# test_link_physics_mass_attr_written
# ---------------------------------------------------------------------------

def test_link_physics_mass_attr_written(tmp_path: Path):
    """physics:mass on the link Xform must be a positive finite value."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd, UsdPhysics

    bodies = [_make_sphere_body("part_a"), _make_sphere_body("part_b")]
    materials = {n: _make_steel_mat(n) for n in ["part_a", "part_b"]}
    topology = _single_link_topology("link_X", ["part_a", "part_b"])

    out_path = _build_stage(tmp_path, bodies, materials, topology)
    stage = Usd.Stage.Open(str(out_path))

    link_prim = stage.GetPrimAtPath("/Root/link_X")
    mass_api = UsdPhysics.MassAPI(link_prim)
    mass = mass_api.GetMassAttr().Get()
    assert mass is not None, "physics:mass attribute missing on link Xform"
    assert math.isfinite(mass) and mass > 0, (
        f"physics:mass must be a positive finite value, got {mass}"
    )


# ---------------------------------------------------------------------------
# test_link_mass_is_sum_of_parts
# ---------------------------------------------------------------------------

def test_link_mass_is_sum_of_parts(tmp_path: Path):
    """Combined link mass must equal the sum of individual part masses (within 0.1%).

    Uses actual trimesh volume (same as the implementation) as the reference,
    avoiding any discrepancy from icosphere discretisation vs. theoretical sphere.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd, UsdPhysics

    radius = 0.05
    part_bodies = [_make_sphere_body("part_a", radius), _make_sphere_body("part_b", radius)]
    materials = {n: _make_steel_mat(n) for n in ["part_a", "part_b"]}
    topology = _single_link_topology("link_X", ["part_a", "part_b"])

    out_path = _build_stage(tmp_path, part_bodies, materials, topology)
    stage = Usd.Stage.Open(str(out_path))

    link_prim = stage.GetPrimAtPath("/Root/link_X")
    link_mass = UsdPhysics.MassAPI(link_prim).GetMassAttr().Get()

    # Expected = sum of actual mesh volumes * density (matches implementation exactly)
    expected = sum(
        trimesh.Trimesh(vertices=b.vertices, faces=b.faces, process=False).volume * _STEEL_DENSITY
        for b in part_bodies
    )
    rel_err = abs(link_mass - expected) / expected
    assert rel_err < 0.001, (
        f"Link mass {link_mass:.6f} kg deviates {rel_err*100:.4f}% from "
        f"expected sum-of-parts {expected:.6f} kg (tolerance 0.1%)"
    )


# ---------------------------------------------------------------------------
# test_body_sub_xforms_have_no_mass_api
# ---------------------------------------------------------------------------

def test_body_sub_xforms_have_no_mass_api(tmp_path: Path):
    """Body sub-Xforms under a link must NOT have PhysicsMassAPI."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd, UsdPhysics

    bodies = [_make_sphere_body("part_a"), _make_sphere_body("part_b")]
    materials = {n: _make_steel_mat(n) for n in ["part_a", "part_b"]}
    topology = _single_link_topology("link_X", ["part_a", "part_b"])

    out_path = _build_stage(tmp_path, bodies, materials, topology)
    stage = Usd.Stage.Open(str(out_path))

    for sub_name in ["part_a", "part_b"]:
        sub_prim = stage.GetPrimAtPath(f"/Root/link_X/{sub_name}")
        assert sub_prim.IsValid(), f"{sub_name} sub-Xform not found under link_X"
        assert not sub_prim.HasAPI(UsdPhysics.MassAPI), (
            f"{sub_name} sub-Xform must NOT carry PhysicsMassAPI — "
            "mass belongs to the parent link Xform only"
        )


# ---------------------------------------------------------------------------
# test_link_customdata_has_provenance
# ---------------------------------------------------------------------------

def test_link_customdata_has_provenance(tmp_path: Path):
    """Link Xform customData must include reasoning_step, materialConfidence, physicsComplete.

    USD stores 'simready:x' as nested dict {'simready': {'x': ...}};
    use GetCustomDataByKey for reliable namespace-aware access.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd

    bodies = [_make_sphere_body("part_a"), _make_sphere_body("part_b")]
    materials = {n: _make_steel_mat(n) for n in ["part_a", "part_b"]}
    topology = _single_link_topology("link_X", ["part_a", "part_b"])

    out_path = _build_stage(tmp_path, bodies, materials, topology)
    stage = Usd.Stage.Open(str(out_path))

    link_prim = stage.GetPrimAtPath("/Root/link_X")

    reasoning = link_prim.GetCustomDataByKey("simready:reasoning_step")
    assert reasoning is not None, (
        "simready:reasoning_step missing from link Xform customData"
    )

    confidence = link_prim.GetCustomDataByKey("simready:materialConfidence")
    assert confidence is not None, (
        "simready:materialConfidence missing from link Xform customData"
    )
    assert confidence > 0.0, "simready:materialConfidence must be a positive float"

    physics_complete = link_prim.GetCustomDataByKey("simready:physicsComplete")
    assert physics_complete is True, (
        "simready:physicsComplete must be True when mass was successfully computed"
    )


# ---------------------------------------------------------------------------
# test_body_sub_xforms_have_no_simready_provenance
# ---------------------------------------------------------------------------

def test_body_sub_xforms_have_no_simready_provenance(tmp_path: Path):
    """Body sub-Xforms must not carry simready physics/provenance customData in Path A.

    They are pure visual/collision containers — provenance lives on the link Xform.
    Uses GetCustomDataByKey for namespace-aware access.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")
    from pxr import Usd

    bodies = [_make_sphere_body("part_a"), _make_sphere_body("part_b")]
    materials = {n: _make_steel_mat(n) for n in ["part_a", "part_b"]}
    topology = _single_link_topology("link_X", ["part_a", "part_b"])

    out_path = _build_stage(tmp_path, bodies, materials, topology)
    stage = Usd.Stage.Open(str(out_path))

    for sub_name in ["part_a", "part_b"]:
        sub_prim = stage.GetPrimAtPath(f"/Root/link_X/{sub_name}")
        assert sub_prim.IsValid(), f"{sub_name} sub-Xform not found"
        for provenance_key in [
            "simready:reasoning_step",
            "simready:primary_material",
            "simready:physicsComplete",
            "simready:materialConfidence",
            "simready:qualityScore",
        ]:
            val = sub_prim.GetCustomDataByKey(provenance_key)
            assert val is None, (
                f"{provenance_key} must NOT be on body sub-Xform {sub_name} in Path A — "
                f"provenance belongs on the link Xform (got {val!r})"
            )
