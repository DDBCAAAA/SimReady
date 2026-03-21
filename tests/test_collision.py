"""Collision mesh regression tests — verifies CoACD triggers for concave
industrial parts (flanges, washers, bearings, pipes, brackets, frames).

TDD-first: these tests verify that 'structural:flange' (and other concave-part
labels) appear in _DECOMPOSE_LABELS and that the USD exporter writes each hull
as an independent Collision_<i> prim with PhysicsCollisionAPI applied.

Key assertion design note:
  content.count("PhysicsCollisionAPI") == number of collision prims
  because CollisionAPI only adds "PhysicsCollisionAPI" to apiSchemas (once per
  prim). MeshCollisionAPI is intentionally NOT applied as an API schema so the
  count remains unambiguous; physics:approximation is set directly via
  CreateAttribute so PhysX still receives the convexHull hint.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from simready.ingestion.step_reader import CADBody


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_body(mesh: trimesh.Trimesh, name: str, semantic_label: str | None = None) -> CADBody:
    """Wrap a trimesh as a minimal CADBody, optionally pinning a semantic label."""
    body = CADBody(
        name=name,
        vertices=mesh.vertices.astype(np.float64),
        faces=mesh.faces.astype(np.int32),
        normals=mesh.vertex_normals.copy(),
    )
    if semantic_label is not None:
        body.metadata["semantic_label"] = semantic_label
    return body


# ---------------------------------------------------------------------------
# Fixture: run create_stage on a torus labeled 'structural:flange'
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flange_usd(tmp_path_factory):
    """Create a USD stage for a torus body with semantic_label='structural:flange'.

    A torus is the canonical ring-with-hole shape — a single convex hull fills
    the center bore entirely, causing interpenetration in Isaac Sim. After the
    fix, CoACD must decompose it into multiple tight-fitting hulls.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")
    pytest.importorskip("coacd", reason="coacd not installed")

    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings
    from simready.materials.material_map import MDLMaterial

    torus = trimesh.creation.torus(major_radius=0.05, minor_radius=0.015)
    body = _make_body(torus, name="DN15_Stamped_Flange", semantic_label="structural:flange")

    mdl_mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        source_material="DN15_Stamped_Flange",
        diffuse_color=(0.55, 0.56, 0.58),
        roughness=0.35,
        metallic=1.0,
        density=7850.0,
    )

    out_dir = tmp_path_factory.mktemp("flange_usd")
    out_path = out_dir / "flange.usda"

    settings = PipelineSettings()
    settings.validation.enabled = False  # skip material validation in test

    create_stage([body], {"DN15_Stamped_Flange": mdl_mat}, out_path, settings)

    assert out_path.exists(), "create_stage did not produce a .usda file"
    return out_path


# ---------------------------------------------------------------------------
# Test 1: core assertion — PhysicsCollisionAPI appears more than once
# ---------------------------------------------------------------------------

def test_flange_multi_hull_collision(flange_usd: Path):
    """A torus labeled 'structural:flange' must produce >1 collision prims.

    count("PhysicsCollisionAPI") == number of Collision_<i> prims because
    CollisionAPI contributes exactly one schema token per prim.
    """
    content = flange_usd.read_text()
    count = content.count("PhysicsCollisionAPI")
    assert count > 1, (
        f"Expected >1 occurrences of 'PhysicsCollisionAPI' for a flange torus body, "
        f"got {count}.\n\n"
        "This means CoACD was NOT triggered for 'structural:flange' — check that\n"
        "'structural:flange' is listed in _DECOMPOSE_LABELS in simready/usd/assembly.py.\n\n"
        "Relevant lines:\n"
        + "\n".join(ln for ln in content.splitlines() if "Collision" in ln or "Physics" in ln)
    )


# ---------------------------------------------------------------------------
# Test 2: each hull must have physics:approximation = "convexHull"
# ---------------------------------------------------------------------------

def test_flange_collision_prims_have_convex_hull_approximation(flange_usd: Path):
    """Every Collision_<i> prim must declare physics:approximation = 'convexHull'."""
    content = flange_usd.read_text()
    # Find blocks introduced by each collision prim definition
    blocks = re.split(r'def Mesh "Collision_\d+"', content)
    # blocks[0] is the preamble; blocks[1..] are the prim bodies
    prim_count = len(blocks) - 1
    assert prim_count > 1, (
        f"Expected >1 Collision_<i> prims, found {prim_count}"
    )
    for i, block in enumerate(blocks[1:], start=0):
        prim_body = block.split("\n}")[0]
        assert "convexHull" in prim_body, (
            f"Collision_{i} prim does not contain physics:approximation = 'convexHull'.\n"
            "Check that _write_collision_prims sets this attribute.\n"
            f"Prim body:\n{prim_body[:400]}"
        )


# ---------------------------------------------------------------------------
# Test 3: each hull must be invisible
# ---------------------------------------------------------------------------

def test_flange_collision_prims_are_invisible(flange_usd: Path):
    """Every Collision_<i> prim must have visibility = invisible."""
    content = flange_usd.read_text()
    blocks = re.split(r'def Mesh "Collision_\d+"', content)
    for i, block in enumerate(blocks[1:], start=0):
        prim_body = block.split("\n}")[0]
        assert "invisible" in prim_body, (
            f"Collision_{i} is missing visibility = 'invisible'"
        )


# ---------------------------------------------------------------------------
# Test 4: whitelist unit tests — all concave labels must be in _DECOMPOSE_LABELS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [
    "mechanical:gear",
    "mechanical:cam",
    "mechanical:bearing",
    "fluid_system:pipe",
    "fluid_system:fitting",
    "fastener:washer",
    "structural:flange",
    "structural:bracket",
    "structural:frame",
])
def test_decompose_label_whitelist(label: str):
    """Each concave-part semantic label must be in _DECOMPOSE_LABELS."""
    from simready.usd.assembly import _DECOMPOSE_LABELS
    assert label in _DECOMPOSE_LABELS, (
        f"'{label}' is missing from _DECOMPOSE_LABELS — "
        "parts with this label will get a single hull that blocks their bore/hole."
    )
