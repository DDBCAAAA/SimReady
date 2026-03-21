"""Test that concave meshes produce multiple collision prims via CoACD decomposition.

Strategy:
- Build a torus in-memory using trimesh (concave — has a central bore).
- Wrap it in a minimal CADBody and call create_stage directly.
- Parse the resulting .usda and verify that multiple Collision_<i> prims exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from simready.ingestion.step_reader import CADBody
from simready.geometry.mesh_processing import decompose_convex


# ---------------------------------------------------------------------------
# Unit test: decompose_convex itself
# ---------------------------------------------------------------------------

def test_decompose_convex_returns_multiple_parts():
    """A torus should decompose into more than one convex part."""
    pytest.importorskip("coacd", reason="coacd not installed")
    torus = trimesh.creation.torus(major_radius=0.05, minor_radius=0.015)
    parts = decompose_convex(torus, threshold=0.05)
    assert len(parts) > 1, (
        f"Expected >1 convex parts for a torus, got {len(parts)}"
    )
    for verts, faces in parts:
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3


def test_decompose_convex_fallback_on_convex_mesh():
    """A convex mesh (box) should still return at least one part even with coacd."""
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    parts = decompose_convex(box, threshold=0.05)
    assert len(parts) >= 1


# ---------------------------------------------------------------------------
# Integration test: USD stage collision prim count
# ---------------------------------------------------------------------------

def _make_gear_body(mesh: trimesh.Trimesh, name: str = "gear_spur") -> CADBody:
    """Wrap a trimesh as a CADBody with a gear semantic name."""
    return CADBody(
        name=name,
        vertices=mesh.vertices.astype(np.float64),
        faces=mesh.faces.astype(np.int32),
        normals=mesh.vertex_normals.copy(),
    )


@pytest.fixture(scope="module")
def torus_usd(tmp_path_factory):
    """Run create_stage on a torus CADBody and return the .usda path."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    pytest.importorskip("coacd", reason="coacd not installed")

    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings
    from simready.materials.material_map import MDLMaterial

    torus = trimesh.creation.torus(major_radius=0.05, minor_radius=0.015)
    body = _make_gear_body(torus, name="gear_spur")

    mdl_mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        source_material="gear_spur",
        diffuse_color=(0.6, 0.6, 0.6),
        roughness=0.4,
        metallic=0.9,
    )

    out_dir = tmp_path_factory.mktemp("torus_usd")
    out_path = out_dir / "torus_gear.usda"

    settings = PipelineSettings()
    settings.validation.enabled = False  # skip material validation for this test

    create_stage([body], {"gear_spur": mdl_mat}, out_path, settings)

    assert out_path.exists(), "create_stage did not produce a .usda file"
    return out_path


def test_multiple_collision_prims_in_usd(torus_usd: Path):
    """The .usda must contain at least 2 Collision_<i> prims for a gear body."""
    text = torus_usd.read_text()

    # Count distinct Collision_<N> prim definitions
    collision_prims = re.findall(r'"Collision_(\d+)"', text)
    count = len(set(collision_prims))

    assert count > 1, (
        f"Expected multiple Collision_<i> prims for a concave gear body, "
        f"found {count}.\n\nRelevant lines:\n"
        + "\n".join(ln for ln in text.splitlines() if "Collision" in ln)
    )


def test_no_single_convex_hull_approximation_on_gear(torus_usd: Path):
    """The old single-mesh convexHull approximation must not appear for gear bodies."""
    text = torus_usd.read_text()
    # The old pattern was a prim named exactly "Collision" (no index suffix)
    assert '"Collision"' not in text, (
        "Found legacy single 'Collision' prim — expected indexed Collision_<i> prims"
    )


def test_collision_prims_are_invisible(torus_usd: Path):
    """Every Collision_<i> prim must have visibility = invisible."""
    text = torus_usd.read_text()
    # Find all Collision_<i> blocks and verify each has the invisible token
    blocks = re.split(r'def Mesh "Collision_\d+"', text)
    # blocks[0] is preamble; blocks[1..] are the prim bodies
    for i, block in enumerate(blocks[1:], 1):
        # Each block ends at the next top-level def — grab up to the first closing brace
        body = block.split("\n}")[0]
        assert "invisible" in body, (
            f"Collision_{i - 1} prim body does not set visibility = invisible"
        )
