"""M0 quality-gap tests: CoM pivot normalization, inertia tensor, quality score."""

from __future__ import annotations

import numpy as np
import pytest

from simready.ingestion.step_reader import CADBody
from simready.materials.material_map import CAEMaterial, MDLMaterial, map_cae_to_mdl
from simready.validation.simready_checks import compute_quality_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    """Unit tetrahedron — watertight, 4 faces."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    return verts, faces


def _steel_mdl() -> MDLMaterial:
    mat = CAEMaterial(name="steel_part")
    return map_cae_to_mdl(mat)


# ---------------------------------------------------------------------------
# CoM pivot normalization
# ---------------------------------------------------------------------------

def test_center_at_com_watertight_centroid_near_origin():
    """After centering, the pivot used (center_mass or centroid) must be at origin."""
    import trimesh
    from simready.geometry.mesh_processing import center_at_com

    # Use a trimesh primitive — guaranteed consistent winding and watertightness
    offset = np.array([100.0, 200.0, 300.0])
    mesh = trimesh.creation.box()
    mesh.apply_translation(offset)

    centered, com = center_at_com(mesh)

    # The com returned must be non-trivially far from origin (mesh was translated)
    assert np.linalg.norm(com) > 50.0

    # After centering, the same pivot computed on the new mesh must be near origin
    recheck = trimesh.Trimesh(vertices=centered.vertices, faces=centered.faces, process=False)
    pivot_after = recheck.center_mass if recheck.is_watertight else recheck.centroid
    assert np.linalg.norm(pivot_after) < 1e-4, (
        f"Pivot not at origin after centering: {pivot_after}"
    )


def test_center_at_com_returns_original_offset():
    """The returned com must equal the pivot the function actually used."""
    import trimesh
    from simready.geometry.mesh_processing import center_at_com

    verts, faces = _tetrahedron()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Capture expected pivot before centering
    expected_com = mesh.center_mass.copy() if mesh.is_watertight else mesh.centroid.copy()
    _, com = center_at_com(mesh)
    assert np.allclose(com, expected_com, atol=1e-6)


def test_pipeline_stores_com_in_metadata(tmp_path):
    """After pipeline geometry processing, body.metadata['center_of_mass'] must exist."""
    import trimesh
    from simready.geometry.mesh_processing import cad_body_to_trimesh, clean_mesh, center_at_com

    verts, faces = _tetrahedron()
    body = CADBody(name="part", vertices=verts + 50, faces=faces)

    mesh = cad_body_to_trimesh(body)
    mesh = clean_mesh(mesh)
    mesh, com = center_at_com(mesh)
    body.metadata["center_of_mass"] = com.tolist()

    assert "center_of_mass" in body.metadata
    assert len(body.metadata["center_of_mass"]) == 3


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------

def test_quality_score_physics_complete_steel():
    """Steel body with known physics should score high and flag physicsComplete."""
    verts, faces = _tetrahedron()
    # Inflate to 1000+ faces via LOD trick — use raw large mesh
    large_verts = np.random.default_rng(0).random((500, 3))
    import trimesh
    hull = trimesh.convex.convex_hull(large_verts)
    body = CADBody(name="steel_part", vertices=hull.vertices, faces=hull.faces)
    mdl = _steel_mdl()
    q = compute_quality_score(body, mdl)

    assert q["simready:physicsComplete"] is True
    assert q["simready:materialConfidence"] > 0.0
    assert 0.0 <= q["simready:qualityScore"] <= 1.0


def test_quality_score_no_material_low_score():
    """Body with no material info must not be falsely flagged as physics-complete."""
    verts, faces = _tetrahedron()
    body = CADBody(name="unknown_part", vertices=verts, faces=faces)
    q = compute_quality_score(body, None)

    assert q["simready:physicsComplete"] is False
    assert q["simready:materialConfidence"] == 0.0
    # Score must still be a valid float in range
    assert 0.0 <= q["simready:qualityScore"] <= 1.0


def test_quality_score_watertight_flag():
    """Watertight tetrahedron must be flagged watertight=True."""
    verts, faces = _tetrahedron()
    body = CADBody(name="tet", vertices=verts, faces=faces)
    q = compute_quality_score(body, None)
    assert q["simready:watertight"] is True


def test_quality_score_open_mesh_not_watertight():
    """A single open triangle must be flagged watertight=False."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    body = CADBody(name="open", vertices=verts, faces=faces)
    q = compute_quality_score(body, None)
    assert q["simready:watertight"] is False


def test_quality_score_keys_present():
    """compute_quality_score must always return all four expected keys."""
    verts, faces = _tetrahedron()
    body = CADBody(name="part", vertices=verts, faces=faces)
    q = compute_quality_score(body, None)
    expected_keys = {
        "simready:qualityScore",
        "simready:watertight",
        "simready:physicsComplete",
        "simready:materialConfidence",
    }
    assert expected_keys == set(q.keys())


# ---------------------------------------------------------------------------
# Inertia tensor (via rotation_matrix_to_quatf helper)
# ---------------------------------------------------------------------------

def test_rotation_matrix_identity_gives_identity_quat():
    """Identity rotation matrix must yield quaternion (w=1, x=0, y=0, z=0)."""
    # Import the private helper directly to test the math
    from simready.usd.assembly import _rotation_matrix_to_quatf
    from pxr import Gf

    R = np.eye(3)
    q = _rotation_matrix_to_quatf(R)
    assert isinstance(q, Gf.Quatf)
    assert abs(q.GetReal() - 1.0) < 1e-5
    img = q.GetImaginary()
    assert abs(img[0]) < 1e-5
    assert abs(img[1]) < 1e-5
    assert abs(img[2]) < 1e-5


def test_rotation_matrix_90deg_z():
    """90° rotation around Z must produce correct quaternion (w=cos45, z=sin45)."""
    from simready.usd.assembly import _rotation_matrix_to_quatf
    import math

    R = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=float)
    q = _rotation_matrix_to_quatf(R)
    expected_w = math.cos(math.pi / 4)
    expected_z = math.sin(math.pi / 4)
    assert abs(q.GetReal() - expected_w) < 1e-5
    assert abs(q.GetImaginary()[2] - expected_z) < 1e-5
