"""Mesh processing utilities — cleanup, normals, and LOD generation."""

from __future__ import annotations

import logging
import time

import numpy as np
import trimesh

from simready.ingestion.step_reader import CADBody

logger = logging.getLogger(__name__)


def get_meters_conversion_factor(mesh_unit: str) -> float:
    """Return the scale factor to convert mesh_unit to meters.

    Handles common engineering unit strings. Falls back to 0.001 (millimeters)
    with a warning if the unit is unrecognized.

    Args:
        mesh_unit: Unit string (e.g. 'mm', 'inch', 'cm', 'm').

    Returns:
        Multiplicative factor such that ``value_in_unit * factor = value_in_meters``.
    """
    u = mesh_unit.lower().strip()
    if u in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
        return 0.001
    if u in ("in", "inch", "inches"):
        return 0.0254
    if u in ("cm", "centimeter", "centimeters", "centimetre", "centimetres"):
        return 0.01
    if u in ("m", "meter", "meters", "metre", "metres"):
        return 1.0
    if u in ("ft", "foot", "feet"):
        return 0.3048
    logger.warning(
        "Unrecognized unit '%s'; defaulting to millimeters (0.001 scale).",
        mesh_unit,
    )
    return 0.001


def cad_body_to_trimesh(body: CADBody) -> trimesh.Trimesh:
    """Convert a CADBody to a trimesh.Trimesh for processing."""
    return trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove degenerate faces, duplicate vertices, and fix winding order."""
    # Remove degenerate faces (API changed in trimesh 4.x)
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    else:
        mask = mesh.nondegenerate_faces()
        mesh.update_faces(mask)
    # Remove duplicate faces
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    else:
        unique = mesh.unique_faces()
        mesh.update_faces(unique)
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def compute_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute per-vertex normals."""
    return mesh.vertex_normals.copy()


def center_at_com(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Translate mesh so its center of mass sits at the origin.

    Uses ``center_mass`` for watertight meshes (volume-weighted) and falls back
    to the surface ``centroid`` for open meshes.

    Returns:
        (centered_mesh, com): Re-centered mesh and the original CoM offset vector.
    """
    com = mesh.center_mass if mesh.is_watertight else mesh.centroid
    centered = trimesh.Trimesh(vertices=mesh.vertices - com, faces=mesh.faces, process=False)
    centered.fix_normals()
    return centered, com


_COACD_MAX_VERTS = 50_000  # pre-decimate collision mesh to this cap before CoACD


def decompose_convex(
    mesh: trimesh.Trimesh,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Approximate convex decomposition using CoACD.

    Returns a list of (vertices, faces) pairs — one per convex part.
    Falls back to a single convex hull if coacd is unavailable or fails.

    Args:
        mesh: Input (potentially concave) mesh.
        threshold: CoACD concavity threshold; lower = more parts, higher fidelity.
        max_convex_hull: Hard cap on output parts (-1 = unlimited).
    """
    try:
        import coacd
        coacd.set_log_level("error")

        # Pre-decimate large meshes — CoACD only needs shape approximation, not
        # full visual fidelity, so capping at _COACD_MAX_VERTS keeps runtime bounded.
        coacd_mesh = mesh
        n_verts = len(mesh.vertices)
        if n_verts > _COACD_MAX_VERTS:
            target_faces = max(4, int(len(mesh.faces) * _COACD_MAX_VERTS / n_verts))
            try:
                # trimesh 4.x: use face_count= keyword (first positional is `percent` 0–1)
                coacd_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                logger.info(
                    "CoACD pre-decimate: %d → %d vertices (%d → %d faces)",
                    n_verts, len(coacd_mesh.vertices), len(mesh.faces), len(coacd_mesh.faces),
                )
            except Exception as dec_exc:
                logger.warning("CoACD pre-decimate failed (%s), using full mesh", dec_exc)
                coacd_mesh = mesh

        cm = coacd.Mesh(
            coacd_mesh.vertices.astype(np.float64),
            coacd_mesh.faces.astype(np.int32),
        )
        logger.info(
            "CoACD: %d vertices / %d faces — decomposing (this may take a while for large meshes)...",
            len(coacd_mesh.vertices), len(coacd_mesh.faces),
        )
        t0 = time.time()
        parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_convex_hull)
        elapsed = time.time() - t0
        result = [(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32))
                  for verts, faces in parts]
        logger.info("CoACD decomposed into %d convex parts in %.1fs", len(result), elapsed)
        return result
    except ImportError:
        logger.warning("coacd not installed; falling back to single convex hull")
    except Exception as exc:
        logger.warning("CoACD decomposition failed (%s); falling back to single convex hull", exc)

    hull = mesh.convex_hull
    return [(hull.vertices.astype(np.float64), hull.faces.astype(np.int32))]


def scale_to_meters(mesh: trimesh.Trimesh, scale: float = 0.001) -> trimesh.Trimesh:
    """Scale vertex coordinates from millimeters to meters.

    Args:
        mesh: Source mesh with vertices in millimeters.
        scale: Multiplicative scale factor (default 0.001 = mm → m).

    Returns:
        New Trimesh with vertices multiplied by scale.
    """
    scaled = trimesh.Trimesh(
        vertices=mesh.vertices * scale,
        faces=mesh.faces,
        process=False,
    )
    scaled.fix_normals()
    return scaled


def generate_lod(mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
    """Generate a LOD by simplifying the mesh to a target face ratio.

    Args:
        mesh: Source mesh.
        ratio: Target ratio of faces to keep (0.0–1.0).

    Returns:
        Simplified mesh.
    """
    if ratio >= 1.0:
        return mesh.copy()

    target_faces = max(4, int(len(mesh.faces) * ratio))
    if target_faces >= len(mesh.faces):
        # Nothing to decimate (mesh already at or below target)
        return mesh.copy()
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    logger.info(
        "LOD %.0f%%: %d → %d faces",
        ratio * 100,
        len(mesh.faces),
        len(simplified.faces),
    )
    return simplified
