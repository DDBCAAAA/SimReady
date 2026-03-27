"""Headless multi-view STL renderer using trimesh + matplotlib Agg.

Produces PNG screenshots for VLM critic input without requiring a GPU or
display server.  Uses mpl_toolkits.mplot3d.art3d.Poly3DCollection with
face-normal shading for a clear, publication-quality view of the geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Camera: (elevation_deg, azimuth_deg)
_VIEW_PARAMS: dict[str, tuple[float, float]] = {
    "isometric": (30, -45),
    "front":     (0, 0),
    "top":       (90, 0),
    "side":      (0, 90),
}

_DEFAULT_VIEWS = ["isometric", "front"]


def render_views(
    stl_path: Path,
    output_dir: Path,
    views: list[str] | None = None,
    resolution: tuple[int, int] = (512, 512),
) -> list[Path]:
    """Render *views* of the mesh at *stl_path* and save PNGs to *output_dir*.

    Args:
        stl_path:   Path to the STL mesh file.
        output_dir: Directory where PNG files will be written.
        views:      List of view names from {"isometric","front","top","side"}.
                    Defaults to ["isometric", "front"].
        resolution: Output image size in pixels (width, height).

    Returns:
        List of Paths to the written PNG files (one per view).
    """
    import matplotlib
    matplotlib.use("Agg")  # must be set before importing pyplot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    if views is None:
        views = _DEFAULT_VIEWS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh — force single mesh even if file contains a scene
    try:
        mesh = trimesh.load(str(stl_path), force="mesh")
    except Exception as exc:
        logger.error("Failed to load STL for rendering: %s", exc)
        return []

    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        logger.warning("Mesh has no faces — skipping render.")
        return []

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces    = np.asarray(mesh.faces,    dtype=int)
    normals  = np.asarray(mesh.face_normals, dtype=float)

    # Simple directional lighting: light from upper-front-right
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir /= np.linalg.norm(light_dir)
    brightness = np.clip(normals @ light_dir, 0.15, 1.0)

    # Steel-blue palette shaded by brightness
    face_colors = np.column_stack([
        0.35 * brightness + 0.30,  # R
        0.55 * brightness + 0.20,  # G
        0.75 * brightness + 0.10,  # B
        np.full(len(faces), 0.92),  # A
    ])

    # Bounding box for axis limits
    bounds = mesh.bounds  # shape (2, 3)
    center = bounds.mean(axis=0)
    half_size = (bounds[1] - bounds[0]).max() * 0.55

    dpi = max(50, resolution[0] // 5)
    fig_size_in = resolution[0] / dpi

    output_paths: list[Path] = []
    for view_name in views:
        elev, azim = _VIEW_PARAMS.get(view_name, _VIEW_PARAMS["isometric"])

        fig = plt.figure(figsize=(fig_size_in, fig_size_in), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        poly_verts = vertices[faces]
        poly = Poly3DCollection(
            poly_verts,
            facecolors=face_colors,
            edgecolors="none",
            linewidth=0,
        )
        ax.add_collection3d(poly)

        ax.set_xlim(center[0] - half_size, center[0] + half_size)
        ax.set_ylim(center[1] - half_size, center[1] + half_size)
        ax.set_zlim(center[2] - half_size, center[2] + half_size)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.patch.set_facecolor("white")

        out_path = output_dir / f"view_{view_name}.png"
        plt.savefig(
            str(out_path),
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            dpi=dpi,
        )
        plt.close(fig)
        output_paths.append(out_path)
        logger.debug("Rendered view '%s' → %s", view_name, out_path.name)

    return output_paths
