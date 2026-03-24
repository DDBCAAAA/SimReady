"""STL/OBJ reader — loads mesh files into the intermediate geometry representation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from simready.ingestion.step_reader import CADAssembly, CADBody

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".stl", ".obj"}


def read_mesh(path: Path) -> CADAssembly:
    """Read an STL or OBJ file into a CADAssembly using trimesh.

    Args:
        path: Path to the mesh file.

    Returns:
        CADAssembly with one CADBody per mesh in the file.
    """
    import trimesh

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"Expected {_SUPPORTED_SUFFIXES} file, got: {suffix}")

    loaded = trimesh.load(str(path), force="mesh")
    assembly = CADAssembly(source_path=path)

    # trimesh sets .units for some formats (e.g. GLTF, 3MF); propagate when present.
    raw_units = getattr(loaded, "units", None)
    if raw_units:
        assembly.units = str(raw_units)
        logger.info("Detected trimesh units '%s' from %s", raw_units, path.name)
    else:
        logger.warning(
            "No unit metadata found in %s. Defaulting to millimeters (0.001 scale).",
            path.name,
        )

    if isinstance(loaded, trimesh.Scene):
        for name, mesh in loaded.geometry.items():
            _append_body(assembly, mesh, name)
    else:
        _append_body(assembly, loaded, path.stem)

    logger.info("Read %d bodies from %s", len(assembly.bodies), path.name)
    return assembly


def _append_body(assembly: CADAssembly, mesh: object, name: str) -> None:
    """Append a single trimesh.Trimesh as a CADBody, skipping empty meshes."""
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return
    assembly.bodies.append(CADBody(
        name=name,
        vertices=np.array(mesh.vertices, dtype=np.float64),
        faces=np.array(mesh.faces, dtype=np.int64),
        normals=np.array(mesh.vertex_normals, dtype=np.float64),
        material_name=name,
    ))
