"""SimReady compliance validation checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from simready.config.settings import PipelineSettings
from simready.ingestion.step_reader import CADBody
from simready.materials.material_map import MDLMaterial

logger = logging.getLogger(__name__)


def compute_quality_score(
    body: "CADBody",
    mdl_mat: "MDLMaterial | None",
) -> dict:
    """Compute a composite quality score for a single asset body.

    Components and weights:
    - watertight        0.30  (1.0 if watertight, 0.5 if open surface)
    - physics_complete  0.40  (density + static/dynamic friction + restitution all present)
    - mat_confidence    0.15  (MDL material source confidence, 0.0–1.0)
    - face_density      0.15  (1.0 at ≥1 000 faces, linear below)

    Returns a dict of customData keys ready to write to USD:
        simready:qualityScore        float 0.0–1.0
        simready:watertight          bool
        simready:physicsComplete     bool
        simready:materialConfidence  float 0.0–1.0
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = False

    physics_complete = (
        mdl_mat is not None
        and mdl_mat.density is not None
        and mdl_mat.friction_static is not None
        and mdl_mat.friction_dynamic is not None
        and mdl_mat.restitution is not None
    )

    mat_confidence = mdl_mat.confidence if mdl_mat is not None else 0.0
    face_score = min(1.0, len(body.faces) / 1000.0)

    quality = (
        0.30 * (1.0 if watertight else 0.5)
        + 0.40 * (1.0 if physics_complete else 0.0)
        + 0.15 * mat_confidence
        + 0.15 * face_score
    )

    return {
        "simready:qualityScore": round(quality, 4),
        "simready:watertight": watertight,
        "simready:physicsComplete": physics_complete,
        "simready:materialConfidence": round(mat_confidence, 4),
    }


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


def validate_geometry(
    bodies: list[CADBody],
    settings: PipelineSettings,
) -> ValidationResult:
    """Check geometry health against SimReady requirements.

    Validates:
    - Minimum face count (degenerate / empty meshes)
    - Bounding box extents are physically plausible in meters (catches wrong units)
    - Mesh watertightness (required for reliable physics simulation)
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Scale sanity bounds in meters (after unit conversion)
    _MIN_EXTENT_M = 0.0001   # 0.1 mm — anything smaller is likely a dust particle / error
    _MAX_EXTENT_M = 100.0    # 100 m — anything larger is likely a wrong unit
    _MIN_FACES = 4

    try:
        import trimesh
        _trimesh_available = True
    except ImportError:
        _trimesh_available = False
        warnings.append("trimesh not available — watertightness checks skipped")

    for body in bodies:
        prefix = f"Body '{body.name}'"

        if len(body.faces) < _MIN_FACES:
            errors.append(f"{prefix}: only {len(body.faces)} faces — mesh is degenerate")
            continue

        # Bounding box scale sanity
        extents = body.vertices.max(axis=0) - body.vertices.min(axis=0)
        max_extent = float(extents.max()) * settings.meters_per_unit
        if max_extent < _MIN_EXTENT_M:
            errors.append(
                f"{prefix}: largest extent {max_extent*1000:.3f} mm is below "
                f"{_MIN_EXTENT_M*1000:.1f} mm — likely wrong unit or empty geometry"
            )
        elif max_extent > _MAX_EXTENT_M:
            warnings.append(
                f"{prefix}: largest extent {max_extent:.1f} m exceeds "
                f"{_MAX_EXTENT_M:.0f} m — check unit conversion"
            )

        # Watertightness — non-watertight meshes produce undefined physics behaviour
        if _trimesh_available:
            mesh = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
            if not mesh.is_watertight:
                warnings.append(
                    f"{prefix}: mesh is not watertight — "
                    "physics simulation may behave incorrectly"
                )

    passed = len(errors) == 0
    if not passed:
        logger.error("Geometry validation FAILED with %d error(s)", len(errors))
    elif warnings:
        logger.warning("Geometry validation passed with %d warning(s)", len(warnings))
    else:
        logger.info("Geometry validation passed")

    return ValidationResult(passed=passed, errors=errors, warnings=warnings)


def validate_materials(
    materials: list[MDLMaterial],
    settings: PipelineSettings,
) -> ValidationResult:
    """Check material fidelity against SimReady requirements.

    Validates:
    - All materials have a source material name (traceability)
    - Confidence threshold is met (material properties came from CAE data)
    - PBR values are within physically plausible ranges
    """
    errors: list[str] = []
    warnings: list[str] = []

    for mat in materials:
        prefix = f"Material '{mat.source_material or 'unknown'}'"

        # Traceability
        if not mat.source_material:
            errors.append(f"{prefix}: missing source material name")

        # Confidence
        if mat.confidence < 0.25:
            errors.append(
                f"{prefix}: confidence {mat.confidence:.0%} is below minimum 25% — "
                "most PBR properties are fabricated"
            )
        elif mat.confidence < 0.5:
            warnings.append(
                f"{prefix}: confidence {mat.confidence:.0%} is low — "
                "consider enriching CAE material data"
            )

        # PBR range checks
        if not (0.0 <= mat.roughness <= 1.0):
            errors.append(f"{prefix}: roughness {mat.roughness} out of range [0, 1]")
        if not (0.0 <= mat.metallic <= 1.0):
            errors.append(f"{prefix}: metallic {mat.metallic} out of range [0, 1]")
        if mat.ior < 1.0:
            errors.append(f"{prefix}: IOR {mat.ior} below 1.0 (physically impossible)")
        if not all(0.0 <= c <= 1.0 for c in mat.diffuse_color):
            errors.append(f"{prefix}: diffuse_color values out of range [0, 1]")

    passed = len(errors) == 0
    if not passed and settings.validation.strict:
        logger.error("Material validation FAILED with %d errors", len(errors))
    elif warnings:
        logger.warning("Material validation passed with %d warnings", len(warnings))
    else:
        logger.info("Material validation passed")

    return ValidationResult(passed=passed, errors=errors, warnings=warnings)
