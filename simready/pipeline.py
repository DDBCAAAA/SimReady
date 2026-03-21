"""Top-level pipeline orchestrator — runs the full CAD/CAE → OpenUSD conversion."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from simready.config.settings import PipelineSettings, load_settings
from simready.ingestion.step_reader import read_step
from simready.ingestion.stl_reader import read_mesh, _SUPPORTED_SUFFIXES as _MESH_SUFFIXES
from simready.geometry.mesh_processing import cad_body_to_trimesh, clean_mesh, compute_normals, generate_lod, center_at_com, scale_to_meters
from simready.materials.material_map import CAEMaterial, map_cae_to_mdl, classify_material
from simready.usd.assembly import create_stage
from simready.validation.simready_checks import validate_geometry, validate_materials, compute_quality_score

logger = logging.getLogger(__name__)


def run(
    input_path: Path,
    output_path: Path,
    config_path: Path | None = None,
    material_overrides: dict[str, str] | None = None,
) -> dict:
    """Execute the full conversion pipeline.

    Args:
        input_path: Path to the source CAD file (.step/.stp).
        output_path: Where to write the output USD file.
        config_path: Optional path to a YAML config file.
        material_overrides: Optional dict mapping body name (or "*" for all)
            to a forced material class string, bypassing auto-classification.

    Returns:
        Summary dict with keys: face_count, quality_score, watertight,
        physics_complete, material_confidence, material_class.

    Raises:
        RuntimeError: If validation fails in strict mode.
    """
    settings = load_settings(config_path)

    # --- 1. Ingestion ---
    logger.info("Reading CAD file: %s", input_path)
    if input_path.suffix.lower() in _MESH_SUFFIXES:
        assembly = read_mesh(input_path)
    else:
        assembly = read_step(input_path, settings.geometry.tessellation_tolerance)
    logger.info("Loaded %d bodies from %s", len(assembly.bodies), input_path.name)

    if not assembly.bodies:
        raise RuntimeError(f"No geometry found in {input_path}")

    # --- 2. Geometry processing ---
    for body in assembly.bodies:
        mesh = cad_body_to_trimesh(body)
        mesh = clean_mesh(mesh)

        # Normalize pivot to center of mass so all LOD variants share a consistent origin
        mesh, com = center_at_com(mesh)
        mesh = scale_to_meters(mesh)  # mm → m; metersPerUnit stays 1.0
        body.metadata["center_of_mass"] = (com * 0.001).tolist()

        body.vertices = mesh.vertices
        body.faces = mesh.faces
        body.normals = compute_normals(mesh)

        # Generate LOD levels if enabled
        if settings.geometry.generate_lods:
            body.lod_meshes = []
            for ratio in settings.geometry.lod_levels:
                lod = generate_lod(mesh, ratio)
                body.lod_meshes.append((ratio, lod.vertices, lod.faces))

    # --- 3. Material mapping ---
    # Build as dict keyed by body name so assembly.py can bind by name.
    # material_name on each body is the lookup key — falls back to body.name.
    mdl_materials: dict[str, object] = {}
    for body in assembly.bodies:
        key = body.material_name or body.name
        if key in mdl_materials:
            continue  # shared material already mapped
        cae_mat = CAEMaterial(
            name=key,
            **body.metadata.get("material_properties", {}),
        )
        # Resolve material override: body-specific key first, then wildcard "*"
        forced: str | None = None
        if material_overrides:
            forced = material_overrides.get(key) or material_overrides.get("*")

        # Compute semantic label and bbox for VLM context (only when VLM enabled)
        _sem: str | None = None
        _bbox: tuple[float, float, float] | None = None
        if settings.materials.enable_vlm:
            try:
                from simready.semantics.classifier import classify as _classify_semantic  # noqa: PLC0415
                _sem = _classify_semantic(key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic classifier unavailable for '%s': %s", key, exc)
            if body.vertices is not None and len(body.vertices) > 0:
                extents = body.vertices.max(axis=0) - body.vertices.min(axis=0)
                _bbox = (float(extents[0]), float(extents[1]), float(extents[2]))

        mdl_mat = map_cae_to_mdl(
            cae_mat,
            fallback_mdl=settings.materials.fallback_mdl,
            forced_class=forced,
            enable_vlm=settings.materials.enable_vlm,
            vlm_model=settings.materials.vlm_model,
            semantic_label=_sem,
            bbox_m=_bbox,
        )
        mdl_materials[key] = mdl_mat

    # --- 3b. Quality gate: material confidence ---
    # Use the minimum confidence across all materials in the assembly — the
    # weakest link determines whether the whole asset meets the production bar.
    # Raises LowConfidenceError (caught by batch._convert_one as "skipped") when
    # any material falls below CONFIDENCE_THRESHOLD; also quarantines the source file.
    # Disabled when settings.validation.enable_confidence_gate is False (dev/test use).
    if settings.validation.enable_confidence_gate:
        from simready.quality_gate import check_material_confidence
        min_confidence = min(
            (m.confidence for m in mdl_materials.values()),
            default=0.0,
        )
        check_material_confidence(
            confidence=min_confidence,
            asset_name=input_path.stem,
            step_path=input_path,
        )

    # --- 4. Validation ---
    mat_list = list(mdl_materials.values())
    if settings.validation.enabled:
        geo_result = validate_geometry(assembly.bodies, settings)
        if not geo_result.passed and settings.validation.strict:
            for err in geo_result.errors:
                logger.error("  %s", err)
            raise RuntimeError(
                f"Geometry validation failed with {len(geo_result.errors)} error(s). "
                "Set validation.strict=false to continue with warnings."
            )

        result = validate_materials(mat_list, settings)
        if not result.passed and settings.validation.strict:
            for err in result.errors:
                logger.error("  %s", err)
            raise RuntimeError(
                f"Material validation failed with {len(result.errors)} error(s). "
                "Set validation.strict=false to continue with warnings."
            )

    # --- 5. USD assembly ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_stage(assembly.bodies, mdl_materials, output_path, settings)

    logger.info("Pipeline complete: %s → %s", input_path, output_path)

    # --- 6. Build summary dict for callers (batch, tag, etc.) ---
    total_faces = sum(len(b.faces) for b in assembly.bodies)
    first_body = assembly.bodies[0]
    first_key = first_body.material_name or first_body.name
    first_mat = mdl_materials.get(first_key)
    quality = compute_quality_score(first_body, first_mat)
    if material_overrides:
        mat_class = material_overrides.get(first_key) or material_overrides.get("*")
    else:
        mat_class = classify_material(CAEMaterial(name=first_key))

    return {
        "face_count": total_faces,
        "quality_score": quality["simready:qualityScore"],
        "watertight": quality["simready:watertight"],
        "physics_complete": quality["simready:physicsComplete"],
        "material_confidence": quality["simready:materialConfidence"],
        "material_class": mat_class,
    }
