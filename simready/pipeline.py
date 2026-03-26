"""Top-level pipeline orchestrator — runs the full CAD/CAE → OpenUSD conversion."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from simready.config.settings import PipelineSettings, load_settings
from simready.ingestion.step_reader import read_step
from simready.ingestion.stl_reader import read_mesh, _SUPPORTED_SUFFIXES as _MESH_SUFFIXES
from simready.geometry.mesh_processing import cad_body_to_trimesh, clean_mesh, compute_normals, generate_lod, center_at_com, scale_to_meters, get_meters_conversion_factor
from simready.materials.material_map import CAEMaterial, map_cae_to_mdl, classify_material
from simready.usd.assembly import create_stage
from simready.validation.simready_checks import validate_geometry, validate_materials, compute_quality_score

logger = logging.getLogger(__name__)


def run(
    input_path: Path,
    output_path: Path,
    config_path: Path | None = None,
    material_overrides: dict[str, str] | None = None,
    asset_metadata: dict | None = None,
    disable_confidence_gate: bool = False,
) -> dict:
    """Execute the full conversion pipeline.

    Args:
        input_path: Path to the source CAD file (.step/.stp).
        output_path: Where to write the output USD file.
        config_path: Optional path to a YAML config file.
        material_overrides: Optional dict mapping body name (or "*" for all)
            to a forced material class string, bypassing auto-classification.
        asset_metadata: Optional flat dict of extra provenance/spec fields to
            embed in the USD /Root prim customData (e.g. TraceParts specs).
        disable_confidence_gate: When True, skip the material confidence gate
            (useful for curated datasets like TraceParts where material is
            supplied externally and source files must not be quarantined).

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

    # Derive the meter conversion factor from the source file's detected unit.
    # scale_factor multiplies native-unit coordinates to produce meters.
    scale_factor = get_meters_conversion_factor(assembly.units)
    logger.info(
        "Source unit: '%s' → scale factor %.6f (file: %s)",
        assembly.units, scale_factor, input_path.name,
    )

    # --- 2. Geometry processing ---
    for body in assembly.bodies:
        mesh = cad_body_to_trimesh(body)
        mesh = clean_mesh(mesh)

        # Normalize pivot to center of mass so all LOD variants share a consistent origin.
        # Store the world-space CoM (in meters) so assembly.py can restore the correct
        # world position via xformOp:translate on each body Xform.
        mesh, com = center_at_com(mesh)
        mesh = scale_to_meters(mesh, scale=scale_factor)  # native units → m
        body.metadata["center_of_mass"] = (com * scale_factor).tolist()
        body.metadata["world_translate_m"] = (com * scale_factor).tolist()

        body.vertices = mesh.vertices
        body.faces = mesh.faces
        body.normals = compute_normals(mesh)

        # Cache geometry properties that downstream stages (VLM, mass-props) need.
        # The mesh object is already available here — no second trimesh build required.
        body.metadata["computed_is_watertight"] = mesh.is_watertight
        if mesh.is_watertight:
            body.metadata["computed_volume_m3"] = float(mesh.volume)

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
        # Use filename stem when body name is generic (e.g. FreeCAD's "body_0")
        # so the VLM gets a meaningful part name rather than a placeholder.
        _is_generic = bool(re.fullmatch(r"body_?\d+|solid_?\d+|shape_?\d+", key, re.IGNORECASE))
        vlm_part_name = input_path.stem if _is_generic else key
        cae_mat = CAEMaterial(
            name=vlm_part_name,
            **body.metadata.get("material_properties", {}),
        )
        # Resolve material override: body-specific key first, then wildcard "*"
        forced: str | None = None
        if material_overrides:
            forced = material_overrides.get(key) or material_overrides.get("*")

        # Compute semantic label, bbox, and volume for VLM context (only when VLM enabled)
        _sem: str | None = None
        _bbox: tuple[float, float, float] | None = None
        _volume_m3: float | None = None
        if settings.materials.enable_vlm:
            try:
                from simready.semantics.classifier import classify as _classify_semantic  # noqa: PLC0415
                _sem = _classify_semantic(vlm_part_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic classifier unavailable for '%s': %s", key, exc)
            if body.vertices is not None and len(body.vertices) > 0:
                extents = body.vertices.max(axis=0) - body.vertices.min(axis=0)
                _bbox = (float(extents[0]), float(extents[1]), float(extents[2]))
                # Volume already computed and cached in the geometry phase — no second trimesh build.
                _volume_m3 = body.metadata.get("computed_volume_m3")

        mdl_mat = map_cae_to_mdl(
            cae_mat,
            fallback_mdl=settings.materials.fallback_mdl,
            forced_class=forced,
            enable_vlm=settings.materials.enable_vlm,
            vlm_model=settings.materials.vlm_model,
            semantic_label=_sem,
            bbox_m=_bbox,
            volume_m3=_volume_m3,
            vlm_max_calls=settings.materials.vlm_max_calls,
        )
        mdl_materials[key] = mdl_mat
        # Propagate VLM-improved semantic label to all bodies sharing this material key
        if mdl_mat.vlm_semantic_label:
            for b in assembly.bodies:
                if (b.material_name or b.name) == key:
                    b.metadata["semantic_label"] = mdl_mat.vlm_semantic_label

    # --- 3b. Quality gate: material confidence ---
    if disable_confidence_gate:
        settings.validation.enable_confidence_gate = False
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

    # --- 4b. Articulation inference (VLM, optional) ---
    # Only runs when enable_vlm=True and ANTHROPIC_API_KEY is set.
    # Builds a kinematic topology (rigid link groups + joints) used by USD assembly.
    topology = None
    if settings.materials.enable_vlm and len(assembly.bodies) > 1:
        try:
            from simready.articulation_inference import infer_kinematic_topology  # noqa: PLC0415
            parts_metadata = {}
            for body in assembly.bodies:
                meta = {}
                if body.vertices is not None and len(body.vertices) > 0:
                    extents = body.vertices.max(axis=0) - body.vertices.min(axis=0)
                    meta["bbox_m"] = (float(extents[0]), float(extents[1]), float(extents[2]))
                vol = body.metadata.get("computed_volume_m3")
                if vol is not None:
                    meta["volume_m3"] = float(vol)
                parts_metadata[body.name] = meta
            topology = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=parts_metadata,
                object_label=input_path.stem,
                model=settings.materials.vlm_model,
            )
            if topology is not None:
                logger.info(
                    "Articulation: base_link='%s', %d links, %d joints",
                    topology.base_link_name, len(topology.rigid_links), len(topology.joints),
                )
        except Exception as exc:
            logger.warning("Articulation inference failed (%s) — exporting as static assembly", exc)

    # --- 5. USD assembly ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_stage(assembly.bodies, mdl_materials, output_path, settings, topology=topology,
                 asset_metadata=asset_metadata)

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
