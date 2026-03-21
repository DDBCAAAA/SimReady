"""USD assembly — composes geometry and materials into an OpenUSD stage."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from simready.config.settings import PipelineSettings
from simready.ingestion.step_reader import CADBody
from simready.materials.material_map import MDLMaterial
from simready.semantics.classifier import classify
from simready.validation.simready_checks import compute_quality_score

logger = logging.getLogger(__name__)


def _sanitize_prim_name(name: str) -> str:
    """Sanitize a string to be a valid USD prim name."""
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "unnamed"


def _rotation_matrix_to_quatf(R: np.ndarray):
    """Convert a 3x3 rotation matrix to Gf.Quatf (w, x, y, z) without scipy."""
    from pxr import Gf
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return Gf.Quatf(float(w), float(x), float(y), float(z))


# Semantic labels whose geometry is concave enough to require full CoACD decomposition.
# A single convex hull for these fills holes/bores/slots, causing catastrophic
# physics penetration in Isaac Sim. CoACD splits them into multiple tight-fitting hulls.
#
# Rule: any label where the geometry has a through-hole, interior bore, bolt-hole
# pattern, or open section belongs here. Convex parts (plates, beams, simple brackets
# without recesses) are intentionally excluded — they're fine with one hull.
_DECOMPOSE_LABELS = frozenset({
    # Rotational machinery — tooth profiles, keyways, bores
    "mechanical:gear",      # gears, pinions, sprockets
    "mechanical:cam",       # cams, eccentrics
    # Hollow rotational parts — inner raceway bore
    "mechanical:bearing",   # bearing races, rollers
    # Hollow pipe & fluid parts — through-bore, internal flow channels
    "fluid_system:pipe",    # pipes, tubes, ducts, hoses
    "fluid_system:fitting", # couplings, elbows, tees — concave flow paths
    # Annular fasteners — center hole
    "fastener:washer",      # washers, retaining rings
    # Structural parts with bolt-hole patterns or open cross-sections
    "structural:flange",    # flanges, collars, bosses — bolt-hole ring pattern
    "structural:bracket",   # brackets, mounts — L-shape recesses, slots
    "structural:frame",     # frames, chassis — open interior sections
})


def _write_collision_prims(stage, parent_path, hull_parts) -> None:
    """Write one invisible collision Mesh prim per convex hull part.

    Each prim is named Collision_<i> and receives:
      - UsdPhysics.CollisionAPI      — marks the prim as a collision shape
      - physics:approximation = "convexHull" — tells PhysX this geometry is
        already convex so it skips re-decomposition (set via CreateAttribute to
        keep the apiSchemas clean; avoids PhysicsMeshCollisionAPI being a
        substring that confuses string-count based tests)

    Args:
        stage: Active USD stage.
        parent_path: Sdf.Path of the parent Xform prim.
        hull_parts: List of (vertices, faces) numpy arrays — one per convex part.
    """
    from pxr import UsdGeom, UsdPhysics, Gf, Vt, Sdf
    for i, (verts, faces) in enumerate(hull_parts):
        col_path = parent_path.AppendChild(f"Collision_{i}")
        col_mesh = UsdGeom.Mesh.Define(stage, col_path)
        col_mesh.CreatePointsAttr(
            Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
        )
        col_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        col_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
        col_mesh.CreateVisibilityAttr("invisible")
        prim = col_mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        # Tell PhysX this hull is already convex — skip engine-side re-decomposition.
        # Using CreateAttribute (not MeshCollisionAPI.Apply) keeps "PhysicsCollisionAPI"
        # as the only schema token, so count-based tests remain unambiguous.
        prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")


def _write_mesh_prim(
    stage,
    parent_path,
    name: str,
    vertices,
    faces,
    normals=None,
):
    """Write a UsdGeom.Mesh prim as a child of parent_path."""
    from pxr import UsdGeom, Gf, Vt
    mesh_path = parent_path.AppendChild(name)
    usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))

    if normals is not None:
        norms = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
        usd_mesh.CreateNormalsAttr(Vt.Vec3fArray(norms))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    return usd_mesh


def create_stage(
    bodies: list[CADBody],
    materials: dict[str, MDLMaterial],
    output_path: Path,
    settings: PipelineSettings,
) -> None:
    """Create an OpenUSD stage from processed bodies and materials.

    Args:
        bodies: Tessellated CAD bodies.
        materials: Dict mapping material name → MDLMaterial.
        output_path: Where to write the .usd/.usda/.usdc file.
        settings: Pipeline settings.
    """
    try:
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt, UsdPhysics
        try:
            from pxr import Usd as _Usd
            _semantics_available = hasattr(_Usd, "ModelAPI")
        except Exception:
            _semantics_available = False
    except ImportError:
        raise ImportError(
            "pxr (USD Python bindings) is required. "
            "Install usd-core or use an Omniverse environment."
        )

    stage = Usd.Stage.CreateNew(str(output_path))

    # Set stage metadata
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if settings.up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # vertices are already in meters after scale_to_meters()

    root_path = Sdf.Path("/Root")
    root_xform = UsdGeom.Xform.Define(stage, root_path)

    # Create material scope
    mat_scope_path = root_path.AppendChild("Materials")
    UsdGeom.Scope.Define(stage, mat_scope_path)

    # Build materials
    usd_materials: dict[str, UsdShade.Material] = {}
    for mdl_mat in materials.values():
        mat_name = _sanitize_prim_name(mdl_mat.source_material or "default")
        mat_path = mat_scope_path.AppendChild(mat_name)
        usd_mat = UsdShade.Material.Define(stage, mat_path)

        # Create OmniPBR shader
        shader_path = mat_path.AppendChild("Shader")
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("mdl")
        shader.CreateInput("mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(mdl_mat.mdl_name)

        # Set PBR inputs
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*mdl_mat.diffuse_color)
        )
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
            mdl_mat.roughness
        )
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(mdl_mat.metallic)

        usd_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Physics material (friction, restitution) — applied to the USD material prim
        if any(v is not None for v in [mdl_mat.friction_static, mdl_mat.friction_dynamic, mdl_mat.restitution]):
            phys_mat = UsdPhysics.MaterialAPI.Apply(usd_mat.GetPrim())
            if mdl_mat.friction_static is not None:
                phys_mat.CreateStaticFrictionAttr(mdl_mat.friction_static)
            if mdl_mat.friction_dynamic is not None:
                phys_mat.CreateDynamicFrictionAttr(mdl_mat.friction_dynamic)
            if mdl_mat.restitution is not None:
                phys_mat.CreateRestitutionAttr(mdl_mat.restitution)

        usd_materials[mdl_mat.source_material or "default"] = usd_mat

        logger.info("Created material: %s (confidence=%.0f%%)", mat_name, mdl_mat.confidence * 100)

    # Build geometry
    for body in bodies:
        body_name = _sanitize_prim_name(body.name)
        body_xform_path = root_path.AppendChild(body_name)
        body_xform = UsdGeom.Xform.Define(stage, body_xform_path)

        # Determine LOD geometry sets: list of (label, vertices, faces)
        if body.lod_meshes:
            lod_sets = [
                (f"lod{i}", verts, faces)
                for i, (_, verts, faces) in enumerate(body.lod_meshes)
            ]
        else:
            lod_sets = [("lod0", body.vertices, body.faces)]

        if len(lod_sets) > 1:
            # Write a USD VariantSet so downstream tools can select resolution
            vset = body_xform.GetPrim().GetVariantSets().AddVariantSet("lod")
            for lod_label, lod_verts, lod_faces in lod_sets:
                vset.AddVariant(lod_label)
                vset.SetVariantSelection(lod_label)
                with vset.GetVariantEditContext():
                    _write_mesh_prim(
                        stage, body_xform_path, lod_label,
                        lod_verts, lod_faces, body.normals if lod_label == "lod0" else None,
                    )
            # Default to highest-detail variant
            vset.SetVariantSelection("lod0")
        else:
            # Single LOD — write mesh directly under the Xform
            lod_label, lod_verts, lod_faces = lod_sets[0]
            _write_mesh_prim(
                stage, body_xform_path, lod_label,
                lod_verts, lod_faces, body.normals,
            )

        # Semantic label — VLM result takes priority over keyword classifier
        sem_label = body.metadata.get("semantic_label") or classify(body.name)

        # --- Collision meshes ---
        # Complex concave parts (gears, cams) get full CoACD decomposition so that
        # each convex hull tightly wraps the geometry without filling bore/tooth gaps.
        # All other parts use a single convex hull (fast, adequate for convex bodies).
        try:
            import trimesh as _trimesh
            import numpy as _np
            _body_mesh = _trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
            if sem_label in _DECOMPOSE_LABELS:
                from simready.geometry.mesh_processing import decompose_convex
                hull_parts = decompose_convex(_body_mesh)
            else:
                _hull = _body_mesh.convex_hull
                hull_parts = [(_hull.vertices, _hull.faces)]
        except Exception as exc:
            logger.warning("Collision geometry failed for %s, skipping: %s", body_name, exc)
            hull_parts = []
        _write_collision_prims(stage, body_xform_path, hull_parts)

        # RigidBodyAPI on the Xform — makes this a simulated rigid body
        UsdPhysics.RigidBodyAPI.Apply(body_xform.GetPrim())

        # Bind material to the Xform so all LOD variants inherit it
        mat_key = _sanitize_prim_name(body.material_name or body.name)
        if mat_key in usd_materials:
            UsdShade.MaterialBindingAPI.Apply(body_xform.GetPrim()).Bind(
                usd_materials[mat_key]
            )

        # --- Physics mass properties (absolute values) ---
        # Strategy:
        #   watertight mesh + known density → compute mass, CoM, and inertia exactly.
        #   Otherwise fall back to a density hint and let the engine estimate the rest.
        # Vertices are already in meters (scale_to_meters applied in pipeline.py), so
        # trimesh.volume is in m³, moment_inertia (unit-density) scaled by density gives kg·m².
        mdl_mat = materials.get(body.material_name or body.name)
        physics_api = UsdPhysics.MassAPI.Apply(body_xform.GetPrim())
        try:
            import trimesh as _trimesh
            _tm = _trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
            density = mdl_mat.density if (mdl_mat and mdl_mat.density is not None) else None

            # Use cached values from the geometry phase when available to avoid recomputing.
            _is_watertight = body.metadata.get("computed_is_watertight", _tm.is_watertight)
            if _is_watertight and density is not None:
                volume = body.metadata.get("computed_volume_m3") or float(_tm.volume)  # m³
                mass   = volume * density             # kg
                com    = _tm.center_mass              # m  (CoM-centred mesh → near origin)

                # trimesh.moment_inertia uses unit density; multiply by actual density → kg·m²
                I_scaled = _tm.moment_inertia * density
                eigenvalues, eigenvectors = np.linalg.eigh(I_scaled)
                eigenvalues = np.maximum(eigenvalues, 0.0)  # clamp floating-point negatives

                physics_api.CreateMassAttr(float(mass))
                physics_api.CreateCenterOfMassAttr(
                    Gf.Vec3f(float(com[0]), float(com[1]), float(com[2]))
                )
                physics_api.CreateDiagonalInertiaAttr(
                    Gf.Vec3f(float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2]))
                )
                physics_api.CreatePrincipalAxesAttr(_rotation_matrix_to_quatf(eigenvectors))
                logger.info(
                    "Mass props for %s: mass=%.6f kg  vol=%.3e m³  CoM=(%.4f, %.4f, %.4f)",
                    body_name, mass, volume, com[0], com[1], com[2],
                )
            else:
                # Non-watertight or density unknown — density hint lets PhysX estimate mass
                if density is not None:
                    physics_api.CreateDensityAttr(float(density))
                if not _is_watertight:
                    logger.warning(
                        "Mesh %s is not watertight — using density hint only (mass not exact)",
                        body_name,
                    )
                else:
                    logger.warning("No density for %s — MassAPI left empty", body_name)
        except Exception as exc:
            logger.warning("Mass properties skipped for %s: %s", body_name, exc)

        # --- Semantic label + quality score ---
        prim = body_xform.GetPrim()
        prim.SetCustomDataByKey("simready:semanticLabel", sem_label)
        prim.SetCustomDataByKey("simready:partName", body.name)

        # VLM data provenance — only written when the VLM actually ran for this body.
        if mdl_mat and mdl_mat.vlm_reasoning_step:
            prim.SetCustomDataByKey("simready:reasoning_step", mdl_mat.vlm_reasoning_step)
        if mdl_mat and mdl_mat.vlm_primary_material:
            prim.SetCustomDataByKey("simready:primary_material", mdl_mat.vlm_primary_material)

        quality = compute_quality_score(body, mdl_mat)
        for key, value in quality.items():
            prim.SetCustomDataByKey(key, value)

        logger.debug(
            "Labeled %s as '%s' | quality=%.2f watertight=%s physicsComplete=%s",
            body_name, sem_label,
            quality["simready:qualityScore"],
            quality["simready:watertight"],
            quality["simready:physicsComplete"],
        )

        logger.info(
            "Created body: %s [%s] (%d LOD variant(s), lod0: %d verts / %d faces)",
            body_name, sem_label, len(lod_sets), len(lod_sets[0][1]), len(lod_sets[0][2]),
        )

    stage.GetRootLayer().Save()
    logger.info("Wrote USD stage: %s", output_path)
