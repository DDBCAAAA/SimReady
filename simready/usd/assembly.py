"""USD assembly — composes geometry and materials into an OpenUSD stage."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

from simready.articulation_inference import ArticulationTopology, JointType, RigidLinkGroup
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


def _compute_link_mass_props(
    constituent_bodies: list[CADBody],
    materials: dict[str, MDLMaterial],
    _trimesh,
) -> dict | None:
    """Compute combined mass, CoM, and principal inertia for a rigid link group.

    Uses the parallel-axis theorem to aggregate inertia tensors from each part.
    Returns None if any part lacks density or is non-watertight.

    Returns dict with keys:
        mass (float)                 — total mass in kg
        com (Gf.Vec3f)               — combined centre of mass
        diagonal_inertia (Gf.Vec3f) — principal inertia components
        principal_axes (Gf.Quatf)   — rotation from world to principal axes
    """
    from pxr import Gf

    if not constituent_bodies or _trimesh is None:
        return None

    parts: list[tuple[float, np.ndarray, np.ndarray]] = []  # (mass, com, I_3x3)
    for body in constituent_bodies:
        mdl_mat = materials.get(body.material_name or body.name)
        density = mdl_mat.density if (mdl_mat and mdl_mat.density is not None) else None
        if density is None:
            return None
        try:
            # Check watertightness from cache before building trimesh —
            # non-watertight bodies get discarded immediately so skip the build.
            cached_watertight = body.metadata.get("computed_is_watertight")
            if cached_watertight is False:
                return None
            tm = _trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
            is_watertight = cached_watertight if cached_watertight is not None else tm.is_watertight
            if not is_watertight:
                return None
            volume = body.metadata.get("computed_volume_m3") or float(tm.volume)
            mass_i = volume * density
            com_i = tm.center_mass.copy()
            I_i = tm.moment_inertia * density  # mass-based inertia about own CoM
            parts.append((mass_i, com_i, I_i))
        except Exception as exc:
            logger.warning("Mass props computation failed for '%s': %s", body.name, exc)
            return None

    total_mass = sum(m for m, _, _ in parts)
    if not parts or total_mass <= 0:
        return None

    # Mass-weighted centre of mass
    combined_com = sum(m * c for m, c, _ in parts) / total_mass

    # Combined inertia via parallel-axis theorem: I_combined = Σ (I_i + m_i*(|d|²I₃ - d⊗d))
    I_combined = np.zeros((3, 3))
    for mass_i, com_i, I_i in parts:
        d = com_i - combined_com
        I_combined += I_i + mass_i * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    eigenvalues, eigenvectors = np.linalg.eigh(I_combined)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return {
        "mass": float(total_mass),
        "com": Gf.Vec3f(float(combined_com[0]), float(combined_com[1]), float(combined_com[2])),
        "diagonal_inertia": Gf.Vec3f(
            float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])
        ),
        "principal_axes": _rotation_matrix_to_quatf(eigenvectors),
    }


def _aggregate_link_provenance(
    constituent_bodies: list[CADBody],
    materials: dict[str, MDLMaterial],
) -> dict:
    """Return provenance fields from the highest-confidence material in the link group."""
    best_mat: MDLMaterial | None = None
    for body in constituent_bodies:
        mat = materials.get(body.material_name or body.name)
        if mat is not None and (best_mat is None or mat.confidence > best_mat.confidence):
            best_mat = mat

    result: dict = {}
    if best_mat is None:
        return result
    result["simready:materialConfidence"] = float(best_mat.confidence)
    if best_mat.vlm_primary_material:
        result["simready:primary_material"] = best_mat.vlm_primary_material
    if best_mat.vlm_reasoning_step:
        result["simready:reasoning_step"] = best_mat.vlm_reasoning_step
    return result


def _gather_vertices(bodies: list[CADBody]) -> np.ndarray | None:
    """Stack vertices from all non-empty constituent bodies into a single array."""
    verts_list = [b.vertices for b in bodies if len(b.vertices) > 0]
    return np.vstack(verts_list) if verts_list else None


def _compute_link_bbox_center(constituent_bodies: list[CADBody]) -> np.ndarray:
    """Return the world-space bounding box center of all vertices in a rigid link group.

    Used by the auto-joint anchor heuristic to compute physics:localPos0/localPos1.
    """
    combined = _gather_vertices(constituent_bodies)
    if combined is None:
        return np.zeros(3, dtype=np.float64)
    return (combined.min(axis=0) + combined.max(axis=0)) * 0.5


def _get_xform_translate(stage, prim_path) -> np.ndarray:
    """Return the xformOp:translate value of a USD prim as a numpy (3,) array.

    Falls back to (0, 0, 0) if the prim has no translate op or the op has no value.
    This is needed to transform world-space pivot points into each body's local frame.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        attr = prim.GetAttribute("xformOp:translate")
        if attr.IsValid():
            val = attr.Get()
            if val is not None:
                return np.array([val[0], val[1], val[2]], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _infer_rotation_axis_from_pca(constituent_bodies: list[CADBody]) -> str | None:
    """Infer the dominant rotation axis of a link from PCA of its vertex cloud.

    For rotationally symmetric parts (fans, shafts, cylinders) the minimum-variance
    principal component is the symmetry axis.  Returns the closest world axis
    ('X', 'Y', 'Z') when that component is unambiguous (|dot| > 0.8), else None.

    Uses the 3×3 covariance matrix so cost is O(N) regardless of vertex count.
    """
    combined = _gather_vertices(constituent_bodies)
    if combined is None or len(combined) < 4:
        return None
    centered = combined - combined.mean(axis=0)
    cov = centered.T @ centered          # 3×3, fast for any N
    _, eigenvectors = np.linalg.eigh(cov)   # ascending order
    # Column 0 = minimum-variance eigenvector = rotation-symmetry axis
    min_var_axis = np.abs(eigenvectors[:, 0])
    dominant_idx = int(np.argmax(min_var_axis))
    if min_var_axis[dominant_idx] > 0.8:
        return ["X", "Y", "Z"][dominant_idx]
    return None


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
        col_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.ravel().tolist()))
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
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.ravel().tolist()))

    if normals is not None:
        norms = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
        usd_mesh.CreateNormalsAttr(Vt.Vec3fArray(norms))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    return usd_mesh


def _write_body_under(
    stage,
    parent_path,
    body: CADBody,
    materials: dict[str, MDLMaterial],
    usd_materials: dict,
    _trimesh,
    _decompose_convex,
    apply_rigid_body: bool = True,
    apply_mass_api: bool = True,
):
    """Create a child Xform for body under parent_path and write its geometry/physics.

    Args:
        stage: Active USD stage.
        parent_path: Sdf.Path of the parent Xform (link Xform or root).
        body: The CADBody to write.
        materials: Mapping body.material_name → MDLMaterial.
        usd_materials: Pre-built USD material prims keyed by sanitized source_material.
        _trimesh: The trimesh module (or None if unavailable).
        _decompose_convex: The decompose_convex function (or None if unavailable).
        apply_rigid_body: If True, apply PhysicsRigidBodyAPI to the created Xform.
            Set to False when the parent link Xform already carries RigidBodyAPI.
        apply_mass_api: If True, apply PhysicsMassAPI and write all simready provenance
            customData to this Xform (standalone rigid-body mode).
            Set to False in Path A — the parent link Xform owns mass/provenance and
            this Xform is a pure visual/collision container.

    Returns:
        Sdf.Path of the created body Xform.
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf

    body_name = _sanitize_prim_name(body.name)
    body_xform_path = parent_path.AppendChild(body_name)
    body_xform = UsdGeom.Xform.Define(stage, body_xform_path)

    # Restore world-space position that was normalised away by center_at_com in the
    # geometry phase.  Without this every body ends up at (0,0,0) and the collision
    # meshes all overlap, causing a NaN physics explosion on simulation start.
    world_t = body.metadata.get("world_translate_m")
    if world_t and any(abs(v) > 1e-9 for v in world_t):
        body_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(world_t[0]), float(world_t[1]), float(world_t[2]))
        )

    # LOD geometry
    if body.lod_meshes:
        lod_sets = [
            (f"lod{i}", verts, faces)
            for i, (_, verts, faces) in enumerate(body.lod_meshes)
        ]
    else:
        lod_sets = [("lod0", body.vertices, body.faces)]

    if len(lod_sets) > 1:
        vset = body_xform.GetPrim().GetVariantSets().AddVariantSet("lod")
        for lod_label, lod_verts, lod_faces in lod_sets:
            vset.AddVariant(lod_label)
            vset.SetVariantSelection(lod_label)
            with vset.GetVariantEditContext():
                _write_mesh_prim(
                    stage, body_xform_path, lod_label,
                    lod_verts, lod_faces, body.normals if lod_label == "lod0" else None,
                )
        vset.SetVariantSelection("lod0")
    else:
        lod_label, lod_verts, lod_faces = lod_sets[0]
        _write_mesh_prim(stage, body_xform_path, lod_label, lod_verts, lod_faces, body.normals)

    # Semantic label — VLM result takes priority over keyword classifier
    sem_label = body.metadata.get("semantic_label") or classify(body.name)

    # Build Trimesh once — reused for both collision decomposition and mass properties.
    _body_mesh = None
    if _trimesh is not None:
        try:
            _body_mesh = _trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
        except Exception as exc:
            logger.warning("Trimesh construction failed for %s: %s", body_name, exc)

    # Collision meshes (CoACD unconditional — concave geometry decomposes into tight hulls)
    try:
        if _body_mesh is None or _decompose_convex is None:
            raise ImportError("trimesh or mesh_processing not available")
        hull_parts = _decompose_convex(_body_mesh)
    except Exception as exc:
        logger.warning("Collision geometry failed for %s, skipping: %s", body_name, exc)
        hull_parts = []
    _write_collision_prims(stage, body_xform_path, hull_parts)

    # RigidBodyAPI — only in per-body (non-grouped) mode
    if apply_rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(body_xform.GetPrim())

    # Material binding — always (needed for visual rendering in all modes)
    mat_key = _sanitize_prim_name(body.material_name or body.name)
    if mat_key in usd_materials:
        UsdShade.MaterialBindingAPI.Apply(body_xform.GetPrim()).Bind(
            usd_materials[mat_key]
        )

    if apply_mass_api:
        # Physics mass properties (absolute values) — standalone rigid-body mode only
        mdl_mat = materials.get(body.material_name or body.name)
        physics_api = UsdPhysics.MassAPI.Apply(body_xform.GetPrim())
        try:
            if _body_mesh is None:
                raise ImportError("trimesh not available")
            density = mdl_mat.density if mdl_mat else None

            _is_watertight = body.metadata.get("computed_is_watertight", _body_mesh.is_watertight)
            if _is_watertight and density is not None:
                volume = body.metadata.get("computed_volume_m3") or float(_body_mesh.volume)
                mass = volume * density
                com = _body_mesh.center_mass
                I_scaled = _body_mesh.moment_inertia * density
                eigenvalues, eigenvectors = np.linalg.eigh(I_scaled)
                eigenvalues = np.maximum(eigenvalues, 0.0)

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

        # SimReady provenance + quality score — standalone rigid-body mode only
        prim = body_xform.GetPrim()
        prim.SetCustomDataByKey("simready:semanticLabel", sem_label)
        prim.SetCustomDataByKey("simready:partName", body.name)
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

    return body_xform_path


def create_stage(
    bodies: list[CADBody],
    materials: dict[str, MDLMaterial],
    output_path: Path,
    settings: PipelineSettings,
    topology: ArticulationTopology | None = None,
) -> None:
    """Create an OpenUSD stage from processed bodies and materials.

    When topology.rigid_links is provided (two-stage architecture):
      - Each RigidLinkGroup becomes one Xform with PhysicsRigidBodyAPI.
      - Constituent body meshes are nested under their link Xform.
      - Joints reference the link Xforms, not individual body Xforms.

    When topology is None or has no rigid_links (fallback):
      - One Xform per CADBody, each carrying PhysicsRigidBodyAPI directly.
      - Original per-body behaviour preserved for pipeline.py and simple exports.

    Args:
        bodies: Tessellated CAD bodies.
        materials: Dict mapping material name → MDLMaterial.
        output_path: Where to write the .usd/.usda/.usdc file.
        settings: Pipeline settings.
        topology: Optional VLM-inferred kinematic topology.
    """
    try:
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt, UsdPhysics
    except ImportError:
        raise ImportError(
            "pxr (USD Python bindings) is required. "
            "Install usd-core or use an Omniverse environment."
        )

    stage = Usd.Stage.CreateNew(str(output_path))

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if settings.up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = Sdf.Path("/Root")
    root_xform = UsdGeom.Xform.Define(stage, root_path)

    if topology is not None and topology.joints:
        UsdPhysics.ArticulationRootAPI.Apply(root_xform.GetPrim())

    # Create material scope
    mat_scope_path = root_path.AppendChild("Materials")
    UsdGeom.Scope.Define(stage, mat_scope_path)

    # Build USD materials
    usd_materials: dict[str, UsdShade.Material] = {}
    for mdl_mat in materials.values():
        mat_name = _sanitize_prim_name(mdl_mat.source_material or "default")
        mat_path = mat_scope_path.AppendChild(mat_name)
        usd_mat = UsdShade.Material.Define(stage, mat_path)

        shader_path = mat_path.AppendChild("Shader")
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("mdl")
        shader.CreateInput("mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(mdl_mat.mdl_name)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*mdl_mat.diffuse_color)
        )
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
            mdl_mat.roughness
        )
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(mdl_mat.metallic)
        usd_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

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

    # Hoist trimesh / decompose_convex imports once
    try:
        import trimesh as _trimesh
        from simready.geometry.mesh_processing import decompose_convex as _decompose_convex
    except ImportError:
        _trimesh = None  # type: ignore[assignment]
        _decompose_convex = None  # type: ignore[assignment]

    # Maps link_name (or body.name in fallback) → Sdf.Path for joint creation
    link_prim_paths: dict[str, Sdf.Path] = {}
    # Maps link_name → constituent CADBody list for joint anchor heuristic
    link_bodies_map: dict[str, list[CADBody]] = {}

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    if topology is not None and topology.rigid_links:
        # ---------------------------------------------------------------
        # PATH A: Two-stage / link-grouped mode
        # One Xform per RigidLinkGroup; body meshes nested under it.
        # RigidBodyAPI + MassAPI applied once per link, not per raw mesh.
        # Body sub-Xforms are pure visual/collision containers.
        # ---------------------------------------------------------------
        body_map = {body.name: body for body in bodies}

        links_iter = (
            _tqdm(topology.rigid_links, desc="Writing USD links", unit="link", dynamic_ncols=True)
            if _tqdm is not None else topology.rigid_links
        )
        for rlg in links_iter:
            link_prim_name = _sanitize_prim_name(rlg.link_name)
            link_xform_path = root_path.AppendChild(link_prim_name)
            link_prim_paths[rlg.link_name] = link_xform_path
            link_xform = UsdGeom.Xform.Define(stage, link_xform_path)
            link_prim = link_xform.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(link_prim)

            # Resolve constituent bodies once (skip unknowns with warning)
            constituent_bodies = []
            for part_name in rlg.constituent_parts:
                b = body_map.get(part_name)
                if b is None:
                    logger.warning(
                        "Link '%s' references unknown part '%s' — skipping",
                        rlg.link_name, part_name,
                    )
                else:
                    constituent_bodies.append(b)
            link_bodies_map[rlg.link_name] = constituent_bodies

            # Combined mass properties at link level (parallel-axis theorem)
            mass_props = _compute_link_mass_props(constituent_bodies, materials, _trimesh)
            if mass_props is not None:
                mass_api = UsdPhysics.MassAPI.Apply(link_prim)
                mass_api.CreateMassAttr(mass_props["mass"])
                mass_api.CreateCenterOfMassAttr(mass_props["com"])
                mass_api.CreateDiagonalInertiaAttr(mass_props["diagonal_inertia"])
                mass_api.CreatePrincipalAxesAttr(mass_props["principal_axes"])
                logger.info(
                    "Link '%s': combined mass=%.4f kg from %d part(s)",
                    rlg.link_name, mass_props["mass"], len(constituent_bodies),
                )
            else:
                logger.warning(
                    "Link '%s': mass props unavailable (non-watertight or no density)",
                    rlg.link_name,
                )

            # Provenance at link level (from highest-confidence constituent material)
            provenance = _aggregate_link_provenance(constituent_bodies, materials)
            link_prim.SetCustomDataByKey("simready:physicsComplete", mass_props is not None)
            for k, v in provenance.items():
                link_prim.SetCustomDataByKey(k, v)

            # Write constituent bodies as child visual/collision containers only
            for body in constituent_bodies:
                _write_body_under(
                    stage, link_xform_path, body, materials, usd_materials,
                    _trimesh, _decompose_convex,
                    apply_rigid_body=False,
                    apply_mass_api=False,
                )

            logger.info("Created link group: %s (parts: %s)", rlg.link_name, rlg.constituent_parts)
    else:
        # ---------------------------------------------------------------
        # PATH B: Fallback — one Xform per body, each is its own rigid body
        # Preserves original behaviour for pipeline.py and simple exports.
        # ---------------------------------------------------------------
        bodies_iter = (
            _tqdm(bodies, desc="Writing USD bodies", unit="body", dynamic_ncols=True)
            if _tqdm is not None else bodies
        )
        for body in bodies_iter:
            body_xform_path = _write_body_under(
                stage, root_path, body, materials, usd_materials,
                _trimesh, _decompose_convex, apply_rigid_body=True,
            )
            link_prim_paths[body.name] = body_xform_path
            link_bodies_map[body.name] = [body]

    # --- Articulation joints ---
    if topology is not None and topology.joints:
        joints_scope_path = root_path.AppendChild("Joints")
        UsdGeom.Scope.Define(stage, joints_scope_path)

        for i, joint_def in enumerate(topology.joints):
            jtype = joint_def.joint_type
            joint_prim_name = f"joint_{i}_{jtype.value}"
            joint_path = joints_scope_path.AppendChild(joint_prim_name)

            if jtype == JointType.revolute:
                usd_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            elif jtype == JointType.prismatic:
                usd_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            else:
                usd_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

            joint_prim = usd_joint.GetPrim()

            parent_path = link_prim_paths.get(joint_def.parent_link)
            child_path = link_prim_paths.get(joint_def.child_link)
            if parent_path:
                joint_prim.CreateRelationship("physics:body0").SetTargets([parent_path])
            if child_path:
                joint_prim.CreateRelationship("physics:body1").SetTargets([child_path])

            # Collision filtering: disable collision between the two linked bodies.
            # Zero-clearance CAD parts produce deeply overlapping collision meshes;
            # without filtering the engine generates infinite repulsive forces → NaN.
            if parent_path and child_path:
                parent_link_prim = stage.GetPrimAtPath(parent_path)
                if parent_link_prim.IsValid():
                    filtered_api = UsdPhysics.FilteredPairsAPI.Apply(parent_link_prim)
                    filtered_api.GetFilteredPairsRel().AddTarget(child_path)

            # Resolve child link bodies — needed for PCA axis inference and anchor calc
            child_bodies = link_bodies_map.get(joint_def.child_link, [])

            if jtype != JointType.fixed:
                # PCA axis override for revolute joints:
                # For rotationally-symmetric parts the min-variance principal axis
                # is more reliable than the VLM's text guess.
                motion_axis = joint_def.motion_axis
                if jtype == JointType.revolute:
                    pca_axis = _infer_rotation_axis_from_pca(child_bodies)
                    if pca_axis is not None and pca_axis != motion_axis:
                        logger.info(
                            "Joint '%s': overriding VLM axis '%s' → '%s' (PCA)",
                            joint_prim_name, motion_axis, pca_axis,
                        )
                        motion_axis = pca_axis
                joint_prim.CreateAttribute(
                    "physics:axis", Sdf.ValueTypeNames.Token
                ).Set(motion_axis)

            if jtype == JointType.revolute:
                if joint_def.lower_limit_deg is not None:
                    joint_prim.CreateAttribute(
                        "physics:lowerLimit", Sdf.ValueTypeNames.Float
                    ).Set(float(joint_def.lower_limit_deg))
                if joint_def.upper_limit_deg is not None:
                    joint_prim.CreateAttribute(
                        "physics:upperLimit", Sdf.ValueTypeNames.Float
                    ).Set(float(joint_def.upper_limit_deg))
            elif jtype == JointType.prismatic:
                # Use a generous safe zone [-0.5, 0.5] m so the joint is never
                # outside its bounds at simulation start (zero-clearance parts would
                # otherwise make auto-computed extents lock the joint immediately).
                joint_prim.CreateAttribute(
                    "physics:lowerLimit", Sdf.ValueTypeNames.Float
                ).Set(-0.5)
                joint_prim.CreateAttribute(
                    "physics:upperLimit", Sdf.ValueTypeNames.Float
                ).Set(0.5)
                logger.info("Prismatic joint '%s': safe zone limits [-0.50, 0.50] m", joint_prim_name)

            # --- Joint Anchor: World-Space Pivot → Body-Local Offsets ---
            # pivot_world = bbox center of the child link's vertex cloud.
            # Each link Xform may have a non-zero translation (parent_origin /
            # child_origin), so we subtract it to get the pivot in each body's
            # local frame.  When Xforms are at the world origin the result is
            # identical to the previous behaviour (pivot == local_pos).
            pivot_world = _compute_link_bbox_center(child_bodies)
            parent_origin = _get_xform_translate(stage, parent_path) if parent_path else np.zeros(3)
            child_origin  = _get_xform_translate(stage, child_path)  if child_path  else np.zeros(3)
            local_pos_0 = pivot_world - parent_origin
            local_pos_1 = pivot_world - child_origin

            joint_prim.CreateAttribute(
                "physics:localPos0", Sdf.ValueTypeNames.Point3f
            ).Set(Gf.Vec3f(float(local_pos_0[0]), float(local_pos_0[1]), float(local_pos_0[2])))
            joint_prim.CreateAttribute(
                "physics:localPos1", Sdf.ValueTypeNames.Point3f
            ).Set(Gf.Vec3f(float(local_pos_1[0]), float(local_pos_1[1]), float(local_pos_1[2])))
            # Explicit identity quaternions via CreateAttribute (not schema API) to
            # guarantee the physics: namespace token is written and never left invalid.
            joint_prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(
                Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            )
            joint_prim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(
                Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            )

            print(
                f"Joint [{joint_prim_name}]: "
                f"Parent_localPos0 = {tuple(float(v) for v in local_pos_0)}, "
                f"Child_localPos1 = {tuple(float(v) for v in local_pos_1)}"
            )
            _log_axis = motion_axis if jtype != JointType.fixed else "N/A"
            logger.info(
                "Created joint: %s [%s] %s → %s (axis=%s, pivot=(%.4f,%.4f,%.4f))",
                joint_prim_name, jtype.value,
                joint_def.parent_link, joint_def.child_link, _log_axis,
                pivot_world[0], pivot_world[1], pivot_world[2],
            )

    stage.GetRootLayer().Save()
    logger.info("Wrote USD stage: %s", output_path)
