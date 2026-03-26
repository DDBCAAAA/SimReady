"""partnet_adapter.py — Bridge between PartNetAsset and the SimReady pipeline.

This module converts a parsed PartNetAsset into the exact internal types that
simready.usd.assembly.create_stage() and simready.pipeline.run() consume:

    PartNetAsset
        │
        ├── to_cad_assembly()          →  CADAssembly  (bodies with meshes)
        ├── to_articulation_topology() →  ArticulationTopology (links + joints)
        ├── build_material_dict()      →  dict[str, MDLMaterial]
        │
        └── run_partnet()              →  calls create_stage() directly,
                                          bypassing pipeline.run() (which
                                          expects a STEP/STL input file)

Integration quick-reference
---------------------------
1. Download   → PartNetDownloader.fetch(object_id)
2. Parse      → PartNetURDFParser(asset_dir).parse()
3. Adapt      → run_partnet(asset, output_path, settings)
                     — or build pieces individually and call create_stage().

PartNet-specific quirks handled here
-------------------------------------
* Y-up meshes in meters: converted to Z-up (USD default) and kept at 1 m/unit.
* Non-convex visual meshes: isolated per-link for CoACD decomposition before
  handing off to create_stage().
* Root link detection: first link with no parent joint gets PhysicsArticulationRoot.
* Continuous joints (wheels): treated as revolute with ±180° limits.
* Fixed joints with zero-volume meshes: flagged for merge into parent link.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from simready.articulation_inference import (
    ArticulationTopology,
    JointDefinition,
    JointType,
    RigidLinkGroup,
)
from simready.ingestion.step_reader import CADAssembly, CADBody
from simready.materials.material_map import CAEMaterial, MDLMaterial, map_cae_to_mdl

logger = logging.getLogger(__name__)

# PartNet meshes are Y-up; USD default is Z-up.
# Rotate -90° around X: (x, y, z) → (x, -z, y)
_YUP_TO_ZUP = np.array(
    [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]],
    dtype=np.float64,
)

# Material hints inferred from PartNet category keywords
_CATEGORY_MATERIAL_HINTS: dict[str, str] = {
    "chair":        "plastic_abs",
    "table":        "plastic_abs",
    "door":         "plastic_abs",
    "window":       "glass",
    "cabinet":      "plastic_abs",
    "laptop":       "plastic_abs",
    "storage":      "plastic_abs",
    "box":          "plastic_abs",
    "dishwasher":   "stainless",
    "washing":      "stainless",
    "refrigerator": "stainless",
    "microwave":    "stainless",
    "oven":         "stainless",
    "safe":         "steel",
    "scissors":     "steel",
    "eyeglasses":   "polycarbonate",
    "lighter":      "steel",
    "pen":          "plastic_abs",
    "suitcase":     "polycarbonate",
    "bag":          "nylon",
    "bucket":       "hdpe",
    "toilet":       "ceramic",
    "faucet":       "chrome",
    "lamp":         "plastic_abs",
}


# ---------------------------------------------------------------------------
# Hook 1 — Root identification
# ---------------------------------------------------------------------------

def identify_articulation_root(asset: "PartNetAsset") -> str:  # noqa: F821
    """Return the link name that should receive PhysicsArticulationRootAPI.

    In PartNet, the root is always the link with no incoming joint (no parent).
    This is pre-computed in PartNetAsset.root_link, but this hook is the
    canonical integration point so your pipeline can override it if needed.

    Usage in your pipeline::

        root_link = identify_articulation_root(asset)
        root_prim = stage.GetPrimAtPath(f"/Root/{root_link}")
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    """
    return asset.root_link


# ---------------------------------------------------------------------------
# Hook 2 — Mesh processing (V-HACD preparation)
# ---------------------------------------------------------------------------

def get_meshes_for_decomposition(
    asset: "PartNetAsset",  # noqa: F821
    link_name: str,
) -> list[Path]:
    """Return the visual mesh paths for a link, ready for CoACD decomposition.

    PartNet visual meshes are:
    - Non-convex (doors, drawers, shelves) → MUST be decomposed before physics.
    - Shared between visual and collision roles (no separate collision .obj).

    Pass the returned paths to your existing decompose_convex() function::

        from simready.geometry.mesh_processing import decompose_convex
        import trimesh

        for path in get_meshes_for_decomposition(asset, link_name):
            mesh = trimesh.load(path, force="mesh")
            hulls = decompose_convex(mesh)          # → list[(verts, faces)]
            # attach hulls as UsdPhysics.CollisionAPI meshes under the link prim
    """
    lnk = asset.link_map.get(link_name)
    if lnk is None:
        logger.warning("Link '%s' not found in asset %s", link_name, asset.object_id)
        return []
    return list(lnk.visual_meshes)


# ---------------------------------------------------------------------------
# Hook 3 — Joint limit mapping
# ---------------------------------------------------------------------------

def map_joint_limits(joint: "PartNetJoint") -> dict:  # noqa: F821
    """Convert PartNet URDF joint limits to USD Physics joint properties.

    Returns a dict ready to be written as prim attributes::

        props = map_joint_limits(joint)
        prim.GetAttribute("physics:lowerLimit").Set(props["lower"])
        prim.GetAttribute("physics:upperLimit").Set(props["upper"])
        prim.GetAttribute("physics:axis").Set(props["axis"])

    Notes
    -----
    - Revolute limits are in degrees (USD convention), converted from radians.
    - Prismatic limits stay in meters.
    - Continuous joints get ±180° limits.
    - The dominant axis in the joint.axis vector selects "X", "Y", or "Z".
    """
    axis_str = _dominant_axis(joint.axis)

    if joint.joint_type == "revolute":
        lower = math.degrees(joint.lower) if joint.lower is not None else -180.0
        upper = math.degrees(joint.upper) if joint.upper is not None else  180.0
        return {"axis": axis_str, "lower": lower, "upper": upper, "type": "revolute"}

    if joint.joint_type == "continuous":
        return {"axis": axis_str, "lower": -180.0, "upper": 180.0, "type": "revolute"}

    if joint.joint_type == "prismatic":
        lower = joint.lower if joint.lower is not None else -0.5
        upper = joint.upper if joint.upper is not None else  0.5
        return {"axis": axis_str, "lower": lower, "upper": upper, "type": "prismatic"}

    # fixed / floating
    return {"axis": axis_str, "lower": 0.0, "upper": 0.0, "type": "fixed"}


# ---------------------------------------------------------------------------
# Conversion: PartNetAsset → CADAssembly
# ---------------------------------------------------------------------------

def to_cad_assembly(asset: "PartNetAsset") -> CADAssembly:  # noqa: F821
    """Load all link meshes from .obj files and produce a CADAssembly.

    Each URDF link becomes one CADBody (merging multi-mesh links).
    Meshes are:
      - Loaded via trimesh (handles .mtl textures gracefully)
      - Rotated from Y-up to Z-up
      - Already in meters (PartNet convention) — no unit scaling needed

    Links with no visual geometry are included as zero-volume placeholders
    so the kinematic topology remains intact.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("pip install trimesh")

    from simready.geometry.mesh_processing import clean_mesh, compute_normals

    bodies: list[CADBody] = []

    for link in asset.links:
        vertices_list: list[np.ndarray] = []
        faces_list: list[np.ndarray]    = []
        vertex_offset = 0

        for obj_path in link.visual_meshes:
            try:
                mesh = trimesh.load(str(obj_path), force="mesh", process=False)
            except Exception as exc:
                logger.warning("Could not load %s: %s", obj_path.name, exc)
                continue

            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                logger.debug("Skipping empty/non-trimesh geometry: %s", obj_path.name)
                continue

            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces,    dtype=np.int64)

            # Y-up → Z-up rotation (PartNet is Y-up, USD is Z-up)
            verts = verts @ _YUP_TO_ZUP.T

            faces = faces + vertex_offset
            vertices_list.append(verts)
            faces_list.append(faces)
            vertex_offset += len(verts)

        if vertices_list:
            all_verts = np.vstack(vertices_list)
            all_faces = np.vstack(faces_list)
        else:
            # Placeholder body (fixed/invisible link — still needed for joint graph)
            all_verts = np.zeros((1, 3), dtype=np.float64)
            all_faces = np.zeros((0, 3), dtype=np.int64)

        # Build trimesh for normal computation + watertight check
        try:
            tm = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
            tm = clean_mesh(tm)
            normals  = compute_normals(tm)
            is_water = tm.is_watertight
            volume   = float(tm.volume) if is_water else None
        except Exception:
            normals  = None
            is_water = False
            volume   = None

        body = CADBody(
            name          = link.name,
            vertices      = all_verts,
            faces         = all_faces,
            normals       = normals,
            material_name = link.name,
            metadata={
                "computed_is_watertight": is_water,
                **({"computed_volume_m3": volume} if volume is not None else {}),
                "world_translate_m": [0.0, 0.0, 0.0],  # PartNet meshes live in world space
                "inertial": link.metadata,
                "partnet_link": link.name,
            },
        )
        bodies.append(body)

    assembly = CADAssembly(
        source_path = asset.asset_dir / "mobility.urdf",
        bodies      = bodies,
        units       = "m",   # PartNet is already in meters
    )
    logger.info(
        "Assembled %d bodies from PartNet object %s", len(bodies), asset.object_id
    )
    return assembly


# ---------------------------------------------------------------------------
# Conversion: PartNetAsset → ArticulationTopology
# ---------------------------------------------------------------------------

def to_articulation_topology(asset: "PartNetAsset") -> ArticulationTopology:  # noqa: F821
    """Convert the URDF kinematic tree to a SimReady ArticulationTopology.

    Mapping rules
    -------------
    - Each URDF link  → one RigidLinkGroup (link_name = URDF link name,
                          constituent_parts = [link_name])
    - Each URDF joint → one JointDefinition
      - revolute / continuous → JointType.revolute
      - prismatic             → JointType.prismatic
      - fixed                 → JointType.fixed
    - Root link (no parent joint) → base_link_name
    """
    rigid_links: list[RigidLinkGroup] = []
    for link in asset.links:
        rigid_links.append(RigidLinkGroup(
            link_name        = link.name,
            constituent_parts = [link.name],
        ))

    joint_defs: list[JointDefinition] = []
    for j in asset.joints:
        limits  = map_joint_limits(j)
        jtype   = _urdf_joint_type_to_simready(j.joint_type)
        axis    = limits["axis"]   # "X", "Y", or "Z"
        lower   = limits["lower"]  # degrees for revolute, meters for prismatic
        upper   = limits["upper"]

        joint_defs.append(JointDefinition(
            parent_link      = j.parent,
            child_link       = j.child,
            joint_type       = jtype,
            motion_axis      = axis,
            lower_limit_deg  = lower if jtype == JointType.revolute else None,
            upper_limit_deg  = upper if jtype == JointType.revolute else None,
            reasoning        = (
                f"URDF {j.joint_type} joint '{j.name}' "
                f"[{j.lower}, {j.upper}] rad"
            ),
        ))

    topology = ArticulationTopology(
        base_link_name = asset.root_link,
        rigid_links    = rigid_links,
        joints         = joint_defs,
    )
    logger.info(
        "Topology: base='%s', %d links, %d joints",
        topology.base_link_name, len(rigid_links), len(joint_defs),
    )
    return topology


# ---------------------------------------------------------------------------
# Material inference
# ---------------------------------------------------------------------------

def build_material_dict(
    asset: "PartNetAsset",  # noqa: F821
    category_hint: str | None = None,
    forced_material: str | None = None,
    enable_vlm: bool = False,
    vlm_model: str  = "claude-haiku-4-5",
) -> dict[str, MDLMaterial]:
    """Produce a {link_name: MDLMaterial} dict for every link in the asset.

    Material resolution order
    -------------------------
    1. ``forced_material`` override (e.g. "steel") applies to every link.
    2. Category hint from ``_CATEGORY_MATERIAL_HINTS`` (e.g. "door" → "wood").
    3. VLM inference when ``enable_vlm=True`` (uses link name + bounding box).
    4. Neutral fallback (OmniPBR grey plastic).
    """
    # Resolve category → material class hint for VLM-off path only.
    # When VLM is on, pass the hint as semantic context so the model can reason
    # freely. When VLM is off, use it as a forced_class fallback.
    category_mat_hint: str | None = None
    if category_hint:
        key = category_hint.lower()
        for kw, hint in _CATEGORY_MATERIAL_HINTS.items():
            if kw in key:
                category_mat_hint = hint
                break

    materials: dict[str, MDLMaterial] = {}
    for link in asset.links:
        cae = CAEMaterial(name=link.name)

        bbox_m: tuple[float, float, float] | None = None
        if link.metadata.get("bbox_m"):
            bbox_m = tuple(link.metadata["bbox_m"])

        # forced_material (explicit --material flag) always wins.
        # category hint only becomes forced_class when VLM is disabled.
        effective_forced = forced_material or (None if enable_vlm else category_mat_hint)

        # Build a rich semantic label for the VLM: "laptop > screen"
        sem_label = f"{category_hint} > {link.name}" if category_hint else link.name

        mat = map_cae_to_mdl(
            cae,
            forced_class   = effective_forced,
            enable_vlm     = enable_vlm,
            vlm_model      = vlm_model,
            semantic_label = sem_label,
            bbox_m         = bbox_m,
        )
        materials[link.name] = mat

    return materials


# ---------------------------------------------------------------------------
# Top-level convenience runner
# ---------------------------------------------------------------------------

def run_partnet(
    asset: "PartNetAsset",  # noqa: F821
    output_path: Path,
    settings=None,
    category_hint: str | None = None,
    forced_material: str | None = None,
    enable_vlm: bool = False,
    asset_metadata: dict | None = None,
) -> dict:
    """Convert a parsed PartNetAsset to a USD file using the SimReady pipeline.

    This function bypasses ``pipeline.run()`` (which reads STEP files) and
    calls ``create_stage()`` directly — the correct integration point for
    pre-parsed articulated datasets.

    Parameters
    ----------
    asset:
        Output of PartNetURDFParser.parse().
    output_path:
        Destination .usd / .usda path.
    settings:
        PipelineSettings instance.  Loads defaults if None.
    category_hint:
        Object category string for material inference (e.g. "StorageFurniture").
    forced_material:
        Override every link's material (e.g. "steel", "wood").
    enable_vlm:
        Run VLM material inference via Claude API.
    asset_metadata:
        Extra key/value pairs embedded in USD /Root customData
        (e.g. PartNet object_id, category, source).

    Returns
    -------
    dict with keys: face_count, quality_score, watertight, physics_complete,
                    material_confidence, material_class
    """
    from simready.config.settings import load_settings
    from simready.geometry.mesh_processing import (
        center_at_com, generate_lod, scale_to_meters,
    )
    from simready.usd.assembly import create_stage
    from simready.validation.simready_checks import compute_quality_score
    from simready.materials.material_map import classify_material

    if settings is None:
        settings = load_settings()
    # PartNet meshes are already in meters — disable unit scaling
    settings.validation.enable_confidence_gate = False

    # --- 1. Load meshes into CADAssembly ---
    assembly = to_cad_assembly(asset)
    if not assembly.bodies:
        raise RuntimeError(f"PartNet {asset.object_id}: no geometry found")

    # --- 2. Geometry processing ---
    for body in assembly.bodies:
        if len(body.faces) == 0:
            # Placeholder (invisible/fixed links) — skip processing
            continue
        try:
            import trimesh
            from simready.geometry.mesh_processing import clean_mesh, compute_normals
            tm = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
            tm = clean_mesh(tm)
            tm, com = center_at_com(tm)
            # Meshes are already in meters — scale factor 1.0
            body.metadata["center_of_mass"]    = com.tolist()
            body.metadata["world_translate_m"] = com.tolist()
            body.vertices = np.asarray(tm.vertices)
            body.faces    = np.asarray(tm.faces)
            body.normals  = compute_normals(tm)
            body.metadata["computed_is_watertight"] = tm.is_watertight
            if tm.is_watertight:
                body.metadata["computed_volume_m3"] = float(tm.volume)

            if settings.geometry.generate_lods:
                body.lod_meshes = []
                for ratio in settings.geometry.lod_levels:
                    lod = generate_lod(tm, ratio)
                    body.lod_meshes.append((ratio, np.asarray(lod.vertices),
                                                   np.asarray(lod.faces)))
        except Exception as exc:
            logger.warning("Geometry processing failed for '%s': %s", body.name, exc)

    # --- 3. Material mapping ---
    materials = build_material_dict(
        asset,
        category_hint    = category_hint,
        forced_material  = forced_material,
        enable_vlm       = enable_vlm,
        vlm_model        = settings.materials.vlm_model,
    )
    # Fill in bbox from processed vertices for any link that needs it
    for body in assembly.bodies:
        if body.name in materials and len(body.vertices) > 1:
            ext = body.vertices.max(axis=0) - body.vertices.min(axis=0)
            body.metadata["bbox_m"] = ext.tolist()

    # --- 4. Build articulation topology ---
    topology = to_articulation_topology(asset)

    # --- 5. Inject PartNet provenance into asset_metadata ---
    meta = {
        "partnet_object_id": asset.object_id,
        "partnet_root_link": asset.root_link,
        "partnet_n_links":   str(len(asset.links)),
        "partnet_n_joints":  str(len(asset.joints)),
    }
    if asset_metadata:
        meta.update(asset_metadata)

    # --- 6. USD assembly ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_stage(
        assembly.bodies, materials, output_path, settings,
        topology       = topology,
        asset_metadata = meta,
    )

    logger.info("PartNet %s → %s", asset.object_id, output_path)

    # --- 7. Summary ---
    total_faces = sum(len(b.faces) for b in assembly.bodies)
    first_body  = next((b for b in assembly.bodies if len(b.faces) > 0), assembly.bodies[0])
    first_mat   = materials.get(first_body.name)
    quality     = compute_quality_score(first_body, first_mat)
    mat_class   = forced_material or classify_material(CAEMaterial(name=first_body.name))

    return {
        "face_count":          total_faces,
        "quality_score":       quality["simready:qualityScore"],
        "watertight":          quality["simready:watertight"],
        "physics_complete":    quality["simready:physicsComplete"],
        "material_confidence": quality["simready:materialConfidence"],
        "material_class":      mat_class,
    }


# ---------------------------------------------------------------------------
# CLI batch helper
# ---------------------------------------------------------------------------

def convert_partnet_batch(
    object_ids: list[str | int],
    data_dir: Path,
    output_dir: Path,
    category_hint: str | None = None,
    forced_material: str | None = None,
    enable_vlm: bool = False,
    hf_token: str | None = None,
) -> list[dict]:
    """Download, parse, and convert a list of PartNet object IDs.

    This is the all-in-one entry point::

        results = convert_partnet_batch(
            object_ids     = [101516, 102379, 102900, 103240, 104035],
            data_dir       = Path("data/partnet"),
            output_dir     = Path("output/partnet"),
            category_hint  = "StorageFurniture",
            forced_material= "wood",
        )

    Returns a list of result dicts (one per object).
    """
    from simready.acquisition.partnet_downloader import PartNetDownloader
    from simready.acquisition.partnet_parser import PartNetURDFParser

    downloader = PartNetDownloader(dest_dir=data_dir, hf_token=hf_token)
    results: list[dict] = []

    for oid in object_ids:
        oid_str = str(oid)
        logger.info("─── PartNet object %s ───", oid_str)

        # Download
        dl_result = downloader.fetch(oid_str)
        if not dl_result.success:
            results.append({"object_id": oid_str, "success": False, "error": dl_result.error})
            continue

        # Parse
        try:
            parser = PartNetURDFParser(dl_result.asset_dir)
            asset  = parser.parse()
        except Exception as exc:
            logger.error("Parse failed for %s: %s", oid_str, exc)
            results.append({"object_id": oid_str, "success": False, "error": str(exc)})
            continue

        # Convert
        usd_path = output_dir / oid_str / f"{oid_str}.usd"
        try:
            summary = run_partnet(
                asset,
                usd_path,
                category_hint   = category_hint,
                forced_material = forced_material,
                enable_vlm      = enable_vlm,
                asset_metadata  = {"partnet_category": category_hint or ""},
            )
            results.append({"object_id": oid_str, "success": True,
                            "usd_path": str(usd_path), **summary})
            logger.info(
                "  → %s  quality=%.2f  mat=%s",
                usd_path.name, summary["quality_score"], summary["material_class"],
            )
        except Exception as exc:
            logger.error("Conversion failed for %s: %s", oid_str, exc)
            results.append({"object_id": oid_str, "success": False, "error": str(exc)})

    passed = sum(1 for r in results if r.get("success"))
    logger.info("PartNet batch: %d/%d converted", passed, len(results))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dominant_axis(axis: tuple[float, float, float]) -> str:
    """Return 'X', 'Y', or 'Z' for the axis vector component with largest magnitude."""
    # PartNet axes are in Y-up space; after Y→Z rotation:
    #   original Y → new Z, original Z → new -Y
    # Apply the same rotation to the axis vector
    ax = np.array(axis, dtype=np.float64)
    ax_zup = _YUP_TO_ZUP @ ax
    idx = int(np.argmax(np.abs(ax_zup)))
    return ["X", "Y", "Z"][idx]


def _urdf_joint_type_to_simready(urdf_type: str) -> JointType:
    _MAP = {
        "revolute":   JointType.revolute,
        "continuous": JointType.revolute,   # continuous = unbounded revolute
        "prismatic":  JointType.prismatic,
        "fixed":      JointType.fixed,
        "floating":   JointType.fixed,      # no floating joint in USD Physics
        "planar":     JointType.fixed,      # not representable — treat as fixed
    }
    jt = _MAP.get(urdf_type)
    if jt is None:
        logger.warning("Unknown URDF joint type '%s' — defaulting to fixed", urdf_type)
        jt = JointType.fixed
    return jt
