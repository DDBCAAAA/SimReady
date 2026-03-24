"""Tests for USD export of articulated objects.

Covers PhysicsArticulationRootAPI, per-link RigidBodyAPI, joint prim creation
(revolute, prismatic) driven by an ArticulationTopology with RigidLinkGroups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared minimal geometry helpers
# ---------------------------------------------------------------------------

# Tetrahedron: four vertices, four triangular faces.
# Convex hull is well-defined; suitable for collision generation.
_VERTS = np.array(
    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.05, 0.1, 0.0], [0.05, 0.05, 0.1]],
    dtype=np.float64,
)
_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)


def _make_body(name: str):
    from simready.ingestion.step_reader import CADBody
    return CADBody(name=name, vertices=_VERTS.copy(), faces=_FACES.copy())


def _make_mat(name: str):
    from simready.materials.material_map import MDLMaterial
    return MDLMaterial(mdl_name="OmniPBR.mdl", source_material=name)


# ---------------------------------------------------------------------------
# test_articulated_scissors_export
# ---------------------------------------------------------------------------

def test_articulated_scissors_export(tmp_path: Path):
    """Scissors articulation export: two blades connected by a revolute/Z joint.

    Each blade is its own RigidLinkGroup (one constituent part each).

    Asserts:
      a) PhysicsArticulationRootAPI applied to the assembly root.
      b) PhysicsRigidBodyAPI present ≥2 times (once per link group).
      c) PhysicsRevoluteJoint prim created.
      d) Joint body0/body1 relationship targets reference /Root/blade_a and /Root/blade_b.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")

    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings
    from simready.articulation_inference import (
        ArticulationTopology,
        JointDefinition,
        JointType,
        RigidLinkGroup,
    )

    bodies = [_make_body("blade_a"), _make_body("blade_b")]
    materials = {
        "blade_a": _make_mat("blade_a"),
        "blade_b": _make_mat("blade_b"),
    }
    topology = ArticulationTopology(
        base_link_name="blade_a",
        rigid_links=[
            RigidLinkGroup(link_name="blade_a", constituent_parts=["blade_a"]),
            RigidLinkGroup(link_name="blade_b", constituent_parts=["blade_b"]),
        ],
        joints=[
            JointDefinition(
                parent_link="blade_a",
                child_link="blade_b",
                joint_type=JointType.revolute,
                motion_axis="Z",
                reasoning="Scissors blades rotate around the pivot pin on the Z axis.",
            )
        ],
    )

    settings = PipelineSettings()
    settings.validation.enabled = False

    out_path = tmp_path / "scissors.usda"
    create_stage(bodies, materials, out_path, settings, topology=topology)

    assert out_path.exists(), "USD file was not created"
    text = out_path.read_text()

    # a) Articulation root applied to the assembly root Xform
    assert "PhysicsArticulationRootAPI" in text, (
        "PhysicsArticulationRootAPI missing — assembly root must be an articulation root"
    )

    # b) Each link group has its own RigidBodyAPI
    rigid_count = text.count("PhysicsRigidBodyAPI")
    assert rigid_count >= 2, (
        f"Expected ≥2 PhysicsRigidBodyAPI entries (one per link group), got {rigid_count}"
    )

    # c) A revolute joint prim was created
    assert "PhysicsRevoluteJoint" in text, (
        "PhysicsRevoluteJoint missing — VLM-inferred revolute joint not written to USD"
    )

    # d) Joint references both blades via body0 / body1 relationship targets
    assert "physics:body0" in text, "physics:body0 relationship missing from joint prim"
    assert "/Root/blade_a" in text, (
        "Joint does not reference /Root/blade_a as body0 target"
    )
    assert "/Root/blade_b" in text, (
        "Joint does not reference /Root/blade_b as body1 target"
    )


# ---------------------------------------------------------------------------
# test_multi_hull_collision_per_body
# ---------------------------------------------------------------------------

def test_multi_hull_collision_per_body(tmp_path: Path):
    """Every body Xform must contain ≥2 PhysicsCollisionAPI meshes when CoACD
    returns multiple convex hulls.

    Asserts:
      a) Collision_0 and Collision_1 prims are both written under the body Xform.
      b) PhysicsCollisionAPI count in the file is ≥2 (one per hull).
    """
    pytest.importorskip("pxr", reason="usd-core not installed")
    from unittest.mock import patch

    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings

    body = _make_body("blade_a")
    materials = {"blade_a": _make_mat("blade_a")}
    settings = PipelineSettings()
    settings.validation.enabled = False

    # Two tiny separate hulls — simulates what CoACD returns for a concave part.
    hull0_v = _VERTS.copy()
    hull0_f = _FACES.copy()
    hull1_v = _VERTS.copy() + np.array([0.5, 0.0, 0.0])
    hull1_f = _FACES.copy()
    two_hulls = [(hull0_v, hull0_f), (hull1_v, hull1_f)]

    out_path = tmp_path / "multi_hull.usda"
    with patch(
        "simready.geometry.mesh_processing.decompose_convex", return_value=two_hulls
    ):
        create_stage([body], materials, out_path, settings)

    assert out_path.exists(), "USD file was not created"
    text = out_path.read_text()

    assert "Collision_0" in text, "Collision_0 prim missing — first hull not written"
    assert "Collision_1" in text, "Collision_1 prim missing — second hull not written"
    collision_count = text.count("PhysicsCollisionAPI")
    assert collision_count >= 2, (
        f"Expected ≥2 PhysicsCollisionAPI entries (one per hull), got {collision_count}"
    )


# ---------------------------------------------------------------------------
# test_revolute_joint_limits_in_usd
# ---------------------------------------------------------------------------

def test_revolute_joint_limits_in_usd(tmp_path: Path):
    """Revolute joint with explicit limits must write physics:lowerLimit and
    physics:upperLimit float attributes to the joint prim.

    Asserts:
      a) physics:lowerLimit present in the USDA text.
      b) physics:upperLimit present in the USDA text.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")

    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings
    from simready.articulation_inference import (
        ArticulationTopology,
        JointDefinition,
        JointType,
        RigidLinkGroup,
    )

    bodies = [_make_body("blade_a"), _make_body("blade_b")]
    materials = {
        "blade_a": _make_mat("blade_a"),
        "blade_b": _make_mat("blade_b"),
    }
    topology = ArticulationTopology(
        base_link_name="blade_a",
        rigid_links=[
            RigidLinkGroup(link_name="blade_a", constituent_parts=["blade_a"]),
            RigidLinkGroup(link_name="blade_b", constituent_parts=["blade_b"]),
        ],
        joints=[
            JointDefinition(
                parent_link="blade_a",
                child_link="blade_b",
                joint_type=JointType.revolute,
                motion_axis="Z",
                lower_limit_deg=0.0,
                upper_limit_deg=60.0,
                reasoning="Scissors pivot is constrained to 0–60 degrees.",
            )
        ],
    )

    settings = PipelineSettings()
    settings.validation.enabled = False

    out_path = tmp_path / "joint_limits.usda"
    create_stage(bodies, materials, out_path, settings, topology=topology)

    assert out_path.exists(), "USD file was not created"
    text = out_path.read_text()

    assert "physics:lowerLimit" in text, (
        "physics:lowerLimit missing — revolute joint rotation lower bound not written to USD"
    )
    assert "physics:upperLimit" in text, (
        "physics:upperLimit missing — revolute joint rotation upper bound not written to USD"
    )


# ---------------------------------------------------------------------------
# test_link_grouped_usd_export
# ---------------------------------------------------------------------------

def test_link_grouped_usd_export(tmp_path: Path):
    """When topology has rigid_links, bodies are grouped under link Xforms.

    Topology: link_A = [body_3, body_4], link_B = [body_5, body_6], one revolute joint.

    Asserts:
      a) Link Xforms /Root/link_A and /Root/link_B exist.
      b) Raw meshes body_3, body_4 are parented under /Root/link_A.
      c) Raw meshes body_5, body_6 are parented under /Root/link_B.
      d) Exactly 2 PhysicsRigidBodyAPI (one per link, NOT per raw mesh).
      e) Body sub-Xforms do NOT have RigidBodyAPI.
      f) PhysicsRevoluteJoint created and references the link Xforms.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")

    from pxr import Usd, UsdPhysics

    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings
    from simready.articulation_inference import (
        ArticulationTopology,
        JointDefinition,
        JointType,
        RigidLinkGroup,
    )

    bodies = [
        _make_body("body_3"),
        _make_body("body_4"),
        _make_body("body_5"),
        _make_body("body_6"),
    ]
    materials = {n: _make_mat(n) for n in ["body_3", "body_4", "body_5", "body_6"]}
    topology = ArticulationTopology(
        base_link_name="link_A",
        rigid_links=[
            RigidLinkGroup(link_name="link_A", constituent_parts=["body_3", "body_4"]),
            RigidLinkGroup(link_name="link_B", constituent_parts=["body_5", "body_6"]),
        ],
        joints=[
            JointDefinition(
                parent_link="link_A",
                child_link="link_B",
                joint_type=JointType.revolute,
                motion_axis="Z",
                reasoning="Pivot between blade assemblies.",
            )
        ],
    )
    settings = PipelineSettings()
    settings.validation.enabled = False

    out_path = tmp_path / "grouped.usda"
    create_stage(bodies, materials, out_path, settings, topology=topology)

    assert out_path.exists(), "USD file was not created"

    # Use the USD Python API to check prim paths precisely
    usd_stage = Usd.Stage.Open(str(out_path))

    # a) Link Xforms exist
    assert usd_stage.GetPrimAtPath("/Root/link_A").IsValid(), "link_A Xform missing"
    assert usd_stage.GetPrimAtPath("/Root/link_B").IsValid(), "link_B Xform missing"

    # b) body_3 and body_4 parented under link_A
    assert usd_stage.GetPrimAtPath("/Root/link_A/body_3").IsValid(), (
        "body_3 not found under /Root/link_A"
    )
    assert usd_stage.GetPrimAtPath("/Root/link_A/body_4").IsValid(), (
        "body_4 not found under /Root/link_A"
    )

    # c) body_5 and body_6 parented under link_B
    assert usd_stage.GetPrimAtPath("/Root/link_B/body_5").IsValid(), (
        "body_5 not found under /Root/link_B"
    )
    assert usd_stage.GetPrimAtPath("/Root/link_B/body_6").IsValid(), (
        "body_6 not found under /Root/link_B"
    )

    # d) Exactly 2 RigidBodyAPI — one per link, not per raw mesh
    text = out_path.read_text()
    rigid_count = text.count("PhysicsRigidBodyAPI")
    assert rigid_count == 2, (
        f"Expected exactly 2 PhysicsRigidBodyAPI (one per link group), got {rigid_count}"
    )

    # e) Body sub-Xforms must NOT have RigidBodyAPI
    body3_prim = usd_stage.GetPrimAtPath("/Root/link_A/body_3")
    assert not body3_prim.HasAPI(UsdPhysics.RigidBodyAPI), (
        "body_3 should NOT have RigidBodyAPI — only the link Xform should"
    )

    # f) Revolute joint exists and references link Xforms
    assert "PhysicsRevoluteJoint" in text, "PhysicsRevoluteJoint missing"
    assert "/Root/link_A" in text, "Joint body0 target /Root/link_A missing"
    assert "/Root/link_B" in text, "Joint body1 target /Root/link_B missing"


# ---------------------------------------------------------------------------
# test_joint_anchor_heuristic
# ---------------------------------------------------------------------------

def test_joint_anchor_heuristic(tmp_path: Path):
    """Auto-joint anchor: localPos0/localPos1 are derived from link bounding box centers.

    Parent link bbox center at (0, 0, 0), child link bbox center at (0, 10, 0).

    Expected:
        localPos0 = (0, 10, 0)  — pivot expressed in parent local frame
        localPos1 = (0,  0, 0)  — pivot expressed in child local frame (always origin)
    """
    pytest.importorskip("pxr", reason="usd-core not installed")

    from pxr import Usd
    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings
    from simready.articulation_inference import (
        ArticulationTopology,
        JointDefinition,
        JointType,
        RigidLinkGroup,
    )
    from simready.ingestion.step_reader import CADBody

    # Tetrahedron faces shared by both bodies
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)

    # Parent: tetrahedron centred at (0, 0, 0)
    # bbox: min=(-0.5,-0.5,-0.5)  max=(0.5, 0.5, 0.5)  → centre=(0,0,0)
    parent_verts = np.array(
        [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.0, 0.5]],
        dtype=np.float64,
    )

    # Child: same shape shifted so bbox centre lands at (0, 10, 0)
    child_verts = parent_verts + np.array([0.0, 10.0, 0.0])

    parent_body = CADBody(name="parent_link", vertices=parent_verts, faces=faces)
    child_body = CADBody(name="child_link", vertices=child_verts, faces=faces)

    materials = {
        "parent_link": _make_mat("parent_link"),
        "child_link": _make_mat("child_link"),
    }
    topology = ArticulationTopology(
        base_link_name="parent_link",
        rigid_links=[
            RigidLinkGroup(link_name="parent_link", constituent_parts=["parent_link"]),
            RigidLinkGroup(link_name="child_link", constituent_parts=["child_link"]),
        ],
        joints=[
            JointDefinition(
                parent_link="parent_link",
                child_link="child_link",
                joint_type=JointType.revolute,
                motion_axis="X",
                reasoning="Test joint for anchor heuristic.",
            )
        ],
    )

    settings = PipelineSettings()
    settings.validation.enabled = False

    out_path = tmp_path / "joint_anchor.usda"
    create_stage([parent_body, child_body], materials, out_path, settings, topology=topology)

    assert out_path.exists(), "USD file was not created"

    usd_stage = Usd.Stage.Open(str(out_path))
    joint_prim = usd_stage.GetPrimAtPath("/Root/Joints/joint_0_revolute")
    assert joint_prim.IsValid(), "Joint prim /Root/Joints/joint_0_revolute not found"

    local_pos0 = joint_prim.GetAttribute("physics:localPos0").Get()
    local_pos1 = joint_prim.GetAttribute("physics:localPos1").Get()

    assert local_pos0 is not None, "physics:localPos0 attribute not set on joint"
    assert local_pos1 is not None, "physics:localPos1 attribute not set on joint"

    # localPos0 = pivot(0,10,0) − parent_centre(0,0,0) = (0,10,0)
    assert abs(local_pos0[0] - 0.0) < 1e-4, f"localPos0.x expected 0.0, got {local_pos0[0]}"
    assert abs(local_pos0[1] - 10.0) < 1e-4, f"localPos0.y expected 10.0, got {local_pos0[1]}"
    assert abs(local_pos0[2] - 0.0) < 1e-4, f"localPos0.z expected 0.0, got {local_pos0[2]}"

    # localPos1 == localPos0 == pivot world coords
    # (both link Xforms are at world origin ⇒ local frame == world frame)
    assert abs(local_pos1[0] - 0.0) < 1e-4, f"localPos1.x expected 0.0, got {local_pos1[0]}"
    assert abs(local_pos1[1] - 10.0) < 1e-4, f"localPos1.y expected 10.0, got {local_pos1[1]}"
    assert abs(local_pos1[2] - 0.0) < 1e-4, f"localPos1.z expected 0.0, got {local_pos1[2]}"


# ---------------------------------------------------------------------------
# test_pca_axis_detection
# ---------------------------------------------------------------------------

def test_pca_axis_detection():
    """_infer_rotation_axis_from_pca identifies Z for a cylinder-like point cloud."""
    from simready.usd.assembly import _infer_rotation_axis_from_pca
    from simready.ingestion.step_reader import CADBody

    rng = np.random.default_rng(42)
    # Disk along Z: radius ~1, thickness ~0.05 — large XY spread, tiny Z spread.
    # Var(X) ≈ Var(Y) ≈ 0.25  >> Var(Z) ≈ 2e-4  → Z is the min-variance axis.
    theta = rng.uniform(0, 2 * np.pi, 2000)
    r = rng.uniform(0, 1.0, 2000)
    z = rng.uniform(-0.025, 0.025, 2000)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    verts = np.column_stack([x, y, z])
    faces = np.zeros((1, 3), dtype=np.int32)   # dummy — not used by helper
    body = CADBody(name="cylinder", vertices=verts, faces=faces)

    axis = _infer_rotation_axis_from_pca([body])
    assert axis == "Z", f"Expected PCA axis 'Z' for cylinder-along-Z, got '{axis}'"


def test_pca_axis_override_in_joint(tmp_path: Path):
    """When the VLM guesses the wrong axis, PCA overrides it to the correct one.

    The child link is a cylinder along Z.  The VLM topology says axis='X'.
    The USD joint must end up with physics:axis = 'Z'.
    """
    pytest.importorskip("pxr", reason="usd-core not installed")

    from pxr import Usd
    from simready.usd.assembly import create_stage
    from simready.config.settings import PipelineSettings
    from simready.articulation_inference import (
        ArticulationTopology, JointDefinition, JointType, RigidLinkGroup,
    )
    from simready.ingestion.step_reader import CADBody

    rng = np.random.default_rng(7)
    theta = rng.uniform(0, 2 * np.pi, 500)
    z = rng.uniform(-0.5, 0.5, 500)
    child_verts = np.column_stack([np.cos(theta), np.sin(theta), z * 0.1]).astype(np.float64)
    parent_verts = np.array(
        [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces_tri = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    # child needs at least 4 faces for trimesh not to choke — use a simple fan
    child_faces = np.array([[0, i, i + 1] for i in range(1, len(child_verts) - 1)], dtype=np.int32)

    bodies = [
        CADBody(name="base", vertices=parent_verts, faces=faces_tri),
        CADBody(name="shaft", vertices=child_verts, faces=child_faces),
    ]
    materials = {"base": _make_mat("base"), "shaft": _make_mat("shaft")}
    topology = ArticulationTopology(
        base_link_name="base",
        rigid_links=[
            RigidLinkGroup(link_name="base", constituent_parts=["base"]),
            RigidLinkGroup(link_name="shaft", constituent_parts=["shaft"]),
        ],
        joints=[
            JointDefinition(
                parent_link="base",
                child_link="shaft",
                joint_type=JointType.revolute,
                motion_axis="X",          # deliberately wrong — PCA should correct to Z
                reasoning="VLM wrong-axis test.",
            )
        ],
    )
    settings = PipelineSettings()
    settings.validation.enabled = False

    out_path = tmp_path / "pca_override.usda"
    create_stage(bodies, materials, out_path, settings, topology=topology)

    assert out_path.exists()
    usd_stage = Usd.Stage.Open(str(out_path))
    joint_prim = usd_stage.GetPrimAtPath("/Root/Joints/joint_0_revolute")
    assert joint_prim.IsValid()

    axis = joint_prim.GetAttribute("physics:axis").Get()
    assert axis == "Z", f"Expected PCA-corrected axis 'Z', got '{axis}'"
