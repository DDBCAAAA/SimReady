"""TDD tests for VLM-based kinematic topology inference (two-stage architecture).

All Anthropic API calls are mocked — no network required.
Tests follow the same mock injection pattern as test_vlm_material.py.

Two-Stage Kinematic Reasoning:
  Stage 1 — Link Aggregation: group raw meshes into rigid sub-assemblies (RigidLinkGroup).
  Stage 2 — Joint Topology: define joints ONLY between RigidLinkGroups.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from simready.articulation_inference import (
    ArticulationTopology,
    JointDefinition,
    JointType,
    RigidLinkGroup,
    clear_articulation_client,
    infer_kinematic_topology,
)


# ---------------------------------------------------------------------------
# Helper: build a fake anthropic module that returns a valid topology JSON
# ---------------------------------------------------------------------------

def _make_anthropic_module(topology_dict: dict):
    """Return (fake_module, client_instance) for injection into sys.modules."""
    text_block = SimpleNamespace(
        type="text",
        text=json.dumps(topology_dict),
    )
    response = SimpleNamespace(content=[text_block])
    client_instance = MagicMock()
    client_instance.messages.create.return_value = response

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = MagicMock(return_value=client_instance)
    return fake_mod, client_instance


def _make_anthropic_module_raising(exc: Exception):
    """Return a fake anthropic module whose client.messages.create raises."""
    client_instance = MagicMock()
    client_instance.messages.create.side_effect = exc
    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = MagicMock(return_value=client_instance)
    return fake_mod, client_instance


# ---------------------------------------------------------------------------
# RigidLinkGroup schema unit tests
# ---------------------------------------------------------------------------

class TestRigidLinkGroup:
    def test_rigid_link_group_fields(self):
        """RigidLinkGroup must expose link_name and constituent_parts."""
        rlg = RigidLinkGroup(
            link_name="left_blade_assembly",
            constituent_parts=["body_3", "body_4"],
        )
        assert rlg.link_name == "left_blade_assembly"
        assert rlg.constituent_parts == ["body_3", "body_4"]

    def test_rigid_link_group_single_part(self):
        """A RigidLinkGroup can contain a single part."""
        rlg = RigidLinkGroup(link_name="base", constituent_parts=["body_0"])
        assert len(rlg.constituent_parts) == 1
        assert rlg.constituent_parts[0] == "body_0"

    def test_articulation_topology_has_rigid_links(self):
        """ArticulationTopology must expose rigid_links and base_link_name."""
        topo = ArticulationTopology(
            base_link_name="link_A",
            rigid_links=[
                RigidLinkGroup(link_name="link_A", constituent_parts=["body_3", "body_4"]),
                RigidLinkGroup(link_name="link_B", constituent_parts=["body_5", "body_6"]),
            ],
            joints=[],
        )
        assert topo.base_link_name == "link_A"
        assert len(topo.rigid_links) == 2
        assert topo.rigid_links[0].link_name == "link_A"
        assert topo.rigid_links[1].link_name == "link_B"

    def test_joint_definition_uses_parent_link_child_link(self):
        """JointDefinition must use parent_link/child_link referencing RigidLinkGroup names."""
        jd = JointDefinition(
            parent_link="link_A",
            child_link="link_B",
            joint_type=JointType.revolute,
            motion_axis="Z",
            reasoning="Pivot between assemblies.",
        )
        assert jd.parent_link == "link_A"
        assert jd.child_link == "link_B"

    def test_scissors_two_link_topology(self):
        """Standard scissors: exactly two RigidLinkGroups, one revolute joint."""
        topo = ArticulationTopology(
            base_link_name="blade_assembly_A",
            rigid_links=[
                RigidLinkGroup(link_name="blade_assembly_A", constituent_parts=["blade_a", "handle_a"]),
                RigidLinkGroup(link_name="blade_assembly_B", constituent_parts=["blade_b", "handle_b"]),
            ],
            joints=[
                JointDefinition(
                    parent_link="blade_assembly_A",
                    child_link="blade_assembly_B",
                    joint_type=JointType.revolute,
                    motion_axis="Z",
                    lower_limit_deg=0.0,
                    upper_limit_deg=60.0,
                    reasoning="Pivot pin connects the two blade assemblies.",
                )
            ],
        )
        assert len(topo.rigid_links) == 2
        assert len(topo.joints) == 1
        assert topo.joints[0].joint_type == JointType.revolute


# ---------------------------------------------------------------------------
# Schema unit tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_joint_type_enum_values(self):
        """JointType must expose fixed, revolute, and prismatic."""
        assert JointType.fixed.value == "fixed"
        assert JointType.revolute.value == "revolute"
        assert JointType.prismatic.value == "prismatic"

    def test_joint_definition_fields(self):
        jd = JointDefinition(
            parent_link="base",
            child_link="slider",
            joint_type=JointType.prismatic,
            motion_axis="X",
            reasoning="Slides along X axis.",
        )
        assert jd.parent_link == "base"
        assert jd.child_link == "slider"
        assert jd.joint_type == JointType.prismatic
        assert jd.motion_axis == "X"

    def test_articulation_topology_fields(self):
        topo = ArticulationTopology(
            base_link_name="vise_base",
            rigid_links=[
                RigidLinkGroup(link_name="vise_base", constituent_parts=["vise_base"]),
                RigidLinkGroup(link_name="vise_slider", constituent_parts=["vise_slider"]),
            ],
            joints=[
                JointDefinition(
                    parent_link="vise_base",
                    child_link="vise_slider",
                    joint_type=JointType.prismatic,
                    motion_axis="X",
                    reasoning="The jaw slides to clamp workpieces.",
                )
            ],
        )
        assert topo.base_link_name == "vise_base"
        assert len(topo.joints) == 1

    def test_motion_axis_valid_values(self):
        """motion_axis accepts X, Y, Z (case-insensitive after normalisation)."""
        for axis in ("X", "Y", "Z"):
            jd = JointDefinition(
                parent_link="a",
                child_link="b",
                joint_type=JointType.revolute,
                motion_axis=axis,
                reasoning="test",
            )
            assert jd.motion_axis in ("X", "Y", "Z")

    def test_motion_axis_rejects_invalid(self):
        """motion_axis must be one of X/Y/Z."""
        with pytest.raises(Exception):
            JointDefinition(
                parent_link="a",
                child_link="b",
                joint_type=JointType.revolute,
                motion_axis="diagonal",
                reasoning="test",
            )

    def test_joint_limits_fields_accepted(self):
        """JointDefinition accepts lower_limit_deg and upper_limit_deg for revolute joints."""
        jd = JointDefinition(
            parent_link="blade_a",
            child_link="blade_b",
            joint_type=JointType.revolute,
            motion_axis="Z",
            lower_limit_deg=0.0,
            upper_limit_deg=60.0,
            reasoning="Scissors pivot.",
        )
        assert jd.lower_limit_deg == 0.0
        assert jd.upper_limit_deg == 60.0

    def test_joint_limits_default_to_none(self):
        """lower_limit_deg and upper_limit_deg default to None when omitted."""
        jd = JointDefinition(
            parent_link="a",
            child_link="b",
            joint_type=JointType.prismatic,
            motion_axis="X",
            reasoning="Slides.",
        )
        assert jd.lower_limit_deg is None
        assert jd.upper_limit_deg is None

    def test_joint_limits_parsed_from_vlm_json(self, monkeypatch):
        """VLM response with lower/upper limits is correctly parsed into the topology."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        scissors_json = {
            "base_link_name": "blade_assembly_A",
            "rigid_links": [
                {"link_name": "blade_assembly_A", "constituent_parts": ["blade_a"]},
                {"link_name": "blade_assembly_B", "constituent_parts": ["blade_b"]},
            ],
            "joints": [
                {
                    "parent_link": "blade_assembly_A",
                    "child_link": "blade_assembly_B",
                    "joint_type": "revolute",
                    "motion_axis": "Z",
                    "lower_limit_deg": 0.0,
                    "upper_limit_deg": 60.0,
                    "reasoning": "Scissors pivot is constrained to 0-60 degrees.",
                }
            ],
        }
        from unittest.mock import MagicMock, patch
        import types
        from types import SimpleNamespace
        import json as _json
        text_block = SimpleNamespace(type="text", text=_json.dumps(scissors_json))
        response = SimpleNamespace(content=[text_block])
        client_instance = MagicMock()
        client_instance.messages.create.return_value = response
        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = MagicMock(return_value=client_instance)
        with patch.dict(__import__("sys").modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata={"blade_a": {}, "blade_b": {}},
                object_label="scissors",
            )
        assert result is not None
        joint = result.joints[0]
        assert joint.lower_limit_deg == 0.0
        assert joint.upper_limit_deg == 60.0


# ---------------------------------------------------------------------------
# infer_kinematic_topology — vise mock test
# ---------------------------------------------------------------------------

VISE_PARTS_METADATA = {
    "vise_base": {
        "bbox_m": (0.20, 0.15, 0.10),
        "volume_m3": 2.5e-3,
    },
    "vise_slider": {
        "bbox_m": (0.12, 0.08, 0.06),
        "volume_m3": 4.0e-4,
    },
}

_VISE_TOPOLOGY_JSON = {
    "base_link_name": "vise_base",
    "rigid_links": [
        {"link_name": "vise_base", "constituent_parts": ["vise_base"]},
        {"link_name": "vise_slider", "constituent_parts": ["vise_slider"]},
    ],
    "joints": [
        {
            "parent_link": "vise_base",
            "child_link": "vise_slider",
            "joint_type": "prismatic",
            "motion_axis": "X",
            "reasoning": (
                "The movable jaw slides along the X axis to clamp workpieces; "
                "driven by a leadscrew anchored in the base casting."
            ),
        }
    ],
}


class TestInferKinematicTopology:
    def setup_method(self):
        clear_articulation_client()
        sys.modules.pop("anthropic", None)

    def test_vise_base_is_base_link(self, monkeypatch):
        """VLM response correctly identifies vise_base as the root anchor."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module(_VISE_TOPOLOGY_JSON)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is not None
        assert result.base_link_name == "vise_base"

    def test_vise_has_one_prismatic_joint(self, monkeypatch):
        """Vise topology must contain exactly one prismatic joint."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module(_VISE_TOPOLOGY_JSON)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is not None
        assert len(result.joints) == 1
        joint = result.joints[0]
        assert joint.joint_type == JointType.prismatic

    def test_vise_joint_axis_is_x(self, monkeypatch):
        """The prismatic joint must be along the X axis."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module(_VISE_TOPOLOGY_JSON)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is not None
        assert result.joints[0].motion_axis == "X"

    def test_vise_joint_parent_child(self, monkeypatch):
        """Joint must connect vise_base (parent) → vise_slider (child)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module(_VISE_TOPOLOGY_JSON)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is not None
        joint = result.joints[0]
        assert joint.parent_link == "vise_base"
        assert joint.child_link == "vise_slider"

    def test_vise_has_two_rigid_links(self, monkeypatch):
        """VLM response for a vise must produce exactly two RigidLinkGroups."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module(_VISE_TOPOLOGY_JSON)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is not None
        assert len(result.rigid_links) == 2
        link_names = {rlg.link_name for rlg in result.rigid_links}
        assert link_names == {"vise_base", "vise_slider"}

    def test_missing_api_key_returns_none(self, monkeypatch):
        """Returns None when ANTHROPIC_API_KEY is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = infer_kinematic_topology(
            image_paths=[],
            parts_metadata=VISE_PARTS_METADATA,
            object_label="industrial_vise",
        )
        assert result is None

    def test_anthropic_not_installed_returns_none(self, monkeypatch):
        """Returns None when the anthropic package is missing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch.dict(sys.modules, {"anthropic": None}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is None

    def test_api_exception_returns_none(self, monkeypatch):
        """Returns None when the API call raises an exception."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module_raising(RuntimeError("timeout"))
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is None

    def test_json_parse_error_returns_none(self, monkeypatch):
        """Returns None when the API response is not valid JSON."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        bad_block = SimpleNamespace(type="text", text="not-json {{bad")
        bad_response = SimpleNamespace(content=[bad_block])
        client_instance = MagicMock()
        client_instance.messages.create.return_value = bad_response
        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = MagicMock(return_value=client_instance)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is None

    def test_parts_metadata_injected_into_prompt(self, monkeypatch):
        """Part names and bbox dimensions appear in the VLM user message."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, client_instance = _make_anthropic_module(_VISE_TOPOLOGY_JSON)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        call_kwargs = client_instance.messages.create.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "vise_base" in user_msg
        assert "vise_slider" in user_msg
        assert "industrial_vise" in user_msg

    def test_revolute_joint_parsed_correctly(self, monkeypatch):
        """A revolute joint in the VLM response is correctly parsed."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        hinge_json = {
            "base_link_name": "door_frame",
            "rigid_links": [
                {"link_name": "door_frame", "constituent_parts": ["door_frame"]},
                {"link_name": "door_panel", "constituent_parts": ["door_panel"]},
            ],
            "joints": [
                {
                    "parent_link": "door_frame",
                    "child_link": "door_panel",
                    "joint_type": "revolute",
                    "motion_axis": "Z",
                    "reasoning": "Door rotates around vertical Z axis on hinges.",
                }
            ],
        }
        fake_mod, _ = _make_anthropic_module(hinge_json)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata={
                    "door_frame": {"bbox_m": (0.1, 2.0, 0.05)},
                    "door_panel": {"bbox_m": (0.05, 1.9, 0.03)},
                },
                object_label="cabinet_door",
            )
        assert result is not None
        assert result.joints[0].joint_type == JointType.revolute
        assert result.joints[0].motion_axis == "Z"

    def test_markdown_fence_stripped(self, monkeypatch):
        """JSON wrapped in ```json ... ``` fences is parsed correctly."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fenced = "```json\n" + json.dumps(_VISE_TOPOLOGY_JSON) + "\n```"
        text_block = SimpleNamespace(type="text", text=fenced)
        response = SimpleNamespace(content=[text_block])
        client_instance = MagicMock()
        client_instance.messages.create.return_value = response
        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = MagicMock(return_value=client_instance)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = infer_kinematic_topology(
                image_paths=[],
                parts_metadata=VISE_PARTS_METADATA,
                object_label="industrial_vise",
            )
        assert result is not None
        assert result.base_link_name == "vise_base"
