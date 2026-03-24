"""VLM-based kinematic topology inference for multi-part articulated objects.

Uses a two-stage reasoning approach (URDF-inspired):
  Stage 1 — Link Aggregation: group raw meshes into rigid sub-assemblies (RigidLinkGroup).
  Stage 2 — Joint Topology: define joints ONLY between the RigidLinkGroups.

Output is an ArticulationTopology — a Pydantic model describing the rigid link groups
and every joint between them, suitable for downstream USD PhysicsArticulationRootAPI /
UsdPhysics joints authoring.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# Kinematic schema
# ---------------------------------------------------------------------------

class JointType(str, Enum):
    fixed = "fixed"
    revolute = "revolute"
    prismatic = "prismatic"


class RigidLinkGroup(BaseModel):
    """A rigid sub-assembly: one or more meshes bolted/welded together as a single body."""

    link_name: str = Field(
        description="Descriptive name for this rigid sub-assembly (e.g. 'left_blade_assembly')."
    )
    constituent_parts: list[str] = Field(
        description="Exact names of the raw meshes that form this rigid link."
    )


class JointDefinition(BaseModel):
    parent_link: str = Field(description="link_name of the parent RigidLinkGroup.")
    child_link: str = Field(description="link_name of the child RigidLinkGroup.")
    joint_type: JointType = Field(description="The inferred type of joint.")
    motion_axis: Literal["X", "Y", "Z"] = Field(
        default="X",
        description="Primary axis of motion: X, Y, or Z. Ignored for fixed joints.",
    )
    lower_limit_deg: float | None = Field(
        default=None,
        description=(
            "Lower rotation limit in degrees for revolute joints (e.g. 0.0). "
            "Null for fixed joints."
        ),
    )
    upper_limit_deg: float | None = Field(
        default=None,
        description=(
            "Upper rotation limit in degrees for revolute joints (e.g. 60.0). "
            "Null for fixed joints."
        ),
    )
    reasoning: str = Field(
        description="Brief explanation of why this joint connects these two links."
    )

    @field_validator("motion_axis", mode="before")
    @classmethod
    def normalise_axis(cls, v: object) -> str:
        if v is None:
            return "X"  # fixed joints have no meaningful axis; default silently
        upper = str(v).strip().upper()
        if upper not in ("X", "Y", "Z"):
            raise ValueError(f"motion_axis must be X, Y, or Z — got '{v}'")
        return upper


class ArticulationTopology(BaseModel):
    base_link_name: str = Field(
        description="link_name of the root/anchor RigidLinkGroup (e.g. the base of a vise)."
    )
    rigid_links: list[RigidLinkGroup] = Field(
        default_factory=list,
        description="All rigid sub-assemblies (Stage 1 aggregation result).",
    )
    joints: list[JointDefinition] = Field(
        default_factory=list,
        description="All inferred joints connecting the RigidLinkGroups.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert Robotics Kinematics Engineer building a URDF/USD kinematic tree. \
You are given a list of raw 3D mesh parts that make up an object. \
You MUST output ONLY a single raw JSON object — no prose, no markdown, no explanation, no code fences.

STAGE 1 — Link Aggregation (Grouping):
Group the raw parts into rigid sub-assemblies (rigid_links). If multiple meshes are bolted, glued, \
or fixed together and DO NOT move relative to each other (e.g. a steel blade and its plastic handle), \
group them into a SINGLE link with a descriptive link_name.
Rule of Thumb: a standard pair of scissors or pliers has exactly TWO rigid_links connected by EXACTLY ONE revolute joint.

STAGE 2 — Joint Topology:
Define the moving joints (revolute or prismatic) ONLY between your newly defined rigid_links.
1. base_link_name = the link_name of the heaviest / most structurally central link.
2. Sliding links (jaw, piston, drawer) → joint_type = "prismatic", axis = direction of travel.
3. Rotating links (handle, blade, wheel) → joint_type = "revolute", axis = axis of rotation.
4. motion_axis MUST be exactly one character: "X", "Y", or "Z".
5. For every revolute joint, estimate realistic rotation limits in degrees:
   - lower_limit_deg: minimum angle in degrees (often 0.0 for one-directional motion).
   - upper_limit_deg: maximum angle in degrees (e.g. scissors ≈ 60°, door hinge ≈ 90°, elbow ≈ 145°).
   For every prismatic joint, estimate realistic travel limits in METERS (not degrees):
   - lower_limit_deg: minimum travel in meters (e.g. -0.05 for a vise jaw that opens 5 cm).
   - upper_limit_deg: maximum travel in meters (e.g. 0.0 when fully closed is the reference).
   - Set both to null only for fixed joints or when limits are genuinely unknown.

OUTPUT FORMAT (raw JSON, nothing else):
{"base_link_name": "<link_name>", "rigid_links": [{"link_name": "<name>", "constituent_parts": ["<part1>", ...]}], "joints": [{"parent_link": "<link_name>", "child_link": "<link_name>", "joint_type": "<fixed|revolute|prismatic>", "motion_axis": "<X|Y|Z>", "lower_limit_deg": <float_or_null>, "upper_limit_deg": <float_or_null>, "reasoning": "<one sentence>"}]}"""

_USER_TEMPLATE = """\
Infer the kinematic tree for this multi-part mechanical object using two-stage reasoning.

Object label: {object_label}

Raw parts and their bounding box dimensions:
{parts_description}

STAGE 1: Group parts into rigid sub-assemblies.
STAGE 2: Define joints between the groups.

Output JSON with exactly this structure:
{{"base_link_name": "<link_name>", "rigid_links": [{{"link_name": "<name>", "constituent_parts": ["<raw_part_name>", ...]}}], "joints": [{{"parent_link": "<link_name>", "child_link": "<link_name>", "joint_type": "<fixed|revolute|prismatic>", "motion_axis": "<X|Y|Z>", "lower_limit_deg": <float_or_null>, "upper_limit_deg": <float_or_null>, "reasoning": "..."}}]}}"""


# ---------------------------------------------------------------------------
# Module-level client singleton
# ---------------------------------------------------------------------------

_client: object | None = None


def _get_client() -> object | None:
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        logger.debug("articulation_inference: anthropic package not installed")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_kinematic_topology(
    image_paths: list[str],
    parts_metadata: dict[str, dict],
    object_label: str,
    model: str = _DEFAULT_MODEL,
) -> ArticulationTopology | None:
    """Infer the kinematic tree of a multi-part object using a VLM.

    Args:
        image_paths: Paths to rendered images of the assembled object (may be empty).
        parts_metadata: Dict mapping part name → dict with optional keys:
            'bbox_m' (tuple of 3 floats in metres), 'volume_m3' (float).
        object_label: Human-readable semantic label for the whole assembly
            (e.g. "industrial_vise", "robotic_arm").
        model: Claude model ID to use.

    Returns:
        An ArticulationTopology Pydantic model, or None on failure.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.debug("articulation_inference: ANTHROPIC_API_KEY not set, skipping")
        return None

    client = _get_client()
    if client is None:
        return None

    # Build human-readable part descriptions for the prompt
    lines: list[str] = []
    for part_name, meta in parts_metadata.items():
        bbox = meta.get("bbox_m")
        vol = meta.get("volume_m3")
        if bbox is not None:
            bbox_str = f"{bbox[0]*1000:.1f} x {bbox[1]*1000:.1f} x {bbox[2]*1000:.1f} mm"
        else:
            bbox_str = "unknown"
        if vol is not None:
            vol_str = f"{vol:.3e} m³"
        else:
            vol_str = "unknown"
        lines.append(f"  - {part_name}: bbox={bbox_str}, volume={vol_str}")
    parts_description = "\n".join(lines) if lines else "  (no metadata provided)"

    user_msg = _USER_TEMPLATE.format(
        object_label=object_label,
        parts_description=parts_description,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()

        # Strip markdown fences (Haiku/Sonnet sometimes wraps JSON)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        topology = ArticulationTopology.model_validate(data)
    except json.JSONDecodeError as exc:
        logger.warning("articulation_inference JSON parse error for '%s': %s", object_label, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("articulation_inference API error for '%s': %s", object_label, exc)
        return None

    logger.info(
        "articulation_inference: '%s' → base_link_name='%s', %d link(s), %d joint(s)",
        object_label, topology.base_link_name, len(topology.rigid_links), len(topology.joints),
    )
    return topology


def clear_articulation_client() -> None:
    """Reset the client singleton — used in tests to ensure a clean state."""
    global _client
    _client = None
