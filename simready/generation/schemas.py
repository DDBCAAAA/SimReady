"""Pydantic models for the procedural generation pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Component(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    shape_hint: str  # box, cylinder, cone, sphere, extrusion, compound
    dimensions_m: dict[str, float]  # {"length": 0.15, "width": 0.10, "height": 0.02}
    position_m: tuple[float, float, float]  # relative to assembly origin
    material_hint: str | None = None


class JointSpec(BaseModel):
    """Informational only — the CadQuery backend ignores joints.
    Fields are optional and extra keys are silently dropped so that
    LLM output variations (e.g. 'type' vs 'joint_type') don't crash."""
    model_config = ConfigDict(extra="ignore")

    parent: str = ""
    child: str = ""
    joint_type: str = "fixed"  # fixed | revolute | prismatic
    axis: str = "Z"
    limits_deg: tuple[float, float] | None = None


class Blueprint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str
    components: list[Component]
    joints: list[JointSpec] = []
    overall_dimensions_m: dict[str, float]  # bounding box: length, width, height


class CriticFeedback(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    issues: list[str] = []
    corrections: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


class GenerationResult(BaseModel):
    prompt: str
    blueprint: Blueprint | None = None
    cadquery_code: str | None = None
    step_path: str | None = None
    usd_path: str | None = None
    iterations: int = 0
    inner_retries: int = 0
    critic_history: list[CriticFeedback] = []
    success: bool = False
    error: str | None = None
    pipeline_summary: dict | None = None
