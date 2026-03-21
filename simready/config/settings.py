"""Pipeline configuration loading and access."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"


@dataclass
class GeometrySettings:
    tessellation_tolerance: float = 0.001
    generate_lods: bool = True
    lod_levels: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])


@dataclass
class MaterialSettings:
    target_format: str = "mdl"
    fallback_mdl: str = "OmniPBR.mdl"
    enable_vlm: bool = False
    vlm_model: str = "claude-haiku-4-5"
    vlm_max_calls: int = 200


@dataclass
class ValidationSettings:
    enabled: bool = True
    strict: bool = True
    material_tolerance: float = 0.05
    enable_confidence_gate: bool = True  # set False to bypass quality gate in tests / dev


@dataclass
class PipelineSettings:
    up_axis: str = "Z"
    meters_per_unit: float = 1.0
    output_format: str = "usdc"
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    materials: MaterialSettings = field(default_factory=MaterialSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)


def load_settings(config_path: Path | None = None) -> PipelineSettings:
    """Load pipeline settings from YAML, falling back to defaults."""
    path = config_path or _DEFAULTS_PATH
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    pipeline_cfg = raw.get("pipeline", {})
    geo_cfg = raw.get("geometry", {})
    mat_cfg = raw.get("materials", {})
    val_cfg = raw.get("validation", {})

    return PipelineSettings(
        up_axis=pipeline_cfg.get("up_axis", "Z"),
        meters_per_unit=pipeline_cfg.get("meters_per_unit", 1.0),
        output_format=pipeline_cfg.get("output_format", "usdc"),
        geometry=GeometrySettings(**geo_cfg),
        materials=MaterialSettings(**mat_cfg),
        validation=ValidationSettings(
            enabled=val_cfg.get("enabled", True),
            strict=val_cfg.get("strict", True),
            material_tolerance=val_cfg.get("material_tolerance", 0.05),
            enable_confidence_gate=val_cfg.get("enable_confidence_gate", True),
        ),
    )
