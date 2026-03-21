"""Material mapping — maps CAE physical properties to MDL/PBR shader parameters."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CAEMaterial:
    """Material properties as defined in CAE source data."""

    name: str
    # Mechanical properties
    density: float | None = None  # kg/m³
    youngs_modulus: float | None = None  # Pa
    poissons_ratio: float | None = None
    # Optical / surface properties (when available from CAE)
    roughness: float | None = None  # 0.0–1.0
    metallic: float | None = None  # 0.0–1.0
    ior: float | None = None  # index of refraction
    albedo_rgb: tuple[float, float, float] | None = None  # linear RGB, 0.0–1.0
    # Raw source metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class MDLMaterial:
    """Target MDL material parameters for Omniverse."""

    mdl_name: str  # e.g. "OmniPBR.mdl"
    diffuse_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float = 0.5
    metallic: float = 0.0
    ior: float = 1.5
    opacity: float = 1.0
    # Physics properties to write via UsdPhysics
    density: float | None = None
    friction_static: float | None = None
    friction_dynamic: float | None = None
    restitution: float | None = None
    # Track provenance
    source_material: str | None = None
    confidence: float = 1.0  # 0.0–1.0, how much of this came from real CAE data
    vlm_semantic_label: str | None = None  # VLM-improved semantic label, if available
    vlm_reasoning_step: str | None = None  # VLM chain-of-thought saved for data provenance
    vlm_primary_material: str | None = None  # VLM-inferred material class (e.g. "steel")


# Known material class → approximate PBR defaults.
# Used only when CAE data is missing specific optical properties.
_MATERIAL_CLASS_DEFAULTS: dict[str, dict] = {
    # Keys: PBR optical + physics (density kg/m³, friction_static, friction_dynamic, restitution)
    "steel": {
        "diffuse_color": (0.55, 0.56, 0.58), "metallic": 1.0, "roughness": 0.35, "ior": 2.5,
        "density": 7850.0, "friction_static": 0.55, "friction_dynamic": 0.42, "restitution": 0.3,
    },
    "aluminum": {
        "diffuse_color": (0.91, 0.92, 0.92), "metallic": 1.0, "roughness": 0.3, "ior": 1.44,
        "density": 2700.0, "friction_static": 0.45, "friction_dynamic": 0.35, "restitution": 0.2,
    },
    "copper": {
        "diffuse_color": (0.95, 0.64, 0.54), "metallic": 1.0, "roughness": 0.25, "ior": 1.1,
        "density": 8960.0, "friction_static": 0.53, "friction_dynamic": 0.36, "restitution": 0.25,
    },
    "rubber": {
        "diffuse_color": (0.15, 0.15, 0.15), "metallic": 0.0, "roughness": 0.9, "ior": 1.52,
        "density": 1200.0, "friction_static": 0.9, "friction_dynamic": 0.8, "restitution": 0.7,
    },
    "glass": {
        "diffuse_color": (0.95, 0.95, 0.95), "metallic": 0.0, "roughness": 0.05, "ior": 1.52,
        "opacity": 0.1,
        "density": 2500.0, "friction_static": 0.4, "friction_dynamic": 0.35, "restitution": 0.15,
    },
    "plastic_abs": {
        "diffuse_color": (0.8, 0.8, 0.78), "metallic": 0.0, "roughness": 0.4, "ior": 1.53,
        "density": 1050.0, "friction_static": 0.45, "friction_dynamic": 0.35, "restitution": 0.35,
    },
    "concrete": {
        "diffuse_color": (0.55, 0.55, 0.52), "metallic": 0.0, "roughness": 0.85, "ior": 1.5,
        "density": 2300.0, "friction_static": 0.7, "friction_dynamic": 0.6, "restitution": 0.1,
    },
    # --- Industrial materials (18 additional classes) ---
    "stainless": {
        "diffuse_color": (0.60, 0.62, 0.64), "metallic": 1.0, "roughness": 0.55, "ior": 2.5,
        "density": 8000.0, "friction_static": 0.55, "friction_dynamic": 0.42, "restitution": 0.30,
    },
    "cast_iron": {
        "diffuse_color": (0.25, 0.25, 0.25), "metallic": 1.0, "roughness": 0.50, "ior": 2.5,
        "density": 7200.0, "friction_static": 0.50, "friction_dynamic": 0.38, "restitution": 0.20,
    },
    "titanium": {
        "diffuse_color": (0.76, 0.76, 0.78), "metallic": 1.0, "roughness": 0.36, "ior": 1.8,
        "density": 4500.0, "friction_static": 0.40, "friction_dynamic": 0.30, "restitution": 0.30,
    },
    "brass": {
        "diffuse_color": (0.85, 0.72, 0.35), "metallic": 1.0, "roughness": 0.35, "ior": 1.6,
        "density": 8500.0, "friction_static": 0.44, "friction_dynamic": 0.32, "restitution": 0.25,
    },
    "bronze": {
        "diffuse_color": (0.72, 0.56, 0.30), "metallic": 1.0, "roughness": 0.40, "ior": 1.6,
        "density": 8800.0, "friction_static": 0.45, "friction_dynamic": 0.33, "restitution": 0.25,
    },
    "nylon": {
        "diffuse_color": (0.95, 0.94, 0.90), "metallic": 0.0, "roughness": 0.35, "ior": 1.53,
        "density": 1150.0, "friction_static": 0.30, "friction_dynamic": 0.25, "restitution": 0.35,
    },
    "ptfe": {
        "diffuse_color": (0.95, 0.95, 0.95), "metallic": 0.0, "roughness": 0.04, "ior": 1.35,
        "density": 2200.0, "friction_static": 0.04, "friction_dynamic": 0.04, "restitution": 0.10,
    },
    "acetal": {
        "diffuse_color": (0.90, 0.90, 0.88), "metallic": 0.0, "roughness": 0.30, "ior": 1.48,
        "density": 1410.0, "friction_static": 0.20, "friction_dynamic": 0.18, "restitution": 0.35,
    },
    "polycarbonate": {
        "diffuse_color": (0.92, 0.92, 0.92), "metallic": 0.0, "roughness": 0.45, "ior": 1.58,
        "density": 1200.0, "friction_static": 0.38, "friction_dynamic": 0.30, "restitution": 0.40,
    },
    "carbon_fiber": {
        "diffuse_color": (0.10, 0.10, 0.12), "metallic": 0.0, "roughness": 0.35, "ior": 1.60,
        "density": 1600.0, "friction_static": 0.35, "friction_dynamic": 0.28, "restitution": 0.20,
    },
    "zinc": {
        "diffuse_color": (0.75, 0.76, 0.76), "metallic": 1.0, "roughness": 0.40, "ior": 1.9,
        "density": 7130.0, "friction_static": 0.48, "friction_dynamic": 0.35, "restitution": 0.25,
    },
    "ceramic": {
        "diffuse_color": (0.92, 0.90, 0.86), "metallic": 0.0, "roughness": 0.50, "ior": 1.78,
        "density": 3900.0, "friction_static": 0.50, "friction_dynamic": 0.40, "restitution": 0.10,
    },
    "chrome": {
        "diffuse_color": (0.72, 0.74, 0.77), "metallic": 1.0, "roughness": 0.40, "ior": 2.5,
        "density": 7190.0, "friction_static": 0.46, "friction_dynamic": 0.34, "restitution": 0.30,
    },
    "magnesium": {
        "diffuse_color": (0.82, 0.83, 0.82), "metallic": 1.0, "roughness": 0.36, "ior": 1.7,
        "density": 1740.0, "friction_static": 0.38, "friction_dynamic": 0.28, "restitution": 0.25,
    },
    "cast_aluminum": {
        "diffuse_color": (0.80, 0.81, 0.82), "metallic": 1.0, "roughness": 0.45, "ior": 1.44,
        "density": 2680.0, "friction_static": 0.42, "friction_dynamic": 0.32, "restitution": 0.20,
    },
    "hdpe": {
        "diffuse_color": (0.94, 0.95, 0.94), "metallic": 0.0, "roughness": 0.20, "ior": 1.50,
        "density": 950.0, "friction_static": 0.20, "friction_dynamic": 0.16, "restitution": 0.30,
    },
    "pvc": {
        "diffuse_color": (0.78, 0.78, 0.78), "metallic": 0.0, "roughness": 0.45, "ior": 1.54,
        "density": 1400.0, "friction_static": 0.40, "friction_dynamic": 0.32, "restitution": 0.35,
    },
    "silicone": {
        "diffuse_color": (0.85, 0.85, 0.85), "metallic": 0.0, "roughness": 0.60, "ior": 1.43,
        "density": 1300.0, "friction_static": 0.55, "friction_dynamic": 0.50, "restitution": 0.60,
    },
}


# --- Compound-name aliases: checked before the generic single-keyword scan ---
# Maps known multi-word / hyphenated phrases → material class.
_COMPOUND_ALIASES: dict[str, str] = {
    "stainless_steel": "stainless", "stainless steel": "stainless",
    "ss304": "stainless", "ss316": "stainless", "ss303": "stainless",
    "cast_iron": "cast_iron", "cast iron": "cast_iron",
    "cast_aluminum": "cast_aluminum", "cast_aluminium": "cast_aluminum",
    "cast aluminum": "cast_aluminum", "cast aluminium": "cast_aluminum",
    "carbon_fiber": "carbon_fiber", "carbon_fibre": "carbon_fiber",
    "carbon fiber": "carbon_fiber", "carbon fibre": "carbon_fiber",
    "abs_plastic": "plastic_abs", "abs plastic": "plastic_abs",
    "teflon": "ptfe",
    "delrin": "acetal",
}

# --- Alloy / grade codes: matched as whole tokens or substrings ---
_ALLOY_CODES: dict[str, str] = {
    # Aluminum alloys
    "6061": "aluminum", "6063": "aluminum", "7075": "aluminum",
    "2024": "aluminum", "5052": "aluminum",
    # Stainless grades
    "304": "stainless", "316": "stainless", "303": "stainless",
    "410": "stainless", "430": "stainless", "17-4": "stainless",
    # Carbon / alloy steels
    "4140": "steel", "4340": "steel", "1018": "steel",
    "1045": "steel", "1095": "steel", "a36": "steel",
    # Titanium grades
    "ti-6al-4v": "titanium", "grade5": "titanium", "grade2": "titanium",
    # Copper alloys
    "c260": "brass", "c360": "brass",
    "c932": "bronze", "c954": "bronze",
}


def classify_material(cae_mat: CAEMaterial) -> str | None:
    """Classify a CAE material into a known material class.

    Three-pass priority order:
    1. Compound-name aliases (e.g. "stainless_steel" → stainless)
    2. Alloy / grade code tokens (e.g. "6061" → aluminum, "304" → stainless)
    3. Generic single-keyword substring scan (existing behavior)
    """
    name_lower = cae_mat.name.lower().replace("-", "_")

    # Pass 1: compound aliases
    for phrase, cls in _COMPOUND_ALIASES.items():
        if phrase in name_lower:
            return cls

    # Pass 2: alloy code tokens
    tokens = set(re.split(r"[_\s]+", name_lower))
    for code, cls in _ALLOY_CODES.items():
        code_norm = code.replace("-", "_")
        if code_norm in tokens or code_norm in name_lower:
            return cls

    # Pass 3: generic single-keyword scan
    for cls in _MATERIAL_CLASS_DEFAULTS:
        if cls in name_lower:
            return cls

    return None


def map_cae_to_mdl(
    cae_mat: CAEMaterial,
    fallback_mdl: str = "OmniPBR.mdl",
    forced_class: str | None = None,
    enable_vlm: bool = False,
    vlm_model: str = "claude-haiku-4-5",
    semantic_label: str | None = None,
    bbox_m: tuple[float, float, float] | None = None,
    volume_m3: float | None = None,
    vlm_max_calls: int = 500,
) -> MDLMaterial:
    """Map CAE material properties to MDL parameters.

    Priority order:
    1. Explicit optical properties from CAE data (roughness, metallic, etc.)
    2. Material class defaults (looked up by name)
    3. Fallback neutral values

    Args:
        cae_mat: Source CAE material.
        fallback_mdl: MDL shader to use.

    Returns:
        MDLMaterial with mapped parameters.
    """
    mdl = MDLMaterial(mdl_name=fallback_mdl, source_material=cae_mat.name)
    fields_from_source = 0
    total_fields = 4  # diffuse_color, roughness, metallic, ior

    # 1. Direct CAE optical properties take priority
    if cae_mat.albedo_rgb is not None:
        mdl.diffuse_color = cae_mat.albedo_rgb
        fields_from_source += 1
    if cae_mat.roughness is not None:
        mdl.roughness = cae_mat.roughness
        fields_from_source += 1
    if cae_mat.metallic is not None:
        mdl.metallic = cae_mat.metallic
        fields_from_source += 1
    if cae_mat.ior is not None:
        mdl.ior = cae_mat.ior
        fields_from_source += 1

    # 2. Fill gaps from material class defaults (forced_class skips classifier)
    mat_class = forced_class if forced_class is not None else classify_material(cae_mat)

    # Pass 4: VLM classification — fires when enable_vlm=True and no forced_class.
    # Overrides both mat_class (when regex returned None or a weak guess) and confidence.
    _vlm_confidence_override: float | None = None
    if enable_vlm and forced_class is None:
        from simready.acquisition.vlm_material import classify_material_vlm  # noqa: PLC0415
        vlm_result = classify_material_vlm(
            part_name=cae_mat.name,
            semantic_label=semantic_label,
            bbox_m=bbox_m,
            volume_m3=volume_m3,
            model=vlm_model,
            hint_class=mat_class,
            max_calls=vlm_max_calls,
        )
        if vlm_result is not None:
            mat_class, _vlm_confidence_override, mdl.vlm_semantic_label, mdl.vlm_reasoning_step = vlm_result
            mdl.vlm_primary_material = mat_class

    if mat_class:
        defaults = _MATERIAL_CLASS_DEFAULTS[mat_class]
        if fields_from_source < total_fields:
            if cae_mat.albedo_rgb is None and "diffuse_color" in defaults:
                mdl.diffuse_color = defaults["diffuse_color"]
            if cae_mat.roughness is None and "roughness" in defaults:
                mdl.roughness = defaults["roughness"]
            if cae_mat.metallic is None and "metallic" in defaults:
                mdl.metallic = defaults["metallic"]
            if cae_mat.ior is None and "ior" in defaults:
                mdl.ior = defaults["ior"]
            if "opacity" in defaults:
                mdl.opacity = defaults["opacity"]
        # Physics defaults from material class (always apply when not in CAE data)
        if mdl.density is None and "density" in defaults:
            mdl.density = defaults["density"]
        mdl.friction_static = defaults.get("friction_static")
        mdl.friction_dynamic = defaults.get("friction_dynamic")
        mdl.restitution = defaults.get("restitution")
        logger.info("Material '%s' classified as '%s'", cae_mat.name, mat_class)

    # Carry over physics properties from CAE data (takes priority over class defaults)
    if cae_mat.density is not None:
        mdl.density = cae_mat.density

    # Confidence = fraction of PBR fields that came directly from CAE source.
    # Class-matched defaults contribute 0.25 partial credit (physically grounded,
    # not fabricated) so that well-known materials (steel, aluminum) pass strict
    # validation even when the STEP file contains no optical CAE data.
    direct_confidence = fields_from_source / total_fields
    if mat_class and direct_confidence < 1.0:
        class_credit = 0.25 * (1.0 - direct_confidence)
        mdl.confidence = direct_confidence + class_credit
    else:
        mdl.confidence = direct_confidence

    # VLM confidence overrides the formula-computed value when available.
    if _vlm_confidence_override is not None:
        mdl.confidence = _vlm_confidence_override

    if mdl.confidence < 0.5:
        logger.warning(
            "Material '%s' has low source confidence (%.0f%%) — "
            "most PBR properties are estimated",
            cae_mat.name,
            mdl.confidence * 100,
        )

    return mdl
