"""VLM-based material classification and semantic labeling using Claude API.

Uses a text-only prompt (part name, semantic label hint, bounding box, volume)
to infer physical material class and a precise SimReady semantic label in one
API call. The model also produces a `reasoning_step` explaining its choice,
which is logged for debugging and audit purposes.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator

# Derive valid classes from the single source of truth: _MATERIAL_CLASS_DEFAULTS.
# Importing here (not lazy) because _VALID_CLASSES is needed at module load time.
# material_map.py only imports from this module inside a function (lazy), so no circular import.
from simready.materials.material_map import _MATERIAL_CLASS_DEFAULTS
from simready.semantics.classifier import _TAXONOMY

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5"

_VALID_CLASSES: frozenset[str] = frozenset(_MATERIAL_CLASS_DEFAULTS)
# Precomputed once — used in every prompt, never changes at runtime.
_VALID_CLASSES_STR: str = ", ".join(sorted(_VALID_CLASSES))

# Valid semantic labels derived from taxonomy — single source of truth.
_VALID_SEMANTIC_LABELS: frozenset[str] = (
    frozenset(label for label, _ in _TAXONOMY) | {"industrial_part:component"}
)
_VALID_SEMANTIC_LABELS_STR: str = ", ".join(sorted(_VALID_SEMANTIC_LABELS))


# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------

class PhysicsMetadata(BaseModel):
    """Structured output from the VLM material classifier."""

    reasoning_step: str = Field(
        default="",
        description="Brief explanation of why this material was chosen based on scale and visuals.",
    )
    material_class: str = Field(
        description="One of the valid SimReady material classes."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score 0.0–1.0. Standard industrial parts should be > 0.85."
    )
    semantic_label: str = Field(
        description="One of the valid SimReady semantic labels."
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert Materials Scientist and Embodied AI Physics Engineer creating Sim-Ready assets for NVIDIA Isaac Lab.
Your task is to infer the exact physical material and properties of the provided object.

CRITICAL PHYSICAL CONTEXT:
You are provided with the exact bounding box dimensions (in meters) and the volume of the object.
- Use the dimensions to judge the scale! A 0.04m (4cm) flange is a small pipe fitting, likely made of dense steel or brass. A 2.0m flange would be massive industrial infrastructure.
- Industrial parts (gears, sprockets, flanges, screws) with metallic visuals MUST be confidently assigned standard industrial metals (e.g., "steel", "cast_iron", "aluminum") rather than generic terms.

INSTRUCTIONS:
1. Analyze the visual texture (shiny, matte, rusty, plastic-like).
2. Analyze the Semantic Label and Bounding Box dimensions to understand its real-world industrial application.
3. Write a brief `reasoning_step` explaining WHY you chose this material based on its scale and visuals.
4. Because you have exact dimensions and clear visuals, BE CONFIDENT. For standard industrial parts, your `materialConfidence` should be > 0.85. Only use low confidence (< 0.8) if the image is highly ambiguous or physically impossible.
5. Output strict JSON matching the required schema."""

_USER_TEMPLATE = """\
Infer the physical material and semantic category for this mechanical part.

Part name: {part_name}
Keyword-based label hint: {semantic_label}
Bounding box (mm): {bbox_str}
Volume: {volume_str}
Regex material hint (may be None): {hint_class}

Valid material classes:
{valid_classes}

Valid semantic labels:
{valid_semantic_labels}

Output JSON with exactly these four keys in this order:
{{"reasoning_step": "<brief reasoning based on scale and part name>", "material_class": "<one of the valid material classes>", "confidence": <float 0.0-1.0>, "semantic_label": "<one of the valid semantic labels>"}}"""

# ---------------------------------------------------------------------------
# Module-level client singleton and call counter
# ---------------------------------------------------------------------------

_client: object | None = None
_call_count: int = 0


def _get_client() -> object | None:
    """Return (or create) the shared Anthropic client. Returns None if not available."""
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        logger.debug("VLM classify skipped: anthropic package not installed")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    _client = anthropic.Anthropic(api_key=api_key)
    return _client


@lru_cache(maxsize=512)
def _cached_vlm_classify(
    part_name: str,
    semantic_label: str,
    bbox_str: str,
    volume_str: str,
    hint_class: str,
    model: str,
) -> tuple[str, float, str | None, str] | None:
    """Inner cached function — all args must be hashable (strings).

    Only called when ANTHROPIC_API_KEY is confirmed set (checked in classify_material_vlm).
    Returns (material_class, confidence, semantic_label, reasoning_step) or None on failure.
    """
    global _call_count
    _call_count += 1
    client = _get_client()
    if client is None:
        return None  # anthropic package not installed

    user_msg = _USER_TEMPLATE.format(
        part_name=part_name,
        semantic_label=semantic_label or "unknown",
        bbox_str=bbox_str,
        volume_str=volume_str,
        hint_class=hint_class or "None",
        valid_classes=_VALID_CLASSES_STR,
        valid_semantic_labels=_VALID_SEMANTIC_LABELS_STR,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()
        # Strip markdown fences if the model wrapped the JSON (e.g. Haiku does this)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        metadata = PhysicsMetadata.model_validate(data)
    except json.JSONDecodeError as exc:
        logger.warning("VLM JSON parse error for '%s': %s", part_name, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("VLM API error for '%s': %s", part_name, exc)
        return None

    mat_class = metadata.material_class.strip()
    confidence = metadata.confidence
    sem_label = metadata.semantic_label.strip() or None

    if mat_class not in _VALID_CLASSES:
        logger.warning(
            "VLM returned unknown material class '%s' for '%s'",
            mat_class, part_name,
        )
        return None

    # Discard semantic label if not in taxonomy (fall back to keyword classifier)
    if sem_label and sem_label not in _VALID_SEMANTIC_LABELS:
        logger.debug("VLM returned unknown semantic label '%s' for '%s', ignoring", sem_label, part_name)
        sem_label = None

    logger.info(
        "VLM classified '%s' → '%s' (confidence=%.2f, label='%s') | reasoning: %s",
        part_name, mat_class, confidence, sem_label or "n/a", metadata.reasoning_step,
    )
    return mat_class, confidence, sem_label, metadata.reasoning_step


def classify_material_vlm(
    part_name: str,
    semantic_label: str | None = None,
    bbox_m: tuple[float, float, float] | None = None,
    volume_m3: float | None = None,
    model: str = _DEFAULT_MODEL,
    hint_class: str | None = None,
    max_calls: int = 500,
) -> tuple[str, float, str | None, str] | None:
    """Classify material and semantic label using a VLM (Claude API).

    Args:
        part_name: Body/part name from the STEP file.
        semantic_label: Keyword-based label hint from the local classifier.
        bbox_m: Bounding box extents in meters (x, y, z).
        volume_m3: Mesh volume in cubic meters (from trimesh, for watertight meshes).
        model: Claude model ID to use.
        hint_class: Material class from regex pass (may be None).
        max_calls: Hard cap on API calls per process (cache hits don't count).

    Returns:
        (material_class, confidence, semantic_label, reasoning_step) tuple, or None on failure/limit reached.
        semantic_label may be None if the VLM response was invalid.
    """
    # Guard here (not inside cached fn) so a missing key doesn't poison the cache.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.debug("VLM classify skipped: ANTHROPIC_API_KEY not set")
        return None
    if _call_count >= max_calls:
        logger.debug(
            "VLM call limit reached (%d/%d), skipping '%s'",
            _call_count, max_calls, part_name,
        )
        return None

    if bbox_m is not None:
        # Convert meters → mm for human-readable prompt
        bbox_str = f"{bbox_m[0]*1000:.1f} x {bbox_m[1]*1000:.1f} x {bbox_m[2]*1000:.1f} mm"
    else:
        bbox_str = "unknown"

    if volume_m3 is not None:
        volume_str = f"{volume_m3:.3e} m³"
    else:
        volume_str = "unknown"

    return _cached_vlm_classify(
        part_name=part_name,
        semantic_label=semantic_label or "",
        bbox_str=bbox_str,
        volume_str=volume_str,
        hint_class=hint_class or "",
        model=model,
    )


def clear_vlm_cache() -> None:
    """Clear the LRU cache, client singleton, and call counter — used in tests to reset state."""
    global _client, _call_count
    _cached_vlm_classify.cache_clear()
    _client = None
    _call_count = 0
