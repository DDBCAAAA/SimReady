"""VLM-based material classification and semantic labeling using Claude API.

Uses a text-only prompt (part name, semantic label hint, bounding box) to infer
physical material class and a precise SimReady semantic label in one API call.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

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

_SYSTEM_PROMPT = (
    "You are a mechanical engineering expert specializing in physical material "
    "identification and part classification from part names and geometry. "
    "You respond ONLY with a single JSON object — no markdown, no explanation. "
    "The JSON must have exactly three keys: "
    '"material_class" (string), "confidence" (float 0.0–1.0), '
    'and "semantic_label" (string).'
)

_USER_TEMPLATE = """\
Infer the physical material and semantic category for this mechanical part.

Part name: {part_name}
Keyword-based label hint: {semantic_label}
Bounding box (mm): {bbox_str}
Regex material hint (may be None): {hint_class}

Valid material classes:
{valid_classes}

Valid semantic labels:
{valid_semantic_labels}

Confidence guidelines:
- 0.90–0.95: part name contains a strong, unambiguous material indicator
  (e.g. "Hex_Socket_Screw_M3" → steel, "Aluminum_Bracket_6061" → aluminum)
- 0.70–0.89: name contains partial evidence (e.g. generic screw → likely steel)
- 0.50–0.69: weak evidence, educated guess
- below 0.50: unknown — use your best guess anyway

Respond with JSON only:
{{"material_class": "<one of the valid material classes>", "confidence": <float>, "semantic_label": "<one of the valid semantic labels>"}}"""

# Module-level Anthropic client singleton — created once per process, reused across all calls.
_client: object | None = None
# Counts actual API calls (cache hits don't increment). Reset by clear_vlm_cache().
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
    hint_class: str,
    model: str,
) -> tuple[str, float, str | None] | None:
    """Inner cached function — all args must be hashable (strings).

    Only called when ANTHROPIC_API_KEY is confirmed set (checked in classify_material_vlm).
    Returns (material_class, confidence, semantic_label) or None on failure.
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
        hint_class=hint_class or "None",
        valid_classes=_VALID_CLASSES_STR,
        valid_semantic_labels=_VALID_SEMANTIC_LABELS_STR,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
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
        mat_class = str(data.get("material_class", "")).strip()
        confidence = float(data.get("confidence", 0.0))
        sem_label = str(data.get("semantic_label", "")).strip() or None
    except json.JSONDecodeError as exc:
        logger.warning("VLM JSON parse error for '%s': %s", part_name, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("VLM API error for '%s': %s", part_name, exc)
        return None

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

    confidence = max(0.0, min(1.0, confidence))
    logger.info(
        "VLM classified '%s' → '%s' (confidence=%.2f, label='%s')",
        part_name, mat_class, confidence, sem_label or "n/a",
    )
    return mat_class, confidence, sem_label


def classify_material_vlm(
    part_name: str,
    semantic_label: str | None = None,
    bbox_m: tuple[float, float, float] | None = None,
    model: str = _DEFAULT_MODEL,
    hint_class: str | None = None,
    max_calls: int = 500,
) -> tuple[str, float, str | None] | None:
    """Classify material and semantic label using a VLM (Claude API).

    Args:
        part_name: Body/part name from the STEP file.
        semantic_label: Keyword-based label hint from the local classifier.
        bbox_m: Bounding box extents in meters (x, y, z).
        model: Claude model ID to use.
        hint_class: Material class from regex pass (may be None).
        max_calls: Hard cap on API calls per process (cache hits don't count).

    Returns:
        (material_class, confidence, semantic_label) tuple, or None on failure/limit reached.
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
        bbox_str = f"{bbox_m[0]*1000:.1f} x {bbox_m[1]*1000:.1f} x {bbox_m[2]*1000:.1f}"
    else:
        bbox_str = "unknown"

    return _cached_vlm_classify(
        part_name=part_name,
        semantic_label=semantic_label or "",
        bbox_str=bbox_str,
        hint_class=hint_class or "",
        model=model,
    )


def clear_vlm_cache() -> None:
    """Clear the LRU cache, client singleton, and call counter — used in tests to reset state."""
    global _client, _call_count
    _cached_vlm_classify.cache_clear()
    _client = None
    _call_count = 0
