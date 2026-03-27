"""VLM Critic: evaluates rendered geometry against blueprint, returns PASS/FAIL.

Sends up to N PNG screenshots to the Claude vision API with the blueprint
and original prompt.  Returns a structured CriticFeedback with actionable
CadQuery-level corrections on FAIL.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

from simready.generation.schemas import Blueprint, CriticFeedback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic client singleton (same pattern as vlm_material.py)
# ---------------------------------------------------------------------------

_client: object | None = None


def _get_client() -> object:
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package not installed — run: pip install anthropic"
        ) from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a 3D geometry quality inspector evaluating CadQuery-generated shapes \
for SimReady physics simulation in NVIDIA Isaac Lab.

Your job: compare the rendered views of a 3D shape against its blueprint \
specification and decide whether it PASSES or FAILS.

Evaluation criteria:
1. All components listed in the blueprint are visibly present.
2. Proportions roughly match overall_dimensions_m (allow ±30% tolerance).
3. No interpenetrating or floating geometry.
4. No obviously missing major features.
5. Shape appears suitable for rigid-body simulation (no major holes or artifacts).

Be strict: FAIL if a component is absent or proportions are wrong by >30%.
PASS if the shape is a reasonable realization of the prompt.

Output a JSON object with EXACTLY these keys:
{
  "verdict": "PASS" or "FAIL",
  "issues": ["<concise description of each problem>"],
  "corrections": ["<actionable CadQuery-level fix for each issue>"],
  "confidence": <float 0.0-1.0 representing your certainty in the verdict>
}

No markdown fences. Only raw JSON."""

_CRITIC_USER = """\
Evaluate the rendered 3D shape shown in the image(s) above.

Original design prompt: {prompt}

Blueprint specification:
{blueprint_json}

Does the rendered shape correctly implement the blueprint? \
Apply the evaluation criteria from the system prompt and return your verdict as JSON."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _encode_image(path: Path) -> str:
    """Base64-encode a PNG file."""
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def critique_geometry(
    images: list[Path],
    prompt: str,
    blueprint: Blueprint,
    model: str = "claude-opus-4-6",
) -> CriticFeedback:
    """Send rendered images + blueprint to the VLM and return PASS/FAIL feedback.

    Args:
        images:    List of PNG screenshot paths (typically 2 views).
        prompt:    Original text prompt used for generation.
        blueprint: Frozen Blueprint from the planning step.
        model:     Anthropic model ID to use.

    Returns:
        CriticFeedback with verdict, issues, corrections, and confidence.
    """
    if not images:
        logger.warning("Critic called with no images — returning FAIL.")
        return CriticFeedback(
            verdict="FAIL",
            issues=["No rendered images were provided for evaluation."],
            corrections=["Ensure the executor produces a valid STL and the renderer succeeds."],
            confidence=0.0,
        )

    client = _get_client()

    # Build multi-modal message content
    content: list[dict] = []
    for img_path in images:
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            continue
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_image(img_path),
            },
        })

    if not content:
        return CriticFeedback(
            verdict="FAIL",
            issues=["All rendered image files were missing."],
            corrections=["Check that the renderer wrote PNG files to the expected paths."],
            confidence=0.0,
        )

    content.append({
        "type": "text",
        "text": _CRITIC_USER.format(
            prompt=prompt,
            blueprint_json=json.dumps(blueprint.model_dump(), indent=2),
        ),
    })

    logger.info("Calling VLM critic (%s) with %d image(s)", model, len(content) - 1)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_CRITIC_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "").strip()
    # Strip markdown fences (same pattern as vlm_material.py)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Critic returned non-JSON: %s\nRaw: %s", exc, raw[:400])
        return CriticFeedback(
            verdict="FAIL",
            issues=[f"Critic LLM returned unparseable response: {exc}"],
            corrections=["Retry generation — this may be a transient API issue."],
            confidence=0.0,
        )

    try:
        feedback = CriticFeedback.model_validate(data)
    except Exception as exc:
        logger.error("CriticFeedback validation failed: %s\nData: %s", exc, data)
        return CriticFeedback(
            verdict="FAIL",
            issues=[f"Could not parse critic response: {exc}"],
            corrections=[],
            confidence=0.0,
        )

    logger.info(
        "Critic verdict: %s (confidence=%.2f, issues=%d)",
        feedback.verdict, feedback.confidence, len(feedback.issues),
    )
    return feedback
