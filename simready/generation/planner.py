"""Planner+Coder LLM agent: text prompt → Blueprint + CadQuery code.

Single LLM call for the first iteration (generates both blueprint and code).
Subsequent calls receive a frozen blueprint and critic feedback — only the
code is revised.  The same function handles both cases, dispatching on
whether *blueprint* is already provided.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from simready.generation.cadquery_reference import CADQUERY_CHEAT_SHEET
from simready.generation.schemas import Blueprint, CriticFeedback

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic client singleton (mirrors vlm_material.py pattern)
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
# Prompt templates
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are an expert CadQuery engineer and 3D geometry designer creating \
simulation-ready mechanical parts for NVIDIA Isaac Lab.

Your task: given a natural-language description, design a complete \
3D object using CadQuery 2.x Python code.

*** CRITICAL UNIT RULE ***
CadQuery / OCC uses MILLIMETERS internally. The SimReady pipeline reads
the STEP file unit header (mm) and automatically converts to meters.
Therefore: ALL values in your CadQuery CODE must be in MILLIMETERS.
  Blueprint overall_dimensions_m uses meters (for documentation).
  Code uses mm: 0.24 m → 240, 0.05 m → 50, 0.003 m → 3.

OTHER RULES:
- Assign the final shape to the variable `result`.
- Do NOT call any export functions — the executor handles that.
- Code must be self-contained and runnable with `import cadquery as cq`.
- Use realistic real-world proportions.

{cheat_sheet}"""

_FIRST_CALL_USER = """\
Design and implement the following 3D object:

Prompt: {prompt}

UNIT REMINDER: blueprint uses meters; CadQuery code uses MILLIMETERS (×1000).
Example: 0.24 m overall length → blueprint "length": 0.24, code box(240, ...).

Return a JSON object with EXACTLY these two top-level keys:

{{
  "blueprint": {{
    "description": "<one-sentence summary>",
    "components": [
      {{
        "name": "<component_name>",
        "shape_hint": "<box|cylinder|cone|sphere|extrusion|compound>",
        "dimensions_m": {{"<dim_key>": <float_meters>, ...}},
        "position_m": [<x>, <y>, <z>],
        "material_hint": "<optional_material_or_null>"
      }}
    ],
    "joints": [],
    "overall_dimensions_m": {{"length": <m>, "width": <m>, "height": <m>}}
  }},
  "code": "import cadquery as cq\\n..."
}}

No markdown fences. Only the raw JSON object."""

_REVISION_USER = """\
The previous CadQuery code was reviewed and needs corrections.

Original prompt: {prompt}

Blueprint (FROZEN — do not change the design intent):
{blueprint_json}

Previous code:
{code}

Critic feedback:
Issues:
{issues}

Corrections to apply:
{corrections}

Return a JSON object with EXACTLY one key:
{{"code": "import cadquery as cq\\n...corrected code..."}}

No markdown fences. Only the raw JSON object."""

_TRACEBACK_SYSTEM = """\
You are an expert CadQuery 2.x Python debugger.
Fix the broken CadQuery script.  Return corrected, complete, runnable code.

RULES:
1. Final shape must be assigned to `result`.
2. Do NOT call any export functions.
3. All dimensions in meters.

{cheat_sheet}"""

_TRACEBACK_USER = """\
This CadQuery script raised an error. Fix it.

Code:
{code}

Error / Traceback:
{traceback}

Return a JSON object with EXACTLY one key:
{{"code": "import cadquery as cq\\n...fixed code..."}}

No markdown fences. Only the raw JSON object."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences (``` or ```json) — same pattern as vlm_material.py."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _extract_text(response: object) -> str:
    return next((b.text for b in response.content if b.type == "text"), "").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_and_code(
    prompt: str,
    model: str = "claude-opus-4-6",
    previous_code: str | None = None,
    critic_feedback: CriticFeedback | None = None,
    blueprint: Blueprint | None = None,
) -> tuple[Blueprint, str]:
    """Generate or revise a Blueprint + CadQuery code string.

    First call (blueprint=None):  returns a new Blueprint and code.
    Revision call (blueprint set): returns the same Blueprint and revised code.
    """
    client = _get_client()
    system = _PLANNER_SYSTEM.format(cheat_sheet=CADQUERY_CHEAT_SHEET)

    is_revision = blueprint is not None and previous_code is not None

    if is_revision:
        issues_text = "\n".join(f"- {i}" for i in (critic_feedback.issues if critic_feedback else []))
        corrections_text = "\n".join(
            f"- {c}" for c in (critic_feedback.corrections if critic_feedback else [])
        )
        user_msg = _REVISION_USER.format(
            prompt=prompt,
            blueprint_json=json.dumps(blueprint.model_dump(), indent=2),
            code=previous_code,
            issues=issues_text or "(none listed)",
            corrections=corrections_text or "(none listed)",
        )
    else:
        user_msg = _FIRST_CALL_USER.format(prompt=prompt)

    logger.info(
        "Calling LLM (%s) — %s", model, "revision" if is_revision else "first call"
    )
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = _strip_fences(_extract_text(response))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned non-JSON response: {exc}\nRaw output (truncated): {raw[:400]}"
        ) from exc

    code: str = data.get("code", "")
    if not code:
        raise RuntimeError("LLM response missing 'code' key.")

    if is_revision:
        return blueprint, code  # type: ignore[return-value]

    blueprint_data = data.get("blueprint")
    if not blueprint_data:
        raise RuntimeError("LLM response missing 'blueprint' key on first call.")
    parsed_blueprint = Blueprint.model_validate(blueprint_data)
    logger.info(
        "Blueprint: %d component(s), overall %s",
        len(parsed_blueprint.components),
        parsed_blueprint.overall_dimensions_m,
    )
    return parsed_blueprint, code


def revise_code(
    code: str,
    traceback: str,
    blueprint: Blueprint,
    model: str = "claude-opus-4-6",
) -> str:
    """Fix a CadQuery script given its traceback.  Returns the corrected code string."""
    client = _get_client()
    system = _TRACEBACK_SYSTEM.format(cheat_sheet=CADQUERY_CHEAT_SHEET)
    user_msg = _TRACEBACK_USER.format(code=code, traceback=traceback[:2000])

    logger.info("Asking LLM to fix traceback (%s)", model)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = _strip_fences(_extract_text(response))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned non-JSON for code fix: {exc}\nRaw: {raw[:400]}"
        ) from exc

    fixed = data.get("code", "")
    if not fixed:
        raise RuntimeError("LLM code-fix response missing 'code' key.")
    return fixed
