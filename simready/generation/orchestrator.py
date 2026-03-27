"""State machine orchestrator for procedural 3D generation.

Flow:
  plan_and_code → [outer loop] → execute_with_retries → render_views
  → critique_geometry → PASS → pipeline.run() → SimReady .usda
                       → FAIL → plan_and_code (revise) → repeat

The existing physics backend (pipeline.run) is called unchanged on the
generated STEP file to inject CoACD collision hulls, mass properties, and
USD physics schemas.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

def _run_pipeline(
    result: object,
    step_path: Path,
    output_path: Path,
    config_path: Path | None,
    material_override: str | None,
    prompt: str,
) -> None:
    """Hand the STEP file to the physics backend and populate *result* in-place."""
    from simready.generation.schemas import GenerationResult  # noqa: PLC0415
    from simready.pipeline import run as pipeline_run          # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mat_overrides = {"*": material_override} if material_override else None
    asset_meta = {
        "aigc:model":         "cadquery-procedural",
        "aigc:prompt":        prompt,
        "aigc:iterations":    str(result.iterations),  # type: ignore[union-attr]
        "aigc:inner_retries": str(result.inner_retries),  # type: ignore[union-attr]
        "aigc:healer_version": "0.1.0",
    }

    summary = pipeline_run(
        step_path,
        output_path,
        config_path=config_path,
        material_overrides=mat_overrides,
        asset_metadata=asset_meta,
        disable_confidence_gate=True,
    )
    result.step_path        = str(step_path)   # type: ignore[union-attr]
    result.usd_path         = str(output_path)  # type: ignore[union-attr]
    result.pipeline_summary = summary           # type: ignore[union-attr]
    result.success          = True              # type: ignore[union-attr]
    logger.info(
        "Generation complete → %s  quality=%.2f  mat=%s",
        output_path, summary.get("quality_score", 0.0),
        summary.get("material_class", "-"),
    )


_DEFAULT_MODEL      = "claude-opus-4-6"
_DEFAULT_OUTER      = 5
_DEFAULT_INNER      = 3
_DEFAULT_VIEWS       = ["isometric", "front"]
_DEFAULT_MIN_CONF   = 0.75  # critic must reach this confidence to count as PASS


def generate(
    prompt: str,
    output_path: Path,
    config_path: Path | None = None,
    material_override: str | None = None,
    model: str = _DEFAULT_MODEL,
    max_outer: int = _DEFAULT_OUTER,
    max_inner: int = _DEFAULT_INNER,
    render_views_list: list[str] | None = None,
    min_confidence: float = _DEFAULT_MIN_CONF,
) -> "GenerationResult":  # noqa: F821 — resolved at runtime
    """Generate a SimReady USD asset from a text prompt.

    Args:
        prompt:           Natural-language description of the 3D object.
        output_path:      Where to write the output .usda file.
        config_path:      Optional path to a pipeline config YAML.
        material_override: Force a single material class for all geometry
                          (e.g. "steel", "cast_iron").
        model:            Anthropic model for all LLM calls.
        max_outer:        Maximum critic-revision cycles (default 5).
        max_inner:        Maximum traceback-fix retries per execution (default 3).
        render_views_list: View names to render (default ["isometric","front"]).
        min_confidence:   Minimum critic confidence to accept a PASS verdict.
                          A PASS below this threshold is treated as FAIL (default 0.75).

    Returns:
        GenerationResult with success flag, paths, and iteration history.
    """
    from simready.generation.critic import critique_geometry
    from simready.generation.executor import execute_with_retries
    from simready.generation.planner import plan_and_code
    from simready.generation.renderer import render_views
    from simready.generation.schemas import GenerationResult

    if render_views_list is None:
        render_views_list = _DEFAULT_VIEWS

    result = GenerationResult(prompt=prompt)
    work_dir = Path(tempfile.mkdtemp(prefix="simready_gen_"))
    logger.info("Generation work dir: %s", work_dir)

    try:
        # ── Step 1: Plan + initial code ────────────────────────────────────
        blueprint, code = plan_and_code(prompt, model=model)
        result.blueprint  = blueprint
        result.cadquery_code = code

        # Track the highest-confidence PASS seen so far (may be below threshold).
        # Used as a fallback if the loop exhausts without hitting min_confidence.
        best_pass_step: Path | None = None
        best_pass_conf: float = -1.0

        # ── Outer loop ─────────────────────────────────────────────────────
        for outer_iter in range(max_outer):
            result.iterations += 1
            logger.info("Outer iteration %d/%d", result.iterations, max_outer)

            # Step 2: Execute (with inner traceback-fix retries)
            step_path, stl_path, code, inner_retries = execute_with_retries(
                code, work_dir, model=model, blueprint=blueprint, max_retries=max_inner
            )
            result.inner_retries += inner_retries
            result.cadquery_code  = code

            if step_path is None:
                result.error = (
                    f"CadQuery execution failed after {max_inner} retries "
                    f"on outer iteration {result.iterations}."
                )
                logger.error(result.error)
                break

            # Step 3: Render views
            image_paths = render_views(
                stl_path, work_dir, views=render_views_list
            )

            # Step 4: VLM critique
            feedback = critique_geometry(image_paths, prompt, blueprint, model=model)
            result.critic_history.append(feedback)

            # Track best PASS seen (regardless of threshold) for timeout fallback
            if feedback.verdict == "PASS" and feedback.confidence > best_pass_conf:
                best_pass_conf = feedback.confidence
                best_pass_step = step_path

            effective_pass = (
                feedback.verdict == "PASS" and feedback.confidence >= min_confidence
            )
            if feedback.verdict == "PASS" and not effective_pass:
                logger.info(
                    "Critic PASS but confidence %.2f < threshold %.2f — treating as FAIL.",
                    feedback.confidence, min_confidence,
                )

            if effective_pass:
                # Step 5: Hand off to physics backend
                logger.info(
                    "Critic PASS (confidence=%.2f ≥ %.2f) — running physics backend pipeline.",
                    feedback.confidence, min_confidence,
                )
                _run_pipeline(
                    result, step_path, output_path, config_path,
                    material_override, prompt,
                )
                break

            # FAIL (or low-confidence PASS) — revise code for next outer iteration
            logger.info(
                "Critic %s (iter %d, conf=%.2f): %d issue(s) — requesting revision.",
                feedback.verdict, result.iterations, feedback.confidence, len(feedback.issues),
            )
            for issue in feedback.issues:
                logger.debug("  Issue: %s", issue)

            if outer_iter < max_outer - 1:
                _, code = plan_and_code(
                    prompt,
                    model=model,
                    previous_code=code,
                    critic_feedback=feedback,
                    blueprint=blueprint,
                )
                result.cadquery_code = code

        else:
            # Loop exhausted without break.
            if not result.success and not result.error:
                if best_pass_step is not None:
                    # Fallback: use the highest-confidence PASS even though it
                    # never cleared min_confidence. Geometry is recognisable —
                    # producing an imperfect asset beats a hard failure.
                    logger.warning(
                        "Max iterations reached; using best PASS (conf=%.2f) as fallback.",
                        best_pass_conf,
                    )
                    _run_pipeline(
                        result, best_pass_step, output_path, config_path,
                        material_override, prompt,
                    )
                else:
                    result.error = (
                        f"Max outer iterations ({max_outer}) reached and no PASS verdict "
                        "was ever returned by the critic."
                    )
                    logger.warning(result.error)

    except Exception as exc:
        result.error = str(exc)
        logger.exception("Generation pipeline raised an exception: %s", exc)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.debug("Cleaned up work dir: %s", work_dir)

    return result
