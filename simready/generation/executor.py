"""Sandbox executor: runs LLM-generated CadQuery scripts in a subprocess.

Follows the same tempfile + subprocess.run pattern as
simready/ingestion/sldprt_converter.py (lines 87–101).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code injection templates
# ---------------------------------------------------------------------------

_HEADER = """\
import os as _simready_os_
_STEP_OUTPUT = _simready_os_.environ['OUTPUT_STEP']
_STL_OUTPUT  = _simready_os_.environ['OUTPUT_STL']

"""

_FOOTER = """

# --- SimReady export footer (injected by executor) ---
_export_result = result
if hasattr(result, 'toCompound'):
    # cq.Assembly → convert to Compound for STEP export
    _export_result = result.toCompound()
import cadquery as _cq_export_mod_
_cq_export_mod_.exporters.export(_export_result, _STEP_OUTPUT)
_cq_export_mod_.exporters.export(_export_result, _STL_OUTPUT, exportType='STL')
print('EXPORT_OK')
"""


def execute_cadquery_script(
    code: str,
    work_dir: Path,
    timeout: int = 120,
) -> tuple[Path | None, Path | None, str | None]:
    """Write *code* to a temp .py, execute it in a subprocess, return outputs.

    Returns:
        (step_path, stl_path, error_string)
        On success: step_path and stl_path are non-None, error_string is None.
        On failure: step_path and stl_path are None, error_string has the traceback.
    """
    uid = uuid.uuid4().hex[:8]
    step_path = work_dir / f"output_{uid}.step"
    stl_path  = work_dir / f"output_{uid}.stl"

    full_code = _HEADER + code + _FOOTER

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir=work_dir
    ) as tf:
        tf.write(full_code)
        script_path = tf.name

    env = os.environ.copy()
    env["OUTPUT_STEP"] = str(step_path)
    env["OUTPUT_STL"]  = str(stl_path)

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if "EXPORT_OK" not in proc.stdout:
            error = (proc.stderr or proc.stdout or "Script produced no output").strip()
            logger.debug("CadQuery script failed:\n%s", error[:500])
            return None, None, error

        if not step_path.exists() or step_path.stat().st_size == 0:
            return None, None, "Script exited cleanly but STEP file is missing or empty."
        if not stl_path.exists() or stl_path.stat().st_size == 0:
            return None, None, "Script exited cleanly but STL file is missing or empty."

        logger.info("CadQuery script executed successfully → %s", step_path.name)
        return step_path, stl_path, None

    except subprocess.TimeoutExpired:
        return None, None, f"Script timed out after {timeout}s."
    except Exception as exc:
        return None, None, str(exc)
    finally:
        Path(script_path).unlink(missing_ok=True)


def execute_with_retries(
    code: str,
    work_dir: Path,
    model: str,
    blueprint: object,
    max_retries: int = 3,
) -> tuple[Path | None, Path | None, str, int]:
    """Inner retry loop: execute, on error ask Coder to fix traceback, retry.

    Returns:
        (step_path, stl_path, latest_code, retry_count)
        step_path/stl_path are None if all retries exhausted.
    """
    from simready.generation.planner import revise_code  # lazy import — avoids import cycles

    current_code = code
    retries = 0

    for attempt in range(max_retries + 1):
        step_path, stl_path, error = execute_cadquery_script(current_code, work_dir)

        if error is None:
            return step_path, stl_path, current_code, retries

        if attempt == max_retries:
            logger.warning(
                "Execution failed after %d retries. Last error: %s", retries, error[:200]
            )
            break

        logger.info(
            "Execution attempt %d/%d failed — asking LLM to fix traceback.",
            attempt + 1, max_retries,
        )
        try:
            current_code = revise_code(current_code, error, blueprint, model=model)
            retries += 1
        except Exception as exc:
            logger.error("Code revision failed: %s", exc)
            break

    return None, None, current_code, retries
