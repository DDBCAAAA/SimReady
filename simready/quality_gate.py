"""Material confidence quality gate.

Intercepts assets whose material confidence is below the production threshold
before USD generation begins.  Low-confidence assets are:
  - Logged to ``output/low_confidence_assets.log`` (append-mode, one line per asset)
  - Quarantined by moving the raw STEP file to ``output/quarantine/``
  - Blocked from USD generation by raising ``LowConfidenceError``

The threshold is set to 0.8 to ensure only assets with strong material
provenance reach the physics simulation dataset.  The 0.25 "class-default"
confidence produced by ``map_cae_to_mdl`` for STEP files with no CAE optical
data is well below this bar.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from simready import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Minimum material confidence required to proceed with USD generation.
CONFIDENCE_THRESHOLD: float = 0.8

#: Default append-mode log file for quarantined assets.
DEFAULT_LOG_PATH: Path = PROJECT_ROOT / "output" / "low_confidence_assets.log"

#: Default quarantine directory for raw STEP files that fail the gate.
DEFAULT_QUARANTINE_DIR: Path = PROJECT_ROOT / "output" / "quarantine"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class LowConfidenceError(RuntimeError):
    """Raised when an asset's material confidence is below the threshold.

    Caught by ``batch._convert_one`` and mapped to ``status="skipped"`` so
    that low-confidence assets do not appear as pipeline failures in reports.
    """

    def __init__(self, asset_name: str, confidence: float, threshold: float) -> None:
        super().__init__(
            f"{asset_name}: material confidence {confidence:.4f} < "
            f"threshold {threshold:.4f} — USD generation skipped"
        )
        self.asset_name = asset_name
        self.confidence = confidence
        self.threshold = threshold


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------

def check_material_confidence(
    confidence: float,
    asset_name: str,
    step_path: Path | None = None,
    log_path: Path | None = None,
    quarantine_dir: Path | None = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> None:
    """Enforce the material confidence threshold.

    If ``confidence >= threshold`` the function returns silently (asset passes).

    If ``confidence < threshold``:
      1. Appends a structured WARNING line to *log_path*.
      2. Moves *step_path* (if it exists) into *quarantine_dir*.
      3. Raises ``LowConfidenceError``.

    Args:
        confidence:    Material confidence score, 0.0–1.0.
        asset_name:    Human-readable name used in log messages.
        step_path:     Path to the raw source file (moved on failure).
                       If ``None`` or the file does not exist, only logging occurs.
        log_path:      Destination for the warning log.
                       Defaults to ``output/low_confidence_assets.log``.
        quarantine_dir: Directory to receive quarantined source files.
                       Defaults to ``output/quarantine/``.
        threshold:     Override the default ``CONFIDENCE_THRESHOLD``.

    Raises:
        LowConfidenceError: When confidence is below threshold.
    """
    if confidence >= threshold:
        return  # asset passes — nothing to do

    resolved_log = log_path if log_path is not None else DEFAULT_LOG_PATH
    resolved_qdir = quarantine_dir if quarantine_dir is not None else DEFAULT_QUARANTINE_DIR

    # --- 1. Write warning log entry ---
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    quarantine_dest: str = "n/a"

    # Determine quarantine destination before moving so we can log it atomically
    if step_path is not None and step_path.exists():
        resolved_qdir.mkdir(parents=True, exist_ok=True)
        dest = resolved_qdir / step_path.name
        # Avoid clobbering an existing quarantine entry with the same filename
        if dest.exists():
            for i in range(1, 10_000):
                candidate = resolved_qdir / f"{step_path.stem}_{i}{step_path.suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
        quarantine_dest = str(dest)

    log_line = (
        f"{timestamp} | WARNING | asset={asset_name} | "
        f"confidence={confidence:.4f} | threshold={threshold:.4f} | "
        f"quarantined={quarantine_dest}\n"
    )
    with resolved_log.open("a", encoding="utf-8") as fh:
        fh.write(log_line)

    logger.warning(
        "LOW CONFIDENCE — %s (%.4f < %.4f): USD generation skipped; "
        "source moved to quarantine",
        asset_name, confidence, threshold,
    )

    # --- 2. Move source file to quarantine ---
    if quarantine_dest != "n/a":
        shutil.move(str(step_path), quarantine_dest)
        logger.info("Quarantined %s → %s", step_path.name, quarantine_dest)

    # --- 3. Block USD generation ---
    raise LowConfidenceError(asset_name, confidence, threshold)
