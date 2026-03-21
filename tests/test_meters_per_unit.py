"""Test that pipeline outputs meters-scale USD with metersPerUnit = 1.

Runs the full pipeline on a real M3 screw STEP file and asserts:
1. The .usda header contains 'metersPerUnit = 1'.
2. The bounding box diagonal is in the range [0.005, 0.05] meters
   (M3x10 screw: ~3 mm wide, 10 mm long → diagonal ≈ 0.011 m).
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Path to a real M3 screw from the FreeCAD batch download
_SCREW_STEP = (
    Path(__file__).parent.parent
    / "data/step_files/fasteners/freecad"
    / "ISO10642_Hex_Socket_Countersunk_Head_Screw_M3x10.step"
)


@pytest.fixture(scope="module")
def m3_usd(tmp_path_factory):
    """Run the pipeline on the M3 screw and return the output .usda path."""
    pytest.importorskip("pxr", reason="usd-core not installed")
    if not _SCREW_STEP.exists():
        pytest.skip(f"M3 screw STEP file not found: {_SCREW_STEP}")

    from simready.pipeline import run

    out_dir = tmp_path_factory.mktemp("usd_out")
    out_path = out_dir / "m3_screw.usda"
    # This fixture tests unit scaling, not confidence; bypass the quality gate.
    with patch("simready.quality_gate.check_material_confidence"):
        run(
            _SCREW_STEP,
            out_path,
            config_path=None,
            material_overrides={"*": "steel"},
        )
    assert out_path.exists(), "pipeline did not produce a .usda file"
    return out_path


def test_meters_per_unit_header(m3_usd: Path):
    """metersPerUnit in the USD header must equal 1 (not 0.001)."""
    text = m3_usd.read_text()
    assert "metersPerUnit = 1" in text, (
        f"Expected 'metersPerUnit = 1' in USD header, got:\n"
        + "\n".join(line for line in text.splitlines()[:20])
    )


def test_bounding_box_meters_scale(m3_usd: Path):
    """Bounding box diagonal must be on the scale of an M3x10 screw in meters.

    Expected: ~11 mm → ~0.011 m.  Accept [0.005, 0.05] to tolerate minor
    tessellation differences.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(m3_usd))
    bb_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bb_cache.ComputeWorldBound(stage.GetPseudoRoot())
    bbox_range = bbox.GetRange()
    size = bbox_range.GetSize()  # Gf.Vec3d
    diagonal = math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)

    assert 0.005 < diagonal < 0.05, (
        f"Bounding box diagonal {diagonal:.6f} m is outside expected range "
        f"[0.005, 0.05] m for an M3x10 screw. "
        f"Size: ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}) m"
    )
