"""Tests for the batch pipeline (simready/batch.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simready.batch import BatchResult, _convert_one, print_batch_summary
from simready.catalog.db import CatalogEntry, open_db, query_assets


# ---------------------------------------------------------------------------
# _convert_one unit tests (no network, no OCC)
# ---------------------------------------------------------------------------

def _fake_step(tmp_path: Path, name: str = "part.step") -> Path:
    p = tmp_path / name
    p.write_text("fake")
    return p


def test_convert_one_passed(tmp_path):
    """Successful conversion with quality above threshold → status=passed."""
    # _convert_one imports pipeline.run inside the function, so patch at source
    with patch("simready.pipeline.run", return_value={
        "face_count": 1200, "quality_score": 0.85,
        "watertight": True, "physics_complete": True,
        "material_confidence": 0.25, "material_class": "steel",
    }):
        result = _convert_one(_fake_step(tmp_path), tmp_path / "out", None, 0.5)

    assert result.status == "passed"
    assert result.quality_score == pytest.approx(0.85)
    assert result.watertight is True
    assert result.physics_complete is True


def test_convert_one_skipped_below_quality_min(tmp_path):
    """Quality below threshold → status=skipped, usd_path=None."""
    with patch("simready.pipeline.run", return_value={
        "face_count": 80, "quality_score": 0.30,
        "watertight": False, "physics_complete": False,
        "material_confidence": 0.0, "material_class": None,
    }):
        result = _convert_one(_fake_step(tmp_path), tmp_path / "out", None, 0.7)

    assert result.status == "skipped"
    assert result.usd_path is None
    assert result.quality_score == pytest.approx(0.30)


def test_convert_one_failed_on_exception(tmp_path):
    """Exception in pipeline.run → status=failed, error set, no raise."""
    with patch("simready.pipeline.run", side_effect=RuntimeError("parse error")):
        result = _convert_one(_fake_step(tmp_path), tmp_path / "out", None, 0.5)

    assert result.status == "failed"
    assert "parse error" in result.error
    assert result.usd_path is None


# ---------------------------------------------------------------------------
# print_batch_summary (smoke test — just must not raise)
# ---------------------------------------------------------------------------

def test_print_batch_summary_no_crash(capsys):
    results = [
        BatchResult("a.step", Path("a.step"), Path("a.usda"), "passed",
                    quality_score=0.9, watertight=True, physics_complete=True,
                    material_class="steel"),
        BatchResult("b.step", Path("b.step"), None, "skipped", quality_score=0.4),
        BatchResult("c.step", Path("c.step"), None, "failed", error="bad"),
    ]
    print_batch_summary(results)
    captured = capsys.readouterr()
    assert "3 processed" in captured.out
    assert "1 passed" in captured.out
    assert "1 failed" in captured.out
    assert "1 skipped" in captured.out


# ---------------------------------------------------------------------------
# pipeline.run return value contract
# ---------------------------------------------------------------------------

def test_pipeline_run_returns_dict(tmp_path):
    """pipeline.run must return a dict with all required keys even on tiny meshes."""
    import numpy as np
    import trimesh
    from simready.ingestion.step_reader import CADBody, CADAssembly
    from simready.config.settings import PipelineSettings

    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=int)

    assembly = CADAssembly(source_path=tmp_path / "fake.step")
    assembly.bodies.append(CADBody(
        name="steel_part", vertices=verts, faces=faces, material_name="steel_part",
    ))

    with patch("simready.pipeline.read_step", return_value=assembly):
        with patch("simready.usd.assembly.create_stage"):
            with patch("simready.pipeline.load_settings") as mock_settings:
                from simready.config.settings import PipelineSettings, ValidationSettings
                s = PipelineSettings()
                s.validation = ValidationSettings(enable_confidence_gate=False)
                mock_settings.return_value = s
                from simready.pipeline import run
                summary = run(
                    tmp_path / "fake.step",
                    tmp_path / "out.usda",
                )

    required_keys = {"face_count", "quality_score", "watertight",
                     "physics_complete", "material_confidence", "material_class"}
    assert required_keys.issubset(set(summary.keys()))
    assert isinstance(summary["face_count"], int)
    assert 0.0 <= summary["quality_score"] <= 1.0
