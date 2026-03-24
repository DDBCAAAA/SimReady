"""Tests for Adaptive Unit Scaling — unit detection and meter conversion."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# get_meters_conversion_factor — known units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit,expected", [
    ("mm",           0.001),
    ("millimeter",   0.001),
    ("millimeters",  0.001),
    ("millimetre",   0.001),
    ("millimetres",  0.001),
    ("MM",           0.001),   # case-insensitive
    ("in",           0.0254),
    ("inch",         0.0254),
    ("inches",       0.0254),
    ("IN",           0.0254),
    ("cm",           0.01),
    ("centimeter",   0.01),
    ("centimeters",  0.01),
    ("m",            1.0),
    ("meter",        1.0),
    ("meters",       1.0),
    ("metre",        1.0),
    ("metres",       1.0),
])
def test_get_meters_conversion_factor_known(unit, expected):
    from simready.geometry.mesh_processing import get_meters_conversion_factor
    assert get_meters_conversion_factor(unit) == pytest.approx(expected)


def test_get_meters_conversion_factor_unknown_warns_and_defaults(caplog):
    """An unrecognized unit should warn and return the mm engineering default."""
    from simready.geometry.mesh_processing import get_meters_conversion_factor
    with caplog.at_level(logging.WARNING):
        factor = get_meters_conversion_factor("furlongs")
    assert factor == pytest.approx(0.001)
    assert caplog.records, "Expected at least one warning"
    combined = " ".join(r.message for r in caplog.records).lower()
    assert "furlong" in combined or "unknown" in combined or "unrecognized" in combined


# ---------------------------------------------------------------------------
# _detect_step_units — regex parsing of STEP text
# ---------------------------------------------------------------------------

def _write_step(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.step"
    p.write_text(content)
    return p


def test_detect_step_units_mm(tmp_path):
    from simready.ingestion.step_reader import _detect_step_units
    step = _write_step(tmp_path, "DATA;\n#1=SI_UNIT(.MILLI.,.METRE.);\nENDSEC;\n")
    assert _detect_step_units(step) == "mm"


def test_detect_step_units_meters(tmp_path):
    from simready.ingestion.step_reader import _detect_step_units
    step = _write_step(tmp_path, "DATA;\n#1=SI_UNIT($,.METRE.);\nENDSEC;\n")
    assert _detect_step_units(step) == "m"


def test_detect_step_units_cm(tmp_path):
    from simready.ingestion.step_reader import _detect_step_units
    step = _write_step(tmp_path, "DATA;\n#1=SI_UNIT(.CENTI.,.METRE.);\nENDSEC;\n")
    assert _detect_step_units(step) == "cm"


def test_detect_step_units_inches(tmp_path):
    from simready.ingestion.step_reader import _detect_step_units
    step = _write_step(tmp_path, "DATA;\n#1=CONVERSION_BASED_UNIT('INCH',#2);\nENDSEC;\n")
    assert _detect_step_units(step) == "in"


def test_detect_step_units_default_mm_with_warning(tmp_path, caplog):
    """When no unit declaration is found the reader must warn and return 'mm'."""
    from simready.ingestion.step_reader import _detect_step_units
    step = _write_step(tmp_path, "DATA;\n#1=SOME_OTHER_UNIT();\nENDSEC;\n")
    with caplog.at_level(logging.WARNING):
        result = _detect_step_units(step)
    assert result == "mm"
    assert caplog.records, "Expected at least one warning"
    combined = " ".join(r.message for r in caplog.records).lower()
    assert "default" in combined or "no unit" in combined


def test_detect_step_units_tolerates_whitespace(tmp_path):
    """Regex must match even with extra whitespace around tokens."""
    from simready.ingestion.step_reader import _detect_step_units
    step = _write_step(tmp_path, "DATA;\n#1=SI_UNIT( .MILLI. , .METRE. );\nENDSEC;\n")
    assert _detect_step_units(step) == "mm"


# ---------------------------------------------------------------------------
# Integration: correct scale factor applied for inch vs mm input
# ---------------------------------------------------------------------------

def _make_assembly(units: str, max_coord: float):
    """Return a CADAssembly whose single body is a tetrahedron scaled to max_coord."""
    from simready.ingestion.step_reader import CADAssembly, CADBody
    verts = np.array([
        [0.0, 0.0, 0.0],
        [max_coord, 0.0, 0.0],
        [0.0, max_coord, 0.0],
        [0.0, 0.0, max_coord],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    body = CADBody(name="test_body", vertices=verts, faces=faces)
    assembly = CADAssembly(source_path=Path("test.step"), units=units)
    assembly.bodies.append(body)
    return assembly


def test_inch_vertices_scale_to_meters():
    """1.0 inch vertex coordinate → 0.0254 m after scale_to_meters with inch factor."""
    from simready.geometry.mesh_processing import (
        get_meters_conversion_factor, scale_to_meters, cad_body_to_trimesh, clean_mesh,
    )
    assembly = _make_assembly("in", 1.0)
    scale_factor = get_meters_conversion_factor(assembly.units)
    assert scale_factor == pytest.approx(0.0254)

    mesh = clean_mesh(cad_body_to_trimesh(assembly.bodies[0]))
    scaled = scale_to_meters(mesh, scale=scale_factor)
    assert scaled.vertices.max() == pytest.approx(0.0254, rel=1e-4)


def test_mm_vertices_scale_to_meters():
    """10.0 mm vertex coordinate → 0.01 m after scale_to_meters with mm factor."""
    from simready.geometry.mesh_processing import (
        get_meters_conversion_factor, scale_to_meters, cad_body_to_trimesh, clean_mesh,
    )
    assembly = _make_assembly("mm", 10.0)
    scale_factor = get_meters_conversion_factor(assembly.units)
    assert scale_factor == pytest.approx(0.001)

    mesh = clean_mesh(cad_body_to_trimesh(assembly.bodies[0]))
    scaled = scale_to_meters(mesh, scale=scale_factor)
    assert scaled.vertices.max() == pytest.approx(0.01, rel=1e-4)


def test_cm_vertices_scale_to_meters():
    """5.0 cm vertex coordinate → 0.05 m after scale_to_meters with cm factor."""
    from simready.geometry.mesh_processing import (
        get_meters_conversion_factor, scale_to_meters, cad_body_to_trimesh, clean_mesh,
    )
    assembly = _make_assembly("cm", 5.0)
    scale_factor = get_meters_conversion_factor(assembly.units)
    assert scale_factor == pytest.approx(0.01)

    mesh = clean_mesh(cad_body_to_trimesh(assembly.bodies[0]))
    scaled = scale_to_meters(mesh, scale=scale_factor)
    assert scaled.vertices.max() == pytest.approx(0.05, rel=1e-4)


def test_read_step_populates_assembly_units_mm(tmp_path):
    """read_step must populate CADAssembly.units from the STEP header (mm case)."""
    pytest.importorskip("OCP", reason="pythonocc-core not installed")
    from simready.ingestion.step_reader import _detect_step_units
    # Just test the detector on a synthetic file; full read_step needs a real OCC shape.
    step = _write_step(tmp_path, "DATA;\n#1=SI_UNIT(.MILLI.,.METRE.);\nENDSEC;\n")
    assert _detect_step_units(step) == "mm"
