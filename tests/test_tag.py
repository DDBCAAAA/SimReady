"""Tests for simready tag command (manual curation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from simready.catalog.db import CatalogEntry, open_db, upsert_asset, get_asset
from simready.cli import main


def _make_entry(tmp_path: Path, step_path: Path | None = None) -> tuple:
    conn = open_db(tmp_path / "catalog.db")
    entry = CatalogEntry(
        name="gear.step",
        source="freecad",
        source_url="https://example.com/gear.step",
        category=None,
        material_class=None,
        local_path=step_path,
        usd_path=tmp_path / "gear.usda" if step_path else None,
    )
    row_id = upsert_asset(conn, entry)
    conn.close()
    return row_id


def test_tag_updates_category(tmp_path):
    row_id = _make_entry(tmp_path)
    main(["tag", str(row_id), "--category", "mechanical:gear",
          "--db", str(tmp_path / "catalog.db")])
    conn = open_db(tmp_path / "catalog.db")
    entry = get_asset(conn, row_id)
    conn.close()
    assert entry.category == "mechanical:gear"


def test_tag_updates_material_class(tmp_path):
    row_id = _make_entry(tmp_path)
    main(["tag", str(row_id), "--material", "steel",
          "--db", str(tmp_path / "catalog.db")])
    conn = open_db(tmp_path / "catalog.db")
    entry = get_asset(conn, row_id)
    conn.close()
    assert entry.material_class == "steel"


def test_tag_unknown_id_exits(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    conn.close()
    with pytest.raises(SystemExit):
        main(["tag", "9999", "--material", "steel",
              "--db", str(tmp_path / "catalog.db")])


def test_tag_reruns_pipeline_when_step_exists(tmp_path):
    # Create a real dummy STEP file
    step = tmp_path / "gear.step"
    step.write_text("fake")
    row_id = _make_entry(tmp_path, step_path=step)

    pipeline_summary = {
        "face_count": 1500, "quality_score": 0.88,
        "watertight": True, "physics_complete": True,
        "material_confidence": 0.25, "material_class": "steel",
    }
    with patch("simready.pipeline.run", return_value=pipeline_summary):
        main(["tag", str(row_id), "--material", "steel",
              "--db", str(tmp_path / "catalog.db")])

    conn = open_db(tmp_path / "catalog.db")
    entry = get_asset(conn, row_id)
    conn.close()
    assert entry.quality_score == pytest.approx(0.88)
    assert entry.physics_complete is True


def test_tag_skips_reexport_when_step_missing(tmp_path, capsys):
    # local_path points to a nonexistent file
    row_id = _make_entry(tmp_path, step_path=tmp_path / "nonexistent.step")
    main(["tag", str(row_id), "--material", "nylon",
          "--db", str(tmp_path / "catalog.db")])
    captured = capsys.readouterr()
    # Should note STEP file not found but not crash
    assert "not found" in captured.out or "updated" in captured.out
