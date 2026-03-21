"""Tests for the SQLite catalog (simready/catalog/db.py)."""

from __future__ import annotations

import json
import pytest

from simready.catalog.db import (
    CatalogEntry,
    open_db,
    upsert_asset,
    get_asset,
    query_assets,
    update_asset_fields,
    export_json,
)


def _make_entry(**kwargs) -> CatalogEntry:
    defaults = dict(
        name="bolt_m8",
        source="freecad",
        source_url="https://example.com/bolt_m8.step",
        category="fastener:bolt",
        material_class="steel",
        face_count=1200,
        quality_score=0.85,
        watertight=True,
        physics_complete=True,
        material_confidence=0.25,
        license="LGPL-2.1",
    )
    defaults.update(kwargs)
    return CatalogEntry(**defaults)


def test_open_db_creates_schema(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    # Table must exist
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='assets'"
    ).fetchone()
    assert row is not None
    conn.close()


def test_upsert_and_get_roundtrip(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    entry = _make_entry()
    row_id = upsert_asset(conn, entry)
    assert isinstance(row_id, int) and row_id > 0

    fetched = get_asset(conn, row_id)
    assert fetched is not None
    assert fetched.name == "bolt_m8"
    assert fetched.material_class == "steel"
    assert fetched.watertight is True
    assert fetched.physics_complete is True
    conn.close()


def test_upsert_deduplicates_by_url(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    entry = _make_entry(quality_score=0.70)
    id1 = upsert_asset(conn, entry)

    # Second upsert with same URL — updates quality_score
    entry2 = _make_entry(quality_score=0.90)
    id2 = upsert_asset(conn, entry2)

    assert id1 == id2  # same row
    fetched = get_asset(conn, id1)
    assert fetched.quality_score == pytest.approx(0.90)
    conn.close()


def test_upsert_does_not_overwrite_with_none(tmp_path):
    """ON CONFLICT COALESCE: existing non-null value must survive a None update."""
    conn = open_db(tmp_path / "catalog.db")
    entry = _make_entry(category="fastener:bolt")
    row_id = upsert_asset(conn, entry)

    # Re-upsert with category=None — existing value should be preserved
    entry2 = _make_entry(category=None)
    upsert_asset(conn, entry2)
    fetched = get_asset(conn, row_id)
    assert fetched.category == "fastener:bolt"
    conn.close()


def test_get_asset_returns_none_for_missing(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    assert get_asset(conn, 9999) is None
    conn.close()


def test_query_filter_category(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    upsert_asset(conn, _make_entry(name="bolt", source_url="u1", category="fastener:bolt"))
    upsert_asset(conn, _make_entry(name="gear", source_url="u2", category="mechanical:gear"))

    results = query_assets(conn, "category='fastener:bolt'")
    assert len(results) == 1
    assert results[0].name == "bolt"
    conn.close()


def test_query_filter_physics_complete(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    upsert_asset(conn, _make_entry(name="a", source_url="u1", physics_complete=True))
    upsert_asset(conn, _make_entry(name="b", source_url="u2", physics_complete=False))

    results = query_assets(conn, "physics_complete=1")
    assert len(results) == 1
    assert results[0].name == "a"
    conn.close()


def test_query_no_filter_returns_all(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    for i in range(5):
        upsert_asset(conn, _make_entry(name=f"part_{i}", source_url=f"u{i}"))
    assert len(query_assets(conn)) == 5
    conn.close()


def test_query_invalid_filter_raises(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    with pytest.raises(ValueError, match="Invalid filter"):
        query_assets(conn, "DROP TABLE assets; --")
    conn.close()


def test_query_ordered_by_quality_desc(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    upsert_asset(conn, _make_entry(name="low",  source_url="u1", quality_score=0.3))
    upsert_asset(conn, _make_entry(name="high", source_url="u2", quality_score=0.9))
    upsert_asset(conn, _make_entry(name="mid",  source_url="u3", quality_score=0.6))

    results = query_assets(conn)
    assert results[0].name == "high"
    assert results[-1].name == "low"
    conn.close()


def test_update_asset_fields(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    row_id = upsert_asset(conn, _make_entry(category=None))
    update_asset_fields(conn, row_id, category="fastener:bolt", quality_score=0.95)
    fetched = get_asset(conn, row_id)
    assert fetched.category == "fastener:bolt"
    assert fetched.quality_score == pytest.approx(0.95)
    conn.close()


def test_update_asset_unknown_field_raises(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    row_id = upsert_asset(conn, _make_entry())
    with pytest.raises(ValueError, match="Unknown catalog fields"):
        update_asset_fields(conn, row_id, nonexistent_col="x")
    conn.close()


def test_export_json(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    upsert_asset(conn, _make_entry(name="a", source_url="u1", category="fastener:bolt"))
    upsert_asset(conn, _make_entry(name="b", source_url="u2", category="mechanical:gear"))

    out = tmp_path / "out.json"
    export_json(conn, out)
    data = json.loads(out.read_text())
    assert len(data) == 2
    conn.close()


def test_export_json_filtered(tmp_path):
    conn = open_db(tmp_path / "catalog.db")
    upsert_asset(conn, _make_entry(name="a", source_url="u1", category="fastener:bolt"))
    upsert_asset(conn, _make_entry(name="b", source_url="u2", category="mechanical:gear"))

    out = tmp_path / "out.json"
    export_json(conn, out, filter_expr="category='fastener:bolt'")
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["name"] == "a"
    conn.close()


def test_from_step_asset(tmp_path):
    from simready.acquisition.sources import STEPAsset
    from simready.catalog.db import from_step_asset

    asset = STEPAsset(
        name="gear.step",
        url="https://example.com/gear.step",
        source="freecad",
        license="LGPL-2.1",
    )
    entry = from_step_asset(asset)
    assert entry.name == "gear.step"
    assert entry.source == "freecad"
    assert entry.source_url == "https://example.com/gear.step"
    assert entry.quality_score is None
    assert entry.physics_complete is None
