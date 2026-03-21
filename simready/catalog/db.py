"""SQLite-backed asset catalog for SimReady.

Replaces catalog.json as the source of truth. JSON export retained for
backward compatibility via export_json().
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

from simready import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "catalog.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL,
    source               TEXT    NOT NULL,
    source_url           TEXT    UNIQUE NOT NULL,
    category             TEXT,
    material_class       TEXT,
    face_count           INTEGER,
    quality_score        REAL,
    watertight           INTEGER,
    physics_complete     INTEGER,
    material_confidence  REAL,
    license              TEXT,
    local_path           TEXT,
    usd_path             TEXT,
    created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_category        ON assets(category);
CREATE INDEX IF NOT EXISTS idx_material_class  ON assets(material_class);
CREATE INDEX IF NOT EXISTS idx_quality_score   ON assets(quality_score);
CREATE INDEX IF NOT EXISTS idx_physics_complete ON assets(physics_complete);
"""


@dataclass
class CatalogEntry:
    """A single cataloged asset with acquisition + quality metadata."""
    name: str
    source: str
    source_url: str
    id: int | None = None
    category: str | None = None
    material_class: str | None = None
    face_count: int | None = None
    quality_score: float | None = None
    watertight: bool | None = None
    physics_complete: bool | None = None
    material_confidence: float | None = None
    license: str | None = None
    local_path: Path | None = None
    usd_path: Path | None = None
    created_at: str | None = None


def open_db(db_path: Path = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (and initialise schema if new) the catalog database."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _entry_to_row(entry: CatalogEntry) -> dict:
    return {
        "name":                entry.name,
        "source":              entry.source,
        "source_url":          entry.source_url,
        "category":            entry.category,
        "material_class":      entry.material_class,
        "face_count":          entry.face_count,
        "quality_score":       entry.quality_score,
        "watertight":          int(entry.watertight) if entry.watertight is not None else None,
        "physics_complete":    int(entry.physics_complete) if entry.physics_complete is not None else None,
        "material_confidence": entry.material_confidence,
        "license":             entry.license,
        "local_path":          str(entry.local_path) if entry.local_path else None,
        "usd_path":            str(entry.usd_path) if entry.usd_path else None,
    }


def _row_to_entry(row: sqlite3.Row) -> CatalogEntry:
    d = dict(row)
    return CatalogEntry(
        id=d["id"],
        name=d["name"],
        source=d["source"],
        source_url=d["source_url"],
        category=d.get("category"),
        material_class=d.get("material_class"),
        face_count=d.get("face_count"),
        quality_score=d.get("quality_score"),
        watertight=bool(d["watertight"]) if d.get("watertight") is not None else None,
        physics_complete=bool(d["physics_complete"]) if d.get("physics_complete") is not None else None,
        material_confidence=d.get("material_confidence"),
        license=d.get("license"),
        local_path=Path(d["local_path"]) if d.get("local_path") else None,
        usd_path=Path(d["usd_path"]) if d.get("usd_path") else None,
        created_at=d.get("created_at"),
    )


def upsert_asset(conn: sqlite3.Connection, entry: CatalogEntry) -> int:
    """Insert or update an asset row. Returns the row id."""
    row = _entry_to_row(entry)
    conn.execute(
        """
        INSERT INTO assets (name, source, source_url, category, material_class,
            face_count, quality_score, watertight, physics_complete,
            material_confidence, license, local_path, usd_path)
        VALUES (:name, :source, :source_url, :category, :material_class,
            :face_count, :quality_score, :watertight, :physics_complete,
            :material_confidence, :license, :local_path, :usd_path)
        ON CONFLICT(source_url) DO UPDATE SET
            name=excluded.name,
            category=COALESCE(excluded.category, assets.category),
            material_class=COALESCE(excluded.material_class, assets.material_class),
            face_count=COALESCE(excluded.face_count, assets.face_count),
            quality_score=COALESCE(excluded.quality_score, assets.quality_score),
            watertight=COALESCE(excluded.watertight, assets.watertight),
            physics_complete=COALESCE(excluded.physics_complete, assets.physics_complete),
            material_confidence=COALESCE(excluded.material_confidence, assets.material_confidence),
            license=COALESCE(excluded.license, assets.license),
            local_path=COALESCE(excluded.local_path, assets.local_path),
            usd_path=COALESCE(excluded.usd_path, assets.usd_path)
        """,
        row,
    )
    conn.commit()
    row_id = conn.execute(
        "SELECT id FROM assets WHERE source_url=?", (entry.source_url,)
    ).fetchone()["id"]
    return row_id


def get_asset(conn: sqlite3.Connection, asset_id: int) -> CatalogEntry | None:
    """Fetch a single asset by id."""
    row = conn.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
    return _row_to_entry(row) if row else None


def query_assets(
    conn: sqlite3.Connection,
    filter_expr: str | None = None,
) -> list[CatalogEntry]:
    """Return assets matching an optional SQL WHERE fragment.

    filter_expr is a SQL WHERE clause, e.g.:
        "category='fastener' AND physics_complete=1 AND quality_score > 0.7"

    The expression is validated via EXPLAIN QUERY PLAN before execution.
    """
    sql = "SELECT * FROM assets"
    if filter_expr:
        sql += f" WHERE {filter_expr}"
    sql += " ORDER BY quality_score DESC NULLS LAST"

    if filter_expr:
        try:
            conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        except sqlite3.Error as e:
            raise ValueError(f"Invalid filter expression: {e}") from e

    rows = conn.execute(sql).fetchall()
    return [_row_to_entry(r) for r in rows]


def update_asset_fields(
    conn: sqlite3.Connection,
    asset_id: int,
    **fields: object,
) -> None:
    """Update one or more named columns for an existing row."""
    if not fields:
        return
    allowed = {
        "name", "category", "material_class", "face_count", "quality_score",
        "watertight", "physics_complete", "material_confidence", "license",
        "local_path", "usd_path",
    }
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"Unknown catalog fields: {bad}")

    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [asset_id]
    conn.execute(f"UPDATE assets SET {set_clause} WHERE id=?", values)
    conn.commit()


def export_json(
    conn: sqlite3.Connection,
    path: Path,
    filter_expr: str | None = None,
) -> None:
    """Write catalog (or filtered subset) to JSON for backward compatibility."""
    entries = query_assets(conn, filter_expr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for e in entries:
        d = asdict(e)
        d["local_path"] = str(e.local_path) if e.local_path else None
        d["usd_path"] = str(e.usd_path) if e.usd_path else None
        records.append(d)

    path.write_text(json.dumps(records, indent=2))
    logger.info("Exported %d entries to %s", len(records), path)


def from_step_asset(asset: object) -> CatalogEntry:
    """Convert a STEPAsset to a CatalogEntry (quality fields default to None)."""
    return CatalogEntry(
        name=asset.name,            # type: ignore[attr-defined]
        source=asset.source,        # type: ignore[attr-defined]
        source_url=asset.url,       # type: ignore[attr-defined]
        license=asset.license,      # type: ignore[attr-defined]
        local_path=asset.local_path,  # type: ignore[attr-defined]
        usd_path=asset.usd_path,    # type: ignore[attr-defined]
    )
