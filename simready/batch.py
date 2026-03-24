"""Batch acquisition and conversion pipeline for SimReady."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from simready import PROJECT_ROOT
from simready.catalog.db import CatalogEntry, open_db, upsert_asset, from_step_asset

logger = logging.getLogger(__name__)

_DEFAULT_STEP_DIR   = PROJECT_ROOT / "data" / "step_files"
_DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "simready-500"
_DEFAULT_DB_PATH    = PROJECT_ROOT / "data" / "catalog.db"

# Canonical category folder names — must match the directory skeleton.
CATEGORIES = (
    "fasteners",
    "gears",
    "brackets_plates",
    "housings_enclosures",
    "connectors_fittings",
    "shafts_bearings",
    "valves_pipe",
    "misc_mechanical",
)

# Keyword → category mapping for routing downloads and outputs.
_CATEGORY_KEYWORDS: dict[str, str] = {
    # fasteners
    "bolt": "fasteners", "screw": "fasteners", "nut": "fasteners",
    "washer": "fasteners", "rivet": "fasteners", "stud": "fasteners",
    "fastener": "fasteners", "thread": "fasteners",
    # gears
    "gear": "gears", "sprocket": "gears", "rack": "gears",
    "pinion": "gears", "worm": "gears", "pulley": "gears",
    # brackets & plates
    "bracket": "brackets_plates", "plate": "brackets_plates",
    "gusset": "brackets_plates", "flange_plate": "brackets_plates",
    "angle": "brackets_plates", "channel": "brackets_plates",
    # housings & enclosures
    "housing": "housings_enclosures", "enclosure": "housings_enclosures",
    "cover": "housings_enclosures", "cap": "housings_enclosures",
    "casing": "housings_enclosures", "box": "housings_enclosures",
    # connectors & fittings
    "connector": "connectors_fittings", "fitting": "connectors_fittings",
    "coupling": "connectors_fittings", "adapter": "connectors_fittings",
    "elbow": "connectors_fittings", "tee": "connectors_fittings",
    "flange": "connectors_fittings", "nipple": "connectors_fittings",
    # shafts, bearings, bushings
    "shaft": "shafts_bearings", "bearing": "shafts_bearings",
    "bushing": "shafts_bearings", "sleeve": "shafts_bearings",
    "spindle": "shafts_bearings", "axle": "shafts_bearings",
    "collar": "shafts_bearings", "key": "shafts_bearings",
    # valves & pipe
    "valve": "valves_pipe", "pipe": "valves_pipe",
    "clamp": "valves_pipe", "manifold": "valves_pipe",
    # misc mechanical
    "spring": "misc_mechanical", "spacer": "misc_mechanical",
    "retainer": "misc_mechanical", "pin": "misc_mechanical",
    "clip": "misc_mechanical", "cam": "misc_mechanical",
    "link": "misc_mechanical", "lever": "misc_mechanical",
}


def category_for(name: str) -> str:
    """Infer category folder from an asset/file name. Falls back to misc_mechanical."""
    name_lower = name.lower()
    for keyword, cat in _CATEGORY_KEYWORDS.items():
        if keyword in name_lower:
            return cat
    return "misc_mechanical"


@dataclass
class BatchResult:
    """Result of converting a single asset in a batch run."""
    asset_name: str
    step_path: Path | None
    usd_path: Path | None
    status: str                      # "passed" | "failed" | "skipped"
    quality_score: float | None = None
    face_count: int | None = None
    watertight: bool | None = None
    physics_complete: bool | None = None
    material_confidence: float | None = None
    material_class: str | None = None
    error: str | None = None


_MAX_STEP_BYTES = 2 * 1024 * 1024  # 2 MB — skip files that would take forever to tessellate


def _convert_one(
    step_path: Path,
    output_dir: Path,
    config_path: Path | None,
    quality_min: float,
    material_default: str | None = None,
) -> BatchResult:
    """Convert one STEP file. Never raises — all exceptions are caught."""
    usd_path = output_dir / (step_path.stem + ".usd")

    file_size = step_path.stat().st_size
    if file_size > _MAX_STEP_BYTES:
        logger.warning(
            "Skipping %s (%.1f MB > %.0f MB limit)",
            step_path.name, file_size / 1e6, _MAX_STEP_BYTES / 1e6,
        )
        return BatchResult(
            asset_name=step_path.name,
            step_path=step_path,
            usd_path=None,
            status="skipped",
            error=f"file too large ({file_size / 1e6:.1f} MB)",
        )
    try:
        from simready.pipeline import run as pipeline_run
        from simready.quality_gate import LowConfidenceError
        overrides = {"*": material_default} if material_default else None
        summary = pipeline_run(step_path, usd_path, config_path, material_overrides=overrides)
        if summary["quality_score"] < quality_min:
            return BatchResult(
                asset_name=step_path.name,
                step_path=step_path,
                usd_path=None,
                status="skipped",
                quality_score=summary["quality_score"],
                face_count=summary["face_count"],
                watertight=summary["watertight"],
                physics_complete=summary["physics_complete"],
                material_confidence=summary["material_confidence"],
                material_class=summary["material_class"],
            )
        return BatchResult(
            asset_name=step_path.name,
            step_path=step_path,
            usd_path=usd_path,
            status="passed",
            quality_score=summary["quality_score"],
            face_count=summary["face_count"],
            watertight=summary["watertight"],
            physics_complete=summary["physics_complete"],
            material_confidence=summary["material_confidence"],
            material_class=summary["material_class"],
        )
    except LowConfidenceError as exc:
        # Quality gate triggered — not a pipeline failure, just a filtered asset
        logger.warning("Skipped (low confidence): %s — %s", step_path.name, exc)
        return BatchResult(
            asset_name=step_path.name,
            step_path=step_path,
            usd_path=None,
            status="skipped",
            error=str(exc),
        )
    except Exception as exc:
        logger.error("Conversion failed for %s: %s", step_path.name, exc)
        return BatchResult(
            asset_name=step_path.name,
            step_path=step_path,
            usd_path=None,
            status="failed",
            error=str(exc),
        )


def run_batch(
    source: str | None = None,
    category: str | None = None,
    max_assets: int = 10,
    quality_min: float = 0.5,
    workers: int = 4,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    db_path: Path = _DEFAULT_DB_PATH,
    config_path: Path | None = None,
    dest_dir: Path = _DEFAULT_STEP_DIR,
    material_default: str | None = None,
) -> list[BatchResult]:
    """Batch acquire, convert, score, and catalog assets.

    Workflow:
    1. Acquire up to max_assets STEP files from the given source.
    2. Convert + score each in parallel using ThreadPoolExecutor.
    3. Write passing assets to the SQLite catalog.
    4. Return results for all processed assets.
    """
    import simready.acquisition.github_source   # noqa: F401
    import simready.acquisition.abc_dataset     # noqa: F401
    import simready.acquisition.nist_source     # noqa: F401
    import simready.acquisition.freecad_source  # noqa: F401
    from simready.acquisition.agent import acquire

    output_dir = Path(output_dir)
    # Ensure all category subdirs exist under both step_files and output
    for cat in CATEGORIES:
        (Path(dest_dir) / cat).mkdir(parents=True, exist_ok=True)
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    # --- Acquire ---
    query = category or source or "mechanical"
    sources = [source] if source else None
    logger.info("Batch acquiring up to %d assets (query=%r, sources=%s)", max_assets, query, sources)
    # Route downloads into the category subdir of step_files
    cat_folder = category if (category and category in CATEGORIES) else category_for(query)
    category_dest = Path(dest_dir) / cat_folder
    category_dest.mkdir(parents=True, exist_ok=True)

    assets = asyncio.run(
        acquire(
            query=query,
            dest_dir=category_dest,
            catalog_path=db_path.with_suffix(".json"),  # legacy JSON kept alongside
            max_per_source=max_assets,
            sources=sources,
        )
    )

    if not assets:
        logger.warning("No assets acquired for query %r", query)
        return []

    # Collect STEP files from all downloaded assets
    step_files: list[tuple] = []  # (step_path, step_asset)
    for asset in assets:
        if not asset.local_path:
            continue
        lp = Path(asset.local_path)
        if lp.is_file() and lp.suffix.lower() in (".step", ".stp", ".stl", ".obj"):
            step_files.append((lp, asset))
        elif lp.is_dir():
            for ext in ("*.step", "*.stp", "*.stl", "*.obj"):
                for f in lp.rglob(ext):
                    step_files.append((f, asset))

    step_files = step_files[:max_assets]
    logger.info("Converting %d STEP file(s) with %d worker(s)", len(step_files), workers)

    # --- Convert in parallel, routing each file to its category subdir ---
    results: list[BatchResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_info = {
            pool.submit(
                _convert_one,
                sp,
                output_dir / category_for(sp.stem),  # category-routed output
                config_path,
                quality_min,
                material_default,
            ): (sp, sa)
            for sp, sa in step_files
        }
        for future in as_completed(future_to_info):
            result = future.result()
            results.append(result)
            logger.info(
                "[%s] %s (score=%.2f)",
                result.status.upper(),
                result.asset_name,
                result.quality_score or 0.0,
            )

    # --- Write passing assets to catalog ---
    conn = open_db(db_path)
    step_path_to_asset = {Path(sa.local_path): sa for _, sa in step_files if sa.local_path}
    for result in results:
        if result.status != "passed" or result.step_path is None:
            continue
        step_asset = step_path_to_asset.get(result.step_path)
        if step_asset is None:
            continue
        entry = from_step_asset(step_asset)
        entry.category = category or category_for(result.step_path.stem)
        entry.material_class = result.material_class
        entry.face_count = result.face_count
        entry.quality_score = result.quality_score
        entry.watertight = result.watertight
        entry.physics_complete = result.physics_complete
        entry.material_confidence = result.material_confidence
        entry.usd_path = result.usd_path
        upsert_asset(conn, entry)

    conn.close()
    return results


def print_batch_summary(results: list[BatchResult]) -> None:
    """Print a human-readable summary table of batch results."""
    passed  = [r for r in results if r.status == "passed"]
    failed  = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]

    print(
        f"\nBatch complete: {len(results)} processed | "
        f"{len(passed)} passed | {len(skipped)} skipped | {len(failed)} failed\n"
    )

    if passed:
        print("Passed assets:")
        print(f"  {'Name':<40} {'Score':>6}  {'Watertight':>10}  {'Physics':>8}  {'MatClass':<16}")
        print("  " + "-" * 84)
        for r in sorted(passed, key=lambda x: -(x.quality_score or 0)):
            print(
                f"  {r.asset_name:<40} {r.quality_score or 0:>6.2f}  "
                f"{'yes' if r.watertight else 'no':>10}  "
                f"{'yes' if r.physics_complete else 'no':>8}  "
                f"{r.material_class or '-':<16}"
            )

    if failed:
        print(f"\nFailed assets ({len(failed)}):")
        for r in failed:
            print(f"  {r.asset_name}: {r.error}")

    print()
