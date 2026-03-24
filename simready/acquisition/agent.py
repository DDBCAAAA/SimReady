"""Acquisition agent — automated search, download, and cataloging of STEP files."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from simready import PROJECT_ROOT
from simready.acquisition.sources import STEPAsset, get_source, list_sources

logger = logging.getLogger(__name__)

# Default paths — absolute so they work regardless of invocation directory
_DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "data" / "step_files"
_DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"


async def search_all_sources(
    query: str,
    max_per_source: int = 10,
    sources: list[str] | None = None,
) -> list[STEPAsset]:
    """Search all registered sources in parallel.

    Args:
        query: Search query string.
        max_per_source: Max results per source.
        sources: Specific sources to search (None = all).

    Returns:
        Combined list of STEPAsset results.
    """
    source_names = sources or list_sources()
    tasks = []

    for name in source_names:
        try:
            src = get_source(name)
            tasks.append(src.search(query, max_per_source))
        except KeyError:
            logger.warning("Unknown source: %s", name)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    assets: list[STEPAsset] = []
    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            logger.error("Search failed for %s: %s", name, result)
        else:
            assets.extend(result)

    logger.info("Found %d total assets across %d sources", len(assets), len(source_names))
    return assets


async def download_assets(
    assets: list[STEPAsset],
    dest_dir: Path = _DEFAULT_DOWNLOAD_DIR,
    max_concurrent: int = 5,
) -> list[STEPAsset]:
    """Download multiple assets with concurrency control.

    Args:
        assets: Assets to download.
        dest_dir: Root download directory.
        max_concurrent: Max parallel downloads.

    Returns:
        Assets with local_path populated (failed ones are excluded).
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    completed: list[STEPAsset] = []

    async def _download_one(asset: STEPAsset) -> STEPAsset | None:
        async with semaphore:
            try:
                src = get_source(asset.source)
                source_dir = dest_dir / asset.source
                await src.download(asset, source_dir)
                return asset
            except Exception as e:
                logger.error("Failed to download %s: %s", asset.name, e)
                return None

    results = await asyncio.gather(*[_download_one(a) for a in assets])
    completed = [r for r in results if r is not None]

    logger.info("Downloaded %d/%d assets", len(completed), len(assets))
    return completed


def save_catalog(assets: list[STEPAsset], catalog_path: Path = _DEFAULT_CATALOG_PATH) -> None:
    """Save asset metadata to a JSON catalog for tracking."""
    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing catalog if present
    existing: list[dict] = []
    if catalog_path.exists():
        existing = json.loads(catalog_path.read_text())

    # Merge — deduplicate by URL, update paths if already present
    existing_by_url = {e["url"]: e for e in existing}
    for asset in assets:
        entry = asdict(asset)
        entry["local_path"] = str(asset.local_path) if asset.local_path else None
        entry["usd_path"] = str(asset.usd_path) if asset.usd_path else None
        existing_by_url[asset.url] = entry

    catalog_path.write_text(json.dumps(list(existing_by_url.values()), indent=2))
    logger.info("Catalog saved: %d entries → %s", len(existing_by_url), catalog_path)


def load_catalog(catalog_path: Path = _DEFAULT_CATALOG_PATH) -> list[STEPAsset]:
    """Load asset catalog from JSON."""
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        return []

    raw = json.loads(catalog_path.read_text())
    assets = []
    for entry in raw:
        local = entry.pop("local_path", None)
        usd = entry.pop("usd_path", None)
        asset = STEPAsset(**entry)
        asset.local_path = Path(local) if local else None
        asset.usd_path = Path(usd) if usd else None
        assets.append(asset)
    return assets


def _step_files_from_path(path: Path) -> list[Path]:
    """Return convertible CAD/mesh files from a file path or a directory (recursive)."""
    if path.is_file():
        return [path]
    files = []
    for ext in ("*.step", "*.stp", "*.stl", "*.obj"):
        files.extend(path.rglob(ext))
    return sorted(files)


def convert_acquired_assets(
    assets: list[STEPAsset],
    output_dir: Path,
    config_path: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Convert downloaded STEP assets to USD.

    Handles both single-file (GitHub) and directory (ABC Dataset) local paths.
    For single-file assets, sets asset.usd_path on success.

    Args:
        assets: Downloaded STEPAssets (local_path must be set).
        output_dir: Where to write .usd output files.
        config_path: Optional pipeline config YAML.

    Returns:
        List of (step_path, usd_path) for each successful conversion.
    """
    from simready.pipeline import run as pipeline_run

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, Path]] = []

    for asset in assets:
        if not asset.local_path:
            logger.warning("Skipping %s — not downloaded", asset.name)
            continue

        step_files = _step_files_from_path(Path(asset.local_path))
        if not step_files:
            logger.warning("No CAD files found at %s", asset.local_path)
            continue

        for step_path in step_files:
            usd_path = output_dir / (step_path.stem + ".usd")
            try:
                pipeline_run(step_path, usd_path, config_path)
                # Track usd_path on single-file assets (local_path is the file itself)
                if Path(asset.local_path).is_file():
                    asset.usd_path = usd_path
                results.append((step_path, usd_path))
                logger.info("Converted %s → %s", step_path.name, usd_path.name)
            except Exception as e:
                logger.error("Conversion failed for %s: %s", step_path.name, e)

    logger.info("Converted %d file(s) to USD in %s", len(results), output_dir)
    return results


async def acquire(
    query: str,
    dest_dir: Path = _DEFAULT_DOWNLOAD_DIR,
    catalog_path: Path = _DEFAULT_CATALOG_PATH,
    max_per_source: int = 10,
    max_concurrent: int = 5,
    sources: list[str] | None = None,
    download: bool = True,
) -> list[STEPAsset]:
    """Full acquisition pipeline: search → download → catalog.

    Args:
        query: Search query (e.g. "gear", "bracket", "automotive").
        dest_dir: Where to save downloaded files.
        catalog_path: Path to the JSON catalog file.
        max_per_source: Max results per source.
        max_concurrent: Max parallel downloads.
        sources: Specific sources to search.
        download: Whether to download (False = search only).

    Returns:
        List of acquired STEPAsset objects.
    """
    logger.info("Acquisition agent starting — query='%s'", query)

    # 1. Search
    assets = await search_all_sources(query, max_per_source, sources)
    if not assets:
        logger.warning("No assets found for query: '%s'", query)
        return []

    # 2. Download
    if download:
        assets = await download_assets(assets, dest_dir, max_concurrent)

    # 3. Catalog
    save_catalog(assets, catalog_path)

    logger.info(
        "Acquisition complete: %d assets cataloged at %s",
        len(assets),
        catalog_path,
    )
    return assets
