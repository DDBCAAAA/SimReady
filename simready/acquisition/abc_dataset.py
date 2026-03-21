"""ABC Dataset source — downloads from the ABC CAD dataset (Koch et al. 2019).

The ABC Dataset contains ~1M CAD models from Onshape, stored as STEP files.
Reference: https://deep-geometry.github.io/abc-dataset/

Chunk URLs and bitstream IDs are sourced from the official manifest:
  https://deep-geometry.github.io/abc-dataset/data/step_v00.txt

Each chunk is a .7z archive containing ~100 STEP files.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import aiohttp

from simready.acquisition.sources import STEPAsset, STEPSource, register_source

logger = logging.getLogger(__name__)

_MANIFEST_URL = "https://deep-geometry.github.io/abc-dataset/data/step_v00.txt"


def _parse_manifest(text: str) -> list[tuple[str, str]]:
    """Parse manifest lines into (url, filename) pairs."""
    pairs = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


@register_source
class ABCDatasetSource(STEPSource):
    """Access STEP files from the ABC Dataset."""

    @property
    def name(self) -> str:
        return "abc_dataset"

    async def _fetch_manifest(self, session: aiohttp.ClientSession) -> list[tuple[str, str]]:
        """Fetch and parse the chunk manifest."""
        async with session.get(_MANIFEST_URL) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch ABC manifest: {resp.status}")
            text = await resp.text()
        return _parse_manifest(text)

    async def search(self, query: str, max_results: int = 10) -> list[STEPAsset]:
        """List available STEP chunks from ABC Dataset manifest.

        Since ABC has no search API, this returns chunks from the manifest
        up to max_results. The query parameter is ignored (no search index).
        """
        async with aiohttp.ClientSession() as session:
            try:
                chunks = await self._fetch_manifest(session)
            except Exception as e:
                logger.error("ABC Dataset manifest unavailable: %s", e)
                return []

        assets: list[STEPAsset] = []
        for url, filename in chunks[:max_results]:
            chunk_id = filename.split("_")[1]  # e.g. "abc_0000_step_v00.7z" → "0000"
            asset = STEPAsset(
                name=filename,
                url=url,
                source="abc_dataset",
                description=f"ABC Dataset chunk {chunk_id} (~100 STEP files)",
                license="MIT",
                tags=["abc-dataset", "cad", "mechanical"],
            )
            assets.append(asset)

        logger.info("ABC Dataset: listed %d chunks", len(assets))
        return assets

    async def download(self, asset: STEPAsset, dest_dir: Path) -> Path:
        """Download and extract a .7z chunk of STEP files from ABC Dataset."""
        try:
            import py7zr
        except ImportError:
            raise ImportError(
                "py7zr is required for ABC Dataset extraction. "
                "Install with: pip install py7zr"
            )

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(asset.url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to download {asset.url}: {resp.status}")
                content = await resp.read()

        with tempfile.NamedTemporaryFile(suffix=".7z", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        stem = asset.name.replace(".7z", "")
        extracted_dir = dest_dir / stem
        extracted_dir.mkdir(parents=True, exist_ok=True)

        try:
            with py7zr.SevenZipFile(tmp_path, mode="r") as archive:
                all_names = archive.getnames()
                step_names = [
                    n for n in all_names
                    if n.lower().endswith((".step", ".stp"))
                ]
                archive.extract(targets=step_names, path=extracted_dir)

            logger.info(
                "Extracted %d STEP files from %s → %s",
                len(step_names),
                asset.name,
                extracted_dir,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        asset.local_path = extracted_dir
        return extracted_dir
