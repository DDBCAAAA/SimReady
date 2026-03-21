"""FreeCAD Library source — STEP files from the FreeCAD/FreeCAD-library GitHub repo.

Contains 2,893 STEP files covering mechanical, architectural, fastener, and hardware parts.
The full git tree is fetched in one unauthenticated API call and filtered by query.
No authentication required (60 req/hr unauthenticated rate limit applies).

Reference: https://github.com/FreeCAD/FreeCAD-library
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiohttp

from simready.acquisition.sources import STEPAsset, STEPSource, register_source

logger = logging.getLogger(__name__)

_REPO = "FreeCAD/FreeCAD-library"
_BRANCH = "master"
_TREE_URL = f"https://api.github.com/repos/{_REPO}/git/trees/{_BRANCH}?recursive=1"
_RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}"


def _classify_license(path: str) -> str:
    """Infer a license note from the file path (FreeCAD library has mixed per-part licenses)."""
    # The FreeCAD library README notes parts have various licenses — LGPLv2+ for FreeCAD files.
    # STEP re-exports have no explicit per-file license; attribution to FreeCAD library repo.
    return "See FreeCAD-library/LICENSE"


@register_source
class FreeCADSource(STEPSource):
    """STEP files from the FreeCAD parts library on GitHub."""

    def __init__(self) -> None:
        self._tree_cache: list[dict] | None = None

    @property
    def name(self) -> str:
        return "freecad"

    async def _fetch_tree(self, session: aiohttp.ClientSession) -> list[dict]:
        """Fetch the full repo tree (cached per instance)."""
        if self._tree_cache is not None:
            return self._tree_cache

        async with session.get(_TREE_URL) as resp:
            if resp.status == 403:
                raise RuntimeError(
                    "GitHub API rate limit hit. Set GITHUB_TOKEN for higher limits."
                )
            if resp.status != 200:
                raise RuntimeError(
                    f"Failed to fetch FreeCAD library tree: HTTP {resp.status}"
                )
            data = await resp.json()

        self._tree_cache = [
            item for item in data.get("tree", [])
            if item.get("path", "").lower().endswith((".step", ".stp"))
        ]
        logger.info("FreeCAD library tree fetched: %d STEP files", len(self._tree_cache))
        return self._tree_cache

    async def search(self, query: str, max_results: int = 10) -> list[STEPAsset]:
        """Search FreeCAD library STEP files by keyword match in file path.

        Args:
            query: Keyword to match against file paths (e.g. 'gear', 'bolt', 'bracket').
            max_results: Maximum number of results to return.
        """
        async with aiohttp.ClientSession() as session:
            try:
                tree = await self._fetch_tree(session)
            except Exception as e:
                logger.error("FreeCAD source unavailable: %s", e)
                return []

        query_lower = query.lower()
        assets: list[STEPAsset] = []

        for item in tree:
            path = item.get("path", "")
            if query_lower in path.lower():
                name = Path(path).name
                assets.append(STEPAsset(
                    name=name,
                    url=f"{_RAW_BASE}/{path}",
                    source="freecad",
                    size_bytes=item.get("size"),
                    license=_classify_license(path),
                    description=path,  # full path as description for navigation context
                    tags=["freecad", "mechanical", Path(path).parts[0].lower()],
                ))
            if len(assets) >= max_results:
                break

        logger.info(
            "FreeCAD search for '%s': %d result(s)", query, len(assets)
        )
        return assets

    async def download(self, asset: STEPAsset, dest_dir: Path) -> Path:
        """Download a STEP file from raw.githubusercontent.com."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / asset.name

        async with aiohttp.ClientSession() as session:
            async with session.get(asset.url) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Failed to download {asset.url}: HTTP {resp.status}"
                    )
                content = await resp.read()

        dest_path.write_bytes(content)
        asset.local_path = dest_path
        logger.info(
            "Downloaded %s (%d bytes) → %s", asset.name, len(content), dest_path
        )
        return dest_path
