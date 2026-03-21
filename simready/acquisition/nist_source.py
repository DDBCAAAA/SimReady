"""NIST CAD source — NIST MTC CRADA manufacturing STEP files from the AR-PMI repository.

Files are NIST-produced physical manufacturing test parts (box, cover, plate, assembly)
hosted on GitHub under usnistgov/AR-PMI. No authentication required.

Reference: https://github.com/usnistgov/AR-PMI
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiohttp

from simready.acquisition.sources import STEPAsset, STEPSource, register_source

logger = logging.getLogger(__name__)

_REPO = "usnistgov/AR-PMI"
_BRANCH = "master"
_STEP_DIR = "Assets/StreamingAssets/STEP"
_RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}/{_STEP_DIR}"

# Hardcoded catalog — all STEP files in the repo, with descriptions.
# These are validated NIST manufacturing test parts used in MBE/PMI conformance testing.
_NIST_FILES: list[dict] = [
    {
        "name": "NIST_MTC_CRADA_PLATE_REV-A.STP",
        "size_bytes": 4454503,
        "description": "NIST MTC CRADA flat plate with GD&T features (Rev A)",
        "tags": ["nist", "plate", "gdt", "manufacturing", "ap242"],
    },
    {
        "name": "NIST_MTC_CRADA_COVER_REV-B.STP",
        "size_bytes": 4330617,
        "description": "NIST MTC CRADA cover component with tolerances (Rev B)",
        "tags": ["nist", "cover", "gdt", "manufacturing", "ap242"],
    },
    {
        "name": "NIST_MTC_CRADA_BOX_REV-D.STP",
        "size_bytes": 4805400,
        "description": "NIST MTC CRADA box component with PMI annotations (Rev D)",
        "tags": ["nist", "box", "pmi", "manufacturing", "ap242"],
    },
    {
        "name": "NIST_MTC_CRADA_ASSEMBLY_REV-D.STP",
        "size_bytes": 10594670,
        "description": "NIST MTC CRADA full assembly (plate + cover + box, Rev D)",
        "tags": ["nist", "assembly", "pmi", "manufacturing", "ap242"],
    },
]


@register_source
class NISTSource(STEPSource):
    """NIST manufacturing STEP files from the AR-PMI GitHub repository."""

    @property
    def name(self) -> str:
        return "nist"

    async def search(self, query: str, max_results: int = 10) -> list[STEPAsset]:
        """Filter NIST STEP files by keyword match in name or description.

        Since only 4 files exist, all are returned when query is empty or matches.
        """
        query_lower = query.lower()
        assets: list[STEPAsset] = []

        for meta in _NIST_FILES:
            name_lower = meta["name"].lower()
            desc_lower = meta["description"].lower()
            tags = meta["tags"]

            if not query_lower or any(
                query_lower in text
                for text in [name_lower, desc_lower] + tags
            ):
                assets.append(STEPAsset(
                    name=meta["name"],
                    url=f"{_RAW_BASE}/{meta['name']}",
                    source="nist",
                    size_bytes=meta["size_bytes"],
                    license="Public Domain",
                    description=meta["description"],
                    tags=meta["tags"],
                ))

            if len(assets) >= max_results:
                break

        logger.info("NIST source: %d file(s) matched query '%s'", len(assets), query)
        return assets

    async def download(self, asset: STEPAsset, dest_dir: Path) -> Path:
        """Download a NIST STEP file from raw.githubusercontent.com."""
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
