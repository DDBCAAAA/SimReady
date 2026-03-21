"""GitHub source — searches public repos for STEP files via the GitHub API."""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any

import aiohttp

from simready.acquisition.sources import STEPAsset, STEPSource, register_source

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_SEARCH_CODE_URL = f"{_GITHUB_API}/search/code"
_BRANCH_FALLBACKS = ("main", "master", "develop", "trunk")


def _branch_fallback_urls(url: str) -> list[str]:
    """Return the URL followed by variants with common branch names substituted.

    raw.githubusercontent.com URLs have the form:
      https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}
    """
    prefix = "https://raw.githubusercontent.com/"
    if not url.startswith(prefix):
        return [url]
    rest = url[len(prefix):]
    parts = rest.split("/", 3)  # owner, repo, branch, path
    if len(parts) != 4:
        return [url]
    owner, repo, current_branch, path = parts
    urls = [url]
    for branch in _BRANCH_FALLBACKS:
        if branch != current_branch:
            urls.append(f"{prefix}{owner}/{repo}/{branch}/{path}")
    return urls


@register_source
class GitHubSource(STEPSource):
    """Search and download STEP files from public GitHub repositories."""

    @property
    def name(self) -> str:
        return "github"

    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self._headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self._token:
            self._headers["Authorization"] = f"token {self._token}"

    async def _fetch_json(self, url: str) -> tuple[int, dict[str, Any]]:
        """Fetch a URL and return (status_code, json_body). Separated for testability."""
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(url) as resp:
                data = await resp.json() if resp.status == 200 else {}
                return resp.status, data

    async def _fetch_bytes(self, url: str) -> tuple[int, bytes]:
        """Fetch raw bytes from a URL. Separated for testability."""
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(url) as resp:
                content = await resp.read()
                return resp.status, content

    def _parse_search_items(self, items: list[dict]) -> list[STEPAsset]:
        """Parse GitHub code search response items into STEPAsset objects."""
        assets = []
        for item in items:
            repo = item["repository"]
            raw_url = (
                f"https://raw.githubusercontent.com/"
                f"{repo['full_name']}/{repo.get('default_branch', 'main')}/{item['path']}"
            )
            asset = STEPAsset(
                name=item["name"],
                url=raw_url,
                source="github",
                size_bytes=item.get("size"),
                license=repo.get("license", {}).get("spdx_id") if repo.get("license") else None,
                description=f"{repo['full_name']}: {repo.get('description', '')}",
                tags=[t["name"] for t in repo.get("topics", [])],
            )
            assets.append(asset)
        return assets

    async def search(self, query: str, max_results: int = 10) -> list[STEPAsset]:
        """Search GitHub for STEP files matching a query.

        Uses the code search API to find .step/.stp files in public repos.
        """
        assets: list[STEPAsset] = []
        # Allocate budget evenly across extensions so STL isn't starved by STEP results.
        extensions = ("step", "stp", "stl")
        per_ext = max(1, (max_results + len(extensions) - 1) // len(extensions))
        per_page = min(per_ext, 30)

        for ext in extensions:
            search_query = f"{query} extension:{ext}"
            params = {"q": search_query, "per_page": per_page}
            url = f"{_SEARCH_CODE_URL}?{urllib.parse.urlencode(params)}"

            status, data = await self._fetch_json(url)

            if status in (401, 403):
                logger.warning(
                    "GitHub code search requires authentication. "
                    "Set the GITHUB_TOKEN environment variable and retry."
                )
                break
            if status != 200:
                logger.error("GitHub search failed: %d", status)
                continue

            new_assets = self._parse_search_items(data.get("items", []))
            assets.extend(new_assets)

        assets = assets[:max_results]

        logger.info("GitHub search for '%s': found %d CAD files", query, len(assets))
        return assets

    async def download(self, asset: STEPAsset, dest_dir: Path) -> Path:
        """Download a CAD file from its raw GitHub URL, retrying with branch fallbacks on 404."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / asset.name

        urls = _branch_fallback_urls(asset.url)
        for url in urls:
            status, content = await self._fetch_bytes(url)
            if status == 200:
                if url != asset.url:
                    logger.debug("Branch fallback succeeded: %s", url)
                    asset.url = url
                break
        else:
            raise RuntimeError(f"Failed to download {asset.url}: {status}")

        dest_path.write_bytes(content)
        asset.local_path = dest_path
        logger.info("Downloaded %s (%d bytes) → %s", asset.name, len(content), dest_path)
        return dest_path
