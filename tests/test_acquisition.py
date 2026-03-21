"""Tests for the acquisition module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from simready.acquisition.sources import STEPAsset, _REGISTRY

# Import sources to trigger registration
import simready.acquisition.github_source  # noqa: F401
import simready.acquisition.abc_dataset  # noqa: F401


def test_sources_registered():
    assert "github" in _REGISTRY
    assert "abc_dataset" in _REGISTRY


def test_step_asset_creation():
    asset = STEPAsset(
        name="gear.step",
        url="https://example.com/gear.step",
        source="github",
        license="MIT",
    )
    assert asset.name == "gear.step"
    assert asset.local_path is None


@pytest.mark.asyncio
async def test_github_search_parses_response():
    from simready.acquisition.github_source import GitHubSource

    mock_response = {
        "items": [
            {
                "name": "bracket.step",
                "path": "models/bracket.step",
                "size": 12345,
                "repository": {
                    "full_name": "user/repo",
                    "default_branch": "main",
                    "description": "A bracket",
                    "license": {"spdx_id": "MIT"},
                    "topics": [],
                },
            }
        ]
    }

    source = GitHubSource()
    source._fetch_json = AsyncMock(return_value=(200, mock_response))

    assets = await source.search("bracket", max_results=5)

    assert len(assets) >= 1
    assert assets[0].name == "bracket.step"
    assert assets[0].source == "github"
    assert assets[0].license == "MIT"


@pytest.mark.asyncio
async def test_github_search_handles_rate_limit():
    from simready.acquisition.github_source import GitHubSource

    source = GitHubSource()
    source._fetch_json = AsyncMock(return_value=(403, {}))

    assets = await source.search("gear", max_results=5)
    assert assets == []


@pytest.mark.asyncio
async def test_github_search_handles_no_auth():
    from simready.acquisition.github_source import GitHubSource

    source = GitHubSource()
    source._fetch_json = AsyncMock(return_value=(401, {}))

    assets = await source.search("gear", max_results=5)
    assert assets == []


@pytest.mark.asyncio
async def test_github_download(tmp_path: Path):
    from simready.acquisition.github_source import GitHubSource

    source = GitHubSource()
    source._fetch_bytes = AsyncMock(return_value=(200, b"STEP FILE CONTENT"))

    asset = STEPAsset(name="test.step", url="https://example.com/test.step", source="github")
    path = await source.download(asset, tmp_path)

    assert path.exists()
    assert path.read_bytes() == b"STEP FILE CONTENT"
    assert asset.local_path == path


def test_abc_manifest_parsing():
    from simready.acquisition.abc_dataset import _parse_manifest

    text = (
        "https://archive.nyu.edu/rest/bitstreams/88598/retrieve abc_0000_step_v00.7z\n"
        "https://archive.nyu.edu/rest/bitstreams/88602/retrieve abc_0001_step_v00.7z\n"
    )
    pairs = _parse_manifest(text)
    assert len(pairs) == 2
    assert pairs[0] == ("https://archive.nyu.edu/rest/bitstreams/88598/retrieve", "abc_0000_step_v00.7z")


@pytest.mark.asyncio
async def test_abc_search_uses_manifest():
    from simready.acquisition.abc_dataset import ABCDatasetSource

    manifest_pairs = [
        ("https://archive.nyu.edu/rest/bitstreams/88598/retrieve", "abc_0000_step_v00.7z"),
        ("https://archive.nyu.edu/rest/bitstreams/88602/retrieve", "abc_0001_step_v00.7z"),
        ("https://archive.nyu.edu/rest/bitstreams/88634/retrieve", "abc_0002_step_v00.7z"),
    ]
    source = ABCDatasetSource()
    source._fetch_manifest = AsyncMock(return_value=manifest_pairs)

    assets = await source.search("gear", max_results=2)

    assert len(assets) == 2
    assert assets[0].url == "https://archive.nyu.edu/rest/bitstreams/88598/retrieve"
    assert assets[0].name == "abc_0000_step_v00.7z"
    assert assets[0].source == "abc_dataset"


@pytest.mark.asyncio
async def test_abc_download_extracts_step_files(tmp_path: Path):
    import io
    import py7zr
    from unittest.mock import MagicMock
    from simready.acquisition.abc_dataset import ABCDatasetSource

    # Build a real in-memory .7z archive with one STEP file
    buf = io.BytesIO()
    with py7zr.SevenZipFile(buf, mode="w") as archive:
        archive.writestr(b"STEP DATA", "part.step")
    archive_bytes = buf.getvalue()

    asset = STEPAsset(
        name="abc_0000_step_v00.7z",
        url="https://archive.nyu.edu/rest/bitstreams/88598/retrieve",
        source="abc_dataset",
    )
    source = ABCDatasetSource()

    # session.get(url) must return an async context manager, not a coroutine
    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 200
    mock_resp.read = AsyncMock(return_value=archive_bytes)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)  # not AsyncMock

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await source.download(asset, tmp_path)

    assert result.is_dir()
    step_files = list(result.rglob("*.step"))
    assert len(step_files) == 1
    assert asset.local_path == result


def test_catalog_round_trip(tmp_path: Path):
    from simready.acquisition.agent import save_catalog, load_catalog

    assets = [
        STEPAsset(name="a.step", url="https://x.com/a.step", source="github"),
        STEPAsset(name="b.stp", url="https://x.com/b.stp", source="abc_dataset", license="MIT"),
    ]
    catalog_path = tmp_path / "catalog.json"

    save_catalog(assets, catalog_path)
    loaded = load_catalog(catalog_path)

    assert len(loaded) == 2
    assert loaded[0].name == "a.step"
    assert loaded[1].license == "MIT"


def test_catalog_deduplicates(tmp_path: Path):
    from simready.acquisition.agent import save_catalog, load_catalog

    asset = STEPAsset(name="a.step", url="https://x.com/a.step", source="github")
    catalog_path = tmp_path / "catalog.json"

    save_catalog([asset], catalog_path)
    save_catalog([asset], catalog_path)  # save again — should not duplicate

    loaded = load_catalog(catalog_path)
    assert len(loaded) == 1
