"""Tests for the NIST and FreeCAD acquisition sources."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simready.acquisition.sources import _REGISTRY


# --- Registration ---

def test_new_sources_registered():
    import simready.acquisition.nist_source    # noqa: F401
    import simready.acquisition.freecad_source # noqa: F401
    assert "nist" in _REGISTRY
    assert "freecad" in _REGISTRY


# --- NIST source ---

@pytest.mark.asyncio
async def test_nist_search_returns_all_on_empty_query():
    from simready.acquisition.nist_source import NISTSource
    source = NISTSource()
    assets = await source.search("", max_results=10)
    assert len(assets) == 4
    assert all(a.source == "nist" for a in assets)
    assert all(a.license == "Public Domain" for a in assets)


@pytest.mark.asyncio
async def test_nist_search_filters_by_keyword():
    from simready.acquisition.nist_source import NISTSource
    source = NISTSource()

    # "plate" matches the PLATE part and the ASSEMBLY (description: "plate + cover + box")
    plate_assets = await source.search("plate")
    assert len(plate_assets) == 2
    assert any("PLATE" in a.name for a in plate_assets)

    assembly_assets = await source.search("assembly")
    assert len(assembly_assets) == 1
    assert "ASSEMBLY" in assembly_assets[0].name


@pytest.mark.asyncio
async def test_nist_search_no_match():
    from simready.acquisition.nist_source import NISTSource
    source = NISTSource()
    assets = await source.search("nonexistent_xyz")
    assert assets == []


@pytest.mark.asyncio
async def test_nist_search_respects_max_results():
    from simready.acquisition.nist_source import NISTSource
    source = NISTSource()
    assets = await source.search("", max_results=2)
    assert len(assets) == 2


@pytest.mark.asyncio
async def test_nist_download(tmp_path: Path):
    from simready.acquisition.nist_source import NISTSource
    from simready.acquisition.sources import STEPAsset

    source = NISTSource()
    asset = STEPAsset(
        name="NIST_MTC_CRADA_PLATE_REV-A.STP",
        url="https://raw.githubusercontent.com/usnistgov/AR-PMI/master/Assets/StreamingAssets/STEP/NIST_MTC_CRADA_PLATE_REV-A.STP",
        source="nist",
    )

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 200
    mock_resp.read = AsyncMock(return_value=b"ISO-10303-21; STEP DATA")

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await source.download(asset, tmp_path)

    assert result.exists()
    assert result.read_bytes() == b"ISO-10303-21; STEP DATA"
    assert asset.local_path == result


# --- FreeCAD source ---

_MOCK_TREE_RESPONSE = {
    "tree": [
        {"path": "Fasteners/DIN/bolt_m8.step", "size": 51200, "type": "blob"},
        {"path": "Fasteners/DIN/bolt_m10.stp", "size": 61200, "type": "blob"},
        {"path": "Mechanical Parts/gear_spur.step", "size": 102400, "type": "blob"},
        {"path": "Mechanical Parts/gear_helical.step", "size": 98000, "type": "blob"},
        {"path": "Architectural Parts/bracket_wall.step", "size": 75000, "type": "blob"},
        {"path": "README.md", "size": 1024, "type": "blob"},
    ]
}


@pytest.mark.asyncio
async def test_freecad_search_filters_by_keyword():
    from simready.acquisition.freecad_source import FreeCADSource

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=_MOCK_TREE_RESPONSE)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    source = FreeCADSource()
    with patch("aiohttp.ClientSession", return_value=mock_session):
        assets = await source.search("gear", max_results=10)

    assert len(assets) == 2
    assert all("gear" in a.name.lower() for a in assets)
    assert all(a.source == "freecad" for a in assets)


@pytest.mark.asyncio
async def test_freecad_search_respects_max_results():
    from simready.acquisition.freecad_source import FreeCADSource

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=_MOCK_TREE_RESPONSE)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    source = FreeCADSource()
    with patch("aiohttp.ClientSession", return_value=mock_session):
        assets = await source.search("bolt", max_results=1)

    assert len(assets) == 1


@pytest.mark.asyncio
async def test_freecad_search_excludes_non_step_files():
    from simready.acquisition.freecad_source import FreeCADSource

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=_MOCK_TREE_RESPONSE)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    source = FreeCADSource()
    with patch("aiohttp.ClientSession", return_value=mock_session):
        # README.md should never appear even with a broad query
        assets = await source.search("readme", max_results=10)

    assert assets == []


@pytest.mark.asyncio
async def test_freecad_tree_is_cached():
    """Tree fetch should happen only once per source instance."""
    from simready.acquisition.freecad_source import FreeCADSource

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=_MOCK_TREE_RESPONSE)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    source = FreeCADSource()
    with patch("aiohttp.ClientSession", return_value=mock_session):
        await source.search("bolt", max_results=5)
        await source.search("gear", max_results=5)

    # json() called only once — tree was cached after first search
    assert mock_resp.json.call_count == 1


@pytest.mark.asyncio
async def test_freecad_download(tmp_path: Path):
    from simready.acquisition.freecad_source import FreeCADSource
    from simready.acquisition.sources import STEPAsset

    source = FreeCADSource()
    asset = STEPAsset(
        name="gear_spur.step",
        url="https://raw.githubusercontent.com/FreeCAD/FreeCAD-library/master/Mechanical Parts/gear_spur.step",
        source="freecad",
    )

    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 200
    mock_resp.read = AsyncMock(return_value=b"ISO-10303-21; STEP DATA")

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await source.download(asset, tmp_path)

    assert result.exists()
    assert result.name == "gear_spur.step"
    assert asset.local_path == result
