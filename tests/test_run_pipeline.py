"""Tests for the end-to-end acquire → convert pipeline (run subcommand)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from simready.acquisition.sources import STEPAsset
from simready.acquisition.agent import convert_acquired_assets, _step_files_from_path


# --- _step_files_from_path ---

def test_step_files_from_single_file(tmp_path):
    f = tmp_path / "part.step"
    f.write_bytes(b"")
    assert _step_files_from_path(f) == [f]


def test_step_files_from_directory(tmp_path):
    (tmp_path / "a.step").write_bytes(b"")
    (tmp_path / "b.stp").write_bytes(b"")
    (tmp_path / "readme.txt").write_bytes(b"")
    files = _step_files_from_path(tmp_path)
    names = {f.name for f in files}
    assert names == {"a.step", "b.stp"}


def test_step_files_from_empty_directory(tmp_path):
    assert _step_files_from_path(tmp_path) == []


# --- convert_acquired_assets ---

def test_convert_single_file_asset(tmp_path):
    step_file = tmp_path / "part.step"
    step_file.write_bytes(b"STEP data")
    asset = STEPAsset(name="part.step", url="http://example.com/part.step", source="github")
    asset.local_path = step_file

    with patch("simready.pipeline.run") as mock_run:
        results = convert_acquired_assets([asset], tmp_path / "usd")

    assert len(results) == 1
    step_path, usd_path = results[0]
    assert step_path == step_file
    assert usd_path == tmp_path / "usd" / "part.usda"
    mock_run.assert_called_once_with(step_file, tmp_path / "usd" / "part.usda", None)
    assert asset.usd_path == usd_path


def test_convert_directory_asset(tmp_path):
    extracted = tmp_path / "abc_chunk"
    extracted.mkdir()
    (extracted / "part_a.step").write_bytes(b"STEP data")
    (extracted / "part_b.stp").write_bytes(b"STP data")
    asset = STEPAsset(name="abc_0000.tar.gz", url="http://example.com/chunk", source="abc_dataset")
    asset.local_path = extracted

    with patch("simready.pipeline.run") as mock_run:
        results = convert_acquired_assets([asset], tmp_path / "usd")

    assert len(results) == 2
    assert mock_run.call_count == 2
    # Directory asset: usd_path stays None (multiple files, no single output)
    assert asset.usd_path is None


def test_convert_skips_asset_without_local_path(tmp_path):
    asset = STEPAsset(name="missing.step", url="http://example.com/x.step", source="github")

    with patch("simready.pipeline.run") as mock_run:
        results = convert_acquired_assets([asset], tmp_path / "usd")

    assert results == []
    mock_run.assert_not_called()


def test_convert_handles_pipeline_error(tmp_path):
    step_file = tmp_path / "broken.step"
    step_file.write_bytes(b"bad")
    asset = STEPAsset(name="broken.step", url="http://example.com/x.step", source="github")
    asset.local_path = step_file

    with patch("simready.pipeline.run", side_effect=RuntimeError("parse error")):
        results = convert_acquired_assets([asset], tmp_path / "usd")

    assert results == []
    assert asset.usd_path is None


def test_convert_passes_config_path(tmp_path):
    step_file = tmp_path / "part.step"
    step_file.write_bytes(b"STEP data")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline:\n  up_axis: Y\n")
    asset = STEPAsset(name="part.step", url="http://example.com/part.step", source="github")
    asset.local_path = step_file

    with patch("simready.pipeline.run") as mock_run:
        convert_acquired_assets([asset], tmp_path / "usd", config_path=config)

    mock_run.assert_called_once_with(step_file, tmp_path / "usd" / "part.usda", config)


# --- CLI run subcommand ---

def test_run_cli_no_assets(tmp_path, capsys):
    from unittest.mock import AsyncMock
    from simready.cli import main

    with patch("simready.acquisition.agent.acquire", new=AsyncMock(return_value=[])):
        main([
            "run", "nonexistent_query",
            "--output", str(tmp_path / "usd"),
            "--dest", str(tmp_path / "steps"),
            "--catalog", str(tmp_path / "cat.json"),
        ])

    out = capsys.readouterr().out
    assert "No assets" in out


def test_run_cli_converts_downloaded_assets(tmp_path, capsys):
    from unittest.mock import AsyncMock
    from simready.cli import main

    step_file = tmp_path / "steps" / "github" / "gear.step"
    step_file.parent.mkdir(parents=True)
    step_file.write_bytes(b"STEP data")

    asset = STEPAsset(name="gear.step", url="http://example.com/gear.step", source="github")
    asset.local_path = step_file

    with (
        patch("simready.acquisition.agent.acquire", new=AsyncMock(return_value=[asset])),
        patch("simready.pipeline.run"),
        patch("simready.acquisition.agent.save_catalog"),
    ):
        main([
            "run", "gear",
            "--output", str(tmp_path / "usd"),
            "--dest", str(tmp_path / "steps"),
            "--catalog", str(tmp_path / "cat.json"),
        ])

    out = capsys.readouterr().out
    assert "gear.step" in out
    assert "gear.usda" in out
