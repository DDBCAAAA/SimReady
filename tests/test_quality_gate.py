"""Tests for the material confidence quality gate.

Verifies that:
- Assets with confidence < 0.8 are intercepted, logged, and quarantined.
- Assets with confidence >= 0.8 pass through without side-effects.
- The exact boundary values (0.79 and 0.81) behave correctly.
- The batch layer maps a gate-blocked asset to status="skipped", not "failed".
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from simready.quality_gate import (
    CONFIDENCE_THRESHOLD,
    LowConfidenceError,
    check_material_confidence,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fake_step(tmp_path: Path, name: str = "part.step") -> Path:
    """Write a tiny fake STEP file and return its path."""
    p = tmp_path / name
    p.write_bytes(b"ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    return p


# ---------------------------------------------------------------------------
# Core gate behaviour
# ---------------------------------------------------------------------------

class TestGateBlocking:
    def test_raises_low_confidence_error_at_079(self, tmp_path):
        """0.79 confidence must raise LowConfidenceError."""
        step = _fake_step(tmp_path)
        with pytest.raises(LowConfidenceError) as exc_info:
            check_material_confidence(
                confidence=0.79,
                asset_name="low_conf_bolt",
                step_path=step,
                log_path=tmp_path / "low_confidence_assets.log",
                quarantine_dir=tmp_path / "quarantine",
            )
        err = exc_info.value
        assert err.confidence == pytest.approx(0.79)
        assert err.threshold == pytest.approx(CONFIDENCE_THRESHOLD)
        assert "low_conf_bolt" in str(err)

    def test_source_file_moved_to_quarantine(self, tmp_path):
        """The raw STEP file must be moved into quarantine/ when blocked."""
        step = _fake_step(tmp_path, "low_conf_bolt.step")
        qdir = tmp_path / "quarantine"

        with pytest.raises(LowConfidenceError):
            check_material_confidence(
                confidence=0.79,
                asset_name="low_conf_bolt",
                step_path=step,
                log_path=tmp_path / "low_confidence_assets.log",
                quarantine_dir=qdir,
            )

        assert not step.exists(), "Source file should have been moved"
        assert (qdir / "low_conf_bolt.step").exists(), "File must appear in quarantine/"

    def test_warning_written_to_log(self, tmp_path):
        """A structured warning line must be appended to the log file."""
        step = _fake_step(tmp_path)
        log = tmp_path / "low_confidence_assets.log"

        with pytest.raises(LowConfidenceError):
            check_material_confidence(
                confidence=0.79,
                asset_name="sprocket_12t",
                step_path=step,
                log_path=log,
                quarantine_dir=tmp_path / "quarantine",
            )

        assert log.exists(), "Log file must be created"
        text = log.read_text()
        assert "sprocket_12t" in text
        assert "0.7900" in text
        assert "WARNING" in text

    def test_log_is_appended_not_overwritten(self, tmp_path):
        """Multiple blocked assets should accumulate in one log file."""
        log = tmp_path / "low_confidence_assets.log"
        qdir = tmp_path / "quarantine"

        for i in range(3):
            step = _fake_step(tmp_path, f"part_{i}.step")
            with pytest.raises(LowConfidenceError):
                check_material_confidence(
                    confidence=0.1 * (i + 1),
                    asset_name=f"part_{i}",
                    step_path=step,
                    log_path=log,
                    quarantine_dir=qdir,
                )

        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3, f"Expected 3 log lines, got {len(lines)}"

    def test_quarantine_name_collision_resolved(self, tmp_path):
        """Two assets with the same filename must both land in quarantine."""
        qdir = tmp_path / "quarantine"
        qdir.mkdir()
        log = tmp_path / "low_confidence_assets.log"

        # First asset
        step1 = tmp_path / "src1" / "part.step"
        step1.parent.mkdir()
        step1.write_bytes(b"first")
        with pytest.raises(LowConfidenceError):
            check_material_confidence(0.1, "part", step1, log, qdir)

        # Second asset — same filename, different directory
        step2 = tmp_path / "src2" / "part.step"
        step2.parent.mkdir()
        step2.write_bytes(b"second")
        with pytest.raises(LowConfidenceError):
            check_material_confidence(0.1, "part", step2, log, qdir)

        quarantined = list(qdir.iterdir())
        assert len(quarantined) == 2, "Both files must be quarantined without overwriting"


class TestGateAllowing:
    def test_no_exception_at_081(self, tmp_path):
        """0.81 confidence must pass without raising."""
        step = _fake_step(tmp_path)
        # Must not raise
        check_material_confidence(
            confidence=0.81,
            asset_name="high_conf_gear",
            step_path=step,
            log_path=tmp_path / "low_confidence_assets.log",
            quarantine_dir=tmp_path / "quarantine",
        )

    def test_source_file_untouched_at_081(self, tmp_path):
        """The source file must remain in place when confidence passes."""
        step = _fake_step(tmp_path, "gear.step")
        check_material_confidence(
            confidence=0.81,
            asset_name="high_conf_gear",
            step_path=step,
            log_path=tmp_path / "low_confidence_assets.log",
            quarantine_dir=tmp_path / "quarantine",
        )
        assert step.exists(), "Passing asset source file must not be moved"

    def test_no_log_entry_at_081(self, tmp_path):
        """No log file should be created for a passing asset."""
        log = tmp_path / "low_confidence_assets.log"
        step = _fake_step(tmp_path)
        check_material_confidence(
            confidence=0.81,
            asset_name="high_conf_gear",
            step_path=step,
            log_path=log,
            quarantine_dir=tmp_path / "quarantine",
        )
        assert not log.exists(), "Log must not be created when asset passes"

    def test_exact_threshold_passes(self, tmp_path):
        """Exactly 0.8 confidence must pass (threshold is inclusive)."""
        step = _fake_step(tmp_path)
        check_material_confidence(
            confidence=0.8,
            asset_name="border_case",
            step_path=step,
            log_path=tmp_path / "low_confidence_assets.log",
            quarantine_dir=tmp_path / "quarantine",
        )


class TestGateEdgeCases:
    def test_no_step_path_still_logs(self, tmp_path):
        """Gate must still log and raise even when step_path is not provided."""
        log = tmp_path / "low_confidence_assets.log"
        with pytest.raises(LowConfidenceError):
            check_material_confidence(
                confidence=0.5,
                asset_name="no_file_asset",
                step_path=None,
                log_path=log,
                quarantine_dir=tmp_path / "quarantine",
            )
        assert "no_file_asset" in log.read_text()

    def test_missing_step_path_does_not_raise_oserror(self, tmp_path):
        """A step_path pointing to a non-existent file must not cause OSError."""
        log = tmp_path / "low_confidence_assets.log"
        ghost = tmp_path / "ghost.step"  # does not exist
        with pytest.raises(LowConfidenceError):
            check_material_confidence(
                confidence=0.1,
                asset_name="ghost_asset",
                step_path=ghost,
                log_path=log,
                quarantine_dir=tmp_path / "quarantine",
            )

    def test_custom_threshold(self, tmp_path):
        """A custom threshold parameter must override the module default."""
        step = _fake_step(tmp_path)
        # With threshold=0.5, confidence=0.6 should pass
        check_material_confidence(
            confidence=0.6,
            asset_name="custom_thresh",
            step_path=step,
            log_path=tmp_path / "low_confidence_assets.log",
            quarantine_dir=tmp_path / "quarantine",
            threshold=0.5,
        )


# ---------------------------------------------------------------------------
# Batch layer integration
# ---------------------------------------------------------------------------

class TestBatchIntegration:
    def test_low_confidence_maps_to_skipped_not_failed(self, tmp_path):
        """_convert_one must return status='skipped' when the gate triggers."""
        from simready.batch import _convert_one
        from simready.quality_gate import LowConfidenceError

        step = _fake_step(tmp_path, "low_bolt.step")

        def _raise_low_confidence(*args, **kwargs):
            raise LowConfidenceError("low_bolt", 0.25, 0.8)

        with patch("simready.pipeline.run", side_effect=_raise_low_confidence):
            result = _convert_one(
                step_path=step,
                output_dir=tmp_path / "out",
                config_path=None,
                quality_min=0.5,
                material_default="steel",
            )

        assert result.status == "skipped", (
            f"Expected status='skipped', got '{result.status}'"
        )
        assert result.usd_path is None

    def test_other_exceptions_remain_failed(self, tmp_path):
        """Non-gate exceptions must still map to status='failed'."""
        from simready.batch import _convert_one

        step = _fake_step(tmp_path, "broken.step")

        with patch("simready.pipeline.run", side_effect=RuntimeError("parse error")):
            result = _convert_one(
                step_path=step,
                output_dir=tmp_path / "out",
                config_path=None,
                quality_min=0.5,
            )

        assert result.status == "failed"
        assert "parse error" in (result.error or "")
