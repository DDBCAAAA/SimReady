"""Tests for VLM-based material classification and semantic labeling.

All Anthropic API calls are mocked — no network required.
The `anthropic` package is not installed in this env; we inject a fake module.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from simready.acquisition.vlm_material import (
    _VALID_CLASSES,
    _VALID_SEMANTIC_LABELS,
    classify_material_vlm,
    clear_vlm_cache,
)
from simready.materials.material_map import CAEMaterial, map_cae_to_mdl


# ---------------------------------------------------------------------------
# Helper: build a fake anthropic module that looks like the real one
# ---------------------------------------------------------------------------

def _make_anthropic_module(
    material_class: str,
    confidence: float,
    semantic_label: str = "fastener:nut",
):
    """Return (fake_module, client_instance) for injection into sys.modules."""
    text_block = SimpleNamespace(
        type="text",
        text=json.dumps({
            "material_class": material_class,
            "confidence": confidence,
            "semantic_label": semantic_label,
        }),
    )
    response = SimpleNamespace(content=[text_block])
    client_instance = MagicMock()
    client_instance.messages.create.return_value = response

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = MagicMock(return_value=client_instance)
    return fake_mod, client_instance


def _make_anthropic_module_raising(exc: Exception):
    """Return a fake anthropic module whose client.messages.create raises."""
    client_instance = MagicMock()
    client_instance.messages.create.side_effect = exc
    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = MagicMock(return_value=client_instance)
    return fake_mod, client_instance


# ---------------------------------------------------------------------------
# classify_material_vlm unit tests
# ---------------------------------------------------------------------------

class TestClassifyMaterialVlm:
    def setup_method(self):
        clear_vlm_cache()
        # Remove any cached 'anthropic' from sys.modules between tests
        sys.modules.pop("anthropic", None)

    def test_returns_class_confidence_and_semantic_label(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("steel", 0.92, "fastener:nut")
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = classify_material_vlm(
                "ISO10642_Hex_Socket_M3x10", bbox_m=(0.003, 0.003, 0.01)
            )
        assert result is not None
        mat_class, confidence, sem_label = result
        assert mat_class == "steel"
        assert confidence == pytest.approx(0.92)
        assert sem_label == "fastener:nut"

    def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = classify_material_vlm("some_part")
        assert result is None

    def test_anthropic_not_installed_returns_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Simulate import failure by setting the module to None in sys.modules
        with patch.dict(sys.modules, {"anthropic": None}):
            result = classify_material_vlm("some_part")
        assert result is None

    def test_json_parse_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        bad_block = SimpleNamespace(type="text", text="not-json {{bad")
        bad_response = SimpleNamespace(content=[bad_block])
        client_instance = MagicMock()
        client_instance.messages.create.return_value = bad_response
        fake_mod = types.ModuleType("anthropic")
        fake_mod.Anthropic = MagicMock(return_value=client_instance)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = classify_material_vlm("some_part")
        assert result is None

    def test_unknown_class_returns_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("unobtanium", 0.99)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = classify_material_vlm("mystery_part")
        assert result is None

    def test_api_exception_returns_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module_raising(RuntimeError("network error"))
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = classify_material_vlm("some_part")
        assert result is None

    def test_confidence_clamped_to_unit_interval(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("aluminum", 1.5)  # out-of-range
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = classify_material_vlm("aluminum_plate")
        assert result is not None
        _, confidence, _ = result
        assert 0.0 <= confidence <= 1.0

    def test_unknown_semantic_label_returns_none_for_label(self, monkeypatch):
        """Invalid semantic label is discarded; material result still returned."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("steel", 0.88, "robot:part")
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = classify_material_vlm("some_part")
        assert result is not None
        mat_class, confidence, sem_label = result
        assert mat_class == "steel"
        assert sem_label is None  # invalid label discarded

    def test_cache_deduplication(self, monkeypatch):
        """API is called only once for identical inputs across two calls."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, client_instance = _make_anthropic_module("steel", 0.91)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            r1 = classify_material_vlm("bolt_m4", bbox_m=(0.004, 0.004, 0.016))
            r2 = classify_material_vlm("bolt_m4", bbox_m=(0.004, 0.004, 0.016))
        assert r1 == r2
        assert client_instance.messages.create.call_count == 1

    def test_all_valid_classes_accepted(self):
        """Sanity check: _VALID_CLASSES is non-empty and all strings."""
        assert len(_VALID_CLASSES) > 0
        for cls in _VALID_CLASSES:
            assert isinstance(cls, str) and cls

    def test_all_valid_semantic_labels_non_empty(self):
        """Sanity check: _VALID_SEMANTIC_LABELS is non-empty and all strings."""
        assert len(_VALID_SEMANTIC_LABELS) > 0
        for label in _VALID_SEMANTIC_LABELS:
            assert isinstance(label, str) and ":" in label


# ---------------------------------------------------------------------------
# map_cae_to_mdl VLM integration tests
# ---------------------------------------------------------------------------

class TestMapCaeToMdlVlm:
    def setup_method(self):
        clear_vlm_cache()
        sys.modules.pop("anthropic", None)

    def test_vlm_disabled_by_default(self, monkeypatch):
        """VLM not called when enable_vlm=False (default)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, client_instance = _make_anthropic_module("steel", 0.95)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            map_cae_to_mdl(CAEMaterial(name="body_0"))
        client_instance.messages.create.assert_not_called()

    def test_vlm_not_called_when_forced_class_set(self, monkeypatch):
        """VLM skipped when forced_class is provided."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, client_instance = _make_anthropic_module("steel", 0.95)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = map_cae_to_mdl(
                CAEMaterial(name="body_0"),
                forced_class="aluminum",
                enable_vlm=True,
            )
        client_instance.messages.create.assert_not_called()
        assert result.source_material == "body_0"

    def test_vlm_overrides_mat_class_when_regex_fails(self, monkeypatch):
        """VLM provides class+confidence when regex returns None."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("steel", 0.91, "fastener:nut")
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = map_cae_to_mdl(
                CAEMaterial(name="body_0"),
                enable_vlm=True,
            )
        assert result.confidence == pytest.approx(0.91)
        assert result.density == pytest.approx(7850.0)

    def test_vlm_semantic_label_stored_on_material(self, monkeypatch):
        """VLM-returned semantic label is stored on the MDLMaterial."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("steel", 0.91, "fastener:bolt")
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = map_cae_to_mdl(
                CAEMaterial(name="body_0"),
                enable_vlm=True,
            )
        assert result.vlm_semantic_label == "fastener:bolt"

    def test_vlm_semantic_label_none_when_vlm_disabled(self, monkeypatch):
        """vlm_semantic_label is None when VLM is not used."""
        result = map_cae_to_mdl(CAEMaterial(name="body_0"))
        assert result.vlm_semantic_label is None

    def test_vlm_overrides_confidence_when_regex_succeeds(self, monkeypatch):
        """VLM provides higher confidence even when regex already found a class."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("steel", 0.93)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            # "steel_rod" matches regex → steel, but formula gives 0.25
            result = map_cae_to_mdl(
                CAEMaterial(name="steel_rod"),
                enable_vlm=True,
            )
        # VLM confidence (0.93) replaces formula-computed 0.25
        assert result.confidence == pytest.approx(0.93)

    def test_vlm_confidence_passes_quality_gate(self, monkeypatch):
        """Asset with VLM confidence=0.90 should pass the 0.8 quality gate."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module("steel", 0.90)
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = map_cae_to_mdl(
                CAEMaterial(name="body_0"),
                enable_vlm=True,
            )
        assert result.confidence >= 0.8, (
            f"Expected confidence ≥ 0.8, got {result.confidence}"
        )

    def test_vlm_failure_falls_back_to_formula_confidence(self, monkeypatch):
        """When VLM returns None, formula-computed confidence is used."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, _ = _make_anthropic_module_raising(RuntimeError("timeout"))
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = map_cae_to_mdl(
                CAEMaterial(name="steel_rod"),
                enable_vlm=True,
            )
        # Formula: 0 direct fields + 0.25 class credit = 0.25
        assert result.confidence == pytest.approx(0.25)

    def test_vlm_bbox_and_semantic_label_passed_through(self, monkeypatch):
        """Verify semantic_label and bbox_m are forwarded in the VLM prompt."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake_mod, client_instance = _make_anthropic_module("titanium", 0.85, "structural:bracket")
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            result = map_cae_to_mdl(
                CAEMaterial(name="bracket_part"),
                enable_vlm=True,
                semantic_label="bracket",
                bbox_m=(0.05, 0.02, 0.01),
            )
        call_kwargs = client_instance.messages.create.call_args
        user_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "bracket" in user_msg
        assert "50.0" in user_msg  # 0.05 m → 50.0 mm
        assert result.confidence == pytest.approx(0.85)
        assert result.vlm_semantic_label == "structural:bracket"
