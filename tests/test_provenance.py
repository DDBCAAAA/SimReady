"""TDD tests for VLM data provenance: reasoning_step and primary_material written to USD.

These tests verify that when the VLM classifies a part, its reasoning and material class
are persisted as custom data inside the USD prim — not just logged and discarded.
"""

from __future__ import annotations

import numpy as np
import pytest

from simready.ingestion.step_reader import CADBody
from simready.materials.material_map import MDLMaterial


def _tetrahedron():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    return verts, faces


# ---------------------------------------------------------------------------
# Phase 1 (RED): MDLMaterial has provenance fields
# ---------------------------------------------------------------------------

def test_mdl_material_has_vlm_reasoning_step_field():
    """MDLMaterial must expose vlm_reasoning_step attribute."""
    mat = MDLMaterial(mdl_name="OmniPBR.mdl")
    assert hasattr(mat, "vlm_reasoning_step")
    assert mat.vlm_reasoning_step is None


def test_mdl_material_has_vlm_primary_material_field():
    """MDLMaterial must expose vlm_primary_material attribute."""
    mat = MDLMaterial(mdl_name="OmniPBR.mdl")
    assert hasattr(mat, "vlm_primary_material")
    assert mat.vlm_primary_material is None


# ---------------------------------------------------------------------------
# Phase 2 (RED): USD export writes provenance into custom data
# ---------------------------------------------------------------------------

@pytest.fixture()
def provenance_usd(tmp_path):
    """Create a minimal USD file whose material carries VLM provenance."""
    from simready.config.settings import PipelineSettings
    from simready.usd.assembly import create_stage

    verts, faces = _tetrahedron()
    body = CADBody(name="flange_part", vertices=verts, faces=faces)

    mdl_mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        source_material="flange_part",
        confidence=0.92,
        vlm_reasoning_step=(
            "DN15 bore diameter confirms a small pipe flange; "
            "stamped manufacturing process confirms steel."
        ),
        vlm_primary_material="steel",
    )

    output_usd = tmp_path / "provenance_test.usda"
    create_stage([body], {"flange_part": mdl_mat}, output_usd, PipelineSettings())
    return output_usd


def test_reasoning_step_key_present_in_usd(provenance_usd):
    """'reasoning_step' key must appear in the USDA custom data."""
    content = provenance_usd.read_text()
    assert "reasoning_step" in content, (
        f"'reasoning_step' not found in USD output:\n{content[:2000]}"
    )


def test_reasoning_step_value_present_in_usd(provenance_usd):
    """The exact reasoning text must be written into the USDA."""
    content = provenance_usd.read_text()
    assert "DN15 bore diameter confirms a small pipe flange" in content, (
        f"reasoning_step value not found in USD output:\n{content[:2000]}"
    )


def test_primary_material_key_present_in_usd(provenance_usd):
    """'primary_material' key must appear in the USDA custom data."""
    content = provenance_usd.read_text()
    assert "primary_material" in content, (
        f"'primary_material' not found in USD output:\n{content[:2000]}"
    )


def test_primary_material_value_present_in_usd(provenance_usd):
    """The material class string ('steel') must be written into the USDA."""
    content = provenance_usd.read_text()
    assert '"steel"' in content, (
        f"primary_material value not found in USD output:\n{content[:2000]}"
    )


def test_provenance_lives_in_simready_dictionary(provenance_usd):
    """Both fields must be nested under the 'simready' custom data dictionary."""
    content = provenance_usd.read_text()
    # USD serialises nested dict keys as: dictionary simready = { ... }
    assert "dictionary simready" in content, (
        "No 'dictionary simready' block found in USD output"
    )


def test_provenance_none_fields_not_written(tmp_path):
    """When vlm_reasoning_step and vlm_primary_material are None, no keys emitted."""
    from simready.config.settings import PipelineSettings
    from simready.usd.assembly import create_stage

    verts, faces = _tetrahedron()
    body = CADBody(name="unknown_part", vertices=verts, faces=faces)
    mdl_mat = MDLMaterial(
        mdl_name="OmniPBR.mdl",
        source_material="unknown_part",
        # vlm_reasoning_step and vlm_primary_material default to None
    )
    output_usd = tmp_path / "no_provenance.usda"
    create_stage([body], {"unknown_part": mdl_mat}, output_usd, PipelineSettings())
    content = output_usd.read_text()
    # Keys must not appear when provenance was never populated
    assert "reasoning_step" not in content
    assert "primary_material" not in content
