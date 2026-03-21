"""Tests for CAE → MDL material mapping."""

import pytest

from simready.materials.material_map import CAEMaterial, MDLMaterial, map_cae_to_mdl, classify_material


def test_classify_steel_plain():
    # Plain "steel" with no grade code → generic steel class
    mat = CAEMaterial(name="steel_bracket")
    assert classify_material(mat) == "steel"


def test_classify_steel_304_is_stainless():
    # Alloy code "304" should resolve to stainless, not generic steel
    mat = CAEMaterial(name="Steel_304_Stainless")
    assert classify_material(mat) == "stainless"


def test_classify_unknown():
    mat = CAEMaterial(name="custom_material_xyz")
    assert classify_material(mat) is None


def test_map_with_full_cae_data():
    cae = CAEMaterial(
        name="TestMetal",
        roughness=0.3,
        metallic=1.0,
        ior=2.5,
        albedo_rgb=(0.8, 0.8, 0.8),
        density=7800.0,
    )
    mdl = map_cae_to_mdl(cae)
    assert mdl.confidence == 1.0
    assert mdl.roughness == 0.3
    assert mdl.metallic == 1.0
    assert mdl.ior == 2.5
    assert mdl.diffuse_color == (0.8, 0.8, 0.8)
    assert mdl.density == 7800.0


def test_map_with_partial_cae_data_classified():
    cae = CAEMaterial(name="aluminum_6061", density=2700.0)
    mdl = map_cae_to_mdl(cae)
    # Should pick up defaults from aluminum class
    assert mdl.metallic == 1.0
    assert mdl.roughness == 0.3
    # Class-matched defaults give 0.25 partial credit (physically grounded, not fabricated)
    assert mdl.confidence == 0.25


def test_map_with_no_data():
    cae = CAEMaterial(name="unknown_material")
    mdl = map_cae_to_mdl(cae)
    assert mdl.confidence == 0.0
    # Should get generic fallback values
    assert mdl.roughness == 0.5
    assert mdl.metallic == 0.0


def test_map_preserves_source_name():
    cae = CAEMaterial(name="MySpecialSteel")
    mdl = map_cae_to_mdl(cae)
    assert mdl.source_material == "MySpecialSteel"


# ---------------------------------------------------------------------------
# 1a: Compound-name and alloy-code classification
# ---------------------------------------------------------------------------

def test_classify_stainless_steel_compound():
    # Compound phrase must beat generic "steel" substring match
    assert classify_material(CAEMaterial(name="stainless_steel_shaft")) == "stainless"


def test_classify_cast_iron_compound():
    assert classify_material(CAEMaterial(name="cast_iron_housing")) == "cast_iron"


def test_classify_carbon_fiber_compound():
    assert classify_material(CAEMaterial(name="carbon_fiber_panel")) == "carbon_fiber"


def test_classify_teflon_alias():
    assert classify_material(CAEMaterial(name="teflon_seal")) == "ptfe"


def test_classify_delrin_alias():
    assert classify_material(CAEMaterial(name="delrin_gear")) == "acetal"


def test_classify_alloy_6061():
    assert classify_material(CAEMaterial(name="aluminum_6061_plate")) == "aluminum"


def test_classify_alloy_7075():
    assert classify_material(CAEMaterial(name="7075_T6_plate")) == "aluminum"


def test_classify_alloy_316_stainless():
    assert classify_material(CAEMaterial(name="SS316_tube")) == "stainless"


def test_classify_alloy_4140_steel():
    assert classify_material(CAEMaterial(name="4140_shaft")) == "steel"


@pytest.mark.parametrize("name,expected", [
    ("nylon_bushing",        "nylon"),
    ("ptfe_seal",            "ptfe"),
    ("titanium_rod",         "titanium"),
    ("brass_fitting",        "brass"),
    ("bronze_bearing",       "bronze"),
    ("acetal_cam",           "acetal"),
    ("polycarbonate_cover",  "polycarbonate"),
    ("zinc_die_cast",        "zinc"),
    ("ceramic_insulator",    "ceramic"),
    ("chrome_shaft",         "chrome"),
    ("magnesium_housing",    "magnesium"),
    ("hdpe_liner",           "hdpe"),
    ("pvc_pipe",             "pvc"),
    ("silicone_gasket",      "silicone"),
])
def test_new_material_classes_classify(name, expected):
    assert classify_material(CAEMaterial(name=name)) == expected


@pytest.mark.parametrize("cls", [
    "stainless", "cast_iron", "titanium", "brass", "bronze", "nylon",
    "ptfe", "acetal", "polycarbonate", "carbon_fiber", "zinc", "ceramic",
    "chrome", "magnesium", "cast_aluminum", "hdpe", "pvc", "silicone",
])
def test_new_classes_have_full_physics(cls):
    mdl = map_cae_to_mdl(CAEMaterial(name=cls))
    assert mdl.density is not None,        f"{cls}: missing density"
    assert mdl.friction_static is not None, f"{cls}: missing friction_static"
    assert mdl.friction_dynamic is not None, f"{cls}: missing friction_dynamic"
    assert mdl.restitution is not None,    f"{cls}: missing restitution"


def test_forced_class_overrides_classifier():
    # Part named "unknown_xyz" with forced_class="steel" must get steel physics
    cae = CAEMaterial(name="unknown_xyz")
    mdl = map_cae_to_mdl(cae, forced_class="steel")
    assert mdl.density == pytest.approx(7850.0)
    assert mdl.friction_static is not None
