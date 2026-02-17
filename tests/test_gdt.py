"""Tests for GD&T support."""

import pytest

from tolerance_stack.gdt import (
    GDTType, MaterialCondition, FeatureControlFrame,
    DatumReferenceFrame, DatumFeature, DatumPrecedence,
    fcf_to_tolerance_parameters,
)
from tolerance_stack.models import Distribution


class TestGDTType:
    def test_all_types(self):
        """All 15 GD&T types are defined."""
        assert len(GDTType) == 15

    def test_form_types(self):
        assert GDTType.FLATNESS.value == "flatness"
        assert GDTType.CYLINDRICITY.value == "cylindricity"

    def test_location_types(self):
        assert GDTType.POSITION.value == "position"
        assert GDTType.CONCENTRICITY.value == "concentricity"


class TestMaterialCondition:
    def test_mmc(self):
        assert MaterialCondition.MMC.value == "mmc"

    def test_lmc(self):
        assert MaterialCondition.LMC.value == "lmc"

    def test_rfs(self):
        assert MaterialCondition.NONE.value == "none"


class TestFeatureControlFrame:
    def test_basic_position(self):
        fcf = FeatureControlFrame(
            name="pos1",
            gdt_type=GDTType.POSITION,
            tolerance_value=0.10,
            is_diametral=True,
            datum_refs=["A", "B"],
        )
        assert fcf.half_tolerance == 0.05
        assert fcf.zone_axes() == 2

    def test_mmc_bonus(self):
        fcf = FeatureControlFrame(
            name="pos_mmc",
            gdt_type=GDTType.POSITION,
            tolerance_value=0.10,
            material_condition=MaterialCondition.MMC,
            feature_size_nominal=10.0,
            feature_size_tol_plus=0.05,
            feature_size_tol_minus=0.03,
        )
        assert fcf.bonus_tolerance == 0.08  # plus + minus
        assert fcf.max_tolerance == 0.18  # 0.10 + 0.08
        assert fcf.effective_tolerance(0.0) == 0.10
        assert fcf.effective_tolerance(0.04) == 0.14
        assert fcf.effective_tolerance(1.0) == 0.18  # Capped at bonus max

    def test_rfs_no_bonus(self):
        fcf = FeatureControlFrame(
            name="pos_rfs",
            gdt_type=GDTType.POSITION,
            tolerance_value=0.10,
            material_condition=MaterialCondition.NONE,
            feature_size_tol_plus=0.05,
            feature_size_tol_minus=0.05,
        )
        assert fcf.bonus_tolerance == 0.0
        assert fcf.effective_tolerance(0.05) == 0.10

    def test_flatness(self):
        fcf = FeatureControlFrame(
            name="flat1",
            gdt_type=GDTType.FLATNESS,
            tolerance_value=0.02,
        )
        assert fcf.zone_axes() == 1
        assert fcf.datum_refs == []

    def test_profile_surface(self):
        fcf = FeatureControlFrame(
            name="prof1",
            gdt_type=GDTType.PROFILE_SURFACE,
            tolerance_value=0.50,
            datum_refs=["A", "B", "C"],
        )
        assert fcf.zone_axes() == 1
        assert len(fcf.datum_refs) == 3

    def test_composite_position(self):
        fcf = FeatureControlFrame(
            name="comp_pos",
            gdt_type=GDTType.POSITION,
            tolerance_value=0.20,
            composite_lower=0.10,
            is_diametral=True,
        )
        assert fcf.composite_lower == 0.10

    def test_serialization(self):
        fcf = FeatureControlFrame(
            name="test_fcf",
            gdt_type=GDTType.PERPENDICULARITY,
            tolerance_value=0.05,
            material_condition=MaterialCondition.MMC,
            datum_refs=["A"],
            composite_lower=0.02,
        )
        d = fcf.to_dict()
        loaded = FeatureControlFrame.from_dict(d)
        assert loaded.name == "test_fcf"
        assert loaded.gdt_type == GDTType.PERPENDICULARITY
        assert loaded.tolerance_value == 0.05
        assert loaded.material_condition == MaterialCondition.MMC
        assert loaded.composite_lower == 0.02


class TestDatumReferenceFrame:
    def test_basic_drf(self):
        drf = DatumReferenceFrame(name="DRF_ABC")
        drf.add_datum(DatumFeature("A", "Block", "bottom_face", DatumPrecedence.PRIMARY))
        drf.add_datum(DatumFeature("B", "Block", "side_face", DatumPrecedence.SECONDARY))
        drf.add_datum(DatumFeature("C", "Block", "end_face", DatumPrecedence.TERTIARY))

        assert drf.constrained_dof == 6
        assert drf.primary.label == "A"
        assert drf.secondary.label == "B"
        assert drf.tertiary.label == "C"

    def test_too_many_datums(self):
        drf = DatumReferenceFrame(name="DRF")
        drf.add_datum(DatumFeature("A", "B", "f", DatumPrecedence.PRIMARY))
        drf.add_datum(DatumFeature("B", "B", "f", DatumPrecedence.SECONDARY))
        drf.add_datum(DatumFeature("C", "B", "f", DatumPrecedence.TERTIARY))
        with pytest.raises(ValueError, match="at most 3"):
            drf.add_datum(DatumFeature("D", "B", "f", DatumPrecedence.PRIMARY))

    def test_duplicate_precedence(self):
        drf = DatumReferenceFrame(name="DRF")
        drf.add_datum(DatumFeature("A", "B", "f", DatumPrecedence.PRIMARY))
        with pytest.raises(ValueError, match="already has"):
            drf.add_datum(DatumFeature("B", "B", "f", DatumPrecedence.PRIMARY))

    def test_serialization(self):
        drf = DatumReferenceFrame(name="DRF1")
        drf.add_datum(DatumFeature("A", "Block", "face", DatumPrecedence.PRIMARY))
        d = drf.to_dict()
        loaded = DatumReferenceFrame.from_dict(d)
        assert loaded.name == "DRF1"
        assert len(loaded.datums) == 1
        assert loaded.primary.label == "A"


class TestFCFToParameters:
    def test_position_diametral(self):
        fcf = FeatureControlFrame(
            name="pos1", gdt_type=GDTType.POSITION,
            tolerance_value=0.10, is_diametral=True,
        )
        params = fcf_to_tolerance_parameters(fcf, "Block", "hole")
        assert len(params) == 3  # x, y, z

    def test_flatness(self):
        fcf = FeatureControlFrame(
            name="flat1", gdt_type=GDTType.FLATNESS,
            tolerance_value=0.02,
        )
        params = fcf_to_tolerance_parameters(fcf, "Block", "face")
        assert len(params) == 1
        assert params[0]["gdt_type"] == "flatness"

    def test_perpendicularity(self):
        fcf = FeatureControlFrame(
            name="perp1", gdt_type=GDTType.PERPENDICULARITY,
            tolerance_value=0.05,
        )
        params = fcf_to_tolerance_parameters(fcf, "Block", "face", (0, 0, 1))
        assert len(params) == 2  # u, v orientation

    def test_runout(self):
        fcf = FeatureControlFrame(
            name="run1", gdt_type=GDTType.CIRCULAR_RUNOUT,
            tolerance_value=0.03,
        )
        params = fcf_to_tolerance_parameters(fcf, "Shaft", "surface")
        assert len(params) == 1
        assert "runout" in params[0]["name"]

    def test_concentricity(self):
        fcf = FeatureControlFrame(
            name="conc1", gdt_type=GDTType.CONCENTRICITY,
            tolerance_value=0.04,
        )
        params = fcf_to_tolerance_parameters(fcf, "Shaft", "bore")
        assert len(params) == 2  # x, y
