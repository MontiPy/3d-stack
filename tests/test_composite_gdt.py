"""Tests for composite GD&T and datum shift features."""

import pytest
import numpy as np

from tolerance_stack.gdt import (
    GDTType, MaterialCondition, FeatureControlFrame,
    CompositeFCF, DatumFeature, DatumPrecedence,
    compute_datum_shift, composite_fcf_to_tolerance_parameters,
)


class TestCompositeFCF:
    """Tests for composite (two-segment) Feature Control Frames."""

    def test_basic_composite_position(self):
        upper = FeatureControlFrame(
            name="upper", gdt_type=GDTType.POSITION,
            tolerance_value=1.0, is_diametral=True,
            datum_refs=["A", "B", "C"],
        )
        lower = FeatureControlFrame(
            name="lower", gdt_type=GDTType.POSITION,
            tolerance_value=0.4, is_diametral=True,
            datum_refs=["A"],
        )
        comp = CompositeFCF(
            name="comp_pos", gdt_type=GDTType.POSITION,
            upper=upper, lower=lower,
        )
        assert comp.pattern_tolerance == 1.0
        assert comp.feature_tolerance == 0.4

    def test_lower_must_be_smaller(self):
        upper = FeatureControlFrame(
            name="upper", gdt_type=GDTType.POSITION,
            tolerance_value=0.5, is_diametral=True,
        )
        lower = FeatureControlFrame(
            name="lower", gdt_type=GDTType.POSITION,
            tolerance_value=0.8, is_diametral=True,
        )
        with pytest.raises(ValueError, match="smaller"):
            CompositeFCF("bad", GDTType.POSITION, upper, lower)

    def test_only_position_or_profile(self):
        upper = FeatureControlFrame(
            name="upper", gdt_type=GDTType.FLATNESS,
            tolerance_value=0.5,
        )
        lower = FeatureControlFrame(
            name="lower", gdt_type=GDTType.FLATNESS,
            tolerance_value=0.2,
        )
        with pytest.raises(ValueError, match="POSITION or PROFILE"):
            CompositeFCF("bad", GDTType.FLATNESS, upper, lower)

    def test_composite_profile(self):
        upper = FeatureControlFrame(
            name="upper", gdt_type=GDTType.PROFILE_SURFACE,
            tolerance_value=1.0,
            datum_refs=["A", "B", "C"],
        )
        lower = FeatureControlFrame(
            name="lower", gdt_type=GDTType.PROFILE_SURFACE,
            tolerance_value=0.5,
            datum_refs=["A"],
        )
        comp = CompositeFCF("prof", GDTType.PROFILE_SURFACE, upper, lower)
        assert comp.pattern_tolerance == 1.0
        assert comp.feature_tolerance == 0.5

    def test_effective_with_mmc_bonus(self):
        upper = FeatureControlFrame(
            name="upper", gdt_type=GDTType.POSITION,
            tolerance_value=1.0, is_diametral=True,
            material_condition=MaterialCondition.MMC,
            feature_size_nominal=10.0,
            feature_size_tol_plus=0.1,
            feature_size_tol_minus=0.1,
        )
        lower = FeatureControlFrame(
            name="lower", gdt_type=GDTType.POSITION,
            tolerance_value=0.4, is_diametral=True,
            material_condition=MaterialCondition.MMC,
            feature_size_nominal=10.0,
            feature_size_tol_plus=0.1,
            feature_size_tol_minus=0.1,
        )
        comp = CompositeFCF("pos", GDTType.POSITION, upper, lower)
        assert comp.effective_upper(0.0) == 1.0
        assert comp.effective_upper(0.1) == 1.1
        assert comp.effective_lower(0.0) == 0.4
        assert comp.effective_lower(0.1) == 0.5

    def test_serialization(self):
        upper = FeatureControlFrame(
            name="u", gdt_type=GDTType.POSITION,
            tolerance_value=1.0, is_diametral=True,
        )
        lower = FeatureControlFrame(
            name="l", gdt_type=GDTType.POSITION,
            tolerance_value=0.3, is_diametral=True,
        )
        comp = CompositeFCF("test", GDTType.POSITION, upper, lower)
        d = comp.to_dict()
        comp2 = CompositeFCF.from_dict(d)
        assert comp2.pattern_tolerance == 1.0
        assert comp2.feature_tolerance == 0.3


class TestDatumShift:
    """Tests for datum feature simulator / datum shift."""

    def test_no_shift_at_rfs(self):
        datum = DatumFeature(
            label="A", body_name="block", feature_name="hole",
            material_condition=MaterialCondition.NONE,
        )
        result = compute_datum_shift(datum, 10.0, 0.1, 0.1)
        assert result.max_shift == 0.0

    def test_mmc_shift_worst_case(self):
        datum = DatumFeature(
            label="A", body_name="block", feature_name="hole",
            material_condition=MaterialCondition.MMC,
        )
        result = compute_datum_shift(datum, 10.0, 0.1, 0.1)
        # Max shift = total_size_tol / 2 = 0.2 / 2 = 0.1
        assert abs(result.max_shift - 0.1) < 1e-10

    def test_lmc_shift_worst_case(self):
        datum = DatumFeature(
            label="B", body_name="shaft", feature_name="pin",
            material_condition=MaterialCondition.LMC,
        )
        result = compute_datum_shift(datum, 5.0, 0.05, 0.05)
        assert abs(result.max_shift - 0.05) < 1e-10

    def test_mmc_shift_with_actual_size(self):
        datum = DatumFeature(
            label="A", body_name="block", feature_name="hole",
            material_condition=MaterialCondition.MMC,
        )
        # MMC = 10.0 - 0.1 = 9.9, actual = 10.05 (at LMC)
        result = compute_datum_shift(datum, 10.0, 0.1, 0.1, actual_size=10.05)
        expected = abs(10.05 - 9.9) / 2.0
        assert abs(result.max_shift - expected) < 1e-10

    def test_summary(self):
        datum = DatumFeature(
            label="C", body_name="block", feature_name="bore",
            material_condition=MaterialCondition.MMC,
        )
        result = compute_datum_shift(datum, 20.0, 0.2, 0.2)
        text = result.summary()
        assert "Datum C" in text


class TestCompositeFCFParameters:
    """Tests for composite_fcf_to_tolerance_parameters()."""

    def test_diametral_composite(self):
        upper = FeatureControlFrame(
            name="upper", gdt_type=GDTType.POSITION,
            tolerance_value=1.0, is_diametral=True,
        )
        lower = FeatureControlFrame(
            name="lower", gdt_type=GDTType.POSITION,
            tolerance_value=0.3, is_diametral=True,
        )
        comp = CompositeFCF("comp", GDTType.POSITION, upper, lower)
        params = composite_fcf_to_tolerance_parameters(comp, "body1", "hole1")
        # 1 pattern_loc + 2 feat_x, feat_y
        assert len(params) == 3
        pattern = [p for p in params if "pattern" in p["name"]]
        feat = [p for p in params if "feat" in p["name"]]
        assert len(pattern) == 1
        assert len(feat) == 2
        assert pattern[0]["half_tol"] == 0.5  # 1.0 / 2
        assert feat[0]["half_tol"] == 0.15  # 0.3 / 2

    def test_bilateral_composite(self):
        upper = FeatureControlFrame(
            name="upper", gdt_type=GDTType.PROFILE_SURFACE,
            tolerance_value=1.0, is_diametral=False,
        )
        lower = FeatureControlFrame(
            name="lower", gdt_type=GDTType.PROFILE_SURFACE,
            tolerance_value=0.4, is_diametral=False,
        )
        comp = CompositeFCF("comp", GDTType.PROFILE_SURFACE, upper, lower)
        params = composite_fcf_to_tolerance_parameters(comp, "body1", "surf1")
        assert len(params) == 2  # 1 pattern_loc + 1 feat
