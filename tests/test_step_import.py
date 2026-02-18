"""Tests for STEP file import."""

import os

import pytest

from tolerance_stack.step_import import (
    import_step, import_step_pmi,
    _parse_step_file, _parse_params, _parse_value,
    StepImportResult,
)


SAMPLE_STEP = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('STEP AP214'),'2;1');
FILE_NAME('test.step','2024-01-01',('Author'),(''),'',' ','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = PRODUCT('Block','Block part','',());
#2 = PRODUCT('Pin','Pin part','',());
#3 = CARTESIAN_POINT('origin',(0.0,0.0,0.0));
#4 = DIRECTION('z_dir',(0.0,0.0,1.0));
#5 = AXIS2_PLACEMENT_3D('',#3,#4,$);
#6 = PLANE('top_face',#5);
#7 = CARTESIAN_POINT('cyl_origin',(50.0,0.0,0.0));
#8 = DIRECTION('cyl_dir',(0.0,0.0,1.0));
#9 = AXIS2_PLACEMENT_3D('',#7,#8,$);
#10 = CYLINDRICAL_SURFACE('bore',#9,5.025);
#11 = POSITION_TOLERANCE('pos_tol',0.05);
#12 = FLATNESS_TOLERANCE('flat_tol',0.02);
ENDSEC;
END-ISO-10303-21;
"""


@pytest.fixture
def step_file(tmp_path):
    path = str(tmp_path / "test.step")
    with open(path, "w") as f:
        f.write(SAMPLE_STEP)
    return path


class TestParseValue:
    def test_string(self):
        assert _parse_value("'hello'") == "hello"

    def test_ref(self):
        assert _parse_value("#42") == 42

    def test_float(self):
        assert _parse_value("3.14") == pytest.approx(3.14)

    def test_int(self):
        assert _parse_value("42") == 42

    def test_none(self):
        assert _parse_value("$") is None

    def test_enum(self):
        assert _parse_value(".FORWARD.") == "FORWARD"


class TestParseParams:
    def test_simple(self):
        result = _parse_params("'name',42,3.14")
        assert result == ["name", 42, pytest.approx(3.14)]

    def test_nested(self):
        result = _parse_params("'point',(1.0,2.0,3.0)")
        assert result[0] == "point"
        assert isinstance(result[1], list)
        assert len(result[1]) == 3


class TestParseStepFile:
    def test_parse(self, step_file):
        entities = _parse_step_file(step_file)
        assert len(entities) > 0
        # Should find PRODUCT, PLANE, CYLINDRICAL_SURFACE, etc.
        types = {e.type_name for e in entities.values()}
        assert "PRODUCT" in types
        assert "PLANE" in types
        assert "CARTESIAN_POINT" in types

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _parse_step_file("/nonexistent/path.step")


class TestImportStep:
    def test_basic(self, step_file):
        result = import_step(step_file)
        assert isinstance(result, StepImportResult)
        assert result.n_entities > 0
        assert len(result.products) == 2
        assert result.products[0]["name"] == "Block"

    def test_features_extracted(self, step_file):
        result = import_step(step_file)
        assert len(result.features) > 0
        types = {f["feature_type"].value for f in result.features}
        assert "plane" in types or "cylinder" in types

    def test_gdt_extracted(self, step_file):
        result = import_step(step_file)
        assert len(result.gdt_callouts) >= 1
        types = {g["gdt_type"].value for g in result.gdt_callouts}
        # Should find position and/or flatness tolerance
        assert len(types) > 0

    def test_assembly_created(self, step_file):
        result = import_step(step_file, assembly_name="TestAssy")
        assert result.assembly is not None
        assert result.assembly.name == "TestAssy"
        assert len(result.assembly.bodies) > 0

    def test_summary(self, step_file):
        result = import_step(step_file)
        s = result.summary()
        assert "STEP Import" in s
        assert "entities" in s.lower()


class TestImportStepPMI:
    def test_pmi(self, step_file):
        fcfs = import_step_pmi(step_file)
        assert len(fcfs) >= 1
        for fcf in fcfs:
            assert fcf.tolerance_value > 0
