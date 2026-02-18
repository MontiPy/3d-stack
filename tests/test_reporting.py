"""Tests for report generation."""

import pytest

from tolerance_stack.reporting import (
    ReportConfig, generate_html_report, generate_text_report,
    generate_apqp_report, save_report,
)


class _FakeResult:
    """Minimal result object for testing reports."""
    def __init__(self):
        self.method = "Worst-Case"
        self.nominal_value = 30.0
        self.value_min = 29.8
        self.value_max = 30.2
        self.plus_tolerance = 0.2
        self.minus_tolerance = 0.2
        self.sensitivity = [("param_A", 1.0), ("param_B", -0.5)]

    def summary(self) -> str:
        return f"Method: {self.method}\nNominal: {self.nominal_value}"


class TestHTMLReport:
    def test_basic(self):
        config = ReportConfig(title="Test Report", project="TestProject")
        results = {"wc": _FakeResult()}
        html = generate_html_report(config, results)
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "TestProject" in html
        assert "Worst-Case" in html

    def test_with_assembly_info(self):
        config = ReportConfig(title="Test")
        assembly_info = {
            "name": "My Assembly",
            "description": "Test assembly",
            "bodies": [{"name": "Block", "features": [], "description": "A block"}],
            "mates": [{"name": "m1", "body_a": "A", "body_b": "B",
                        "mate_type": "coplanar", "distance_tol": 0.01}],
            "measurement": {"name": "gap", "body_a": "A", "feature_a": "f1",
                             "body_b": "B", "feature_b": "f2",
                             "measurement_type": "distance"},
        }
        html = generate_html_report(config, {"wc": _FakeResult()},
                                     assembly_info=assembly_info)
        assert "My Assembly" in html
        assert "Bodies" in html
        assert "Mates" in html

    def test_sensitivity_table(self):
        config = ReportConfig(include_sensitivity=True)
        html = generate_html_report(config, {"wc": _FakeResult()})
        assert "param_A" in html
        assert "param_B" in html


class TestTextReport:
    def test_basic(self):
        config = ReportConfig(title="Test", project="P1")
        text = generate_text_report(config, {"wc": _FakeResult()})
        assert "Test" in text
        assert "Worst-Case" in text
        assert "END OF REPORT" in text


class TestAPQPReport:
    def test_basic(self):
        config = ReportConfig(title="APQP Test", project="Auto")
        html = generate_apqp_report(config, {"wc": _FakeResult()},
                                     spec_limits=(29.5, 30.5))
        assert "APQP" in html
        assert "Design Verification" in html
        assert "Dimensional Analysis" in html
        assert "29.5" in html
        assert "Approval" in html


class TestSaveReport:
    def test_save(self, tmp_path):
        config = ReportConfig(title="Save Test")
        html = generate_html_report(config, {"wc": _FakeResult()})
        path = str(tmp_path / "report.html")
        save_report(html, path)
        with open(path) as f:
            content = f.read()
        assert "Save Test" in content
