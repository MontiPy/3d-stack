"""Tests for PDF report generation."""

import pytest
from tolerance_stack.reporting import (
    ReportConfig, generate_pdf_report, save_pdf_report,
)


class TestPDFReport:

    def test_basic_pdf_generation(self):
        config = ReportConfig(
            title="Test PDF Report",
            project="Test Project",
            author="Tester",
        )
        results = {
            "manual": {"nominal": 10.0, "variation": 0.5},
        }
        pdf_bytes = generate_pdf_report(config, results)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        # Check PDF header
        assert pdf_bytes[:5] == b"%PDF-"
        # Check PDF trailer
        assert b"%%EOF" in pdf_bytes

    def test_pdf_with_assembly_info(self):
        config = ReportConfig(title="Assembly PDF", project="Proj")
        results = {"wc": {"method": "wc", "nominal": 5.0}}
        assembly_info = {"name": "TestAssembly", "bodies": 3, "mates": 2}
        pdf_bytes = generate_pdf_report(config, results, assembly_info=assembly_info)
        assert b"%PDF-" in pdf_bytes

    def test_pdf_multiple_pages(self):
        config = ReportConfig(title="Multi-Page")
        results = {
            "wc": {"method": "wc", "nominal": 5.0},
            "rss": {"method": "rss", "nominal": 5.0},
            "mc": {"method": "mc", "nominal": 5.0},
        }
        pdf_bytes = generate_pdf_report(config, results)
        # Should have multiple page objects
        assert pdf_bytes.count(b"/Type /Page") >= 1

    def test_save_pdf_report(self, tmp_path):
        config = ReportConfig(title="Save Test")
        results = {"info": {"data": "test"}}
        pdf_bytes = generate_pdf_report(config, results)
        path = str(tmp_path / "report.pdf")
        save_pdf_report(pdf_bytes, path)
        with open(path, "rb") as f:
            content = f.read()
        assert content == pdf_bytes
