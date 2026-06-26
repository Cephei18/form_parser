"""Tests for the HTML dashboard builder (Task 3)."""
from __future__ import annotations

from src.diagnostics_dashboard import build_dashboard_html


def test_dashboard_is_self_contained():
    summary = {
        "form_diagnostics_summary": {
            "total_widgets": 100, "checkbox_total": 80, "checkbox_share": 0.8,
            "suspicious_checkboxes": 10, "duplicate_rate": 0.2, "ambiguity_count": 5,
            "comb_group_count": 3, "comb_cell_count": 12, "table_types": {"MATRIX": 1},
            "page_checkbox_counts": {"3": 60}, "confidence_distribution": {"HIGH": 30, "MEDIUM": 50, "LOW": 20},
            "penalty_saturation": {"percent_penalized": 90.0, "saturated": True, "penalty_histogram": {}, "penalties_by_page": {}},
            "dotted_leader_recall": {"leaders_detected": 10, "filter_reason_counts": {}},
        }
    }
    html = build_dashboard_html(summary)
    assert html.startswith("<!doctype html>")
    assert "function render" in html
    # No network dependencies.
    assert "http://" not in html and "https://" not in html
    assert "cdn" not in html.lower()
    # The embedded data is present.
    assert "form_diagnostics_summary" in html


def test_dashboard_slims_large_payload():
    # A huge nested array must NOT bloat the HTML — the builder embeds only the
    # curated subset.
    summary = {
        "form_diagnostics_summary": {"total_widgets": 1},
        "anchoring": {"anchors": [{"x": i} for i in range(100000)]},
    }
    html = build_dashboard_html(summary)
    assert "anchors" not in html
    assert len(html) < 50_000
