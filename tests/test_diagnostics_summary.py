"""Tests for the curated form diagnostics summary (Task 2)."""
from __future__ import annotations

from src.diagnostics_summary import build_form_diagnostics_summary


def _diag():
    return {
        "anchoring": {
            "checkbox_validation": {"enabled": True, "mode": "observe", "would_reject_count": 3, "rejected": []},
            "comb_diagnostics": {"enabled": True, "likely_comb_group_count": 2, "total_cells_in_groups": 9},
            "dotted_leader_source": {"enabled": True, "recall": {"leaders_detected": 10, "leaders_selected": 1}},
            "table_intelligence": {
                "enabled": True,
                "table_count": 4,
                "table_types": {"LAYOUT_TABLE": 3, "MATRIX": 1},
                "input_cells": [{}, {}],
            },
        },
        "confidence_calibration": {
            "enabled": True,
            "duplicate_mappings": 4,
            "ambiguous_mappings": 6,
            "penalty_saturation": {"percent_penalized": 80.0, "confidence_distribution": {"HIGH": 1, "MEDIUM": 2, "LOW": 1}},
        },
    }


def _mappings():
    m = []
    for i in range(4):
        m.append({"field_id": i, "field_type": "checkbox", "page": 3, "bbox": {}})
    m.append({"field_id": 99, "field_type": "text", "page": 1, "bbox": {}})
    return m


def test_summary_keys_present():
    s = build_form_diagnostics_summary(_diag(), _mappings())
    for k in (
        "suspicious_checkboxes", "duplicate_rate", "ambiguity_count", "comb_group_count",
        "comb_cell_count", "table_types", "page_checkbox_counts", "confidence_distribution",
        "penalty_saturation", "dotted_leader_recall", "table_intelligence_summary",
    ):
        assert k in s


def test_summary_values():
    s = build_form_diagnostics_summary(_diag(), _mappings())
    assert s["total_widgets"] == 5
    assert s["checkbox_total"] == 4
    assert s["suspicious_checkboxes"] == 3
    assert s["duplicate_rate"] == round(4 / 5, 4)
    assert s["ambiguity_count"] == 6
    assert s["comb_group_count"] == 2 and s["comb_cell_count"] == 9
    assert s["table_types"] == {"LAYOUT_TABLE": 3, "MATRIX": 1}
    assert s["page_checkbox_counts"] == {"3": 4}
    assert s["table_intelligence_summary"]["input_cell_count"] == 2
    assert s["dotted_leader_recall"]["leaders_detected"] == 10


def test_summary_tolerates_flags_off():
    # Empty diagnostics (all flags off) must not raise and must give safe defaults.
    s = build_form_diagnostics_summary({}, [])
    assert s["total_widgets"] == 0
    assert s["comb_group_count"] == 0
    assert s["table_types"] == {}
    assert s["penalty_saturation"] == {"enabled": False}
