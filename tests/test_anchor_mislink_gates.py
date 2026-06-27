"""Tests for the anchor mis-link gates added for form_2 page-3 bugs:
- checkbox-label-prefix classification ("[ ] Enclosed ...")
- table label-cell vertical-proximity gate (no cross-page text matches)
"""
from __future__ import annotations

from src.field_anchor_engine import _classify_field, _matching_label_cell

_METRICS = {"line_height": 0.012}


def test_label_prefix_checkbox_classified():
    t, _ = _classify_field("[ ] Enclosed (Please tick): Bank Account Details Proof Provided.", "", None, _METRICS)
    assert t == "checkbox"


def test_checked_label_prefix_checkbox():
    t, _ = _classify_field("[X] I confirm", "", None, _METRICS)
    assert t == "checkbox"


def test_caption_value_not_checkbox():
    # "Total Cash Component (Rs.)" with caption value "(in words)" stays text.
    t, _ = _classify_field("Total Cash Component (Rs.)", "(in words)", None, _METRICS)
    assert t == "text"


def test_normal_label_not_checkbox():
    t, _ = _classify_field("First Name", "", None, _METRICS)
    assert t == "text"


def _cell(text, x, y, w=0.1, h=0.02):
    return {"text": text, "bbox": {"x": x, "y": y, "width": w, "height": h}, "page": 1,
            "table_id": "t", "row_index": 1, "column_index": 1}


def test_label_cell_match_rejects_distant_cell():
    # A label at y0.65 must NOT match a same-text cell at y0.06 (cross-page).
    label_box = {"x": 0.10, "y": 0.65, "width": 0.20, "height": 0.015}
    far = _cell("Total Cash Component", 0.10, 0.06)
    assert _matching_label_cell("Total Cash Component", 1, label_box, [far]) is None


def test_label_cell_match_accepts_near_cell():
    label_box = {"x": 0.10, "y": 0.65, "width": 0.20, "height": 0.015}
    near = _cell("Total Cash Component", 0.10, 0.652)
    assert _matching_label_cell("Total Cash Component", 1, label_box, [near]) is not None
