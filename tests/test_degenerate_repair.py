"""Tests for degenerate-value-anchor repair (src/field_anchor_engine).

Covers the inline "Label :......" case where a dotted leader touches the colon
and Textract collapses the value to a ~1px point: the anchor is repaired into a
real writing region extending to the next text on the row.
"""
from __future__ import annotations

from src.field_anchor_engine import _repair_degenerate_value_anchor

_METRICS = {
    "x_gap": 0.008,
    "page_right": 0.95,
    "wide_row_tolerance": 0.032,
    "text_width": 0.047,
    "line_height": 0.011,
}


def _tb(text, x, y, w, h=0.011):
    return {"bbox": {"x": x, "y": y, "width": w, "height": h}, "text": text, "page": 1}


def test_repairs_inline_collapsed_value_to_next_text():
    # "Mobile No. :........Alternate No" — value collapsed onto the colon.
    anchor = {"x": 0.158, "y": 0.411, "width": 0.005, "height": 0.004}
    text_boxes = [
        _tb("Mobile No. :", 0.070, 0.407, 0.093),
        _tb(":", 0.159, 0.410, 0.004),            # the collapsed-onto colon (ignored)
        _tb("Alternate No", 0.286, 0.407, 0.093),  # the right obstacle
    ]
    rep = _repair_degenerate_value_anchor(anchor, 1, text_boxes, _METRICS, multiline_hint=False)
    assert rep is not None
    assert rep["anchor_type"] == "value_block"
    assert "repaired_degenerate_value" in rep["reasons"]
    b = rep["bbox"]
    assert abs(b["x"] - 0.158) < 0.01          # starts at the anchor
    assert 0.10 < b["width"] < 0.14            # extends to just before "Alternate No"
    assert b["x"] + b["width"] <= 0.286        # does not cross the next label


def test_extends_to_page_right_when_no_obstacle():
    anchor = {"x": 0.505, "y": 0.866, "width": 0.001, "height": 0.001}
    text_boxes = [_tb("Sign. Patient", 0.40, 0.860, 0.10)]
    rep = _repair_degenerate_value_anchor(anchor, 1, text_boxes, _METRICS, multiline_hint=False)
    assert rep is not None
    assert rep["bbox"]["width"] > 0.3          # runs out to near the page margin


def test_returns_none_when_no_room():
    anchor = {"x": 0.90, "y": 0.40, "width": 0.001, "height": 0.001}
    text_boxes = [_tb("X", 0.905, 0.40, 0.02)]
    rep = _repair_degenerate_value_anchor(anchor, 1, text_boxes, _METRICS, multiline_hint=False)
    assert rep is None
