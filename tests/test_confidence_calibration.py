"""Tests for document-level confidence calibration (src/confidence_calibration.py)."""
from __future__ import annotations

from src.confidence_calibration import apply_confidence_calibration


def _m(field_id, box, field_type="text", score=0.8, anchoring=None, page=1):
    m = {"field_id": field_id, "field_type": field_type, "page": page, "bbox": box, "confidence_score": score}
    if anchoring:
        m["anchoring"] = anchoring
    return m


def test_disabled_returns_unchanged():
    m = _m(1, {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05})
    out = apply_confidence_calibration([m], enabled=False)
    assert out["diagnostics"]["enabled"] is False
    assert "calibration" not in m  # no mutation


def test_duplicate_region_penalized():
    box = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}
    a = _m(1, dict(box))
    b = _m(2, dict(box))  # fully overlapping -> duplicate
    out = apply_confidence_calibration([a, b], enabled=True)
    assert out["diagnostics"]["duplicate_mappings"] == 2
    assert "duplicate_region" in a["calibration"]["reasons"]
    # Original score untouched; calibrated score is lower.
    assert a["confidence_score"] == 0.8
    assert a["calibrated_confidence_score"] < 0.8


def test_checkbox_explosion_penalized():
    boxes = [
        _m(i, {"x": 0.01 * i, "y": 0.1, "width": 0.005, "height": 0.005}, field_type="checkbox")
        for i in range(1, 15)  # 14 checkboxes on one page > default limit 12
    ]
    out = apply_confidence_calibration(boxes, enabled=True)
    assert out["diagnostics"]["checkbox_explosion_mappings"] == 14
    assert "checkbox_explosion" in boxes[0]["calibration"]["reasons"]


def test_ambiguous_assignment_penalized():
    m = _m(1, {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}, anchoring={"candidate_count": 7})
    out = apply_confidence_calibration([m], enabled=True)
    assert out["diagnostics"]["ambiguous_mappings"] == 1
    assert "ambiguous_assignment" in m["calibration"]["reasons"]


def test_clean_mapping_no_penalty():
    m = _m(1, {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05})
    out = apply_confidence_calibration([m], enabled=True)
    assert m["calibration"]["penalty"] == 0.0
    assert m["calibrated_confidence_score"] == 0.8  # base preserved
    assert out["diagnostics"]["penalized_mappings"] == 0


def test_penalty_saturation_block():
    box = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05}
    a = _m(1, dict(box), field_type="checkbox")
    b = _m(2, dict(box), field_type="checkbox")  # duplicate -> both penalized
    clean = _m(3, {"x": 0.6, "y": 0.6, "width": 0.2, "height": 0.05}, field_type="text")
    sat = apply_confidence_calibration([a, b, clean], enabled=True)["diagnostics"]["penalty_saturation"]
    assert sat["percent_penalized"] == round(100 * 2 / 3, 2)
    assert set(sat["penalty_histogram"]) == {"0", "0-0.1", "0.1-0.2", "0.2-0.3", "0.3+"}
    assert sat["confidence_distribution"]["MEDIUM"] == 3  # base 0.8 buckets to MEDIUM (>=0.55, <0.82)
    assert "checkbox" in sat["penalties_by_type"]
    assert sat["penalties_by_type"]["checkbox"]["penalized"] == 2
    assert "1" in sat["penalties_by_page"]
