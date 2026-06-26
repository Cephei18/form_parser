"""Tests for false-checkbox rejection (src/checkbox_validator.py)."""
from __future__ import annotations

import cv2
import numpy as np

from src.checkbox_validator import validate_checkboxes


def _checkbox(field_id, box, page=1):
    return {"field_id": field_id, "field_type": "checkbox", "page": page, "label": f"cb{field_id}", "bbox": box}


def test_real_square_checkbox_kept():
    cb = _checkbox(1, {"x": 0.10, "y": 0.10, "width": 0.02, "height": 0.02})
    kept, diag = validate_checkboxes([cb], text_boxes=[])
    assert len(kept) == 1
    assert diag["evaluations"][0]["decision"] == "kept"


def test_stroke_text_checkbox_rejected():
    # Tall, narrow box with an overlapping OCR token "11" -> vertical-stroke text.
    box = {"x": 0.10, "y": 0.10, "width": 0.008, "height": 0.03}
    cb = _checkbox(1, box)
    text_boxes = [{"bbox": box, "text": "11", "page": 1}]
    kept, diag = validate_checkboxes([cb], text_boxes=text_boxes)
    assert kept == []
    assert diag["rejected_count"] == 1
    assert any("ocr_text_stroke_glyphs" in r for r in diag["rejected"][0]["reasons"])


def test_non_checkbox_passes_through_unchanged():
    text = {"field_id": 9, "field_type": "text", "page": 1, "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.03}}
    kept, diag = validate_checkboxes([text], text_boxes=[])
    assert kept == [text]
    assert diag["checkbox_count"] == 0


def test_diagnostics_shape():
    cb = _checkbox(1, {"x": 0.1, "y": 0.1, "width": 0.02, "height": 0.02})
    _, diag = validate_checkboxes([cb], text_boxes=[])
    assert diag["enabled"] is True
    assert diag["feature_flag"] == "FORM_PARSER_CHECKBOX_VALIDATION_ENABLED"
    ev = diag["evaluations"][0]
    assert set(ev) >= {"field_id", "authenticity_score", "aspect", "ocr_text", "closed_contour", "reasons", "decision"}


def test_closed_contour_detected_with_image(tmp_path):
    # Draw a clear black square outline; the validator should see a closed contour.
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (90, 90), (0, 0, 0), 2)  # ~box at x[0.1..0.225], y[0.1..0.225]
    path = tmp_path / "page_1.png"
    cv2.imwrite(str(path), img)
    box = {"x": 0.10, "y": 0.10, "width": 0.125, "height": 0.125}
    cb = _checkbox(1, box)
    kept, diag = validate_checkboxes(
        [cb],
        page_images={1: path},
        text_boxes=[],
        image_sizes={1: {"width": 400, "height": 400}},
    )
    assert len(kept) == 1
    assert diag["evaluations"][0]["closed_contour"] is True


def test_threshold_override_rejects_borderline():
    cb = _checkbox(1, {"x": 0.1, "y": 0.1, "width": 0.02, "height": 0.02})
    # An impossibly high threshold rejects even a clean square.
    kept, diag = validate_checkboxes([cb], text_boxes=[], reject_threshold=0.99)
    assert kept == []
    assert diag["reject_threshold"] == 0.99


def test_observe_mode_drops_nothing():
    # A box that WOULD be rejected (impossible threshold) is kept in observe mode.
    cb = _checkbox(1, {"x": 0.1, "y": 0.1, "width": 0.02, "height": 0.02})
    kept, diag = validate_checkboxes([cb], text_boxes=[], reject_threshold=0.99, observe=True)
    assert kept == [cb]                      # nothing dropped
    assert diag["mode"] == "observe"
    assert diag["would_reject_count"] == 1
    assert diag["kept_count"] == 1
    assert diag["evaluations"][0]["decision"] == "would_reject"


def test_component_count_present_with_image(tmp_path):
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (90, 90), (0, 0, 0), 2)
    path = tmp_path / "page_1.png"
    cv2.imwrite(str(path), img)
    cb = _checkbox(1, {"x": 0.10, "y": 0.10, "width": 0.125, "height": 0.125})
    _, diag = validate_checkboxes([cb], page_images={1: path}, text_boxes=[], image_sizes={1: {"width": 400, "height": 400}})
    assert diag["evaluations"][0]["component_count"] is not None
