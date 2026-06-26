"""Tests for dotted-leader source diagnostics (src/dotted_leader_diagnostics.py)."""
from __future__ import annotations

from src.dotted_leader_diagnostics import _is_dot_run, analyze_dotted_leaders


def _page(detected, page=1):
    return {"page": page, "source_image": "p.png", "detected": detected, "rejected": []}


def test_is_dot_run():
    assert _is_dot_run("......")
    assert _is_dot_run(". . . . .")
    assert _is_dot_run("………")  # three ellipsis glyphs
    assert not _is_dot_run("Name")
    assert not _is_dot_run("..")  # too few


def test_text_origin_classification():
    box = {"x": 0.2, "y": 0.5, "width": 0.3, "height": 0.01}
    pages = [_page([{"bbox": box}])]
    text_boxes = [{"bbox": {"x": 0.2, "y": 0.5, "width": 0.3, "height": 0.012}, "text": "........", "page": 1}]
    diag = analyze_dotted_leaders(pages, text_boxes=text_boxes)
    assert diag["text_origin_count"] == 1
    assert diag["graphics_origin_count"] == 0
    assert diag["classified"][0]["origin"] == "text"


def test_graphics_origin_classification():
    box = {"x": 0.2, "y": 0.5, "width": 0.3, "height": 0.01}
    pages = [_page([{"bbox": box}])]
    # OCR token present but it's a real label, not dots.
    text_boxes = [{"bbox": {"x": 0.2, "y": 0.5, "width": 0.3, "height": 0.012}, "text": "Signature", "page": 1}]
    diag = analyze_dotted_leaders(pages, text_boxes=text_boxes)
    assert diag["graphics_origin_count"] == 1
    assert diag["text_origin_count"] == 0


def test_ocr_only_dot_run_surfaced():
    # A dot run that no CV detection covers.
    text_boxes = [{"bbox": {"x": 0.6, "y": 0.8, "width": 0.2, "height": 0.01}, "text": "......", "page": 1}]
    diag = analyze_dotted_leaders([], text_boxes=text_boxes)
    assert diag["ocr_only_dot_run_count"] == 1
    assert diag["cv_detected_count"] == 0


def test_no_behavior_change_flag():
    diag = analyze_dotted_leaders([], text_boxes=[])
    assert diag["behavior_change"] is False
    assert diag["enabled"] is True


def test_recall_accounting():
    # 3 leaders; one sits just right of a label (associates); one mapping selected
    # a dotted underline. Expect: detected=3, associated=1, selected=1.
    leaders = [
        {"bbox": {"x": 0.40, "y": 0.30, "width": 0.30, "height": 0.01}},  # associates with label
        {"bbox": {"x": 0.40, "y": 0.60, "width": 0.30, "height": 0.01}},  # no label
        {"bbox": {"x": 0.40, "y": 0.80, "width": 0.30, "height": 0.01}},  # no label
    ]
    pages = [_page(leaders)]
    field_labels = [{"page": 1, "bbox": {"x": 0.10, "y": 0.295, "width": 0.25, "height": 0.02}}]
    mappings = [{"anchoring": {"anchor_type": "dotted_underline"}}]
    diag = analyze_dotted_leaders(pages, field_labels=field_labels, mappings=mappings)
    rec = diag["recall"]
    assert rec["leaders_detected"] == 3
    assert rec["leaders_associated_to_a_label"] == 1
    assert rec["leaders_selected"] == 1
    assert rec["filter_reason_counts"]["no_field_label_association"] == 2
    assert rec["filter_reason_counts"]["selected_as_answer_region"] == 1


def test_recall_absent_when_not_requested():
    diag = analyze_dotted_leaders([_page([])], text_boxes=[])
    assert "recall" not in diag
