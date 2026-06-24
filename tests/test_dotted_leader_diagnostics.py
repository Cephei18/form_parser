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
