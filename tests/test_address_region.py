"""Tests for key-less address-region synthesis (src/address_region.py)."""
from __future__ import annotations

import os

import pytest

from src.address_region import address_region_enabled, emit_address_regions
from src.section_detector import SectionIndex

_METRICS = {1: {"line_height": 0.012, "page_right": 0.95, "page_left": 0.05}}


def _line(text, x, y, w=0.39, h=0.012, page=1):
    return {"text": text, "bbox": {"x": x, "y": y, "width": w, "height": h}, "page": page}


def _emit(lines, text_boxes=None, mappings=None, nonfill=None):
    return emit_address_regions(
        lines,
        text_boxes=text_boxes or [],
        mappings=mappings or [],
        metrics_by_page=_METRICS,
        section_index=SectionIndex([]),
        page_px=lambda p: (1000, 1400),
        non_fillable_pages=nonfill or set(),
    )


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("FORM_PARSER_ADDRESS_REGION_ENABLED", None)
    yield
    if saved is not None:
        os.environ["FORM_PARSER_ADDRESS_REGION_ENABLED"] = saved
    else:
        os.environ.pop("FORM_PARSER_ADDRESS_REGION_ENABLED", None)


def test_enabled_default_and_kill_switch():
    assert address_region_enabled() is True
    os.environ["FORM_PARSER_ADDRESS_REGION_ENABLED"] = "false"
    assert address_region_enabled() is False


def test_synthesizes_keyless_mailing_address():
    lines = [_line("MAILING ADDRESS OF FIRST / SOLE APPLICANT (Mandatory)", 0.08, 0.584)]
    # CITY row below bounds the band.
    text = [{"text": "CITY", "bbox": {"x": 0.09, "y": 0.622, "width": 0.03, "height": 0.012}, "page": 1}]
    out = _emit(lines, text_boxes=text)
    assert len(out) == 1
    m = out[0]
    assert m["field_type"] == "multiline"
    assert m["source"] == "textract_address_region"
    assert 0.584 < m["bbox"]["y"] < 0.622
    assert m["bbox"]["width"] > 0.5


def test_skipped_when_field_already_covers_label():
    # An address Textract keyed (a field's label_bbox overlaps the heading).
    lines = [_line("Present Address", 0.08, 0.30)]
    text = [{"text": "Pin", "bbox": {"x": 0.2, "y": 0.34, "width": 0.03, "height": 0.012}, "page": 1}]
    mappings = [{"page": 1, "label": "Present Address :", "field_type": "multiline",
                 "label_bbox": {"x": 0.08, "y": 0.30, "width": 0.10, "height": 0.012},
                 "bbox": {"x": 0.20, "y": 0.30, "width": 0.40, "height": 0.02}}]
    assert _emit(lines, text_boxes=text, mappings=mappings) == []


def test_address_type_prompt_not_matched():
    # "Address Type:" is a checkbox prompt, not a write-in address.
    lines = [_line("Address Type: Residential or Business", 0.05, 0.51)]
    assert _emit(lines) == []


def test_skipped_when_band_occupied():
    lines = [_line("Mailing Address", 0.08, 0.40)]
    text = [{"text": "CITY", "bbox": {"x": 0.09, "y": 0.44, "width": 0.03, "height": 0.012}, "page": 1}]
    # A field already sits in the writing band.
    mappings = [{"page": 1, "label": "X", "field_type": "text", "label_bbox": None,
                 "bbox": {"x": 0.3, "y": 0.42, "width": 0.2, "height": 0.012}}]
    assert _emit(lines, text_boxes=text, mappings=mappings) == []


def test_skipped_on_non_fillable_page():
    lines = [_line("Mailing Address", 0.08, 0.40, page=2)]
    text = [{"text": "CITY", "bbox": {"x": 0.09, "y": 0.44, "width": 0.03, "height": 0.012}, "page": 2}]
    _METRICS[2] = _METRICS[1]
    assert _emit(lines, text_boxes=text, nonfill={2}) == []


def test_disabled_emits_nothing():
    os.environ["FORM_PARSER_ADDRESS_REGION_ENABLED"] = "false"
    lines = [_line("Mailing Address", 0.08, 0.584)]
    text = [{"text": "CITY", "bbox": {"x": 0.09, "y": 0.622, "width": 0.03, "height": 0.012}, "page": 1}]
    assert _emit(lines, text_boxes=text) == []
