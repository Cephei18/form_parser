"""Tests for non-fillable page detection (src/page_classifier.py)."""
from __future__ import annotations

import os

import pytest

from src.page_classifier import non_fillable_pages, page_gating_enabled


def _word(text, page, top=0.5):
    return {"BlockType": "WORD", "Text": text, "Page": page,
            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": top, "Width": 0.1, "Height": 0.02}}}


def _line(text, page, top):
    return {"BlockType": "LINE", "Text": text, "Page": page,
            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": top, "Width": 0.3, "Height": 0.02}}}


def _key(page):
    return {"BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"], "Page": page,
            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.3, "Width": 0.1, "Height": 0.02}}}


def _raw(blocks):
    return {"Blocks": blocks}


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("FORM_PARSER_PAGE_GATING_ENABLED", None)
    yield
    if saved is not None:
        os.environ["FORM_PARSER_PAGE_GATING_ENABLED"] = saved
    else:
        os.environ.pop("FORM_PARSER_PAGE_GATING_ENABLED", None)


def test_enabled_by_default_and_kill_switch():
    assert page_gating_enabled() is True
    os.environ["FORM_PARSER_PAGE_GATING_ENABLED"] = "false"
    assert page_gating_enabled() is False


def test_blank_page_detected():
    raw = _raw([{"BlockType": "PAGE", "Page": 1}] + [_word("INTENTIONALLY", 1)] + [_word("BLANK", 1)])
    assert non_fillable_pages(raw) == {1: "blank"}


def test_blank_not_flagged_when_checkboxes_present():
    # Few words but a checkbox present -> a real (sparse) input page, not blank.
    raw = _raw([_word("Agree", 1), {"BlockType": "SELECTION_ELEMENT", "Page": 1,
                "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.3, "Width": 0.02, "Height": 0.02}}}])
    assert non_fillable_pages(raw) == {}


def test_blank_not_flagged_when_parsed_fields_present():
    raw = _raw([_word("PAN", 1)])
    parsed = {"field_items": [{"key": "PAN", "page": 1}]}
    assert non_fillable_pages(raw, parsed) == {}


def test_checklist_banner_detected():
    blocks = [_line("CHECKLIST (FOR OFFICE USE)", 1, 0.02)] + [_word("x", 1) for _ in range(40)]
    assert non_fillable_pages(_raw(blocks)) == {1: "banner:checklist"}


def test_riskometer_banner_detected():
    blocks = [_line("Scheme Riskometer & Benchmark Riskometer", 1, 0.03)] + [_word("x", 1) for _ in range(40)]
    assert non_fillable_pages(_raw(blocks)) == {1: "banner:riskometer"}


def test_long_sentence_mentioning_riskometer_not_flagged():
    # A fillable page whose subtitle merely references the riskometer must stay.
    line = _line("For Scheme Riskometer and Benchmark Riskometer refer last page of application form", 1, 0.05)
    blocks = [line, _key(1)] + [_word("x", 1) for _ in range(60)]
    assert non_fillable_pages(_raw(blocks)) == {}


def test_refer_instruction_heading_not_flagged():
    # "7. TRANSACTION DETAILS (refer instruction 7)" must not match the
    # 'instructions' title marker (it is not a short title starting with it).
    line = _line("7. TRANSACTION DETAILS (refer instruction 7)", 1, 0.03)
    blocks = [line, _key(1)] + [_word("x", 1) for _ in range(60)]
    assert non_fillable_pages(_raw(blocks)) == {}


def test_instructions_title_detected():
    blocks = [_line("Instructions", 1, 0.02)] + [_word("x", 1) for _ in range(40)]
    assert non_fillable_pages(_raw(blocks)) == {1: "title:instructions"}


def test_multi_page_mixed():
    blocks = (
        [_line("Application Form", 1, 0.03), _key(1)] + [_word("x", 1) for _ in range(60)]
        + [_line("CHECKLIST (FOR OFFICE USE)", 2, 0.02)] + [_word("y", 2) for _ in range(40)]
        + [_word("INTENTIONALLY LEFT BLANK", 3)]
    )
    assert non_fillable_pages(_raw(blocks)) == {2: "banner:checklist", 3: "blank"}
