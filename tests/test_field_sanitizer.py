"""Tests for the generalized field-hygiene pass (src/field_sanitizer.py)."""
from __future__ import annotations

import os

import pytest

from src.field_sanitizer import (
    DEGENERATE_MIN_HEIGHT,
    DEGENERATE_MIN_WIDTH,
    dedupe_overlapping_fields,
    drop_degenerate_candidates,
    field_hygiene_enabled,
    is_boilerplate_label,
    is_degenerate_box,
    is_page_furniture,
)


def _box(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("FORM_PARSER_FIELD_HYGIENE_ENABLED", None)
    yield
    if saved is not None:
        os.environ["FORM_PARSER_FIELD_HYGIENE_ENABLED"] = saved
    else:
        os.environ.pop("FORM_PARSER_FIELD_HYGIENE_ENABLED", None)


# --- flag --------------------------------------------------------------------
def test_enabled_by_default():
    assert field_hygiene_enabled() is True


def test_kill_switch():
    os.environ["FORM_PARSER_FIELD_HYGIENE_ENABLED"] = "false"
    assert field_hygiene_enabled() is False


# --- rule 1: degenerate boxes ------------------------------------------------
def test_degenerate_box_detection():
    assert is_degenerate_box(_box(0.4, 0.3, 0.001, 0.001)) is True
    assert is_degenerate_box(_box(0.4, 0.3, DEGENERATE_MIN_WIDTH / 2, 0.02)) is True
    assert is_degenerate_box(_box(0.4, 0.3, 0.02, DEGENERATE_MIN_HEIGHT / 2)) is True
    assert is_degenerate_box(_box(0.4, 0.3, 0.30, 0.017)) is False
    assert is_degenerate_box(None) is True


def test_drop_degenerate_prefers_real_feature():
    # The 1px value block (the bug) vs the genuine underline next to the label.
    ghost = {"anchor_type": "value_block", "bbox": _box(0.40, 0.32, 0.001, 0.001), "score": 0.94}
    underline = {"anchor_type": "underline", "bbox": _box(0.59, 0.30, 0.30, 0.017), "score": 0.88}
    kept = drop_degenerate_candidates([ghost, underline])
    assert kept == [underline]


def test_drop_degenerate_keeps_all_when_every_candidate_degenerate():
    only = [{"anchor_type": "value_block", "bbox": _box(0.4, 0.3, 0.001, 0.001), "score": 0.94}]
    assert drop_degenerate_candidates(only) == only


def test_drop_degenerate_exempts_checkbox_glyph():
    cb = {"anchor_type": "checkbox_region", "bbox": _box(0.1, 0.1, 0.018, 0.018), "score": 0.7}
    assert drop_degenerate_candidates([cb]) == [cb]


def test_drop_degenerate_noop_when_disabled():
    os.environ["FORM_PARSER_FIELD_HYGIENE_ENABLED"] = "false"
    cands = [{"anchor_type": "value_block", "bbox": _box(0.4, 0.3, 0.001, 0.001), "score": 0.94}]
    assert drop_degenerate_candidates(cands) == cands


# --- rule 2: page furniture --------------------------------------------------
def test_boilerplate_labels():
    assert is_boilerplate_label("SampleWords") is True
    assert is_boilerplate_label("© 2010 SampleForms LLC") is True
    assert is_boilerplate_label("www.sampleforms.com") is True
    assert is_boilerplate_label("Page 1 of 3") is True
    assert is_boilerplate_label("First Name") is False


def test_margin_furniture_in_footer_with_no_value():
    is_f, reason = is_page_furniture(
        "downlost 5", _box(0.40, 0.962, 0.05, 0.008), _box(0.45, 0.963, 0.10, 0.008), ""
    )
    assert is_f is True
    assert reason == "margin_furniture"


def test_instruction_note_suppressed_when_value_is_a_sentence():
    is_f, reason = is_page_furniture(
        "Note :", _box(0.14, 0.93, 0.04, 0.012),
        _box(0.20, 0.93, 0.50, 0.012),
        "For any change in demographic data, Please contact the admission",
    )
    assert is_f is True
    assert reason == "instruction_note"


def test_blank_notes_field_preserved():
    # An empty "Notes:" input box (no pre-printed sentence) is a real field.
    is_f, _ = is_page_furniture("Notes", _box(0.1, 0.5, 0.06, 0.012), _box(0.2, 0.5, 0.4, 0.02), "")
    assert is_f is False


def test_margin_furniture_preserves_low_field_with_value():
    # A real answer that happens to sit low on the page must survive.
    is_f, _ = is_page_furniture(
        "Signature", _box(0.10, 0.96, 0.08, 0.012), _box(0.30, 0.96, 0.30, 0.02), "John Doe"
    )
    assert is_f is False


def test_furniture_never_drops_checkbox():
    is_f, _ = is_page_furniture(
        "SampleWords", _box(0.79, 0.96, 0.10, 0.014), _box(0.78, 0.98, 0.04, 0.02), "", field_type="checkbox"
    )
    assert is_f is False


def test_furniture_keeps_normal_body_field():
    is_f, _ = is_page_furniture(
        "Email", _box(0.36, 0.31, 0.04, 0.011), _box(0.59, 0.30, 0.30, 0.017), ""
    )
    assert is_f is False


def test_furniture_noop_when_disabled():
    os.environ["FORM_PARSER_FIELD_HYGIENE_ENABLED"] = "false"
    is_f, _ = is_page_furniture("SampleWords", _box(0.79, 0.96, 0.1, 0.014), None, "")
    assert is_f is False


# --- rule 3: duplicate dedupe ------------------------------------------------
def _field(fid, box, conf, ftype="text", **extra):
    return {"field_id": fid, "bbox": box, "confidence": conf, "field_type": ftype, "page": 1, **extra}


def test_dedupe_keeps_higher_confidence():
    a = _field("f1", _box(0.175, 0.211, 0.718, 0.029), 0.94)  # Address
    b = _field("f2", _box(0.175, 0.211, 0.718, 0.014), 0.90)  # sub-caption duplicate
    kept, dropped = dedupe_overlapping_fields([a, b])
    assert dropped == ["f2"]
    assert [m["field_id"] for m in kept] == ["f1"]


def test_dedupe_keeps_big_region_containing_small_field():
    # A tall multiline address region geometrically contains a small Pin field on
    # its last line: smaller-box overlap is ~1.0 but they are different widgets.
    addr = _field("addr", _box(0.073, 0.305, 0.400, 0.060), 0.90)
    pin = _field("pin", _box(0.232, 0.355, 0.240, 0.014), 0.94)
    kept, dropped = dedupe_overlapping_fields([addr, pin])
    assert dropped == []
    assert len(kept) == 2


def test_boilerplate_form_codes():
    assert is_boilerplate_label("FO/Reg.") is True
    assert is_boilerplate_label("Form/Ver. 2.0/April'23") is True
    assert is_boilerplate_label("Ver. 3") is True
    assert is_boilerplate_label("Reverse") is False
    assert is_boilerplate_label("Registration") is False


def test_dedupe_leaves_distinct_regions():
    a = _field("f1", _box(0.10, 0.20, 0.30, 0.02), 0.9)
    b = _field("f2", _box(0.10, 0.40, 0.30, 0.02), 0.9)
    kept, dropped = dedupe_overlapping_fields([a, b])
    assert dropped == []
    assert len(kept) == 2


def test_dedupe_protects_checkboxes_and_groups():
    a = _field("f1", _box(0.1, 0.1, 0.02, 0.02), 0.9, ftype="checkbox")
    b = _field("f2", _box(0.1, 0.1, 0.02, 0.02), 0.9, ftype="checkbox")
    kept, dropped = dedupe_overlapping_fields([a, b])
    assert dropped == []
    # comb/radio widgets carry widget_type and are also left intact
    c = _field("f3", _box(0.2, 0.2, 0.3, 0.02), 0.9, widget_type="comb")
    d = _field("f4", _box(0.2, 0.2, 0.3, 0.02), 0.8, widget_type="comb")
    kept, dropped = dedupe_overlapping_fields([c, d])
    assert dropped == []


def test_dedupe_noop_when_disabled():
    os.environ["FORM_PARSER_FIELD_HYGIENE_ENABLED"] = "false"
    a = _field("f1", _box(0.175, 0.211, 0.718, 0.029), 0.94)
    b = _field("f2", _box(0.175, 0.211, 0.718, 0.014), 0.90)
    kept, dropped = dedupe_overlapping_fields([a, b])
    assert dropped == []
    assert len(kept) == 2
