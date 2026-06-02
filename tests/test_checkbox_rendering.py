"""
Targeted regression tests for Textract checkbox handling.

Guards against the regression where checkboxes Textract could not associate with
an owner (standalone / agreement / option-list boxes) were silently dropped and
never rendered. Covers: checked, unchecked, grouped sections, associated vs
unassociated ownership, label alignment, and AcroForm widget rendering in the PDF.

Runnable directly (``python tests/test_checkbox_rendering.py``) or via pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging
logging.disable(logging.CRITICAL)

from src.textract_parser import parse_textract_response
from src import field_anchor_engine as fae
from src.pdf_generator import create_pdf_with_fields

# A real form image to use as the render background (geometry is normalised, so
# any same-pipeline image works).
_IMAGE = next((REPO_ROOT / "output" / "runs").glob("*/mapping.png"))


def _bbox(left, top, w=0.02, h=0.02):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}


def _selection(block_id, left, top, selected):
    return {
        "Id": block_id,
        "BlockType": "SELECTION_ELEMENT",
        "SelectionStatus": "SELECTED" if selected else "NOT_SELECTED",
        "Confidence": 95.0,
        "Geometry": _bbox(left, top),
    }


def _word(block_id, text, left, top):
    return {"Id": block_id, "BlockType": "WORD", "Text": text, "Confidence": 99.0, "Geometry": _bbox(left, top, 0.08)}


def _standalone_response():
    """Two unassociated checkboxes (parent = PAGE): one checked, one unchecked."""
    return {
        "Blocks": [
            {"Id": "page", "BlockType": "PAGE", "Geometry": _bbox(0, 0, 1, 1),
             "Relationships": [{"Type": "CHILD", "Ids": ["s1", "s2", "w1", "w2"]}]},
            _selection("s1", 0.10, 0.20, True),
            _selection("s2", 0.10, 0.25, False),
            _word("w1", "I agree", 0.13, 0.20),
            _word("w2", "I disagree", 0.13, 0.25),
        ]
    }


def _associated_response():
    """One checkbox tied to a KEY via VALUE relationship (value_to_key_mapping)."""
    return {
        "Blocks": [
            {"Id": "page", "BlockType": "PAGE", "Geometry": _bbox(0, 0, 1, 1),
             "Relationships": [{"Type": "CHILD", "Ids": ["k", "v", "kw", "s1"]}]},
            {"Id": "k", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"], "Confidence": 90.0,
             "Geometry": _bbox(0.10, 0.30, 0.10),
             "Relationships": [{"Type": "CHILD", "Ids": ["kw"]}, {"Type": "VALUE", "Ids": ["v"]}]},
            {"Id": "kw", "BlockType": "WORD", "Text": "Married", "Confidence": 99.0, "Geometry": _bbox(0.10, 0.30, 0.07)},
            {"Id": "v", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"], "Confidence": 90.0,
             "Geometry": _bbox(0.22, 0.30), "Relationships": [{"Type": "CHILD", "Ids": ["s1"]}]},
            _selection("s1", 0.22, 0.30, True),
        ]
    }


def _grouped_response(n=4):
    """A horizontal group of n checkboxes (an option row)."""
    blocks = [{"Id": "page", "BlockType": "PAGE", "Geometry": _bbox(0, 0, 1, 1),
               "Relationships": [{"Type": "CHILD", "Ids": [f"s{i}" for i in range(n)]}]}]
    for i in range(n):
        blocks.append(_selection(f"s{i}", 0.10 + i * 0.12, 0.40, selected=(i == 1)))
    return {"Blocks": blocks}


def _associated_checkbox_blocks(idx, left, top, selected, label):
    """A KEY_VALUE_SET whose VALUE is a SELECTION_ELEMENT — Textract emits this as
    BOTH a native checkbox AND a "[X]"/"[ ]" token field item (the duplication)."""
    kid, vid, kw, sid = f"k{idx}", f"v{idx}", f"kw{idx}", f"s{idx}"
    blocks = [
        {"Id": kid, "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"], "Confidence": 90.0,
         "Geometry": _bbox(left, top, 0.10),
         "Relationships": [{"Type": "CHILD", "Ids": [kw]}, {"Type": "VALUE", "Ids": [vid]}]},
        _word(kw, label, left, top),
        {"Id": vid, "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"], "Confidence": 90.0,
         "Geometry": _bbox(left + 0.12, top),
         "Relationships": [{"Type": "CHILD", "Ids": [sid]}]},
        _selection(sid, left + 0.12, top, selected),
    ]
    return blocks, [kid, kw, vid, sid]


def _associated_group_response(specs):
    """specs: list of (selected, label). Each entry becomes one duplicated checkbox."""
    all_blocks, child_ids = [], []
    for i, (sel, label) in enumerate(specs):
        blocks, ids = _associated_checkbox_blocks(i, 0.10, 0.20 + i * 0.06, sel, label)
        all_blocks += blocks
        child_ids += ids
    page = {"Id": "page", "BlockType": "PAGE", "Geometry": _bbox(0, 0, 1, 1),
            "Relationships": [{"Type": "CHILD", "Ids": child_ids}]}
    return {"Blocks": [page] + all_blocks}


def _checkbox_mappings(response):
    parsed = parse_textract_response(response)
    out = fae.build_anchored_mappings(response, parsed, _IMAGE)
    cbs = [m for m in out["mappings"] if m.get("field_type") == "checkbox"]
    return parsed, out, cbs


def _count_pdf_checkbox_widgets(mappings) -> int:
    out_pdf = Path("/tmp/_cb_test.pdf")
    create_pdf_with_fields(str(_IMAGE), mappings, str(out_pdf))
    data = out_pdf.read_bytes()
    # AcroForm checkbox fields are button fields: /FT /Btn
    return data.count(b"/Btn")


def _pdf_bytes_for_mappings(mappings) -> bytes:
    out_pdf = Path("/tmp/_cb_state_test.pdf")
    create_pdf_with_fields(str(_IMAGE), mappings, str(out_pdf))
    return out_pdf.read_bytes()


def test_unassociated_checkboxes_are_not_dropped():
    parsed, out, cbs = _checkbox_mappings(_standalone_response())
    assert len(parsed["checkboxes"]) == 2
    assert all(c["ownership"]["ownership_type"] == "unassociated" for c in parsed["checkboxes"])
    assert len(cbs) == 2, "standalone (unassociated) checkboxes must still render"
    assert out["diagnostics"]["unassociated_checkbox_count"] == 2
    assert out["diagnostics"]["dropped_unassociated_checkboxes"] is False


def test_checked_and_unchecked_state_preserved():
    _, _, cbs = _checkbox_mappings(_standalone_response())
    values = sorted(c["value"] for c in cbs)
    assert values == ["[ ]", "[X]"], f"expected one checked + one unchecked, got {values}"


def test_associated_checkbox_keeps_label():
    parsed, _, cbs = _checkbox_mappings(_associated_response())
    assert len(cbs) >= 1
    labels = " ".join(str(c.get("label", "")) for c in cbs).lower()
    assert "married" in labels, f"associated checkbox should carry its owner label, got {labels}"


def test_grouped_checkbox_section_all_render():
    _, _, cbs = _checkbox_mappings(_grouped_response(4))
    assert len(cbs) == 4, "every checkbox in a group must render"
    selected = [c for c in cbs if c["value"] == "[X]"]
    assert len(selected) == 1


def test_checkbox_label_alignment_geometry():
    # Each checkbox bbox must be a valid, on-page region aligned to its glyph.
    _, _, cbs = _checkbox_mappings(_grouped_response(4))
    xs = []
    for c in cbs:
        b = c["bbox"]
        assert 0 <= b["x"] <= 1 and 0 <= b["y"] <= 1 and 0 < b["width"] <= 1 and 0 < b["height"] <= 1
        xs.append(b["x"])
    assert xs == sorted(xs), "grouped checkboxes should preserve left-to-right order"


def test_pdf_renders_checkbox_widgets():
    _, _, cbs = _checkbox_mappings(_standalone_response())
    widgets = _count_pdf_checkbox_widgets(cbs)
    assert widgets >= len(cbs), f"expected >= {len(cbs)} /Btn widgets in PDF, found {widgets}"


def test_pdf_checkbox_widget_state_matches_detected_value():
    _, _, cbs = _checkbox_mappings(_standalone_response())
    data = _pdf_bytes_for_mappings(cbs)
    expected_checked = sum(1 for c in cbs if c["value"] == "[X]")
    expected_unchecked = sum(1 for c in cbs if c["value"] == "[ ]")
    assert data.count(b"/V /Yes") == expected_checked
    assert data.count(b"/AS /Yes") == expected_checked
    assert data.count(b"/V /Off") == expected_unchecked
    assert data.count(b"/AS /Off") == expected_unchecked


def test_single_checkbox_duplication_is_deduped():
    parsed, out, cbs = _checkbox_mappings(_associated_group_response([(True, "Married")]))
    assert out["diagnostics"]["deduped_token_checkbox_count"] == 1
    assert len(cbs) == 1, "one logical checkbox must render exactly one widget"
    assert cbs[0]["value"] == "[X]"
    assert "married" in str(cbs[0]["label"]).lower(), "canonical checkbox keeps the owner label"


def test_grouped_checkbox_duplication_is_deduped():
    specs = [(False, "Option A"), (True, "Option B"), (False, "Option C")]
    parsed, out, cbs = _checkbox_mappings(_associated_group_response(specs))
    assert len(cbs) == 3, "a 3-option group must render 3 widgets, not 6"
    assert out["diagnostics"]["deduped_token_checkbox_count"] == 3
    assert sum(1 for c in cbs if c["value"] == "[X]") == 1


def test_nearby_independent_checkboxes_not_deduped():
    # Two distinct standalone checkboxes near each other, with NO token source.
    resp = {"Blocks": [
        {"Id": "page", "BlockType": "PAGE", "Geometry": _bbox(0, 0, 1, 1),
         "Relationships": [{"Type": "CHILD", "Ids": ["s0", "s1"]}]},
        _selection("s0", 0.10, 0.40, True),
        _selection("s1", 0.14, 0.40, False),
    ]}
    parsed, out, cbs = _checkbox_mappings(resp)
    assert len(cbs) == 2, "independent adjacent checkboxes must both survive"
    assert out["diagnostics"]["deduped_token_checkbox_count"] == 0


def test_mixed_states_preserved_after_dedup():
    parsed, out, cbs = _checkbox_mappings(_associated_group_response([(True, "Yes"), (False, "No")]))
    assert len(cbs) == 2
    assert sorted(c["value"] for c in cbs) == ["[ ]", "[X]"], "checked/unchecked states must survive dedup"


def test_dedup_never_removes_a_selection_element():
    # Invariant: rendered checkbox count == number of SELECTION_ELEMENTs.
    resp = _associated_group_response([(True, "A"), (False, "B"), (True, "C")])
    n_sel = sum(1 for b in resp["Blocks"] if b.get("BlockType") == "SELECTION_ELEMENT")
    _, _, cbs = _checkbox_mappings(resp)
    assert len(cbs) == n_sel


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} checkbox tests passed")


if __name__ == "__main__":
    _run_all()
