from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src.pdf_generator import create_pdf_with_fields
from src.pipelines.textract_pipeline import run_textract_pipeline
from src.radio_grouper import apply_radio_grouping


RADIO_FLAG = 1 << 15


def _section(section_id: str = "section_holding", title: str = "Holding Mode", page: int = 1) -> dict:
    return {
        "section_id": section_id,
        "title": title,
        "type": section_id.replace("section_", ""),
        "page": page,
        "confidence": 0.9,
    }


def _checkbox(label: str, x: float, y: float, *, selected: bool = False, page: int = 1, section: dict | None = None) -> dict:
    bbox = {"x": x, "y": y, "width": 0.022, "height": 0.022}
    return {
        "field_id": f"cb_{page}_{label.lower().replace(' ', '_').replace('/', '_')}_{int(x * 1000)}_{int(y * 1000)}",
        "label": label,
        "value": "[X]" if selected else "[ ]",
        "field_type": "checkbox",
        "bbox": bbox,
        "page": page,
        "section": section or _section(page=page),
        "answer_region": {"bbox": bbox, "type": "checkbox_region", "confidence": 0.92},
    }


def _group(labels: list[str], *, vertical: bool = False, selected_index: int | None = None, page: int = 1, section: dict | None = None) -> list[dict]:
    out = []
    for index, label in enumerate(labels):
        x = 0.20 if vertical else 0.20 + index * 0.12
        y = 0.20 + index * 0.05 if vertical else 0.20
        out.append(_checkbox(label, x, y, selected=selected_index == index, page=page, section=section))
    return out


@pytest.mark.parametrize(
    "labels",
    [
        ["Yes", "No"],
        ["Male", "Female"],
        ["Single", "Joint"],
        ["Savings", "Current"],
        ["Resident", "NRI"],
    ],
)
def test_vocabulary_radio_groups(labels):
    mappings, diagnostics = apply_radio_grouping(_group(labels, selected_index=0), enabled=True, threshold=0.86)

    assert diagnostics["group_count"] == 1
    assert all(mapping["widget_type"] == "radio" for mapping in mappings)
    assert {mapping["export_value"] for mapping in mappings} == {label.lower().replace(" ", "_") for label in labels}
    assert sum(1 for mapping in mappings if mapping["radio_selected"]) == 1


def test_three_option_group():
    mappings, diagnostics = apply_radio_grouping(_group(["Mr", "Mrs", "Ms"], selected_index=2), enabled=True)

    assert diagnostics["group_count"] == 1
    assert {mapping["export_value"] for mapping in mappings} == {"mr", "mrs", "ms"}
    assert [mapping["radio_selected"] for mapping in mappings] == [False, False, True]


def test_four_option_group():
    mappings, diagnostics = apply_radio_grouping(_group(["Mr", "Mrs", "Ms", "Dr"]), enabled=True)

    assert diagnostics["group_count"] == 1
    assert len({mapping["radio_group"] for mapping in mappings}) == 1


def test_vertical_layout():
    mappings, diagnostics = apply_radio_grouping(_group(["Yes", "No"], vertical=True), enabled=True)

    assert diagnostics["group_count"] == 1
    assert diagnostics["groups"][0]["layout"] == "vertical"


def test_horizontal_layout():
    _, diagnostics = apply_radio_grouping(_group(["Single", "Joint"]), enabled=True)

    assert diagnostics["groups"][0]["layout"] == "horizontal"


def test_section_ownership_blocks_cross_section_grouping():
    first = _checkbox("Yes", 0.20, 0.20, section=_section("section_a", "Applicant"))
    second = _checkbox("No", 0.32, 0.20, section=_section("section_b", "Nominee"))
    mappings, diagnostics = apply_radio_grouping([first, second], enabled=True)

    assert diagnostics["group_count"] == 0
    assert all(mapping.get("widget_type") is None for mapping in mappings)


def test_multipage_forms_do_not_cross_group():
    mappings, diagnostics = apply_radio_grouping(
        [
            *_group(["Yes", "No"], page=1, section=_section("section_p1", "Question", page=1)),
            *_group(["Yes", "No"], page=2, section=_section("section_p2", "Question", page=2)),
        ],
        enabled=True,
    )

    assert diagnostics["group_count"] == 2
    assert len({mapping["radio_group"] for mapping in mappings}) == 2


def test_false_positive_sentences_are_not_grouped():
    mappings, diagnostics = apply_radio_grouping(
        [
            _checkbox("I agree to terms", 0.20, 0.20),
            _checkbox("Send me updates", 0.32, 0.20),
        ],
        enabled=True,
    )

    assert diagnostics["group_count"] == 0
    assert all(mapping.get("widget_type") is None for mapping in mappings)


def test_generic_checkboxes_can_use_nearby_text_labels():
    mappings = [
        _checkbox("Checkbox 1", 0.20, 0.20),
        _checkbox("Checkbox 2", 0.32, 0.20),
    ]
    text_boxes = [
        {"page": 1, "text": "Yes", "bbox": {"x": 0.23, "y": 0.198, "width": 0.04, "height": 0.02}},
        {"page": 1, "text": "No", "bbox": {"x": 0.35, "y": 0.198, "width": 0.04, "height": 0.02}},
    ]

    enriched, diagnostics = apply_radio_grouping(mappings, text_boxes=text_boxes, enabled=True)

    assert diagnostics["group_count"] == 1
    assert {mapping["option_label"] for mapping in enriched} == {"Yes", "No"}


def test_multiple_selected_radios_are_normalized_to_one():
    mappings, diagnostics = apply_radio_grouping(_group(["Yes", "No"], selected_index=None), enabled=True)
    mappings[0]["value"] = "[X]"
    mappings[1]["value"] = "[X]"
    remapped, diagnostics = apply_radio_grouping(mappings, enabled=True)

    assert diagnostics["group_count"] == 1
    assert sum(1 for mapping in remapped if mapping["radio_selected"]) == 1


def test_native_radio_renderer(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")
    bg = tmp_path / "bg.png"
    cv2.imwrite(str(bg), np.full((400, 500, 3), 255, np.uint8))
    mappings, _ = apply_radio_grouping(_group(["Yes", "No"], selected_index=0), enabled=True)
    out_pdf = tmp_path / "radio.pdf"

    create_pdf_with_fields(str(bg), mappings, str(out_pdf))

    data = out_pdf.read_bytes()
    assert b"/FT /Btn" in data
    doc = fitz.open(str(out_pdf))
    try:
        widgets = list(doc.load_page(0).widgets() or [])
        assert len(widgets) == 2
        assert all(int(widget.field_flags) & RADIO_FLAG for widget in widgets)
        assert len({widget.field_name for widget in widgets}) == 1
    finally:
        doc.close()


def test_singleton_radio_render_falls_back_to_checkbox(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")
    bg = tmp_path / "bg.png"
    cv2.imwrite(str(bg), np.full((400, 500, 3), 255, np.uint8))
    mappings, _ = apply_radio_grouping(_group(["Attached", "Please"], selected_index=0), enabled=True, threshold=0.80)
    out_pdf = tmp_path / "singleton_radio.pdf"

    create_pdf_with_fields(str(bg), [mappings[0]], str(out_pdf))

    doc = fitz.open(str(out_pdf))
    try:
        widgets = list(doc.load_page(0).widgets() or [])
        assert len(widgets) == 1
        assert not (int(widgets[0].field_flags) & RADIO_FLAG)
    finally:
        doc.close()


def _bbox(left, top, w=0.02, h=0.02):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}


def _selection(block_id, left, top, selected):
    return {
        "Id": block_id,
        "BlockType": "SELECTION_ELEMENT",
        "SelectionStatus": "SELECTED" if selected else "NOT_SELECTED",
        "Confidence": 95.0,
        "Page": 1,
        "Geometry": _bbox(left, top),
    }


def _word(block_id, text, left, top, w=0.06):
    return {"Id": block_id, "BlockType": "WORD", "Text": text, "Confidence": 99.0, "Page": 1, "Geometry": _bbox(left, top, w)}


def _associated_checkbox_blocks(idx, left, top, selected, label):
    kid, vid, kw, sid = f"k{idx}", f"v{idx}", f"kw{idx}", f"s{idx}"
    return [
        {
            "Id": kid,
            "BlockType": "KEY_VALUE_SET",
            "EntityTypes": ["KEY"],
            "Confidence": 90.0,
            "Page": 1,
            "Geometry": _bbox(left, top, 0.08),
            "Relationships": [{"Type": "CHILD", "Ids": [kw]}, {"Type": "VALUE", "Ids": [vid]}],
        },
        _word(kw, label, left, top),
        {
            "Id": vid,
            "BlockType": "KEY_VALUE_SET",
            "EntityTypes": ["VALUE"],
            "Confidence": 90.0,
            "Page": 1,
            "Geometry": _bbox(left + 0.09, top),
            "Relationships": [{"Type": "CHILD", "Ids": [sid]}],
        },
        _selection(sid, left + 0.09, top, selected),
    ], [kid, vid, kw, sid]


def _yes_no_response() -> dict:
    blocks, ids = [], []
    for idx, (label, selected, left) in enumerate([("Yes", True, 0.10), ("No", False, 0.24)]):
        group, group_ids = _associated_checkbox_blocks(idx, left, 0.20, selected, label)
        blocks.extend(group)
        ids.extend(group_ids)
    return {
        "Blocks": [
            {
                "Id": "page",
                "BlockType": "PAGE",
                "Page": 1,
                "Geometry": _bbox(0, 0, 1, 1),
                "Relationships": [{"Type": "CHILD", "Ids": ids}],
            },
            *blocks,
        ],
        "DocumentMetadata": {"Pages": 1},
    }


def test_pipeline_writes_radio_debug_json(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_RADIO_GROUPING_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_yes_no_response()), encoding="utf-8")
    image = tmp_path / "page_1.png"
    cv2.imwrite(str(image), np.full((800, 600, 3), 255, np.uint8))
    out_dir = tmp_path / "run"

    result = run_textract_pipeline(
        str(raw),
        str(out_dir),
        reference_image_path=str(image),
        page_images=[(1, str(image))],
    )

    radios = [mapping for mapping in result["mappings"] if mapping.get("widget_type") == "radio"]
    assert len(radios) == 2
    debug = json.loads((out_dir / "radio_debug.json").read_text(encoding="utf-8"))
    assert debug["group_count"] == 1
    assert debug["groups"][0]["labels"] == ["Yes", "No"]
