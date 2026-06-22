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

from src.comb_detector import apply_comb_detection, expected_length_for_label
from src.pdf_generator import create_pdf_with_fields
from src.pipelines.textract_pipeline import run_textract_pipeline


A4_PORTRAIT = (595.28, 841.89)
COMB_FLAG = 1 << 24


def _cell_features(page: int, *, x: float = 0.32, y: float = 0.20, n: int = 10, w: float = 0.025, h: float = 0.030, gap: float = 0.004) -> dict:
    empty_boxes = []
    for index in range(n):
        box = {"x": x + index * (w + gap), "y": y, "width": w, "height": h}
        empty_boxes.append(
            {
                "bbox": box,
                "page": page,
                "type": "empty_rectangle",
                "confidence": 0.86,
                "text_count": 0,
                "aspect_ratio": round(w / h, 4),
                "rectangularity": 0.9,
            }
        )
    return {"underlines": [], "empty_boxes": empty_boxes, "photo_regions": []}


def _union_for(features: dict) -> dict:
    boxes = [item["bbox"] for item in features["empty_boxes"]]
    x1 = min(box["x"] for box in boxes)
    y1 = min(box["y"] for box in boxes)
    x2 = max(box["x"] + box["width"] for box in boxes)
    y2 = max(box["y"] + box["height"] for box in boxes)
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def _mapping(label: str, features: dict, *, page: int = 1, field_type: str = "text", bbox: dict | None = None) -> dict:
    answer_box = bbox or _union_for(features)
    return {
        "field_id": f"field_{page}_{label.lower().replace(' ', '_')}",
        "label": label,
        "field_type": field_type,
        "bbox": answer_box,
        "label_bbox": {"x": 0.10, "y": answer_box["y"], "width": 0.16, "height": 0.025},
        "answer_region": {"bbox": answer_box, "type": "empty_rectangle", "confidence": 0.86},
        "page": page,
        "value": "",
        "field_bboxes": [],
        "source": "test",
    }


def _detect(mapping: dict, features: dict, **kwargs):
    mappings, diagnostics = apply_comb_detection([mapping], features, enabled=True, threshold=0.84, **kwargs)
    return mappings[0], diagnostics


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("PAN", 10),
        ("Aadhaar Number", 12),
        ("DOB", 8),
        ("Date of Birth", 8),
        ("IFSC Code", 11),
    ],
)
def test_label_driven_lengths(label, expected):
    assert expected_length_for_label(label)["expected_length"] == expected


@pytest.mark.parametrize(
    ("label", "cells"),
    [
        ("PAN", 10),
        ("Aadhaar Number", 12),
        ("DOB", 8),
        ("IFSC Code", 11),
        ("Account Number", 16),
    ],
)
def test_comb_detection_for_enterprise_labels(label, cells):
    features = _cell_features(1, n=cells)
    mapping, diagnostics = _detect(_mapping(label, features), features)

    assert mapping["field_type"] == "text"
    assert mapping["widget_type"] == "comb"
    assert mapping["comb_cells"] == cells
    assert len(mapping["comb_boxes"]) == cells
    assert diagnostics["detected_count"] == 1


def test_geometry_driven_detects_strong_unlabeled_comb_when_threshold_allows():
    features = _cell_features(1, n=10)
    mapping, diagnostics = apply_comb_detection(
        [_mapping("Reference", features)],
        features,
        enabled=True,
        threshold=0.80,
    )

    assert mapping[0]["widget_type"] == "comb"
    assert diagnostics["detected_count"] == 1


def test_geometry_only_stays_below_default_threshold_for_generic_label():
    features = _cell_features(1, n=10)
    mapping, diagnostics = _detect(_mapping("Reference", features), features)

    assert mapping.get("widget_type") is None
    assert diagnostics["detected_count"] == 0
    assert diagnostics["rejected_count"] == 1


def test_short_box_rows_are_not_comb_fields():
    features = _cell_features(1, n=4)
    mapping, diagnostics = _detect(_mapping("Reference", features), features)

    assert mapping.get("widget_type") is None
    assert diagnostics["detected_count"] == 0


def test_table_cells_are_rejected():
    features = _cell_features(1, n=10)
    table_cells = [{"page": 1, "bbox": box["bbox"]} for box in features["empty_boxes"]]
    mapping, diagnostics = _detect(_mapping("PAN", features), features, table_cells=table_cells)

    assert mapping.get("widget_type") is None
    assert diagnostics["detected_count"] == 0


def test_checkbox_groups_are_rejected():
    features = _cell_features(1, n=5, w=0.020, h=0.020)
    selection_regions = [{"page": 1, "bbox": box["bbox"]} for box in features["empty_boxes"]]
    mapping, diagnostics = _detect(_mapping("Options", features), features, selection_regions=selection_regions)

    assert mapping.get("widget_type") is None
    assert diagnostics["detected_count"] == 0


def test_multipage_comb_detection_uses_page_local_features():
    page1 = _cell_features(1, n=10, y=0.20)
    page2 = _cell_features(2, n=12, y=0.30)
    features = {"empty_boxes": page1["empty_boxes"] + page2["empty_boxes"], "underlines": [], "photo_regions": []}
    mappings, diagnostics = apply_comb_detection(
        [
            _mapping("Name", page1, page=1),
            _mapping("Aadhaar Number", page2, page=2),
        ],
        features,
        enabled=True,
        threshold=0.84,
    )

    assert mappings[0].get("widget_type") is None
    assert mappings[1]["widget_type"] == "comb"
    assert mappings[1]["comb_cells"] == 12
    assert diagnostics["detected_fields"][0]["page"] == 2


def test_comb_renderer_emits_native_acroform_comb(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")
    bg = tmp_path / "bg.png"
    cv2.imwrite(str(bg), np.full((800, 600, 3), 255, np.uint8))
    features = _cell_features(1, n=10)
    mapping = _mapping("PAN", features)
    mapping["widget_type"] = "comb"
    mapping["comb_cells"] = 10
    mapping["comb_boxes"] = [box["bbox"] for box in features["empty_boxes"]]
    out_pdf = tmp_path / "comb.pdf"

    create_pdf_with_fields(str(bg), [mapping], str(out_pdf), page_sizes={1: A4_PORTRAIT})

    data = out_pdf.read_bytes()
    assert b"/MaxLen 10" in data
    assert b"/Ff 16777216" in data
    doc = fitz.open(str(out_pdf))
    try:
        assert doc.load_page(0).rect.width == pytest.approx(A4_PORTRAIT[0], abs=4.0)
        widget = list(doc.load_page(0).widgets() or [])[0]
        assert int(widget.field_flags) & COMB_FLAG
    finally:
        doc.close()


def _bbox(left, top, w=0.20, h=0.03):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}


def _single_pan_response() -> dict:
    return {
        "Blocks": [
            {
                "Id": "page1",
                "BlockType": "PAGE",
                "Page": 1,
                "Geometry": _bbox(0, 0, 1, 1),
                "Relationships": [{"Type": "CHILD", "Ids": ["k", "kw", "v"]}],
            },
            {
                "Id": "k",
                "BlockType": "KEY_VALUE_SET",
                "EntityTypes": ["KEY"],
                "Confidence": 95.0,
                "Page": 1,
                "Geometry": _bbox(0.10, 0.20, 0.12, 0.03),
                "Relationships": [{"Type": "CHILD", "Ids": ["kw"]}, {"Type": "VALUE", "Ids": ["v"]}],
            },
            {"Id": "kw", "BlockType": "WORD", "Text": "PAN", "Confidence": 99.0, "Page": 1, "Geometry": _bbox(0.10, 0.20, 0.08, 0.03)},
            {
                "Id": "v",
                "BlockType": "KEY_VALUE_SET",
                "EntityTypes": ["VALUE"],
                "Confidence": 95.0,
                "Page": 1,
                "Geometry": _bbox(0.32, 0.20, 0.025, 0.03),
            },
        ],
        "DocumentMetadata": {"Pages": 1},
    }


def _draw_comb_image(path: Path, cells: int = 10) -> Path:
    image = np.full((800, 600, 3), 255, np.uint8)
    x, y, w, h, gap = 192, 160, 15, 24, 3
    for index in range(cells):
        left = x + index * (w + gap)
        cv2.rectangle(image, (left, y), (left + w, y + h), (0, 0, 0), 1)
    cv2.imwrite(str(path), image)
    return path


def test_pipeline_writes_comb_debug_json(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_COMB_DETECTION_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_single_pan_response()), encoding="utf-8")
    image = _draw_comb_image(tmp_path / "page_1.png")
    out_dir = tmp_path / "run"

    result = run_textract_pipeline(
        str(raw),
        str(out_dir),
        reference_image_path=str(image),
        page_images=[(1, str(image))],
    )

    mappings = result["mappings"]
    assert mappings[0]["widget_type"] == "comb"
    debug = json.loads((out_dir / "comb_debug.json").read_text(encoding="utf-8"))
    assert debug["detected_count"] == 1
    assert debug["detected_fields"][0]["comb_cells"] == 10
