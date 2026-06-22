from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src.confidence_pipeline import apply_confidence_pipeline, draw_confidence_overlay
from src.pipeline_config import PipelineConfig
from src.pipelines.textract_pipeline import run_textract_pipeline


def _mapping(
    label: str = "Name",
    *,
    score: float = 0.9,
    anchor_type: str = "value_block",
    page: int = 1,
    section_confidence: float = 0.9,
    widget_type: str | None = None,
) -> dict:
    bbox = {"x": 0.25, "y": 0.20 + (page - 1) * 0.02, "width": 0.25, "height": 0.035}
    mapping = {
        "field_id": f"field_{page}_{label.lower().replace(' ', '_')}",
        "label": label,
        "value": "",
        "field_type": "checkbox" if widget_type == "radio" else "text",
        "bbox": bbox,
        "label_bbox": {"x": 0.10, "y": bbox["y"], "width": 0.10, "height": 0.025},
        "answer_region": {"bbox": bbox, "type": anchor_type, "confidence": score},
        "page": page,
        "section": {
            "section_id": "section_profile",
            "title": "Profile",
            "page": page,
            "confidence": section_confidence,
        },
        "confidence": score,
        "candidate_score": score,
        "confidence_class": "high" if score >= 0.82 else "medium" if score >= 0.62 else "low",
        "field_bboxes": [{"x": 150, "y": 160, "width": 150, "height": 28}],
        "anchoring": {
            "anchor_type": anchor_type,
            "candidate_count": 1,
            "top_candidates": [{"score": score, "anchor_type": anchor_type}],
            "label_overlap_ratio": 0.0,
        },
    }
    if widget_type == "comb":
        mapping.update({"widget_type": "comb", "comb_confidence": 0.92, "comb_cells": 10})
    if widget_type == "radio":
        mapping.update(
            {
                "widget_type": "radio",
                "radio_confidence": 0.91,
                "radio_group": "profile_gender",
                "export_value": label,
            }
        )
    return mapping


def _bbox(left: float, top: float, w: float = 0.16, h: float = 0.03) -> dict:
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}


def _unresolved_label_response() -> dict:
    return {
        "Blocks": [
            {
                "Id": "page1",
                "BlockType": "PAGE",
                "Page": 1,
                "Geometry": _bbox(0, 0, 1, 1),
                "Relationships": [{"Type": "CHILD", "Ids": ["k", "kw"]}],
            },
            {
                "Id": "k",
                "BlockType": "KEY_VALUE_SET",
                "EntityTypes": ["KEY"],
                "Confidence": 82.0,
                "Page": 1,
                "Geometry": _bbox(0.10, 0.20, 0.11, 0.03),
                "Relationships": [{"Type": "CHILD", "Ids": ["kw"]}],
            },
            {
                "Id": "kw",
                "BlockType": "WORD",
                "Text": "Reference",
                "Confidence": 98.0,
                "Page": 1,
                "Geometry": _bbox(0.10, 0.20, 0.11, 0.03),
            },
        ],
        "DocumentMetadata": {"Pages": 1},
    }


def test_feature_flags_default_off(monkeypatch):
    monkeypatch.delenv("FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED", raising=False)
    monkeypatch.delenv("FORM_PARSER_REVIEW_QUEUE_ENABLED", raising=False)

    config = PipelineConfig.from_env()

    assert config.confidence_pipeline_enabled is False
    assert config.review_queue_enabled is False


def test_thresholds_are_configurable(monkeypatch):
    monkeypatch.setenv("FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_REVIEW_QUEUE_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_CONFIDENCE_HIGH_THRESHOLD", "0.9")
    monkeypatch.setenv("FORM_PARSER_CONFIDENCE_MEDIUM_THRESHOLD", "0.6")
    monkeypatch.setenv("FORM_PARSER_CONFIDENCE_LOW_THRESHOLD", "0.2")

    config = PipelineConfig.from_env()

    assert config.confidence_pipeline_enabled is True
    assert config.review_queue_enabled is True
    assert config.confidence_high_threshold == 0.9
    assert config.confidence_medium_threshold == 0.6
    assert config.confidence_low_threshold == 0.2


def test_disabled_pipeline_leaves_mappings_unchanged():
    mappings = [_mapping()]

    result = apply_confidence_pipeline(mappings, enabled=False, review_enabled=True)

    assert result["mappings"] == mappings
    assert result["render_mappings"] == mappings
    assert result["diagnostics"]["enabled"] is False


def test_high_confidence_renders_normally():
    result = apply_confidence_pipeline([_mapping(score=0.93)], enabled=True, review_enabled=True)
    mapping = result["mappings"][0]

    assert mapping["confidence_level"] == "HIGH"
    assert mapping["needs_review"] is False
    assert mapping["auto_render"] is True
    assert result["render_mappings"] == result["mappings"]


def test_medium_confidence_renders_and_marks_review():
    result = apply_confidence_pipeline(
        [_mapping(score=0.62, anchor_type="adjacent_estimate", section_confidence=0.74)],
        enabled=True,
        review_enabled=True,
        high_threshold=0.82,
        medium_threshold=0.50,
    )
    mapping = result["mappings"][0]

    assert mapping["confidence_level"] == "MEDIUM"
    assert mapping["needs_review"] is True
    assert mapping["auto_render"] is True
    assert result["review_artifacts"]["medium_count"] == 1
    assert result["render_mappings"] == result["mappings"]


def test_low_confidence_is_review_artifact_only():
    result = apply_confidence_pipeline(
        [_mapping(score=0.18, anchor_type="unresolved_label_region", section_confidence=0.72)],
        enabled=True,
        review_enabled=True,
    )
    mapping = result["mappings"][0]

    assert mapping["confidence_level"] == "LOW"
    assert mapping["needs_review"] is True
    assert mapping["auto_render"] is False
    assert result["render_mappings"] == []
    assert result["review_artifacts"]["low_count"] == 1
    assert result["diagnostics"]["render_skipped_count"] == 1


def test_ambiguous_mapping_gets_reviewed():
    ambiguous = _mapping(score=0.88)
    ambiguous["anchoring"]["candidate_count"] = 8
    ambiguous["anchoring"]["top_candidates"] = [{"score": 0.88}, {"score": 0.86}]

    result = apply_confidence_pipeline(
        [ambiguous],
        enabled=True,
        review_enabled=True,
        high_threshold=0.86,
        medium_threshold=0.55,
    )

    assert result["mappings"][0]["confidence_level"] == "MEDIUM"
    assert result["mappings"][0]["needs_review"] is True
    assert result["diagnostics"]["ambiguity_count"] >= 1


def test_multipage_render_policy_only_suppresses_low_page():
    high = _mapping("Applicant Name", page=1, score=0.91)
    low = _mapping("Nominee Name", page=2, score=0.16, anchor_type="unresolved_label_region")

    result = apply_confidence_pipeline([high, low], enabled=True, review_enabled=True)

    assert [mapping["page"] for mapping in result["mappings"]] == [1, 2]
    assert [mapping["page"] for mapping in result["render_mappings"]] == [1]
    assert result["confidence_report"]["fields_by_confidence"]["LOW"] == 1


def test_comb_and_radio_confidence_are_used():
    result = apply_confidence_pipeline(
        [
            _mapping("PAN", widget_type="comb", score=0.87),
            _mapping("Yes", widget_type="radio", score=0.86),
        ],
        enabled=True,
        review_enabled=True,
    )

    assert {mapping["widget_type"]: mapping["confidence_level"] for mapping in result["mappings"]} == {
        "comb": "HIGH",
        "radio": "HIGH",
    }
    assert result["confidence_report"]["field_count"] == 2


def test_confidence_overlay_is_written(tmp_path):
    image = tmp_path / "page_1.png"
    cv2.imwrite(str(image), np.full((500, 400, 3), 255, np.uint8))
    result = apply_confidence_pipeline(
        [
            _mapping("High", score=0.92),
            _mapping("Medium", score=0.62, anchor_type="adjacent_estimate"),
            _mapping("Low", score=0.18, anchor_type="unresolved_label_region"),
        ],
        enabled=True,
        review_enabled=True,
    )
    out = tmp_path / "confidence_overlay.png"

    assert draw_confidence_overlay(image, result["mappings"], out) is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_textract_pipeline_writes_review_artifacts_and_suppresses_low_render(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED", "1")
    monkeypatch.setenv("FORM_PARSER_REVIEW_QUEUE_ENABLED", "1")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_unresolved_label_response()), encoding="utf-8")
    image = tmp_path / "page_1.png"
    cv2.imwrite(str(image), np.full((800, 600, 3), 255, np.uint8))
    out_dir = tmp_path / "run"

    result = run_textract_pipeline(
        str(raw),
        str(out_dir),
        reference_image_path=str(image),
        page_images=[(1, str(image))],
    )

    assert len(result["mappings"]) == 1
    assert result["mappings"][0]["confidence_level"] == "LOW"
    assert result["mappings"][0]["auto_render"] is False
    review = json.loads((out_dir / "review_artifacts.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "confidence_report.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((out_dir / "mapping_diagnostics.json").read_text(encoding="utf-8"))
    assert review["field_count"] == 1
    assert review["render_skipped_count"] == 1
    assert report["fields_by_confidence"]["LOW"] == 1
    assert diagnostics["confidence"]["render_skipped_count"] == 1
    assert (out_dir / "confidence_overlay.png").exists()

    doc = fitz.open(result["pdf_output_path"])
    try:
        assert list(doc.load_page(0).widgets() or []) == []
    finally:
        doc.close()


def test_unresolved_fields_still_drop_when_phase_d_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED", raising=False)
    monkeypatch.delenv("FORM_PARSER_REVIEW_QUEUE_ENABLED", raising=False)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_unresolved_label_response()), encoding="utf-8")
    image = tmp_path / "page_1.png"
    cv2.imwrite(str(image), np.full((800, 600, 3), 255, np.uint8))

    result = run_textract_pipeline(
        str(raw),
        str(tmp_path / "run"),
        reference_image_path=str(image),
        page_images=[(1, str(image))],
    )

    assert result["mappings"] == []
