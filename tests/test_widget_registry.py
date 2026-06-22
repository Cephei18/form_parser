from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src.pdf_generator import create_pdf_with_fields, get_widget_renderer_registry
from src.pipeline_config import PipelineConfig
from src.widget_model import SUPPORTED_WIDGET_TYPES, build_widget_diagnostics


def _blank_png(path: Path) -> Path:
    cv2.imwrite(str(path), np.full((240, 320, 3), 255, np.uint8))
    return path


def _mapping(field_type: str = "text", widget_type: str | None = None) -> dict:
    payload = {
        "field_id": "field_1",
        "label": "Applicant Name",
        "field_type": field_type,
        "bbox": {"x": 0.20, "y": 0.20, "width": 0.30, "height": 0.04},
        "page": 1,
        "value": "",
    }
    if widget_type is not None:
        payload["widget_type"] = widget_type
    return payload


def _render_pdf_bytes(tmp_path: Path, mapping: dict) -> bytes:
    bg = _blank_png(tmp_path / "bg.png")
    out_pdf = tmp_path / "out.pdf"
    create_pdf_with_fields(str(bg), [mapping], str(out_pdf))
    return out_pdf.read_bytes()


def test_supported_widget_model_includes_foundational_types():
    assert SUPPORTED_WIDGET_TYPES == {
        "text",
        "multiline",
        "checkbox",
        "radio",
        "comb",
        "signature",
    }


def test_renderer_registry_registers_foundational_widget_types():
    assert set(get_widget_renderer_registry().widget_types) == SUPPORTED_WIDGET_TYPES


def test_widget_registry_config_default_off(monkeypatch):
    monkeypatch.delenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", raising=False)
    assert PipelineConfig.from_env().widget_registry_enabled is False


def test_widget_registry_config_can_be_enabled(monkeypatch):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "true")
    assert PipelineConfig.from_env().widget_registry_enabled is True


def test_widget_registry_default_off_ignores_widget_type(monkeypatch, tmp_path):
    monkeypatch.delenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", raising=False)

    data = _render_pdf_bytes(tmp_path, _mapping(field_type="text", widget_type="checkbox"))

    assert b"/FT /Tx" in data
    assert b"/FT /Btn" not in data


def test_widget_registry_enabled_dispatches_widget_type(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")

    data = _render_pdf_bytes(tmp_path, _mapping(field_type="text", widget_type="checkbox"))

    assert b"/FT /Btn" in data


def test_missing_widget_type_uses_legacy_renderer_when_flag_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")

    data = _render_pdf_bytes(tmp_path, _mapping(field_type="text"))

    assert b"/FT /Tx" in data
    assert b"/FT /Btn" not in data


def test_invalid_widget_type_falls_back_to_legacy_renderer(monkeypatch, tmp_path):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")

    data = _render_pdf_bytes(tmp_path, _mapping(field_type="text", widget_type="slider"))

    assert b"/FT /Tx" in data
    assert b"/FT /Btn" not in data


def test_widget_diagnostics_explain_registry_state(monkeypatch):
    monkeypatch.setenv("FORM_PARSER_WIDGET_REGISTRY_ENABLED", "1")

    diagnostics = build_widget_diagnostics(
        [
            _mapping(field_type="text"),
            _mapping(field_type="text", widget_type="comb"),
            _mapping(field_type="checkbox", widget_type="radio"),
            _mapping(field_type="text", widget_type="slider"),
        ]
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["explicit_widget_type_count"] == 3
    assert diagnostics["legacy_mapping_count"] == 1
    assert diagnostics["widget_type_counts"] == {"comb": 1, "radio": 1}
    assert diagnostics["unsupported_widget_type_count"] == 1
    assert diagnostics["unsupported_widget_types"] == ["slider"]
