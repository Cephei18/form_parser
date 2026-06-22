"""Validation suite for Issue 1 — page-geometry preservation (Phase 1).

No AWS, no Poppler: synthetic mappings + blank PNG backgrounds drive the
renderer, and PyMuPDF (already a dependency) reads back the output PDF's per-page
size and widget placement.

Covered:
  * point-size derivation from raster pixels + DPI (incl. embedded DPI)
  * the FORM_PARSER_PRESERVE_PAGE_SIZE flag (default OFF)
  * renderer sizes each page to A4 / Legal / landscape when page_sizes given
  * mixed-size documents (portrait page 1, landscape page 2)
  * widget alignment: fraction bbox maps to the *page's* width, not Letter
  * regression: omitting page_sizes keeps byte-compatible US-Letter output
  * end-to-end pipeline: flag ON sizes pages + surfaces page_geometry diagnostics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging
logging.disable(logging.CRITICAL)

import fitz  # PyMuPDF, already in requirements
from PIL import Image

from src import page_geometry as pg
from src.document_render import DEFAULT_DPI
from src.pdf_generator import create_pdf_with_fields
from src.pipelines.textract_pipeline import run_textract_pipeline

# Common page sizes in points.
LETTER = (612.0, 792.0)
A4_PORTRAIT = (595.28, 841.89)
A4_LANDSCAPE = (841.89, 595.28)
LEGAL = (612.0, 1008.0)

_TOL = 4.0  # points; tight enough to distinguish A4 (595) from Letter (612)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _blank_png(path: Path, width_px: int = 600, height_px: int = 800, dpi: int | None = None) -> Path:
    cv2.imwrite(str(path), np.full((height_px, width_px, 3), 255, np.uint8))
    if dpi is not None:
        # Re-save with embedded DPI metadata so the embedded-DPI path is exercised.
        with Image.open(str(path)) as im:
            im.save(str(path), dpi=(dpi, dpi))
    return path


def _text_mapping(page: int, x: float, y: float, w: float, h: float, label: str = "F") -> dict:
    bbox = {"x": x, "y": y, "width": w, "height": h}
    return {
        "field_id": f"f_{page}_{label}",
        "label": label,
        "field_type": "text",
        "bbox": bbox,
        "page": page,
        "answer_region": {"bbox": bbox, "type": "value_block", "confidence": 0.9},
    }


def _page_rect(doc, page_index: int) -> fitz.Rect:
    return doc.load_page(page_index).rect


# --------------------------------------------------------------------------- #
# Unit: point-size derivation
# --------------------------------------------------------------------------- #
def test_image_point_size_uses_default_dpi_when_no_metadata(tmp_path):
    # 600x800 px at the rasteriser's DEFAULT_DPI -> exact points.
    img = _blank_png(tmp_path / "p.png", 600, 800)
    size = pg.image_point_size(img)
    assert size is not None
    w, h = size
    assert w == pytest.approx(600 / DEFAULT_DPI * 72.0, abs=0.01)
    assert h == pytest.approx(800 / DEFAULT_DPI * 72.0, abs=0.01)
    assert pg.orientation(size) == "portrait"


def test_image_point_size_honours_embedded_dpi(tmp_path):
    img = _blank_png(tmp_path / "p300.png", 2480, 3508, dpi=300)  # A4 @ 300 DPI
    size = pg.image_point_size(img)
    assert size is not None
    assert size[0] == pytest.approx(A4_PORTRAIT[0], abs=1.0)
    assert size[1] == pytest.approx(A4_PORTRAIT[1], abs=1.0)


def test_image_point_size_dpi_override(tmp_path):
    img = _blank_png(tmp_path / "p.png", 600, 800, dpi=300)
    # Explicit override beats embedded metadata.
    size = pg.image_point_size(img, dpi=DEFAULT_DPI)
    assert size[0] == pytest.approx(600 / DEFAULT_DPI * 72.0, abs=0.01)


def test_image_point_size_returns_none_for_bad_path(tmp_path):
    assert pg.image_point_size(tmp_path / "missing.png") is None


def test_page_sizes_from_images_skips_underivable(tmp_path):
    good = _blank_png(tmp_path / "g.png", 600, 800)
    sizes = pg.page_sizes_from_images({1: good, 2: tmp_path / "missing.png"})
    assert set(sizes) == {1}  # page 2 omitted -> renderer falls back to Letter


def test_orientation():
    assert pg.orientation((800, 600)) == "landscape"
    assert pg.orientation((600, 800)) == "portrait"
    assert pg.orientation((500, 500)) == "square"
    assert pg.orientation(None) == "unknown"


# --------------------------------------------------------------------------- #
# Unit: the flag (default OFF)
# --------------------------------------------------------------------------- #
def test_preserve_flag_default_off(monkeypatch):
    monkeypatch.delenv("FORM_PARSER_PRESERVE_PAGE_SIZE", raising=False)
    assert pg.preserve_page_size_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_preserve_flag_on_values(monkeypatch, value):
    monkeypatch.setenv("FORM_PARSER_PRESERVE_PAGE_SIZE", value)
    assert pg.preserve_page_size_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_preserve_flag_off_values(monkeypatch, value):
    monkeypatch.setenv("FORM_PARSER_PRESERVE_PAGE_SIZE", value)
    assert pg.preserve_page_size_enabled() is False


# --------------------------------------------------------------------------- #
# Renderer: per-page sizing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [A4_PORTRAIT, LEGAL, A4_LANDSCAPE])
def test_renderer_sizes_page_to_target(tmp_path, target):
    bg = _blank_png(tmp_path / "bg.png", 600, 800)
    mappings = [_text_mapping(1, 0.2, 0.2, 0.3, 0.03)]
    out_pdf = tmp_path / "out.pdf"

    create_pdf_with_fields(str(bg), mappings, str(out_pdf), page_sizes={1: target})

    doc = fitz.open(str(out_pdf))
    try:
        rect = _page_rect(doc, 0)
        assert rect.width == pytest.approx(target[0], abs=_TOL)
        assert rect.height == pytest.approx(target[1], abs=_TOL)
    finally:
        doc.close()


def test_renderer_defaults_to_letter_without_page_sizes(tmp_path):
    bg = _blank_png(tmp_path / "bg.png", 600, 800)
    mappings = [_text_mapping(1, 0.2, 0.2, 0.3, 0.03)]
    out_pdf = tmp_path / "out.pdf"

    create_pdf_with_fields(str(bg), mappings, str(out_pdf))  # no page_sizes

    doc = fitz.open(str(out_pdf))
    try:
        rect = _page_rect(doc, 0)
        assert rect.width == pytest.approx(LETTER[0], abs=_TOL)
        assert rect.height == pytest.approx(LETTER[1], abs=_TOL)
    finally:
        doc.close()


def test_mixed_size_document(tmp_path):
    bg1 = _blank_png(tmp_path / "p1.png", 600, 800)
    bg2 = _blank_png(tmp_path / "p2.png", 800, 600)
    page_images = {1: bg1, 2: bg2}
    page_sizes = {1: A4_PORTRAIT, 2: A4_LANDSCAPE}
    mappings = [_text_mapping(1, 0.2, 0.2, 0.3, 0.03), _text_mapping(2, 0.2, 0.2, 0.3, 0.03)]
    out_pdf = tmp_path / "out.pdf"

    create_pdf_with_fields(str(bg1), mappings, str(out_pdf), page_images=page_images, page_sizes=page_sizes)

    doc = fitz.open(str(out_pdf))
    try:
        assert doc.page_count == 2
        p1, p2 = _page_rect(doc, 0), _page_rect(doc, 1)
        assert (p1.width, p1.height) == pytest.approx(A4_PORTRAIT, abs=_TOL)
        assert (p2.width, p2.height) == pytest.approx(A4_LANDSCAPE, abs=_TOL)
        assert p1.width < p1.height and p2.width > p2.height  # orientation preserved
    finally:
        doc.close()


def test_page_absent_from_sizes_falls_back_to_letter(tmp_path):
    bg1 = _blank_png(tmp_path / "p1.png", 600, 800)
    bg2 = _blank_png(tmp_path / "p2.png", 600, 800)
    page_images = {1: bg1, 2: bg2}
    page_sizes = {1: A4_PORTRAIT}  # page 2 intentionally missing
    mappings = [_text_mapping(1, 0.2, 0.2, 0.3, 0.03), _text_mapping(2, 0.2, 0.2, 0.3, 0.03)]
    out_pdf = tmp_path / "out.pdf"

    create_pdf_with_fields(str(bg1), mappings, str(out_pdf), page_images=page_images, page_sizes=page_sizes)

    doc = fitz.open(str(out_pdf))
    try:
        assert (_page_rect(doc, 0).width, _page_rect(doc, 0).height) == pytest.approx(A4_PORTRAIT, abs=_TOL)
        assert (_page_rect(doc, 1).width, _page_rect(doc, 1).height) == pytest.approx(LETTER, abs=_TOL)
    finally:
        doc.close()


def test_widget_alignment_scales_with_page_width(tmp_path):
    """A fraction bbox at x=0.5 must land at ~0.5*page_width — proving widgets
    track the actual (A4) page width, not the Letter default."""
    bg = _blank_png(tmp_path / "bg.png", 600, 800)
    mappings = [_text_mapping(1, 0.5, 0.5, 0.2, 0.03)]
    out_pdf = tmp_path / "out.pdf"

    create_pdf_with_fields(str(bg), mappings, str(out_pdf), page_sizes={1: A4_PORTRAIT})

    doc = fitz.open(str(out_pdf))
    try:
        widgets = list(doc.load_page(0).widgets() or [])
        assert len(widgets) == 1
        x0 = widgets[0].rect.x0
        a4_expected = 0.5 * A4_PORTRAIT[0]      # ~297.6
        letter_expected = 0.5 * LETTER[0]       # 306.0
        assert abs(x0 - a4_expected) < abs(x0 - letter_expected)
        assert x0 == pytest.approx(a4_expected, abs=3.0)  # +small x-pad
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# End-to-end pipeline: flag ON sizes pages + emits diagnostics
# --------------------------------------------------------------------------- #
def _single_field_response() -> dict:
    def _bbox(left, top, w, h):
        return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}
    return {
        "Blocks": [
            {"Id": "page1", "BlockType": "PAGE", "Page": 1, "Geometry": _bbox(0, 0, 1, 1),
             "Relationships": [{"Type": "CHILD", "Ids": ["k", "kw", "v", "vw"]}]},
            {"Id": "k", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"], "Confidence": 95.0, "Page": 1,
             "Geometry": _bbox(0.10, 0.20, 0.15, 0.02),
             "Relationships": [{"Type": "CHILD", "Ids": ["kw"]}, {"Type": "VALUE", "Ids": ["v"]}]},
            {"Id": "kw", "BlockType": "WORD", "Text": "Name", "Confidence": 99.0, "Page": 1,
             "Geometry": _bbox(0.10, 0.20, 0.12, 0.02)},
            {"Id": "v", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"], "Confidence": 95.0, "Page": 1,
             "Geometry": _bbox(0.32, 0.20, 0.20, 0.02),
             "Relationships": [{"Type": "CHILD", "Ids": ["vw"]}]},
            {"Id": "vw", "BlockType": "WORD", "Text": "x", "Confidence": 99.0, "Page": 1,
             "Geometry": _bbox(0.32, 0.20, 0.18, 0.02)},
        ],
        "DocumentMetadata": {"Pages": 1},
    }


def test_pipeline_preserves_page_size_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FORM_PARSER_PRESERVE_PAGE_SIZE", "1")
    # 800x600 px @ DEFAULT_DPI -> 288x216 pt (landscape).
    bg = _blank_png(tmp_path / "page_1.png", 800, 600)
    expected = (800 / DEFAULT_DPI * 72.0, 600 / DEFAULT_DPI * 72.0)

    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_single_field_response()), encoding="utf-8")
    out_dir = tmp_path / "run"

    result = run_textract_pipeline(
        str(raw), str(out_dir),
        reference_image_path=str(bg),
        page_images=[(1, str(bg))],
    )

    geom = result["page_observability"]["page_geometry"]
    assert geom["preserve_page_size"] is True
    assert geom["page_sizes_pt"]["1"] == pytest.approx(list(expected), abs=0.1)
    assert geom["orientations"]["1"] == "landscape"

    doc = fitz.open(str(out_dir / "output.pdf"))
    try:
        rect = _page_rect(doc, 0)
        assert (rect.width, rect.height) == pytest.approx(expected, abs=_TOL)
    finally:
        doc.close()

    # Diagnostics artifact carries it too.
    diag = json.loads((out_dir / "mapping_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["page_observability"]["page_geometry"]["preserve_page_size"] is True


def test_pipeline_letter_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("FORM_PARSER_PRESERVE_PAGE_SIZE", raising=False)
    bg = _blank_png(tmp_path / "page_1.png", 800, 600)

    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(_single_field_response()), encoding="utf-8")
    out_dir = tmp_path / "run"

    result = run_textract_pipeline(
        str(raw), str(out_dir),
        reference_image_path=str(bg),
        page_images=[(1, str(bg))],
    )

    geom = result["page_observability"]["page_geometry"]
    assert geom["preserve_page_size"] is False
    assert geom["page_sizes_pt"] == {}

    doc = fitz.open(str(out_dir / "output.pdf"))
    try:
        rect = _page_rect(doc, 0)
        assert (rect.width, rect.height) == pytest.approx(LETTER, abs=_TOL)
    finally:
        doc.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
