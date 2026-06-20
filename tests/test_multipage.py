"""Validation suite for production-grade multi-page document support.

No AWS, no Poppler: synthetic multi-page Textract responses drive the page-aware
parser/anchor/render path, and blank PNGs stand in for rasterised page
backgrounds. PyMuPDF (already a dependency) is used to assert the rendered PDF's
page count and per-page widget placement.

Covered (1 / 2 / 5 / 10 pages):
  * page count            -> rendered PDF page_count == source page count
  * extraction count      -> every field on every page is mapped
  * page-local coordinates-> page-N widgets land ONLY on page N (no leakage)
  * rendering correctness  -> each page carries exactly its own widgets
Plus: per-page-response merge, single-page regression, observability.
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

from src import field_anchor_engine as fae
from src.document_render import ensure_page_images
from src.pdf_generator import create_pdf_with_fields
from src.pipelines.textract_pipeline import _merge_page_responses, run_textract_pipeline
from src.textract_parser import parse_textract_response


# --------------------------------------------------------------------------- #
# Synthetic Textract fixtures
# --------------------------------------------------------------------------- #
def _bbox(left, top, w=0.2, h=0.02):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}


def _field_blocks(page: int, idx: int, top: float):
    """One KEY_VALUE_SET field (label + value) fully on ``page``."""
    kid, kw = f"k{page}_{idx}", f"kw{page}_{idx}"
    vid, vw = f"v{page}_{idx}", f"vw{page}_{idx}"
    blocks = [
        {"Id": kid, "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"], "Confidence": 95.0, "Page": page,
         "Geometry": _bbox(0.10, top, 0.15),
         "Relationships": [{"Type": "CHILD", "Ids": [kw]}, {"Type": "VALUE", "Ids": [vid]}]},
        {"Id": kw, "BlockType": "WORD", "Text": f"Field{idx}", "Confidence": 99.0, "Page": page,
         "Geometry": _bbox(0.10, top, 0.12)},
        {"Id": vid, "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"], "Confidence": 95.0, "Page": page,
         "Geometry": _bbox(0.32, top, 0.20),
         "Relationships": [{"Type": "CHILD", "Ids": [vw]}]},
        {"Id": vw, "BlockType": "WORD", "Text": f"val{page}_{idx}", "Confidence": 99.0, "Page": page,
         "Geometry": _bbox(0.32, top, 0.18)},
    ]
    return blocks, [kid, kw, vid, vw]


def _multipage_response(n_pages: int, fields_per_page: int = 3) -> dict:
    blocks: list[dict] = []
    for page in range(1, n_pages + 1):
        child_ids: list[str] = []
        page_blocks: list[dict] = []
        for idx in range(fields_per_page):
            fb, ids = _field_blocks(page, idx, top=0.20 + idx * 0.15)
            page_blocks += fb
            child_ids += ids
        page_block = {
            "Id": f"page{page}", "BlockType": "PAGE", "Page": page, "Geometry": _bbox(0, 0, 1, 1),
            "Relationships": [{"Type": "CHILD", "Ids": child_ids}],
        }
        blocks += [page_block] + page_blocks
    return {"Blocks": blocks, "DocumentMetadata": {"Pages": n_pages}}


def _make_page_images(tmp_path: Path, n_pages: int) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for page in range(1, n_pages + 1):
        path = tmp_path / f"page_{page}.png"
        cv2.imwrite(str(path), np.full((800, 600, 3), 255, np.uint8))
        images[page] = path
    return images


FIELDS_PER_PAGE = 3
PAGE_COUNTS = [1, 2, 5, 10]


# --------------------------------------------------------------------------- #
# Extraction: per-page response merge
# --------------------------------------------------------------------------- #
def test_merge_page_responses_stamps_pages_and_counts():
    singles = [
        (1, {"Blocks": [{"Id": "a", "BlockType": "WORD"}], "AnalyzeDocumentModelVersion": "1.0"}),
        (2, {"Blocks": [{"Id": "b", "BlockType": "WORD"}]}),
        (3, {"Blocks": [{"Id": "c", "BlockType": "WORD"}]}),
    ]
    merged = _merge_page_responses(singles)
    assert merged["DocumentMetadata"]["Pages"] == 3
    assert merged["JobStatus"] == "SUCCEEDED"
    assert merged["AnalyzeDocumentModelVersion"] == "1.0"
    assert {b["Id"]: b["Page"] for b in merged["Blocks"]} == {"a": 1, "b": 2, "c": 3}


def test_ensure_page_images_passthrough_for_image(tmp_path):
    img = tmp_path / "scan.png"
    cv2.imwrite(str(img), np.full((50, 50, 3), 255, np.uint8))
    assert ensure_page_images(img, tmp_path) == [(1, img)]


# --------------------------------------------------------------------------- #
# Mapping: page-aware extraction + page-local coordinates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_pages", PAGE_COUNTS)
def test_mapping_is_page_aware(tmp_path, n_pages):
    response = _multipage_response(n_pages, FIELDS_PER_PAGE)
    parsed = parse_textract_response(response)
    page_images = _make_page_images(tmp_path, n_pages)

    out = fae.build_anchored_mappings(response, parsed, page_images[1], page_images=page_images)
    mappings = [m for m in out["mappings"] if m.get("field_type") != "photo"]

    # extraction count: every field on every page is mapped
    assert len(mappings) == n_pages * FIELDS_PER_PAGE

    # fields are spread across exactly the expected pages
    assert {m["page"] for m in mappings} == set(range(1, n_pages + 1))

    # page-local coordinates: every bbox is a valid normalised on-page region
    for m in mappings:
        b = m["bbox"]
        assert 0.0 <= b["x"] <= 1.0 and 0.0 <= b["y"] <= 1.0
        assert 0.0 < b["width"] <= 1.0 and 0.0 < b["height"] <= 1.0

    # observability surfaced by the engine
    diag = out["diagnostics"]
    assert diag["page_count"] == n_pages
    assert diag["analyzed_pages"] == list(range(1, n_pages + 1))
    assert diag["fields_per_page"] == {str(p): FIELDS_PER_PAGE for p in range(1, n_pages + 1)}


def test_single_page_mapping_unchanged_without_page_images(tmp_path):
    """Regression: omitting page_images keeps the legacy single-page behaviour."""
    response = _multipage_response(1, FIELDS_PER_PAGE)
    parsed = parse_textract_response(response)
    img = _make_page_images(tmp_path, 1)[1]

    out = fae.build_anchored_mappings(response, parsed, img)  # no page_images kwarg
    mappings = [m for m in out["mappings"] if m.get("field_type") != "photo"]
    assert len(mappings) == FIELDS_PER_PAGE
    assert out["diagnostics"]["page_count"] == 1
    assert all(m["page"] == 1 for m in mappings)


# --------------------------------------------------------------------------- #
# Rendering: one background per page, correct page count, no widget leakage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_pages", PAGE_COUNTS)
def test_render_page_count_and_locality(tmp_path, n_pages):
    response = _multipage_response(n_pages, FIELDS_PER_PAGE)
    parsed = parse_textract_response(response)
    page_images = _make_page_images(tmp_path, n_pages)
    mappings = fae.build_anchored_mappings(response, parsed, page_images[1], page_images=page_images)["mappings"]

    out_pdf = tmp_path / "output.pdf"
    create_pdf_with_fields(page_images[1], mappings, out_pdf, page_images=page_images)

    doc = fitz.open(str(out_pdf))
    try:
        # page count == source page count
        assert doc.page_count == n_pages
        # page-local widget placement: page N carries exactly its own fields,
        # proving no page-2+ widget bleeds onto an earlier page's background.
        per_page = [len(list(doc.load_page(p).widgets() or [])) for p in range(doc.page_count)]
        assert per_page == [FIELDS_PER_PAGE] * n_pages
    finally:
        doc.close()


def test_single_page_render_unchanged_without_page_images(tmp_path):
    response = _multipage_response(1, FIELDS_PER_PAGE)
    parsed = parse_textract_response(response)
    img = _make_page_images(tmp_path, 1)[1]
    mappings = fae.build_anchored_mappings(response, parsed, img)["mappings"]

    out_pdf = tmp_path / "single.pdf"
    create_pdf_with_fields(img, mappings, out_pdf)  # no page_images kwarg

    doc = fitz.open(str(out_pdf))
    try:
        assert doc.page_count == 1
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# End-to-end pipeline (JSON replay, no AWS): extraction -> mapping -> render
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_pages", PAGE_COUNTS)
def test_pipeline_end_to_end_multipage(tmp_path, n_pages):
    response = _multipage_response(n_pages, FIELDS_PER_PAGE)
    page_images = _make_page_images(tmp_path, n_pages)

    # `.json` source => pipeline replays this response instead of calling Textract.
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(response), encoding="utf-8")

    out_dir = tmp_path / "run"
    result = run_textract_pipeline(
        str(raw_path),
        str(out_dir),
        reference_image_path=str(page_images[1]),
        page_images=[(p, str(path)) for p, path in page_images.items()],
    )

    obs = result["page_observability"]
    assert obs["multipage_mode"] is True
    assert obs["pages_detected"] == n_pages
    assert obs["pages_rendered"] == list(range(1, n_pages + 1))
    assert obs["pages_analyzed"] == list(range(1, n_pages + 1))
    assert obs["missing_page_warnings"] == []
    assert obs["fields_per_page"] == {str(p): FIELDS_PER_PAGE for p in range(1, n_pages + 1)}

    # rendered PDF really has all pages
    doc = fitz.open(str(out_dir / "output.pdf"))
    try:
        assert doc.page_count == n_pages
    finally:
        doc.close()

    # observability persisted to the diagnostics artifact
    diag = json.loads((out_dir / "mapping_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["page_observability"]["pages_rendered"] == list(range(1, n_pages + 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
