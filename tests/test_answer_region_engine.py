"""Phase I — answer region engine validation.

Engine-level tests drive ``build_answer_regions`` with synthetic feature dicts
(no rasters, no AWS) for every required scenario: dotted leaders, broken
underlines, multiline stacks, grouped lines, table cells, signatures, freeform
whitespace, multipage scoping and diagnostics shape. Integration tests confirm
the ``FORM_PARSER_ANSWER_REGION_V2_ENABLED`` flag is fully reversible and emit a
validation artifact.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src import field_anchor_engine as fae
from src.answer_region_engine import (
    BROKEN,
    DOTTED,
    FREEFORM,
    MULTILINE,
    SIGNATURE,
    TABLE_CELL,
    UNDERLINE,
    build_answer_regions,
)

FLAG = "FORM_PARSER_ANSWER_REGION_V2_ENABLED"
METRICS = {1: {"line_height": 0.02, "text_width": 0.12, "page_left": 0.05, "page_right": 0.95},
           2: {"line_height": 0.02, "text_width": 0.12, "page_left": 0.05, "page_right": 0.95}}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _feat(x, y, w, h, page=1, **extra):
    return {"bbox": {"x": x, "y": y, "width": w, "height": h}, "page": page, **extra}


def _types(result):
    return {r.region_type for r in result.regions}


def _by_type(result, rtype):
    return [r for r in result.regions if r.region_type == rtype]


# --------------------------------------------------------------------------- #
# Engine-level scenarios
# --------------------------------------------------------------------------- #
def test_dotted_leaders():
    vf = {"synthetic_underlines": [_feat(0.3, 0.4, 0.4, 0.01, source_type="dotted", confidence=0.8)]}
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    dotted = _by_type(result, DOTTED)
    assert len(dotted) == 1
    assert dotted[0].confidence == 0.8


def test_broken_underline_from_synthetic():
    vf = {"synthetic_underlines": [_feat(0.3, 0.4, 0.4, 0.01, source_type="broken", confidence=0.7)]}
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    assert len(_by_type(result, BROKEN)) == 1


def test_broken_underline_merges_fragments():
    """Three solid fragments on one y band with small gaps -> one BROKEN region."""
    vf = {"underlines": [
        _feat(0.30, 0.40, 0.10, 0.004),
        _feat(0.41, 0.40, 0.08, 0.004),
        _feat(0.50, 0.40, 0.10, 0.004),
    ]}
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    broken = _by_type(result, BROKEN)
    assert len(broken) == 1
    assert broken[0].metadata["segment_count"] == 3
    # The three fragments are recorded as merged, not emitted as 3 underlines.
    assert len(result.merged_regions) == 3
    assert _by_type(result, UNDERLINE) == []


def test_multiline_stacked_lines():
    """Three equal-width lines stacked with even spacing -> one MULTILINE."""
    vf = {"underlines": [
        _feat(0.20, 0.40, 0.50, 0.004),
        _feat(0.20, 0.44, 0.50, 0.004),
        _feat(0.20, 0.48, 0.50, 0.004),
    ]}
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    ml = _by_type(result, MULTILINE)
    assert len(ml) == 1
    assert ml[0].metadata["line_count"] == 3
    # Combined bbox spans all three lines.
    assert ml[0].bbox[3] > 0.07
    assert _by_type(result, UNDERLINE) == []  # not three separate fields


def test_grouped_lines_single_region():
    """Grouped answer lines under a label collapse to one multiline region."""
    vf = {"underlines": [
        _feat(0.25, 0.30, 0.40, 0.004),
        _feat(0.25, 0.34, 0.40, 0.004),
    ]}
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    assert len(_by_type(result, MULTILINE)) == 1


def test_single_underline_not_multiline():
    vf = {"underlines": [_feat(0.20, 0.40, 0.50, 0.004)]}
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    assert _by_type(result, MULTILINE) == []
    assert len(_by_type(result, UNDERLINE)) == 1


def test_table_cells():
    cells = [
        {"bbox": {"x": 0.1, "y": 0.5, "width": 0.2, "height": 0.04}, "page": 1, "text": "School",
         "table_id": "t1", "row_index": 0, "column_index": 0},
        {"bbox": {"x": 0.3, "y": 0.5, "width": 0.2, "height": 0.04}, "page": 1, "text": "",
         "table_id": "t1", "row_index": 0, "column_index": 1},
        {"bbox": {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.04}, "page": 1, "text": "",
         "table_id": "t1", "row_index": 0, "column_index": 2},
    ]
    result = build_answer_regions({}, table_cells=cells, metrics_by_page=METRICS)
    tc = _by_type(result, TABLE_CELL)
    assert len(tc) == 2  # only the two empty cells
    assert all(r.metadata["table_id"] == "t1" for r in tc)


def test_signature_region_binds_to_line():
    text_boxes = [{"bbox": {"x": 0.1, "y": 0.80, "width": 0.12, "height": 0.02}, "text": "Signature", "page": 1}]
    vf = {"underlines": [_feat(0.30, 0.805, 0.40, 0.004)]}
    result = build_answer_regions(vf, text_boxes=text_boxes, metrics_by_page=METRICS)
    sig = _by_type(result, SIGNATURE)
    assert len(sig) == 1
    # The line was reclassified as the signature region, not left as UNDERLINE.
    assert _by_type(result, UNDERLINE) == []


def test_signature_region_fallback_without_line():
    text_boxes = [{"bbox": {"x": 0.1, "y": 0.80, "width": 0.12, "height": 0.02}, "text": "Sign Here", "page": 1}]
    result = build_answer_regions({}, text_boxes=text_boxes, metrics_by_page=METRICS)
    assert len(_by_type(result, SIGNATURE)) == 1


def test_freeform_whitespace():
    """A 'Comments:' label with a large blank area below -> FREEFORM."""
    text_boxes = [{"bbox": {"x": 0.1, "y": 0.30, "width": 0.18, "height": 0.02}, "text": "Comments:", "page": 1}]
    result = build_answer_regions({}, text_boxes=text_boxes, metrics_by_page=METRICS)
    ff = _by_type(result, FREEFORM)
    assert len(ff) == 1
    assert ff[0].bbox[3] >= 0.05  # tall blank band


def test_freeform_suppressed_when_blocked():
    """Text directly below the label means it is not a blank freeform area."""
    text_boxes = [
        {"bbox": {"x": 0.1, "y": 0.30, "width": 0.18, "height": 0.02}, "text": "Comments:", "page": 1},
        {"bbox": {"x": 0.1, "y": 0.34, "width": 0.40, "height": 0.02}, "text": "already filled in", "page": 1},
    ]
    result = build_answer_regions({}, text_boxes=text_boxes, metrics_by_page=METRICS)
    assert _by_type(result, FREEFORM) == []


def test_multipage_scoping():
    vf = {"underlines": [
        _feat(0.20, 0.40, 0.50, 0.004, page=1),
        _feat(0.20, 0.40, 0.50, 0.004, page=2),
    ]}
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    pages = {r.page for r in result.regions}
    assert pages == {1, 2}
    ids = {r.region_id for r in result.regions}
    assert len(ids) == 2  # identical geometry on different pages stays distinct


def test_diagnostics_shape():
    vf = {
        "underlines": [_feat(0.20, 0.40, 0.50, 0.004), _feat(0.20, 0.44, 0.50, 0.004)],
        "synthetic_underlines": [_feat(0.3, 0.6, 0.4, 0.01, source_type="dotted")],
    }
    result = build_answer_regions(vf, metrics_by_page=METRICS)
    debug = result.to_debug_dict()
    for key in ("enabled", "region_count", "regions", "counts_by_type", "merged_regions", "rejected_regions"):
        assert key in debug
    assert debug["counts_by_type"].get(MULTILINE) == 1
    assert debug["counts_by_type"].get(DOTTED) == 1
    assert len(debug["merged_regions"]) == 2  # two lines consumed by the multiline


def test_empty_inputs():
    result = build_answer_regions({}, text_boxes=[], table_cells=[], metrics_by_page=METRICS)
    assert result.regions == []
    assert result.counts_by_type == {}


# --------------------------------------------------------------------------- #
# Integration through build_anchored_mappings
# --------------------------------------------------------------------------- #
def _blank_png(tmp_path) -> Path:
    img = np.full((1000, 800, 3), 255, dtype=np.uint8)
    path = tmp_path / "page1.png"
    cv2.imwrite(str(path), img)
    return path


def _word(bid, text, x, y, w=0.12, h=0.02, page=1):
    return {"Id": bid, "BlockType": "WORD", "Text": text, "Page": page,
            "Geometry": {"BoundingBox": {"Left": x, "Top": y, "Width": w, "Height": h}}}


def _response(words):
    page = {"Id": "PAGE1", "BlockType": "PAGE", "Page": 1,
            "Geometry": {"BoundingBox": {"Left": 0, "Top": 0, "Width": 1, "Height": 1}},
            "Relationships": [{"Type": "CHILD", "Ids": [w["Id"] for w in words]}]}
    return {"Blocks": [page, *words]}


def _run(raw, parsed, image_path, enabled):
    if enabled:
        os.environ[FLAG] = "true"
    else:
        os.environ.pop(FLAG, None)
    try:
        return fae.build_anchored_mappings(raw, parsed, image_path)
    finally:
        os.environ.pop(FLAG, None)


def _signature(result):
    return sorted(
        (m.get("label"), m.get("anchoring", {}).get("anchor_type"), tuple(sorted(m["bbox"].items())))
        for m in result["mappings"]
    )


def test_integration_flag_off_is_unchanged(tmp_path):
    image_path = _blank_png(tmp_path)
    words = [_word("K1", "First Name", 0.10, 0.20), _word("V1", "", 0.55, 0.20, w=0.25)]
    raw = _response(words)
    parsed = {"field_items": [{"key": "First Name", "key_block_id": "K1", "value_block_ids": ["V1"], "value": ""}],
              "tables": [], "checkboxes": []}

    off = _run(raw, parsed, image_path, enabled=False)
    on = _run(raw, parsed, image_path, enabled=True)

    assert off["diagnostics"]["answer_region_engine"] == {"enabled": False}
    assert on["diagnostics"]["answer_region_engine"]["enabled"] is True
    # No CV features on a blank page + a real value block -> identical selection.
    assert _signature(off) == _signature(on)


def test_integration_diagnostics_present(tmp_path):
    image_path = _blank_png(tmp_path)
    words = [_word("K1", "Name", 0.10, 0.20), _word("V1", "", 0.55, 0.20, w=0.25)]
    raw = _response(words)
    parsed = {"field_items": [{"key": "Name", "key_block_id": "K1", "value_block_ids": ["V1"], "value": ""}],
              "tables": [], "checkboxes": []}
    on = _run(raw, parsed, image_path, enabled=True)
    diag = on["diagnostics"]["answer_region_engine"]
    assert "counts_by_type" in diag and "regions" in diag


def test_emit_answer_region_debug_artifact():
    vf = {
        "underlines": [
            _feat(0.20, 0.40, 0.50, 0.004),
            _feat(0.20, 0.44, 0.50, 0.004),
            _feat(0.20, 0.48, 0.50, 0.004),
        ],
        "synthetic_underlines": [_feat(0.3, 0.65, 0.4, 0.01, source_type="dotted", confidence=0.8)],
    }
    text_boxes = [{"bbox": {"x": 0.1, "y": 0.80, "width": 0.12, "height": 0.02}, "text": "Signature", "page": 1}]
    result = build_answer_regions(vf, text_boxes=text_boxes, metrics_by_page=METRICS)

    out_dir = REPO_ROOT / "output" / "phase_i_answer_region_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answer_region_debug.json").write_text(
        json.dumps(result.to_debug_dict(), indent=2), encoding="utf-8"
    )
    assert (out_dir / "answer_region_debug.json").exists()
    assert result.counts_by_type.get(MULTILINE) == 1
