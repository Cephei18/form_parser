"""Validation suite for document hierarchy / section detection (Phase 2.2).

No AWS: synthetic Textract responses (heading LINE blocks + KEY_VALUE_SET fields
+ SELECTION_ELEMENTs + TABLE blocks) drive the section detector and the
section-aware anchor engine. Blank PNGs stand in for page rasters.

Covers: repeated-label disambiguation, multi-page (continuation) sections,
section boundaries, checkbox ownership, table ownership, orphan fields, and the
heading-less no-regression case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging
logging.disable(logging.CRITICAL)

from src import field_anchor_engine as fae
from src.section_detector import detect_sections, SectionIndex, qualify_label, build_hierarchy
from src.textract_parser import parse_textract_response


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _bbox(left, top, w, h):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}}


def _line(block_id, text, top, page=1, height=0.03):
    """Raw Textract LINE block (for full-response engine tests)."""
    return {"Id": block_id, "BlockType": "LINE", "Text": text, "Confidence": 99.0, "Page": page,
            "Geometry": _bbox(0.08, top, 0.5, height)}


def _nl(text, top, page=1, height=0.03):
    """Normalised line dict, the shape detect_sections() consumes directly."""
    return {"block_id": text, "text": text, "page": page,
            "bbox": {"x": 0.08, "y": top, "width": 0.5, "height": height}}


def _kv_field(page, fid, key_text, top):
    kid, kw, vid, vw = f"k{fid}", f"kw{fid}", f"v{fid}", f"vw{fid}"
    blocks = [
        {"Id": kid, "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"], "Confidence": 95.0, "Page": page,
         "Geometry": _bbox(0.10, top, 0.15, 0.02),
         "Relationships": [{"Type": "CHILD", "Ids": [kw]}, {"Type": "VALUE", "Ids": [vid]}]},
        {"Id": kw, "BlockType": "WORD", "Text": key_text, "Confidence": 99.0, "Page": page,
         "Geometry": _bbox(0.10, top, 0.12, 0.02)},
        {"Id": vid, "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"], "Confidence": 95.0, "Page": page,
         "Geometry": _bbox(0.32, top, 0.20, 0.02), "Relationships": [{"Type": "CHILD", "Ids": [vw]}]},
        {"Id": vw, "BlockType": "WORD", "Text": "John", "Confidence": 99.0, "Page": page,
         "Geometry": _bbox(0.32, top, 0.18, 0.02)},
    ]
    return blocks, [kid, kw, vid, vw]


def _selection(block_id, top, selected=True, page=1):
    return {"Id": block_id, "BlockType": "SELECTION_ELEMENT",
            "SelectionStatus": "SELECTED" if selected else "NOT_SELECTED",
            "Confidence": 95.0, "Page": page, "Geometry": _bbox(0.10, top, 0.02, 0.02)}


def _img(tmp_path: Path, name="bg.png") -> Path:
    path = tmp_path / name
    cv2.imwrite(str(path), np.full((800, 600, 3), 255, np.uint8))
    return path


def _build(blocks):
    response = {"Blocks": blocks, "DocumentMetadata": {"Pages": max((b.get("Page", 1) for b in blocks), default=1)}}
    return response, parse_textract_response(response)


def _fields(mappings):
    return [m for m in mappings if m.get("field_type") == "text" or m.get("field_type") == "multiline"]


# --------------------------------------------------------------------------- #
# Unit: detector + ownership
# --------------------------------------------------------------------------- #
def test_detect_named_section_types():
    lines = [
        _nl("Applicant Details", 0.10),
        _nl("Guardian Details", 0.30),
        _nl("Nominee Details", 0.50),
        _nl("FATCA Declaration", 0.70),
    ]
    sections = detect_sections(lines, field_key_slugs=set(), metrics_by_page={1: {"line_height": 0.02}})
    assert [s["type"] for s in sections] == ["applicant", "guardian", "nominee", "fatca"]
    # bands are contiguous in reading order
    assert sections[0]["y_end"] == sections[1]["y_start"]


def test_field_label_is_not_a_heading():
    # A line that is also a field key must never become a section.
    lines = [_nl("Name", 0.10, height=0.02)]
    sections = detect_sections(lines, field_key_slugs={"name"}, metrics_by_page={1: {"line_height": 0.02}})
    assert sections == []


def test_section_index_owner_and_root():
    sections = detect_sections(
        [_nl("Applicant Details", 0.10), _nl("Nominee Details", 0.50)],
        set(), {1: {"line_height": 0.02}},
    )
    idx = SectionIndex(sections)
    assert idx.owner(1, 0.05)["type"] == "document"   # above first heading -> root
    assert idx.owner(1, 0.20)["type"] == "applicant"
    assert idx.owner(1, 0.60)["type"] == "nominee"
    assert qualify_label(idx.owner(1, 0.20), "Name") == "Applicant Details › Name"
    assert qualify_label(idx.root, "Name") == "Name"   # root never prefixes


# --------------------------------------------------------------------------- #
# Repeated labels are disambiguated
# --------------------------------------------------------------------------- #
def test_repeated_labels_disambiguated(tmp_path):
    blocks = [_line("h1", "Applicant Details", 0.10)]
    f, _ = _kv_field(1, "a", "Name", 0.15); blocks += f
    blocks += [_line("h2", "Guardian Details", 0.30)]
    f, _ = _kv_field(1, "b", "Name", 0.35); blocks += f
    blocks += [_line("h3", "Nominee Details", 0.50)]
    f, _ = _kv_field(1, "c", "Name", 0.55); blocks += f

    response, parsed = _build(blocks)
    out = fae.build_anchored_mappings(response, parsed, _img(tmp_path))
    fields = _fields(out["mappings"])

    assert len(fields) == 3
    # raw label is identical, but section + qualified_label disambiguate them
    assert {m["label"] for m in fields} == {"Name"}
    by_type = {m["section"]["type"]: m["qualified_label"] for m in fields}
    assert by_type == {
        "applicant": "Applicant Details › Name",
        "guardian": "Guardian Details › Name",
        "nominee": "Nominee Details › Name",
    }
    assert out["diagnostics"]["hierarchy"]["section_count"] == 3
    assert out["diagnostics"]["hierarchy"]["orphan_field_count"] == 0


# --------------------------------------------------------------------------- #
# Multi-page: section continues onto the next page (no heading on page 2)
# --------------------------------------------------------------------------- #
def test_section_continues_across_pages(tmp_path):
    blocks = [_line("h1", "Applicant Details", 0.10, page=1)]
    f, _ = _kv_field(1, "p1", "Name", 0.15); blocks += f
    f, _ = _kv_field(2, "p2", "Mobile", 0.10); blocks += f  # page 2, no heading above it

    response, parsed = _build(blocks)
    page_images = {1: _img(tmp_path, "p1.png"), 2: _img(tmp_path, "p2.png")}
    out = fae.build_anchored_mappings(response, parsed, page_images[1], page_images=page_images)
    fields = _fields(out["mappings"])

    page2 = next(m for m in fields if m["page"] == 2)
    assert page2["section"]["type"] == "applicant", "page-2 field inherits the page-1 section"
    assert page2["qualified_label"] == "Applicant Details › Mobile"


# --------------------------------------------------------------------------- #
# Section boundaries within a page
# --------------------------------------------------------------------------- #
def test_section_boundaries_assign_by_band(tmp_path):
    blocks = [_line("h1", "Applicant Details", 0.10)]
    f, _ = _kv_field(1, "a", "PAN", 0.20); blocks += f
    f, _ = _kv_field(1, "b", "Mobile", 0.30); blocks += f       # still under Applicant
    blocks += [_line("h2", "Bank Details", 0.40)]
    f, _ = _kv_field(1, "c", "PAN", 0.50); blocks += f          # under Bank

    response, parsed = _build(blocks)
    out = fae.build_anchored_mappings(response, parsed, _img(tmp_path))
    fields = _fields(out["mappings"])

    pan_sections = sorted(m["section"]["type"] for m in fields if m["label"] == "PAN")
    assert pan_sections == ["applicant", "bank"], "the two PANs land in different sections"


# --------------------------------------------------------------------------- #
# Checkbox ownership inherits section
# --------------------------------------------------------------------------- #
def test_checkbox_inherits_section(tmp_path):
    blocks = [
        _line("h1", "Holding Mode", 0.10),
        {"Id": "page", "BlockType": "PAGE", "Page": 1, "Geometry": _bbox(0, 0, 1, 1),
         "Relationships": [{"Type": "CHILD", "Ids": ["h1", "s1", "s2"]}]},
        _selection("s1", 0.15, selected=True),
        _selection("s2", 0.20, selected=False),
    ]
    response, parsed = _build(blocks)
    out = fae.build_anchored_mappings(response, parsed, _img(tmp_path))
    checkboxes = [m for m in out["mappings"] if m["field_type"] == "checkbox"]

    assert len(checkboxes) == 2
    assert all(c["section"]["type"] == "holding_mode" for c in checkboxes)
    assert all(c["qualified_label"].startswith("Holding Mode ›") for c in checkboxes)


# --------------------------------------------------------------------------- #
# Table ownership inherits section
# --------------------------------------------------------------------------- #
def test_table_inherits_section(tmp_path):
    blocks = [
        _line("h1", "Bank Account Details", 0.10),
        {"Id": "tbl", "BlockType": "TABLE", "Page": 1, "Confidence": 90.0, "Geometry": _bbox(0.1, 0.15, 0.8, 0.2),
         "Relationships": [{"Type": "CHILD", "Ids": ["c1"]}]},
        {"Id": "c1", "BlockType": "CELL", "RowIndex": 1, "ColumnIndex": 1, "Page": 1,
         "Geometry": _bbox(0.1, 0.15, 0.2, 0.03), "Relationships": [{"Type": "CHILD", "Ids": ["cw1"]}]},
        {"Id": "cw1", "BlockType": "WORD", "Text": "Acc No", "Page": 1, "Geometry": _bbox(0.1, 0.15, 0.15, 0.02)},
    ]
    response, parsed = _build(blocks)
    out = fae.build_anchored_mappings(response, parsed, _img(tmp_path))
    table_sections = out["diagnostics"]["hierarchy"]["table_sections"]
    assert table_sections["tbl"]["type"] == "bank"


# --------------------------------------------------------------------------- #
# Orphans + heading-less no-regression
# --------------------------------------------------------------------------- #
def test_orphan_field_without_heading(tmp_path):
    f, _ = _kv_field(1, "a", "Name", 0.10)
    response, parsed = _build(f)
    out = fae.build_anchored_mappings(response, parsed, _img(tmp_path))
    field = _fields(out["mappings"])[0]
    assert field["section"]["type"] == "document"
    assert field["qualified_label"] == "Name"            # no prefix at root
    assert out["diagnostics"]["hierarchy"]["section_count"] == 0
    assert out["diagnostics"]["hierarchy"]["orphan_field_count"] == 1


def test_hierarchy_tree_built(tmp_path):
    blocks = [_line("h1", "Applicant Details", 0.10)]
    f, _ = _kv_field(1, "a", "Name", 0.15); blocks += f
    response, parsed = _build(blocks)
    out = fae.build_anchored_mappings(response, parsed, _img(tmp_path))
    tree = build_hierarchy(out["sections"], out["mappings"])
    assert tree["section_count"] == 1
    applicant = next(n for n in tree["sections"] if n["type"] == "applicant")
    assert any(child["label"] == "Name" for child in applicant["children"])
