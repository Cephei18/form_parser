"""Phase J — table intelligence validation.

Engine-level tests drive ``build_tables`` with synthetic normalised cell lists
(the same shape ``field_anchor_engine._table_cells`` produces) for every
required scenario: input table, employment history, nominee table, matrix,
layout table, empty input cells, repeating rows, multipage and diagnostics.
Integration tests confirm the ``FORM_PARSER_TABLE_INTELLIGENCE_ENABLED`` flag is
fully reversible and emit validation artifacts.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src import field_anchor_engine as fae
from src.table_intelligence import (
    INPUT_TABLE,
    LAYOUT_TABLE,
    LINE_ITEM,
    MATRIX,
    REPEATING_GROUP,
    build_tables,
)

FLAG = "FORM_PARSER_TABLE_INTELLIGENCE_ENABLED"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _cell(table_id, row, col, text, x, y, w=0.18, h=0.04, page=1):
    return {
        "table_id": table_id, "cell_id": f"{table_id}_r{row}c{col}", "page": page,
        "row_index": row, "column_index": col, "row_span": 1, "column_span": 1,
        "text": text, "confidence": 90.0,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
    }


def _grid(table_id, header, body_rows, page=1, y0=0.30, col_w=0.20, row_h=0.05):
    """Build a header row + body rows of cells. '' = empty cell."""
    cells = []
    rows = [header] + body_rows
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cells.append(_cell(table_id, r, c, text, x=0.1 + c * col_w, y=y0 + r * row_h,
                               w=col_w * 0.95, h=row_h * 0.9, page=page))
    return cells


def _one(result, table_id="t1"):
    return next(t for t in result.tables if t.table_id == table_id)


# --------------------------------------------------------------------------- #
# Engine-level scenarios
# --------------------------------------------------------------------------- #
def test_input_table_school_degree_year():
    cells = _grid("t1", ["School", "Degree", "Year"], [["", "", ""]])
    result = build_tables(cells)
    t = _one(result)
    assert t.table_type == INPUT_TABLE
    assert t.metadata["header_rows"] == [0]
    assert sum(1 for c in t.cells if c.is_input) == 3


def test_employment_history_table():
    cells = _grid("t1", ["Employer", "Start", "End"], [["", "", ""], ["", "", ""]])
    result = build_tables(cells)
    t = _one(result)
    assert t.table_type == INPUT_TABLE
    assert sum(1 for c in t.cells if c.is_input) == 6


def test_nominee_table():
    cells = _grid("t1", ["Nominee Name", "DOB", "Relationship"], [["", "", ""]])
    result = build_tables(cells)
    t = _one(result)
    assert t.table_type == INPUT_TABLE
    # Header cells are labels, not inputs.
    assert all(not c.is_input for c in t.cells if c.metadata["is_header"])


def test_matrix_table():
    cells = [
        _cell("t1", 0, 0, "Male", 0.1, 0.3), _cell("t1", 0, 1, "[ ]", 0.3, 0.3, w=0.05),
        _cell("t1", 1, 0, "Female", 0.1, 0.36), _cell("t1", 1, 1, "[ ]", 0.3, 0.36, w=0.05),
    ]
    result = build_tables(cells)
    t = _one(result)
    assert t.table_type == MATRIX
    assert any(c.is_input for c in t.cells)


def test_layout_table_all_filled():
    cells = _grid("t1", ["A", "B"], [["C", "D"], ["E", "F"]])
    result = build_tables(cells)
    t = _one(result)
    assert t.table_type == LAYOUT_TABLE
    assert all(not c.is_input for c in t.cells)


def test_line_item_table():
    cells = _grid("t1", ["Description", "Qty", "Amount"], [["", "", ""], ["", "", ""]])
    result = build_tables(cells)
    t = _one(result)
    assert t.table_type == LINE_ITEM
    assert t.metadata["has_line_item_cue"] is True


def test_empty_input_cells_flagged():
    cells = _grid("t1", ["School", "Degree", "Year"], [["", "", ""]])
    result = build_tables(cells)
    inputs = [c for c in result.input_cells]
    assert len(inputs) == 3
    assert all(c.is_input and c.metadata["is_empty"] for c in inputs)
    assert all(c.metadata["cell_id"] for c in inputs)


def test_repeating_rows():
    # A single label column with Name/DOB/Relationship repeated twice.
    rows = ["Name", "DOB", "Relationship", "Name", "DOB", "Relationship"]
    cells = [_cell("t1", r, 0, text, 0.1, 0.30 + r * 0.05) for r, text in enumerate(rows)]
    result = build_tables(cells)
    t = _one(result)
    assert t.table_type == REPEATING_GROUP
    assert t.metadata["repeating_period"] == 3
    assert t.metadata["group_count"] == 2
    assert any(g["table_id"] == "t1" and g["group_size"] == 3 for g in result.repeating_groups)


def test_repeating_group_from_sections():
    sections = [
        {"section_id": "s1", "title": "Nominee 1"},
        {"section_id": "s2", "title": "Nominee 2"},
        {"section_id": "s3", "title": "Nominee 3"},
    ]
    result = build_tables([], sections=sections)
    section_groups = [g for g in result.repeating_groups if g["source"] == "section_index"]
    assert len(section_groups) == 1
    assert section_groups[0]["instance_count"] == 3


def test_multipage_tables():
    cells = _grid("t1", ["School", "Degree", "Year"], [["", "", ""]], page=1)
    cells += _grid("t2", ["Employer", "Start", "End"], [["", "", ""]], page=2, y0=0.10)
    result = build_tables(cells)
    pages = {t.page for t in result.tables}
    assert pages == {1, 2}
    assert all(t.table_type == INPUT_TABLE for t in result.tables)


def test_diagnostics_shape():
    cells = _grid("t1", ["School", "Degree", "Year"], [["", "", ""]])
    result = build_tables(cells)
    debug = result.to_debug_dict()
    for key in ("enabled", "table_count", "tables_detected", "table_types", "input_cells",
                "repeating_groups", "rejected_tables"):
        assert key in debug
    assert debug["table_types"].get(INPUT_TABLE) == 1
    assert len(debug["input_cells"]) == 3


def test_empty_input():
    result = build_tables([])
    assert result.tables == []
    assert result.input_cells == []


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


def _table_block(parsed_cells):
    cells = []
    for c in parsed_cells:
        cells.append({
            "cell_id": c["cell_id"], "row_index": c["row_index"], "column_index": c["column_index"],
            "row_span": 1, "column_span": 1, "text": c["text"], "confidence": 90.0,
            "geometry": {"bounding_box": c["bbox"]},
        })
    bx = min(c["bbox"]["x"] for c in parsed_cells)
    by = min(c["bbox"]["y"] for c in parsed_cells)
    return {"table_block_id": "t1", "page": 1, "cells": cells,
            "geometry": {"bounding_box": {"x": bx, "y": by, "width": 0.6, "height": 0.2}}}


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
    raw = {"Blocks": [
        {"Id": "PAGE1", "BlockType": "PAGE", "Page": 1,
         "Geometry": {"BoundingBox": {"Left": 0, "Top": 0, "Width": 1, "Height": 1}},
         "Relationships": [{"Type": "CHILD", "Ids": ["K1", "V1"]}]},
        *words,
    ]}
    table_cells = _grid("t1", ["School", "Degree", "Year"], [["", "", ""]])
    parsed = {
        "field_items": [{"key": "First Name", "key_block_id": "K1", "value_block_ids": ["V1"], "value": ""}],
        "tables": [_table_block(table_cells)], "checkboxes": [],
    }

    off = _run(raw, parsed, image_path, enabled=False)
    on = _run(raw, parsed, image_path, enabled=True)

    assert off["diagnostics"]["table_intelligence"] == {"enabled": False}
    diag = on["diagnostics"]["table_intelligence"]
    assert diag["enabled"] is True
    assert diag["table_types"].get(INPUT_TABLE) == 1
    assert len(diag["input_cells"]) == 3
    # The field's selection is unaffected here (its value block beats far-away
    # table cells), so OFF and ON agree on the rendered widgets.
    assert _signature(off) == _signature(on)


def test_emit_validation_artifacts():
    before_cells = _grid("t1", ["School", "Degree", "Year"], [["", "", ""]])
    by_table = defaultdict(list)
    for c in before_cells:
        by_table[c["table_id"]].append(c)
    result = build_tables(before_cells)

    out_dir = REPO_ROOT / "output" / "phase_j_table_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name, payload):
        (out_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _dump("before_tables.json", {"tables": {tid: cs for tid, cs in by_table.items()}})
    _dump("after_tables.json", {"tables": [t.to_dict() for t in result.tables]})
    _dump("table_debug.json", result.to_debug_dict())
    _dump("validation_metrics.json", {
        "tables_detected": len(result.tables),
        "table_types": result.to_debug_dict()["table_types"],
        "input_cell_count": len(result.input_cells),
        "repeating_group_count": len(result.repeating_groups),
        "rejected_table_count": len(result.rejected_tables),
    })

    for name in ("before_tables.json", "after_tables.json", "table_debug.json", "validation_metrics.json"):
        assert (out_dir / name).exists()
    assert len(result.input_cells) == 3
