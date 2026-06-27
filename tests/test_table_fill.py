"""Tests for empty input-grid cell emission (src/table_fill.py)."""
from __future__ import annotations

import os

import pytest

from src.section_detector import SectionIndex
from src.table_fill import emit_table_input_cells, table_fill_enabled


def _cell(r, c, text, x, y, w=0.15, h=0.04):
    return {
        "cell_id": f"R{r}C{c}",
        "row_index": r,
        "column_index": c,
        "row_span": 1,
        "column_span": 1,
        "text": text,
        "confidence": 90.0,
        "geometry": {"bounding_box": {"Left": x, "Top": y, "Width": w, "Height": h}},
    }


def _table(cells, rows, cols, page=1, tid="t1"):
    return {
        "table_block_id": tid,
        "page": page,
        "dimensions": {"rows": rows, "columns": cols},
        "cells": cells,
    }


def _emit(tables, existing=None):
    return emit_table_input_cells(
        tables,
        section_index=SectionIndex([]),
        page_px=lambda p: (1000, 1300),
        existing_field_boxes=existing or [],
    )


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.pop("FORM_PARSER_TABLE_FILL_ENABLED", None)
    yield
    if saved is not None:
        os.environ["FORM_PARSER_TABLE_FILL_ENABLED"] = saved
    else:
        os.environ.pop("FORM_PARSER_TABLE_FILL_ENABLED", None)


def test_enabled_by_default_and_kill_switch():
    assert table_fill_enabled() is True
    os.environ["FORM_PARSER_TABLE_FILL_ENABLED"] = "false"
    assert table_fill_enabled() is False


def test_label_column_grid_no_header_row():
    # form_3 shape: column 1 = row labels, columns 2-3 empty data cells, no
    # internal header row. Every empty data cell becomes a widget.
    cells = []
    labels = ["High School", "College", "Other"]
    for i, lab in enumerate(labels, start=1):
        y = 0.50 + i * 0.05
        cells.append(_cell(i, 1, lab, 0.10, y))
        cells.append(_cell(i, 2, "", 0.30, y))
        cells.append(_cell(i, 3, "", 0.50, y))
    out = _emit([_table(cells, 3, 3)])
    assert len(out) == 6  # 3 rows x 2 data columns
    assert all(m["field_type"] == "text" for m in out)
    assert all(m["anchoring"]["anchor_type"] == "table_cell" for m in out)
    assert all(m["render_border"] is False for m in out)
    labels_out = {m["label"] for m in out}
    assert "High School (column 2)" in labels_out
    assert "College (column 3)" in labels_out


def test_header_row_only_grid_emits_blank_rows():
    # references shape: header row + two blank input rows, no row-label column.
    cells = [
        _cell(1, 1, "Name", 0.10, 0.30),
        _cell(1, 2, "Phone", 0.40, 0.30),
        _cell(2, 1, "", 0.10, 0.35),
        _cell(2, 2, "", 0.40, 0.35),
        _cell(3, 1, "", 0.10, 0.40),
        _cell(3, 2, "", 0.40, 0.40),
    ]
    out = _emit([_table(cells, 3, 2)])
    assert len(out) == 4  # 2 blank rows x 2 columns
    labels_out = {m["label"] for m in out}
    assert "Name (row 2)" in labels_out
    assert "Phone (row 3)" in labels_out


def test_internal_header_row_used_for_column_labels():
    # Realistic internal header row: mostly filled (Degree + Year), labelling the
    # data columns of a single blank data row beneath it.
    cells = [
        _cell(1, 1, "", 0.10, 0.30),
        _cell(1, 2, "Degree", 0.40, 0.30),
        _cell(1, 3, "Year", 0.60, 0.30),
        _cell(2, 1, "College", 0.10, 0.35),
        _cell(2, 2, "", 0.40, 0.35),
        _cell(2, 3, "", 0.60, 0.35),
    ]
    out = _emit([_table(cells, 2, 3)])
    assert [m["label"] for m in out] == ["College - Degree", "College - Year"]


def test_fully_populated_table_emits_nothing():
    cells = [_cell(r, c, f"v{r}{c}", 0.1 + c * 0.2, 0.3 + r * 0.05) for r in (1, 2) for c in (1, 2)]
    assert _emit([_table(cells, 2, 2)]) == []


def test_layout_table_without_structure_skipped():
    # No header row, no label column (each column half-filled) -> not an input grid.
    cells = [
        _cell(1, 1, "a", 0.10, 0.30),
        _cell(1, 2, "", 0.40, 0.30),
        _cell(2, 1, "", 0.10, 0.35),
        _cell(2, 2, "b", 0.40, 0.35),
    ]
    assert _emit([_table(cells, 2, 2)]) == []


def test_covered_cells_skipped():
    cells = [
        _cell(1, 1, "College", 0.10, 0.35),
        _cell(1, 2, "", 0.40, 0.35, w=0.15, h=0.04),
    ]
    # An existing widget already covers the only data cell.
    existing = [(1, {"x": 0.40, "y": 0.35, "width": 0.15, "height": 0.04})]
    assert _emit([_table(cells, 2, 2)], existing=existing) == []


def test_disabled_emits_nothing():
    os.environ["FORM_PARSER_TABLE_FILL_ENABLED"] = "false"
    cells = [
        _cell(1, 1, "College", 0.10, 0.35),
        _cell(2, 1, "School", 0.10, 0.40),
        _cell(1, 2, "", 0.40, 0.35),
        _cell(2, 2, "", 0.40, 0.40),
    ]
    assert _emit([_table(cells, 2, 2)]) == []


def test_small_table_ignored():
    assert _emit([_table([_cell(1, 1, "x", 0.1, 0.3)], 1, 1)]) == []
